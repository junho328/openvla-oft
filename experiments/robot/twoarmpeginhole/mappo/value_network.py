"""
Centralized Value Network for MAPPO training.

Implements a centralized value function that takes the global state
(combining observations from all agents) and predicts state values
for training the multi-agent policy.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import MAPPOConfig, TWOARM_MAPPO_CONSTANTS


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ImageEncoder(nn.Module):
    """
    Simple CNN encoder for processing images for value function.
    
    Uses a lightweight architecture to extract features from images
    without the overhead of the full VLA vision backbone.
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        input_channels: int = 3,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 28 -> 14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 14 -> 7
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to feature vector.
        
        Args:
            x: Image tensor (B, C, H, W) or (B, H, W, C)
            
        Returns:
            Feature tensor (B, output_dim)
        """
        # Handle different input formats
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return self.encoder(x)


class CentralizedValueNetwork(nn.Module):
    """
    Centralized value function for MAPPO.
    
    Takes global state (combining both agents' observations) and
    outputs value estimates for each agent. In MAPPO, we typically
    use a centralized critic that has access to all agents' information
    during training, while actors only see local observations.
    
    Architecture:
    1. Encode images from all cameras
    2. Encode proprioceptive states from all agents
    3. Combine encodings through MLP
    4. Output value for the joint state
    """
    
    def __init__(
        self,
        cfg: MAPPOConfig,
        num_agents: int = 2,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_images: bool = True,
        image_encoder_dim: int = 256,
    ):
        """
        Initialize centralized value network.
        
        Args:
            cfg: MAPPO configuration
            num_agents: Number of agents
            proprio_dim: Proprioceptive state dimension per agent
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of residual blocks
            use_images: Whether to use images in value function
            image_encoder_dim: Output dimension of image encoder
        """
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = num_agents
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.use_images = use_images
        
        # Calculate input dimension
        # Proprio: proprio_dim * num_agents * history_length
        proprio_total_dim = proprio_dim * num_agents * cfg.history_length
        
        # Images: We'll encode each image and concatenate
        # (front + wrist * num_agents) * history_length
        num_images = (1 + num_agents) * cfg.history_length
        
        if use_images:
            self.image_encoder = ImageEncoder(output_dim=image_encoder_dim)
            image_feature_dim = image_encoder_dim * num_images
        else:
            self.image_encoder = None
            image_feature_dim = 0
        
        # Proprio encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Combine encoders
        combined_dim = hidden_dim + image_feature_dim
        
        # Main value network
        self.input_proj = nn.Linear(combined_dim, hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Value head - single value for joint state
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Per-agent value heads (optional, for decomposed value)
        self.per_agent_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_agents)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_global_state(
        self,
        front_images: Optional[torch.Tensor] = None,
        wrist_images: Optional[List[torch.Tensor]] = None,
        proprio_states: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode global state from all observations.
        
        Args:
            front_images: Front camera images (B, T, H, W, C) where T is history
            wrist_images: List of wrist images per agent (B, T, H, W, C)
            proprio_states: List of proprio states per agent (B, T, proprio_dim)
            
        Returns:
            Encoded global state (B, hidden_dim)
        """
        batch_size = proprio_states[0].shape[0]
        device = proprio_states[0].device
        
        features = []
        
        # Encode images if used
        if self.use_images and self.image_encoder is not None:
            image_features = []
            
            # Encode front images
            if front_images is not None:
                # Flatten batch and time dimensions
                B, T = front_images.shape[:2]
                front_flat = front_images.reshape(B * T, *front_images.shape[2:])
                front_encoded = self.image_encoder(front_flat.float() / 255.0)
                front_encoded = front_encoded.reshape(B, T, -1)
                image_features.append(front_encoded.reshape(B, -1))
            
            # Encode wrist images for each agent
            if wrist_images is not None:
                for wrist_imgs in wrist_images:
                    B, T = wrist_imgs.shape[:2]
                    wrist_flat = wrist_imgs.reshape(B * T, *wrist_imgs.shape[2:])
                    wrist_encoded = self.image_encoder(wrist_flat.float() / 255.0)
                    wrist_encoded = wrist_encoded.reshape(B, T, -1)
                    image_features.append(wrist_encoded.reshape(B, -1))
            
            if image_features:
                features.append(torch.cat(image_features, dim=-1))
        
        # Encode proprio states
        # Concatenate all agents' proprio states
        proprio_concat = []
        for proprio in proprio_states:
            # proprio: (B, T, proprio_dim) -> (B, T * proprio_dim)
            proprio_concat.append(proprio.reshape(batch_size, -1))
        
        proprio_all = torch.cat(proprio_concat, dim=-1)  # (B, T * proprio_dim * num_agents)
        proprio_encoded = self.proprio_encoder(proprio_all)
        features.append(proprio_encoded)
        
        # Combine all features
        combined = torch.cat(features, dim=-1)
        
        # Project to hidden dimension
        hidden = self.input_proj(combined)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            hidden = block(hidden)
        
        return hidden
    
    def forward(
        self,
        front_images: Optional[torch.Tensor] = None,
        wrist_images: Optional[List[torch.Tensor]] = None,
        proprio_states: List[torch.Tensor] = None,
        return_per_agent: bool = False,
    ) -> torch.Tensor:
        """
        Compute value for global state.
        
        Args:
            front_images: Front camera images with history
            wrist_images: List of wrist images per agent with history
            proprio_states: List of proprio states per agent with history
            return_per_agent: If True, return per-agent values
            
        Returns:
            Value tensor (B,) or (B, num_agents) if return_per_agent
        """
        # Encode global state
        hidden = self.encode_global_state(
            front_images=front_images,
            wrist_images=wrist_images,
            proprio_states=proprio_states,
        )
        
        # Compute joint value
        joint_value = self.value_head(hidden).squeeze(-1)
        
        if return_per_agent:
            # Compute per-agent values
            agent_values = []
            for head in self.per_agent_value_heads:
                agent_values.append(head(hidden).squeeze(-1))
            return joint_value, torch.stack(agent_values, dim=-1)
        
        return joint_value


class LightweightCentralizedCritic(nn.Module):
    """
    Lightweight centralized critic using only proprioceptive states.
    
    This is a simpler version that doesn't use image encoders,
    making it faster to train and compute.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        proprio_dim: int = 8,
        history_length: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        
        # Input: proprio from all agents with history
        input_dim = proprio_dim * num_agents * history_length
        
        # MLP layers
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        
        self.mlp = nn.Sequential(*layers)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        proprio_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute value from proprioceptive states.
        
        Args:
            proprio_states: List of proprio states per agent (B, T, proprio_dim)
            
        Returns:
            Value tensor (B,)
        """
        batch_size = proprio_states[0].shape[0]
        
        # Concatenate all proprio states
        proprio_concat = []
        for proprio in proprio_states:
            proprio_concat.append(proprio.reshape(batch_size, -1))
        
        x = torch.cat(proprio_concat, dim=-1)
        
        # Forward through MLP
        hidden = self.mlp(x)
        
        # Get value
        value = self.value_head(hidden).squeeze(-1)
        
        return value


def create_value_network(
    cfg: MAPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """
    Factory function to create the appropriate value network.
    
    Args:
        cfg: MAPPO configuration
        device: Target device
        
    Returns:
        Value network module
    """
    if cfg.use_global_state:
        # Full centralized value network with images
        value_net = CentralizedValueNetwork(
            cfg=cfg,
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            proprio_dim=TWOARM_MAPPO_CONSTANTS["PROPRIO_DIM"],
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
            use_images=True,
        )
    else:
        # Lightweight critic with proprio only
        value_net = LightweightCentralizedCritic(
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            proprio_dim=TWOARM_MAPPO_CONSTANTS["PROPRIO_DIM"],
            history_length=cfg.history_length,
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
        )
    
    return value_net.to(device)
