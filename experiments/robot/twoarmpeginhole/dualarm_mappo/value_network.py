"""
Centralized Value Network for Dual-Arm MAPPO training.

Implements a centralized value function that takes the global state
(combining observations from all agents) and predicts state values
for training the multi-agent policy.

Key differences from single-arm:
- Handles 14-dim bimanual proprio
- 3 images per agent (agentview + 2 wrist cameras)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import DualArmMAPPOConfig, DUALARM_MAPPO_CONSTANTS


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
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return self.encoder(x)


class DualArmCentralizedValueNetwork(nn.Module):
    """
    Centralized value function for Dual-Arm MAPPO.
    
    Takes global state (combining both agents' observations) and
    outputs value estimates. Handles 14-dim bimanual proprio and
    3 images per agent (agentview + left_wrist + right_wrist).
    
    Architecture:
    1. Encode images from all cameras (agentview + 2 wrist)
    2. Encode 14-dim proprioceptive state
    3. Combine encodings through MLP
    4. Output value for the joint state
    """
    
    def __init__(
        self,
        cfg: DualArmMAPPOConfig,
        num_agents: int = 2,
        proprio_dim: int = 14,  # 14-dim bimanual proprio
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_images: bool = True,
        image_encoder_dim: int = 256,
    ):
        """
        Initialize centralized value network.
        
        Args:
            cfg: Dual-Arm MAPPO configuration
            num_agents: Number of agents
            proprio_dim: Proprioceptive state dimension (14 for bimanual)
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
        # Proprio: proprio_dim * history_length (full bimanual proprio)
        proprio_total_dim = proprio_dim * cfg.history_length
        
        # Images: (agentview + left_wrist + right_wrist) * history_length = 3 * history_length
        num_images = 3 * cfg.history_length
        
        if use_images:
            self.image_encoder = ImageEncoder(output_dim=image_encoder_dim)
            image_feature_dim = image_encoder_dim * num_images
        else:
            self.image_encoder = None
            image_feature_dim = 0
        
        # Proprio encoder (14-dim bimanual)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
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
        agentview_images: Optional[torch.Tensor] = None,
        left_wrist_images: Optional[torch.Tensor] = None,
        right_wrist_images: Optional[torch.Tensor] = None,
        proprio_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode global state from all observations.
        
        Args:
            agentview_images: Agent view images (B, T, H, W, C) where T is history
            left_wrist_images: Left wrist images (B, T, H, W, C)
            right_wrist_images: Right wrist images (B, T, H, W, C)
            proprio_state: Full 14-dim bimanual proprio (B, T, 14)
            
        Returns:
            Encoded global state (B, hidden_dim)
        """
        batch_size = proprio_state.shape[0]
        device = proprio_state.device
        
        features = []
        
        # Encode images if used
        if self.use_images and self.image_encoder is not None:
            image_features = []
            
            # Encode agentview images
            if agentview_images is not None:
                B, T = agentview_images.shape[:2]
                agentview_flat = agentview_images.reshape(B * T, *agentview_images.shape[2:])
                agentview_encoded = self.image_encoder(agentview_flat.float() / 255.0)
                agentview_encoded = agentview_encoded.reshape(B, T, -1)
                image_features.append(agentview_encoded.reshape(B, -1))
            
            # Encode left wrist images
            if left_wrist_images is not None:
                B, T = left_wrist_images.shape[:2]
                left_flat = left_wrist_images.reshape(B * T, *left_wrist_images.shape[2:])
                left_encoded = self.image_encoder(left_flat.float() / 255.0)
                left_encoded = left_encoded.reshape(B, T, -1)
                image_features.append(left_encoded.reshape(B, -1))
            
            # Encode right wrist images
            if right_wrist_images is not None:
                B, T = right_wrist_images.shape[:2]
                right_flat = right_wrist_images.reshape(B * T, *right_wrist_images.shape[2:])
                right_encoded = self.image_encoder(right_flat.float() / 255.0)
                right_encoded = right_encoded.reshape(B, T, -1)
                image_features.append(right_encoded.reshape(B, -1))
            
            if image_features:
                features.append(torch.cat(image_features, dim=-1))
        
        # Encode proprio state (full 14-dim bimanual)
        proprio_flat = proprio_state.reshape(batch_size, -1)
        proprio_encoded = self.proprio_encoder(proprio_flat)
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
        agentview_images: Optional[torch.Tensor] = None,
        left_wrist_images: Optional[torch.Tensor] = None,
        right_wrist_images: Optional[torch.Tensor] = None,
        proprio_state: torch.Tensor = None,
        return_per_agent: bool = False,
    ) -> torch.Tensor:
        """
        Compute value for global state.
        
        Args:
            agentview_images: Agent view images with history
            left_wrist_images: Left wrist images with history
            right_wrist_images: Right wrist images with history
            proprio_state: Full 14-dim bimanual proprio with history
            return_per_agent: If True, return per-agent values
            
        Returns:
            Value tensor (B,) or (B, num_agents) if return_per_agent
        """
        hidden = self.encode_global_state(
            agentview_images=agentview_images,
            left_wrist_images=left_wrist_images,
            right_wrist_images=right_wrist_images,
            proprio_state=proprio_state,
        )
        
        joint_value = self.value_head(hidden).squeeze(-1)
        
        if return_per_agent:
            agent_values = []
            for head in self.per_agent_value_heads:
                agent_values.append(head(hidden).squeeze(-1))
            return joint_value, torch.stack(agent_values, dim=-1)
        
        return joint_value


class LightweightDualArmCritic(nn.Module):
    """
    Lightweight centralized critic using only proprioceptive states.
    
    Uses full 14-dim bimanual proprio without images for faster computation.
    """
    
    def __init__(
        self,
        proprio_dim: int = 14,  # 14-dim bimanual proprio
        history_length: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        
        # Input: full bimanual proprio with history
        input_dim = proprio_dim * history_length
        
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
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        proprio_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value from full bimanual proprio state.
        
        Args:
            proprio_state: Full 14-dim bimanual proprio (B, T, 14)
            
        Returns:
            Value tensor (B,)
        """
        batch_size = proprio_state.shape[0]
        
        x = proprio_state.reshape(batch_size, -1)
        
        hidden = self.mlp(x)
        
        value = self.value_head(hidden).squeeze(-1)
        
        return value


def create_value_network(
    cfg: DualArmMAPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """
    Factory function to create the appropriate value network.
    
    Args:
        cfg: Dual-Arm MAPPO configuration
        device: Target device
        
    Returns:
        Value network module
    """
    if cfg.use_global_state:
        value_net = DualArmCentralizedValueNetwork(
            cfg=cfg,
            num_agents=DUALARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            proprio_dim=cfg.model_proprio_dim,  # 14-dim bimanual
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
            use_images=True,
        )
    else:
        value_net = LightweightDualArmCritic(
            proprio_dim=cfg.model_proprio_dim,  # 14-dim bimanual
            history_length=cfg.history_length,
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
        )
    
    return value_net.to(device)
