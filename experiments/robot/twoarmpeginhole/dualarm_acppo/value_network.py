"""
Centralized Value Network for Dual-Arm ACPPO training.

Implements a centralized value function that takes the global state
(combining observations from all agents) and predicts state values
for training the multi-agent policy.

Key differences from single-arm ACPPO:
- Handles 14-dim bimanual proprio
- 3 images per agent (agentview + 2 wrist cameras)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import DualArmACPPOConfig, DUALARM_ACPPO_CONSTANTS


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
        """
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        return self.encoder(x)


class DualArmCentralizedValueNetwork(nn.Module):
    """
    Centralized value function for Dual-Arm ACPPO.
    
    Takes global state (combining both agents' observations) and
    outputs value estimates. Handles 14-dim bimanual proprio and
    3 images per agent (agentview + left_wrist + right_wrist).
    """
    
    def __init__(
        self,
        cfg: DualArmACPPOConfig,
        num_agents: int = 2,
        proprio_dim: int = 14,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_images: bool = True,
        image_encoder_dim: int = 256,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = num_agents
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.use_images = use_images
        
        proprio_total_dim = proprio_dim * cfg.history_length
        num_images = 3 * cfg.history_length
        
        if use_images:
            self.image_encoder = ImageEncoder(output_dim=image_encoder_dim)
            image_feature_dim = image_encoder_dim * num_images
        else:
            self.image_encoder = None
            image_feature_dim = 0
        
        input_dim = proprio_total_dim + image_feature_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        last_linear = list(self.value_head.modules())[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.orthogonal_(last_linear.weight, gain=0.01)
    
    def encode_global_state(
        self,
        proprio_state: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Encode global state into feature vector.
        """
        batch_size = proprio_state.shape[0]
        
        if proprio_state.dim() == 3:
            proprio_flat = proprio_state.reshape(batch_size, -1)
        else:
            proprio_flat = proprio_state
        
        if self.use_images and images is not None and len(images) > 0:
            image_features = []
            for img in images:
                if img.dim() == 5:
                    B, T, H, W, C = img.shape
                    img = img.reshape(B * T, H, W, C)
                    feat = self.image_encoder(img)
                    feat = feat.reshape(B, -1)
                else:
                    feat = self.image_encoder(img)
                image_features.append(feat)
            
            combined_image_features = torch.cat(image_features, dim=-1)
            global_features = torch.cat([proprio_flat, combined_image_features], dim=-1)
        else:
            global_features = proprio_flat
        
        return global_features
    
    def forward(
        self,
        proprio_state: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute state value from global state.
        """
        global_features = self.encode_global_state(proprio_state, images)
        
        x = self.input_projection(global_features)
        
        for block in self.residual_blocks:
            x = block(x)
        
        value = self.value_head(x).squeeze(-1)
        
        return value


class LightweightDualArmCritic(nn.Module):
    """
    Lightweight critic using only proprio for faster training.
    """
    
    def __init__(
        self,
        cfg: DualArmACPPOConfig,
        proprio_dim: int = 14,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        
        input_dim = proprio_dim * cfg.history_length
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        last_linear = list(self.mlp.modules())[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.orthogonal_(last_linear.weight, gain=0.01)
    
    def forward(
        self,
        proprio_state: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = proprio_state.shape[0]
        
        if proprio_state.dim() == 3:
            proprio_flat = proprio_state.reshape(batch_size, -1)
        else:
            proprio_flat = proprio_state
        
        value = self.mlp(proprio_flat).squeeze(-1)
        
        return value


def create_value_network(
    cfg: DualArmACPPOConfig,
    device: torch.device,
    use_images: bool = False,
    lightweight: bool = True,
) -> nn.Module:
    """
    Factory function to create value network.
    """
    proprio_dim = DUALARM_ACPPO_CONSTANTS["MODEL_PROPRIO_DIM"]
    
    if lightweight:
        critic = LightweightDualArmCritic(
            cfg=cfg,
            proprio_dim=proprio_dim,
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
        )
    else:
        critic = DualArmCentralizedValueNetwork(
            cfg=cfg,
            num_agents=DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            proprio_dim=proprio_dim,
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
            use_images=use_images,
        )
    
    critic = critic.to(device)
    
    return critic
