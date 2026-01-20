"""
Value Network for ACPPO training.

For ACPPO, we use per-agent value functions:
- V^(1)(s_t) for agent 0: Value based on state only
- V^(2)([s_t, b_t^(2)]) for agent 1: Value based on state + estimated action dist from agent 0

The value functions are implemented as part of the VLA policy (shared backbone with Value Head).
This module provides additional standalone value networks if needed.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import ACPPOConfig, TWOARM_ACPPO_CONSTANTS


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
    """Simple CNN encoder for processing images for value function."""
    
    def __init__(
        self,
        output_dim: int = 256,
        input_channels: int = 3,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        return self.encoder(x)


class PerAgentValueNetwork(nn.Module):
    """
    Per-agent value function for ACPPO.
    
    For agent i:
    - Takes state s_t and belief b_t^(i) as input
    - b_t^(i) contains information about previous agents' actions
    
    Agent 0: V^(1)(s_t) - only state
    Agent 1: V^(2)([s_t, b_t^(2)]) - state + estimated action dist from agent 0
    """
    
    def __init__(
        self,
        agent_idx: int,
        proprio_dim: int = 8,
        action_dist_dim: int = 48,  # For agent 1: mu + sigma from agent 0
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_images: bool = False,
        image_encoder_dim: int = 256,
        history_length: int = 2,
    ):
        super().__init__()
        
        self.agent_idx = agent_idx
        self.proprio_dim = proprio_dim
        self.action_dist_dim = action_dist_dim
        self.hidden_dim = hidden_dim
        self.use_images = use_images
        self.history_length = history_length
        
        # Input dimension depends on agent
        if agent_idx == 0:
            # Agent 0: only proprio (no action dist from previous agent)
            input_dim = proprio_dim * history_length
        else:
            # Agent 1+: proprio + action distribution from previous agents
            input_dim = proprio_dim * history_length + action_dist_dim
        
        # Image encoder (optional)
        if use_images:
            self.image_encoder = ImageEncoder(output_dim=image_encoder_dim)
            input_dim += image_encoder_dim * (1 + 1) * history_length  # front + wrist
        else:
            self.image_encoder = None
        
        # MLP for value estimation
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
        
        # Small initialization for value head
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
    
    def forward(
        self,
        proprio: torch.Tensor,
        action_dist: Optional[torch.Tensor] = None,
        front_images: Optional[torch.Tensor] = None,
        wrist_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value for agent's state.
        
        Args:
            proprio: Proprioceptive state (B, T, proprio_dim) or (B, proprio_dim)
            action_dist: Estimated action distribution from previous agents (B, action_dist_dim)
                        Only used for agent_idx > 0
            front_images: Front view images (B, T, H, W, C) - optional
            wrist_images: Wrist images (B, T, H, W, C) - optional
            
        Returns:
            Value tensor (B,)
        """
        batch_size = proprio.shape[0]
        
        # Flatten proprio if needed
        if proprio.dim() == 3:
            proprio = proprio.reshape(batch_size, -1)
        
        features = [proprio]
        
        # Add action distribution for agent 1+
        if self.agent_idx > 0 and action_dist is not None:
            features.append(action_dist)
        
        # Encode images if used
        if self.use_images and self.image_encoder is not None:
            if front_images is not None:
                B, T = front_images.shape[:2]
                front_flat = front_images.reshape(B * T, *front_images.shape[2:])
                front_encoded = self.image_encoder(front_flat.float() / 255.0)
                front_encoded = front_encoded.reshape(B, -1)
                features.append(front_encoded)
            
            if wrist_images is not None:
                B, T = wrist_images.shape[:2]
                wrist_flat = wrist_images.reshape(B * T, *wrist_images.shape[2:])
                wrist_encoded = self.image_encoder(wrist_flat.float() / 255.0)
                wrist_encoded = wrist_encoded.reshape(B, -1)
                features.append(wrist_encoded)
        
        # Concatenate all features
        x = torch.cat(features, dim=-1)
        
        # MLP forward
        hidden = self.mlp(x)
        value = self.value_head(hidden).squeeze(-1)
        
        return value


class ACPPOCentralizedCritic(nn.Module):
    """
    Centralized critic that maintains per-agent value heads for ACPPO.
    
    This is an alternative to using the VLA's built-in Value Head.
    It maintains separate value functions for each agent.
    """
    
    def __init__(
        self,
        cfg: ACPPOConfig,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]
        self.device = device
        
        # Create per-agent value networks
        self.value_networks = nn.ModuleList()
        
        for agent_idx in range(self.num_agents):
            value_net = PerAgentValueNetwork(
                agent_idx=agent_idx,
                proprio_dim=TWOARM_ACPPO_CONSTANTS["PROPRIO_DIM"],
                action_dist_dim=cfg.action_dist_dim,
                hidden_dim=cfg.value_hidden_dim,
                num_layers=cfg.value_num_layers,
                use_images=False,  # Lightweight version without images
                history_length=cfg.history_length,
            )
            self.value_networks.append(value_net)
        
        self.to(device)
    
    def forward(
        self,
        agent_proprios: List[torch.Tensor],
        action_dists: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Compute values for all agents.
        
        Args:
            agent_proprios: List of proprio states per agent
            action_dists: List of action distributions (only agent 0's used for agent 1)
            
        Returns:
            List of value tensors per agent
        """
        values = []
        
        for agent_idx in range(self.num_agents):
            proprio = agent_proprios[agent_idx]
            
            # Get action distribution for agent 1+
            if agent_idx > 0 and action_dists is not None:
                # Use previous agent's action distribution
                action_dist = action_dists[agent_idx - 1]
            else:
                action_dist = None
            
            value = self.value_networks[agent_idx](
                proprio=proprio,
                action_dist=action_dist,
            )
            values.append(value)
        
        return values
    
    def get_agent_value(
        self,
        agent_idx: int,
        proprio: torch.Tensor,
        action_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value for a specific agent."""
        return self.value_networks[agent_idx](
            proprio=proprio,
            action_dist=action_dist,
        )


class LightweightACPPOCritic(nn.Module):
    """
    Lightweight centralized critic using only proprioceptive states.
    
    Simpler version without image encoders for faster training.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        proprio_dim: int = 8,
        action_dist_dim: int = 48,
        history_length: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.proprio_dim = proprio_dim
        self.action_dist_dim = action_dist_dim
        self.history_length = history_length
        
        # Shared encoder for proprio
        proprio_total_dim = proprio_dim * history_length
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Per-agent value heads
        self.agent_value_heads = nn.ModuleList()
        
        for agent_idx in range(num_agents):
            if agent_idx == 0:
                # Agent 0: only encoded proprio
                head_input_dim = hidden_dim
            else:
                # Agent 1+: encoded proprio + action dist
                head_input_dim = hidden_dim + action_dist_dim
            
            head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.agent_value_heads.append(head)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        agent_proprios: List[torch.Tensor],
        action_dists: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Compute values for all agents."""
        values = []
        
        for agent_idx in range(self.num_agents):
            proprio = agent_proprios[agent_idx]
            batch_size = proprio.shape[0]
            
            # Flatten proprio
            if proprio.dim() == 3:
                proprio = proprio.reshape(batch_size, -1)
            
            # Encode proprio
            proprio_encoded = self.proprio_encoder(proprio)
            
            # Add action dist for agent 1+
            if agent_idx > 0 and action_dists is not None:
                action_dist = action_dists[agent_idx - 1]
                if action_dist.dim() == 3:
                    action_dist = action_dist.reshape(batch_size, -1)
                features = torch.cat([proprio_encoded, action_dist], dim=-1)
            else:
                features = proprio_encoded
            
            value = self.agent_value_heads[agent_idx](features).squeeze(-1)
            values.append(value)
        
        return values


def create_acppo_value_network(
    cfg: ACPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """
    Factory function to create the appropriate value network for ACPPO.
    
    Note: In the main ACPPO implementation, we use the VLA's built-in Value Head
    which shares the backbone with the policy. This function creates standalone
    value networks if a separate critic is desired.
    """
    if cfg.use_global_state:
        value_net = ACPPOCentralizedCritic(
            cfg=cfg,
            device=device,
        )
    else:
        value_net = LightweightACPPOCritic(
            num_agents=TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            proprio_dim=TWOARM_ACPPO_CONSTANTS["PROPRIO_DIM"],
            action_dist_dim=cfg.action_dist_dim,
            history_length=cfg.history_length,
            hidden_dim=cfg.value_hidden_dim,
            num_layers=cfg.value_num_layers,
        ).to(device)
    
    return value_net
