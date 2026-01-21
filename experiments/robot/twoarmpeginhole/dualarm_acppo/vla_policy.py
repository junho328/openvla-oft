"""
Multi-Agent VLA Policy wrapper for Dual-Arm ACPPO training.

This module provides the actor policy using OpenVLA-OFT bimanual models for
multi-agent reinforcement learning in the TwoArmPegInHole environment.

Key differences from single-arm ACPPO:
- VLA outputs 14-dim action (bimanual), split into 7-dim per agent
- Agent 0 uses action[:7], Agent 1 uses action[7:]
- Action head is shared, Value heads are per-agent
- Each agent receives padded proprio (14-dim with 7-dim real, 7-dim zeros)
- Agent 1 additionally receives estimated action distribution from Agent 0

ACPPO-specific features:
1. All agents act SIMULTANEOUSLY - they all decide and execute actions at the same time
2. Agent 0: Uses only its own observation
3. Agent 1: Additionally uses ESTIMATED action distribution from Agent 0
4. No gradient flows through the estimation process (detached)
5. All components (value head, proprio_projector, action_head) are shared

Architecture:
    VLA Backbone ─┬─────────→ Action Head → 14-dim Action → Split to 7-dim per agent
                  └─────────→ Value Heads → Per-agent Value (critic)
    
    For Agent 1: proprio input = [padded_proprio(14), action_dist_from_agent0(action_dist_dim)]
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import DualArmACPPOConfig, DUALARM_ACPPO_CONSTANTS


def is_main_process():
    """Check if current process is main (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class PassThroughProjector(nn.Module):
    """
    Pass-through projector that returns input as-is.
    
    Used in ACPPO when combining separately projected proprio and action_dist.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class ValueHead(nn.Module):
    """
    Value Head MLP for estimating state value from VLA hidden states.
    
    For ACPPO, we have per-agent value functions:
    - V^(0)(s_t) for agent 0
    - V^(1)([s_t, b_t^(1)]) for agent 1, where b_t^(1) includes estimated action dist
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        
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
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        value = self.mlp(pooled).squeeze(-1)
        return value


class ActionDistributionProjector(nn.Module):
    """
    Projects action distribution (mu, sigma) to be concatenated with proprio.
    
    This allows the estimated action distribution from agent 0 to be used
    as additional context for agent 1's decision making.
    
    For bimanual model:
    - Agent 0's action is 7-dim per chunk step
    - action_dist_dim = 7 * num_actions_chunk * 2 (mu + sigma)
    """
    
    def __init__(
        self,
        action_dist_dim: int,
        output_dim: int,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(action_dist_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),  # Normalize to [-1, 1] range similar to proprio
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Project action distribution to embedding.
        
        Args:
            mu: Action mean (B, action_dim * chunk_size) or (B, chunk_size, action_dim)
            sigma: Action std (B, action_dim * chunk_size) or (B, chunk_size, action_dim)
            
        Returns:
            Projected embedding (B, output_dim)
        """
        # Flatten if needed
        if mu.dim() == 3:
            mu = mu.reshape(mu.shape[0], -1)
        if sigma.dim() == 3:
            sigma = sigma.reshape(sigma.shape[0], -1)
        
        # Concatenate mu and sigma
        action_dist = torch.cat([mu, sigma], dim=-1)
        
        return self.projection(action_dist)


class DualArmVLAAgentACPPO(nn.Module):
    """
    Dual-Arm VLA agent wrapper for ACPPO (Actor-Critic).
    
    Key features:
    - VLA outputs 14-dim action (bimanual model)
    - Each agent extracts their 7-dim slice from the full action
    - Action head is shared between agents
    - Value heads are per-agent (V^(0), V^(1))
    - Supports extended proprio input (proprio + action_dist for agent 1)
    - Can estimate action distribution with front-view only input
    """
    
    def __init__(
        self,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        model_action_dim: int = 14,      # Full bimanual action dim
        agent_action_dim: int = 7,       # Per-agent action dim
        num_actions_chunk: int = 25,
        action_std_init: float = 0.5,
        min_action_std: float = 0.01,
        device: torch.device = torch.device("cuda"),
        freeze_vla_backbone: bool = True,
        train_proprio_projector: bool = False,
        train_action_head: bool = True,
        train_value_head: bool = True,
        train_action_dist_projector: bool = True,
        value_head_hidden_dim: int = 512,
        value_head_num_layers: int = 2,
        num_agents: int = 2,
        action_dist_dim: int = 350,      # 7 * 25 * 2 for bimanual
    ):
        super().__init__()
        
        self.vla = vla_model
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.processor = processor
        
        self.model_action_dim = model_action_dim
        self.agent_action_dim = agent_action_dim
        self.num_actions_chunk = num_actions_chunk
        self.total_action_dim = model_action_dim * num_actions_chunk
        self.device = device
        self.num_agents = num_agents
        self.action_dist_dim = action_dist_dim
        
        # Store training flags
        self.freeze_vla_backbone = freeze_vla_backbone
        self.train_proprio_projector = train_proprio_projector
        self.train_action_head = train_action_head
        self.train_value_head = train_value_head
        self.train_action_dist_projector = train_action_dist_projector
        
        # Get VLA hidden dimension
        llm_dim = vla_model.llm_dim if hasattr(vla_model, 'llm_dim') else 4096
        
        # Create per-agent Value Heads
        self.value_heads = nn.ModuleList([
            ValueHead(
                input_dim=llm_dim,
                hidden_dim=value_head_hidden_dim,
                num_layers=value_head_num_layers,
            ).to(torch.bfloat16).to(device)
            for _ in range(num_agents)
        ])
        self.value_head = self.value_heads[0]  # Backward compatibility
        
        # ACPPO: Action distribution projector for agent 1
        # This projects the estimated action distribution from agent 0 to llm_dim
        # Initialize in float32 first (orthogonal_ doesn't support bfloat16)
        self.action_dist_projector = nn.Linear(
            action_dist_dim,
            llm_dim,
        ).to(device)  # Keep float32 for initialization
        nn.init.orthogonal_(self.action_dist_projector.weight, gain=1.0)
        nn.init.zeros_(self.action_dist_projector.bias)
        # Convert to bfloat16 after initialization
        self.action_dist_projector = self.action_dist_projector.to(torch.bfloat16)
        
        print_rank0(f"\nCreated {num_agents} per-agent Value Heads for Dual-Arm ACPPO:")
        print_rank0(f"  Input dim: {llm_dim}")
        print_rank0(f"  Hidden dim: {value_head_hidden_dim}")
        print_rank0(f"  Num layers: {value_head_num_layers}")
        print_rank0(f"  Model action dim: {model_action_dim} (14 for bimanual)")
        print_rank0(f"  Per-agent action dim: {agent_action_dim} (7 per arm)")
        print_rank0(f"  Action dist dim: {action_dist_dim}")
        
        # Apply freeze settings
        self._setup_trainable_parameters()
        
        # Learnable action standard deviation (for 14-dim bimanual action)
        self.log_std = nn.Parameter(
            torch.ones(self.total_action_dim, device=device, dtype=torch.bfloat16) * np.log(action_std_init)
        )
        self.min_log_std = np.log(min_action_std)
        
        self._num_patches = None
        self._print_trainable_summary()
    
    def _setup_trainable_parameters(self):
        """Setup which parameters are trainable based on config."""
        if self.freeze_vla_backbone:
            print_rank0("Freezing VLA backbone...")
            for param in self.vla.parameters():
                param.requires_grad = False
        
        if self.proprio_projector is not None:
            for param in self.proprio_projector.parameters():
                param.requires_grad = self.train_proprio_projector
        
        if self.action_head is not None:
            for param in self.action_head.parameters():
                param.requires_grad = self.train_action_head
        
        for value_head in self.value_heads:
            for param in value_head.parameters():
                param.requires_grad = self.train_value_head
        
        # ACPPO: Action dist projector
        for param in self.action_dist_projector.parameters():
            param.requires_grad = self.train_action_dist_projector
    
    def _print_trainable_summary(self):
        """Print summary of trainable parameters."""
        total_params = 0
        trainable_params = 0
        
        print_rank0("\n" + "-"*50)
        print_rank0("Trainable Parameters Summary (Dual-Arm ACPPO):")
        print_rank0("-"*50)
        
        # VLA backbone
        vla_total = sum(p.numel() for p in self.vla.parameters())
        vla_trainable = sum(p.numel() for p in self.vla.parameters() if p.requires_grad)
        total_params += vla_total
        trainable_params += vla_trainable
        print_rank0(f"  VLA backbone: {vla_trainable:,} / {vla_total:,} trainable")
        
        # Action head
        if self.action_head is not None:
            ah_total = sum(p.numel() for p in self.action_head.parameters())
            ah_trainable = sum(p.numel() for p in self.action_head.parameters() if p.requires_grad)
            total_params += ah_total
            trainable_params += ah_trainable
            print_rank0(f"  Action head (14-dim): {ah_trainable:,} / {ah_total:,} trainable")
        
        # Per-agent value heads
        for i, vh in enumerate(self.value_heads):
            vh_total = sum(p.numel() for p in vh.parameters())
            vh_trainable = sum(p.numel() for p in vh.parameters() if p.requires_grad)
            total_params += vh_total
            trainable_params += vh_trainable
            print_rank0(f"  Value head (agent {i}): {vh_trainable:,} / {vh_total:,} trainable")
        
        # Proprio projector
        if self.proprio_projector is not None:
            pp_total = sum(p.numel() for p in self.proprio_projector.parameters())
            pp_trainable = sum(p.numel() for p in self.proprio_projector.parameters() if p.requires_grad)
            total_params += pp_total
            trainable_params += pp_trainable
            print_rank0(f"  Proprio projector (14-dim): {pp_trainable:,} / {pp_total:,} trainable")
        
        # Action dist projector
        adp_total = sum(p.numel() for p in self.action_dist_projector.parameters())
        adp_trainable = sum(p.numel() for p in self.action_dist_projector.parameters() if p.requires_grad)
        total_params += adp_total
        trainable_params += adp_trainable
        print_rank0(f"  Action dist projector: {adp_trainable:,} / {adp_total:,} trainable")
        
        # log_std
        log_std_params = self.total_action_dim
        trainable_params += log_std_params
        total_params += log_std_params
        print_rank0(f"  Log std (14-dim * chunk): {log_std_params:,} trainable")
        
        print_rank0("-"*50)
        print_rank0(f"  TOTAL: {trainable_params:,} / {total_params:,} trainable parameters")
        print_rank0(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print_rank0("-"*50 + "\n")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only the trainable parameters for optimizer."""
        params = []
        
        if not self.freeze_vla_backbone:
            params.extend([p for p in self.vla.parameters() if p.requires_grad])
        
        if self.action_head is not None and self.train_action_head:
            params.extend([p for p in self.action_head.parameters() if p.requires_grad])
        
        if self.value_heads is not None and self.train_value_head:
            for vh in self.value_heads:
                params.extend([p for p in vh.parameters() if p.requires_grad])
        
        if self.proprio_projector is not None and self.train_proprio_projector:
            params.extend([p for p in self.proprio_projector.parameters() if p.requires_grad])
        
        if self.train_action_dist_projector:
            params.extend([p for p in self.action_dist_projector.parameters() if p.requires_grad])
        
        params.append(self.log_std)
        
        return params
    
    def get_actor_parameters(self) -> List[nn.Parameter]:
        """Get actor (policy) parameters."""
        params = []
        
        if not self.freeze_vla_backbone:
            params.extend([p for p in self.vla.parameters() if p.requires_grad])
        
        if self.action_head is not None and self.train_action_head:
            params.extend([p for p in self.action_head.parameters() if p.requires_grad])
        
        if self.proprio_projector is not None and self.train_proprio_projector:
            params.extend([p for p in self.proprio_projector.parameters() if p.requires_grad])
        
        if self.train_action_dist_projector:
            params.extend([p for p in self.action_dist_projector.parameters() if p.requires_grad])
        
        params.append(self.log_std)
        
        return params
    
    def get_critic_parameters(self) -> List[nn.Parameter]:
        """Get critic (value function) parameters."""
        params = []
        
        if self.value_heads is not None and self.train_value_head:
            for vh in self.value_heads:
                params.extend([p for p in vh.parameters() if p.requires_grad])
        
        return params
    
    def _get_num_patches(self, use_proprio: bool = False, use_diffusion: bool = False) -> int:
        """Calculate number of vision patches."""
        if self._num_patches is None:
            num_patches = self.vla.vision_backbone.get_num_patches()
            num_patches *= self.vla.vision_backbone.get_num_images_in_input()
            if use_proprio:
                num_patches += 1
            if use_diffusion:
                num_patches += 1
            self._num_patches = num_patches
        return self._num_patches
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VLA to get action hidden states.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dummy_labels = torch.full_like(inputs["input_ids"], -100)
            
            output = self.vla(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                labels=dummy_labels,
                output_hidden_states=True,
                proprio=proprio if use_proprio and self.proprio_projector else None,
                proprio_projector=self.proprio_projector if use_proprio else None,
                use_film=use_film,
            )
        
        last_hidden_states = output.hidden_states[-1]
        num_patches = self._get_num_patches(use_proprio)
        
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        
        return text_hidden_states, num_patches
    
    def forward_with_action_dist_estimation(
        self,
        inputs: Dict[str, torch.Tensor],
        agent0_inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        agent0_proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        detach_action_dist: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Agent 1 with action distribution estimation from Agent 0.
        
        This is the ACPPO-specific forward pass where:
        1. First, we estimate Agent 0's action distribution using Agent 0's observation
        2. Then, we project the action distribution and add it to Agent 1's proprio embedding
        3. Finally, we do the forward pass for Agent 1
        
        Args:
            inputs: Agent 1's VLA inputs
            agent0_inputs: Agent 0's VLA inputs (for action dist estimation)
            proprio: Agent 1's proprio (14-dim padded)
            agent0_proprio: Agent 0's proprio (14-dim padded)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            detach_action_dist: If True, no gradient through action dist estimation
            
        Returns:
            Tuple of (text_hidden_states, num_patches, action_dist_mu, action_dist_sigma)
        """
        # Step 1: Estimate Agent 0's action distribution (no grad)
        with torch.no_grad() if detach_action_dist else torch.enable_grad():
            agent0_hidden, _ = self.forward(
                inputs=agent0_inputs,
                proprio=agent0_proprio,
                use_proprio=use_proprio,
                use_film=use_film,
            )
            
            batch_size = agent0_inputs["input_ids"].shape[0]
            action_hidden_dim = self.num_actions_chunk * self.model_action_dim
            
            if agent0_hidden.shape[1] >= action_hidden_dim:
                agent0_action_hidden = agent0_hidden[:, :action_hidden_dim, :]
            else:
                pad_len = action_hidden_dim - agent0_hidden.shape[1]
                padding = torch.zeros(
                    batch_size, pad_len, agent0_hidden.shape[-1],
                    device=agent0_hidden.device, dtype=agent0_hidden.dtype
                )
                agent0_action_hidden = torch.cat([agent0_hidden, padding], dim=1)
            
            # Get action distribution from Agent 0 (14-dim -> extract 7-dim for Agent 0)
            dist = self.get_action_distribution(agent0_action_hidden)
            
            # Extract Agent 0's 7-dim slice
            action_dist_mu_full = dist.mean  # (B, total_action_dim = 14 * chunk)
            action_dist_sigma_full = dist.stddev  # (B, total_action_dim)
            
            # Reshape to (B, chunk, 14) and take first 7 dims for Agent 0
            action_dist_mu_14 = action_dist_mu_full.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
            action_dist_sigma_14 = action_dist_sigma_full.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
            
            action_dist_mu = action_dist_mu_14[..., :self.agent_action_dim]  # (B, chunk, 7)
            action_dist_sigma = action_dist_sigma_14[..., :self.agent_action_dim]  # (B, chunk, 7)
            
            # Flatten for projection: (B, 7 * chunk * 2)
            action_dist_mu_flat = action_dist_mu.reshape(batch_size, -1)  # (B, 7 * chunk)
            action_dist_sigma_flat = action_dist_sigma.reshape(batch_size, -1)  # (B, 7 * chunk)
            action_dist_concat = torch.cat([action_dist_mu_flat, action_dist_sigma_flat], dim=-1)  # (B, 7 * chunk * 2)
        
        # Step 2: Project action distribution to llm_dim and add to proprio embedding
        # We project the action dist and add it to the proprio embedding
        action_dist_embedding = self.action_dist_projector(action_dist_concat)  # (B, llm_dim)
        
        # Step 3: Forward pass for Agent 1
        # Instead of directly modifying proprio, we'll do the forward pass and then
        # add the action_dist_embedding to the hidden states at the appropriate position
        
        # First, do a regular forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dummy_labels = torch.full_like(inputs["input_ids"], -100)
            
            # Get proprio embedding
            proprio_embedding = None
            if use_proprio and self.proprio_projector is not None and proprio is not None:
                proprio_embedding = self.proprio_projector(proprio)  # (B, llm_dim)
                # Add action dist embedding to proprio embedding
                proprio_embedding = proprio_embedding + action_dist_embedding
            
            # Do forward pass with modified proprio embedding
            # We need to pass the combined embedding, so we use a PassThroughProjector
            original_proprio_projector = self.proprio_projector
            self.proprio_projector = PassThroughProjector()
            
            output = self.vla(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                labels=dummy_labels,
                output_hidden_states=True,
                proprio=proprio_embedding if use_proprio else None,
                proprio_projector=self.proprio_projector if use_proprio and proprio_embedding is not None else None,
                use_film=use_film,
            )
            
            # Restore original proprio projector
            self.proprio_projector = original_proprio_projector
        
        last_hidden_states = output.hidden_states[-1]
        num_patches = self._get_num_patches(use_proprio)
        
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        
        return text_hidden_states, num_patches, action_dist_mu, action_dist_sigma
    
    def get_action_distribution(
        self,
        action_hidden_states: torch.Tensor,
    ) -> Normal:
        """
        Get action distribution from hidden states.
        
        Returns Normal distribution over 14-dim bimanual action.
        """
        batch_size = action_hidden_states.shape[0]
        
        if hasattr(self.action_head, 'module'):
            action_mean = self.action_head.module.predict_action(action_hidden_states)
        else:
            action_mean = self.action_head.predict_action(action_hidden_states)
        
        action_mean = action_mean.reshape(batch_size, -1)
        
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        action_std = log_std.exp().expand_as(action_mean)
        
        return Normal(action_mean, action_std)
    
    def get_action_and_log_prob(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Returns 14-dim bimanual action.
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = torch.clamp(action, -1.0, 1.0)
        action = action.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
        
        return action, log_prob, entropy
    
    def get_action_and_log_prob_with_action_dist(
        self,
        inputs: Dict[str, torch.Tensor],
        agent0_inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        agent0_proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        deterministic: bool = False,
        detach_action_dist: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For Agent 1: Get action with estimated action distribution from Agent 0.
        
        Returns:
            Tuple of (action [B, chunk, 14], log_prob [B], entropy [B], 
                     action_dist_mu [B, chunk, 7], action_dist_sigma [B, chunk, 7])
        """
        text_hidden_states, _, action_dist_mu, action_dist_sigma = self.forward_with_action_dist_estimation(
            inputs=inputs,
            agent0_inputs=agent0_inputs,
            proprio=proprio,
            agent0_proprio=agent0_proprio,
            use_proprio=use_proprio,
            use_film=use_film,
            detach_action_dist=detach_action_dist,
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = torch.clamp(action, -1.0, 1.0)
        action = action.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
        
        return action, log_prob, entropy, action_dist_mu, action_dist_sigma
    
    def evaluate_actions(
        self,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = self.get_action_distribution(action_hidden_states)
        
        actions_flat = actions.reshape(batch_size, -1)
        
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def get_value(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        agent_idx: int = 0,
    ) -> torch.Tensor:
        """
        Get state value from per-agent Value Head.
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return value
    
    def get_action_and_value(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        deterministic: bool = False,
        agent_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, AND value in one forward pass.
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = torch.clamp(action, -1.0, 1.0)
        action = action.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
        
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return action, log_prob, entropy, value
    
    def evaluate_actions_and_value(
        self,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        agent_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions AND get value in one forward pass (for PPO update).
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = self.get_action_distribution(action_hidden_states)
        actions_flat = actions.reshape(batch_size, -1)
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return log_prob, entropy, value


class DualArmMultiAgentVLAPolicyACPPO(nn.Module):
    """
    Multi-Agent VLA Policy for Dual-Arm ACPPO.
    
    Manages the bimanual VLA model with ACPPO's action distribution chaining.
    Key difference from MAPPO: Agent 1 receives estimated action distribution from Agent 0.
    """
    
    def __init__(
        self,
        cfg: DualArmACPPOConfig,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]
        self.share_policy = cfg.share_policy
        self.device = device
        self.model_action_dim = cfg.model_action_dim  # 14
        self.agent_action_dim = cfg.agent_action_dim  # 7
        self.use_action_dist_input = cfg.use_action_dist_input
        self.detach_action_dist_grad = cfg.detach_action_dist_grad
        
        print_rank0("\n" + "="*60)
        print_rank0("Initializing Dual-Arm Multi-Agent VLA Policy for ACPPO")
        print_rank0("="*60)
        print_rank0(f"  VLA backbone frozen: {cfg.freeze_vla_backbone}")
        print_rank0(f"  Train action head: {cfg.train_action_head}")
        print_rank0(f"  Train proprio projector: {cfg.train_proprio_projector}")
        print_rank0(f"  Train action dist projector: {cfg.train_action_dist_projector}")
        print_rank0(f"  Shared policy: {cfg.share_policy}")
        print_rank0(f"  Per-agent Value Heads: {self.num_agents}")
        print_rank0(f"  Model action dim: {cfg.model_action_dim} (bimanual)")
        print_rank0(f"  Agent action dim: {cfg.agent_action_dim} (per arm)")
        print_rank0(f"  Action dist input for Agent 1: {cfg.use_action_dist_input}")
        print_rank0(f"  Detach action dist gradient: {cfg.detach_action_dist_grad}")
        print_rank0("="*60 + "\n")
        
        # Create shared agent with per-agent Value Heads
        self.shared_agent = DualArmVLAAgentACPPO(
            vla_model=vla_model,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            processor=processor,
            model_action_dim=cfg.model_action_dim,
            agent_action_dim=cfg.agent_action_dim,
            num_actions_chunk=cfg.num_actions_chunk,
            device=device,
            freeze_vla_backbone=cfg.freeze_vla_backbone,
            train_proprio_projector=cfg.train_proprio_projector,
            train_action_head=cfg.train_action_head,
            train_value_head=cfg.train_value_head,
            train_action_dist_projector=cfg.train_action_dist_projector,
            value_head_hidden_dim=cfg.value_hidden_dim,
            value_head_num_layers=cfg.value_num_layers,
            num_agents=self.num_agents,
            action_dist_dim=cfg.action_dist_dim,
        )
        self.agents = [self.shared_agent] * self.num_agents
    
    def _split_action_for_agent(self, full_action: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """
        Split 14-dim bimanual action for specific agent.
        """
        if agent_idx == 0:
            return full_action[..., :self.agent_action_dim]
        else:
            return full_action[..., self.agent_action_dim:]
    
    def get_actions_and_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        front_view_only_input: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get actions AND values for all agents in one efficient forward pass.
        
        ACPPO-specific: Agent 1 receives estimated action distribution from Agent 0.
        
        Args:
            agent_inputs: List of VLA inputs for each agent
            agent_proprios: List of proprioceptive states for each agent
            front_view_only_input: Front-view only input for estimating Agent 0's action dist
            deterministic: Whether to use deterministic actions
        
        Returns:
            Tuple of (actions [7-dim], log_probs, entropies, values, action_dist) lists for all agents
            action_dist is (mu, sigma) for Agent 0's estimated action distribution
        """
        actions = []
        log_probs = []
        entropies = []
        values = []
        action_dist = None  # (mu, sigma) for Agent 0
        
        # ========== Agent 0: Regular forward pass ==========
        agent_idx = 0
        agent = self.agents[agent_idx]
        inputs = agent_inputs[agent_idx]
        proprio = agent_proprios[agent_idx] if agent_proprios else None
        
        text_hidden_states, _ = agent.forward(
            inputs=inputs,
            proprio=proprio,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = agent.num_actions_chunk * agent.model_action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        dist = agent.get_action_distribution(action_hidden_states)
        
        if deterministic:
            full_action = dist.mean
        else:
            full_action = dist.rsample()
        
        log_prob = dist.log_prob(full_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        full_action_clamped = torch.clamp(full_action, -1.0, 1.0)
        full_action_reshaped = full_action_clamped.reshape(batch_size, agent.num_actions_chunk, agent.model_action_dim)
        
        # Split to get 7-dim per-agent action
        agent_action = self._split_action_for_agent(full_action_reshaped, agent_idx)
        
        # Get Agent 0's value
        value = agent.value_heads[agent_idx](text_hidden_states)
        
        actions.append(agent_action)
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        
        # Store Agent 0's action distribution for Agent 1
        action_dist_mu = dist.mean.reshape(batch_size, agent.num_actions_chunk, agent.model_action_dim)
        action_dist_mu = action_dist_mu[..., :self.agent_action_dim]  # (B, chunk, 7) for Agent 0
        action_dist_sigma = dist.stddev.reshape(batch_size, agent.num_actions_chunk, agent.model_action_dim)
        action_dist_sigma = action_dist_sigma[..., :self.agent_action_dim]  # (B, chunk, 7)
        action_dist = (action_dist_mu, action_dist_sigma)
        
        # ========== Agent 1: Forward with action distribution from Agent 0 ==========
        agent_idx = 1
        agent = self.agents[agent_idx]
        inputs = agent_inputs[agent_idx]
        proprio = agent_proprios[agent_idx] if agent_proprios else None
        
        if self.use_action_dist_input:
            # Use ACPPO's action distribution chaining
            # Use front_view_only_input if provided, otherwise fall back to agent_inputs[0]
            agent0_inputs_for_dist = front_view_only_input if front_view_only_input is not None else agent_inputs[0]
            
            full_action, log_prob, entropy, _, _ = agent.get_action_and_log_prob_with_action_dist(
                inputs=inputs,
                agent0_inputs=agent0_inputs_for_dist,
                proprio=proprio,
                agent0_proprio=agent_proprios[0] if agent_proprios else None,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
                detach_action_dist=self.detach_action_dist_grad,
            )
            
            # Get value (need separate forward for value)
            text_hidden_states, _, _, _ = agent.forward_with_action_dist_estimation(
                inputs=inputs,
                agent0_inputs=agent0_inputs_for_dist,
                proprio=proprio,
                agent0_proprio=agent_proprios[0] if agent_proprios else None,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                detach_action_dist=self.detach_action_dist_grad,
            )
            value = agent.value_heads[agent_idx](text_hidden_states)
        else:
            # Regular forward without action distribution (like MAPPO)
            full_action, log_prob, entropy, value = agent.get_action_and_value(
                inputs=inputs,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
                agent_idx=agent_idx,
            )
        
        # Split to get 7-dim per-agent action
        agent_action = self._split_action_for_agent(full_action, agent_idx)
        
        actions.append(agent_action)
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        
        return actions, log_probs, entropies, values, action_dist
    
    def evaluate_actions_and_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_actions: List[torch.Tensor],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        stored_action_dist: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Evaluate actions AND get values in one forward pass (for PPO update).
        """
        log_probs = []
        entropies = []
        values = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            actions = agent_actions[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            log_prob, entropy, value = agent.evaluate_actions_and_value(
                inputs=inputs,
                actions=actions,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                agent_idx=agent_idx,
            )
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        
        return log_probs, entropies, values
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only the trainable parameters for optimizer."""
        return self.shared_agent.get_trainable_parameters()
    
    def get_actor_parameters(self) -> List[nn.Parameter]:
        """Get actor (policy) parameters."""
        return self.shared_agent.get_actor_parameters()
    
    def get_critic_parameters(self) -> List[nn.Parameter]:
        """Get critic (value function) parameters."""
        return self.shared_agent.get_critic_parameters()
    
    def forward_evaluate_agent(
        self,
        agent_idx: int,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for evaluating a single agent's actions.
        """
        agent = self.agents[agent_idx]
        
        return agent.evaluate_actions_and_value(
            inputs=inputs,
            actions=actions,
            proprio=proprio,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
            agent_idx=agent_idx,
        )
    
    def forward(
        self,
        mode: str,
        **kwargs,
    ):
        """
        Generic forward method for DDP compatibility.
        """
        if mode == 'evaluate_agent':
            return self.forward_evaluate_agent(
                agent_idx=kwargs['agent_idx'],
                inputs=kwargs['inputs'],
                actions=kwargs['actions'],
                proprio=kwargs.get('proprio'),
            )
        else:
            raise ValueError(f"Unknown forward mode: {mode}")


def _is_bimanual_checkpoint(checkpoint_path: str) -> bool:
    """Check if the checkpoint is a bimanual (ALOHA-style) model."""
    aloha_keywords = ["aloha", "bimanual", "dual", "dualarm"]
    
    checkpoint_lower = checkpoint_path.lower()
    for keyword in aloha_keywords:
        if keyword in checkpoint_lower:
            return True
    
    return False


def load_vla_for_dualarm_acppo(
    cfg: DualArmACPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module], Any, Optional[Dict]]:
    """
    Load VLA model and components for Dual-Arm ACPPO training.
    
    This loads a bimanual VLA model (ALOHA-style) that outputs 14-dim actions.
    
    Returns:
        Tuple of (vla_model, action_head, proprio_projector, noisy_action_projector, processor, norm_stats)
    """
    import sys
    sys.path.append("../../..")
    
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
    
    print_rank0("Loading VLA bimanual model for Dual-Arm ACPPO training...")
    
    is_bimanual = _is_bimanual_checkpoint(cfg.pretrained_checkpoint)
    
    if is_bimanual:
        print_rank0(f"  Detected BIMANUAL checkpoint: {cfg.pretrained_checkpoint}")
    else:
        print_rank0(f"  Checkpoint: {cfg.pretrained_checkpoint}")
        print_rank0("  WARNING: This may not be a bimanual model!")
    
    # Load base VLA model
    vla = get_model(cfg)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    
    # Enable gradient checkpointing to reduce memory usage
    # This trades compute for memory by recomputing activations during backward pass
    use_gradient_checkpointing = getattr(cfg, 'use_gradient_checkpointing', True)
    if use_gradient_checkpointing:
        print_rank0("Enabling gradient checkpointing for VLA model...")
        if hasattr(vla, 'gradient_checkpointing_enable'):
            vla.gradient_checkpointing_enable()
        elif hasattr(vla, 'language_model') and hasattr(vla.language_model, 'gradient_checkpointing_enable'):
            vla.language_model.gradient_checkpointing_enable()
        print_rank0("  Gradient checkpointing enabled!")
    
    vla = vla.to(device)
    
    torch.cuda.empty_cache()
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # Get processor
    processor = get_processor(cfg)
    
    # Get normalization statistics
    norm_stats = getattr(vla, 'norm_stats', None)
    if norm_stats is not None:
        print_rank0("Loaded action normalization statistics from VLA model")
        if len(norm_stats) > 0:
            first_key = list(norm_stats.keys())[0]
            print_rank0(f"  Using normalization key: {first_key}")
    else:
        print_rank0("WARNING: No normalization statistics found in VLA model!")
    
    # Load or initialize proprio projector (14-dim for bimanual)
    proprio_projector = None
    if cfg.use_proprio:
        from experiments.robot.openvla_utils import get_proprio_projector
        print_rank0("Loading/initializing proprio projector (14-dim for bimanual)...")
        
        try:
            proprio_projector = get_proprio_projector(
                cfg,
                vla.llm_dim,
                proprio_dim=cfg.model_proprio_dim,  # 14 for bimanual
            )
            print_rank0(f"  Loaded proprio projector: input_dim={cfg.model_proprio_dim}, output_dim={vla.llm_dim}")
        except:
            # Initialize fresh
            print_rank0("Initializing NEW proprio projector (14-dim)")
            proprio_projector = ProprioProjector(
                llm_dim=vla.llm_dim,
                proprio_dim=cfg.model_proprio_dim,
            )
            proprio_projector = proprio_projector.to(torch.bfloat16).to(device)
            print_rank0(f"  Initialized proprio projector: input_dim={cfg.model_proprio_dim}, output_dim={vla.llm_dim}")
    
    # Initialize action head (14-dim for bimanual)
    action_head = None
    if cfg.use_l1_regression:
        print_rank0(f"Initializing L1 regression action head (action_dim={cfg.model_action_dim}, num_actions_chunk={cfg.num_actions_chunk})")
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=cfg.model_action_dim,  # 14 for bimanual
            num_actions_chunk=cfg.num_actions_chunk,
        )
        action_head = action_head.to(torch.bfloat16).to(device)
    elif cfg.use_diffusion:
        print_rank0(f"Initializing diffusion action head (action_dim={cfg.model_action_dim}, num_actions_chunk={cfg.num_actions_chunk})")
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=cfg.model_action_dim,  # 14 for bimanual
            num_diffusion_steps_train=cfg.num_diffusion_steps_train,
            num_actions_chunk=cfg.num_actions_chunk,
        )
        action_head = action_head.to(torch.bfloat16).to(device)
    
    # Noisy action projector (only for diffusion)
    noisy_action_projector = None
    if cfg.use_diffusion:
        from prismatic.models.projectors import NoisyActionProjector
        print_rank0("Initializing NEW noisy action projector")
        noisy_action_projector = NoisyActionProjector(llm_dim=vla.llm_dim)
        noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(device)
    
    return vla, action_head, proprio_projector, noisy_action_projector, processor, norm_stats
