"""
Multi-Agent VLA Policy wrapper for Dual-Arm MAPPO training.

This module provides the actor policy using OpenVLA-OFT bimanual models for
multi-agent reinforcement learning in the TwoArmPegInHole environment.

Key differences from single-arm MAPPO:
- VLA outputs 14-dim action (bimanual), split into 7-dim per agent
- Agent 0 uses action[:7], Agent 1 uses action[7:]
- Action head is shared, Value heads are per-agent
- Each agent receives padded proprio (14-dim with 7-dim real, 7-dim zeros)

Architecture:
    VLA Backbone ─┬─────────→ Action Head → 14-dim Action → Split to 7-dim per agent
                  └─────────→ Value Heads → Per-agent Value (critic)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import DualArmMAPPOConfig, DUALARM_MAPPO_CONSTANTS


def is_main_process():
    """Check if current process is main (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class ValueHead(nn.Module):
    """
    Value Head MLP for estimating state value from VLA hidden states.
    
    Takes the same hidden states as Action Head and outputs a scalar value.
    This enables shared representation learning between actor and critic.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        """
        Initialize Value Head.
        
        Args:
            input_dim: Input dimension (VLA hidden dim * action tokens)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        
        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Output layer (scalar value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Last layer with smaller gain for value estimation
        last_linear = list(self.mlp.modules())[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.orthogonal_(last_linear.weight, gain=0.01)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute value from hidden states.
        
        Args:
            hidden_states: VLA hidden states (B, seq_len, hidden_dim)
            
        Returns:
            Value tensor (B,)
        """
        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)  # (B, hidden_dim)
        
        # Get value
        value = self.mlp(pooled).squeeze(-1)  # (B,)
        
        return value


class DualArmVLAAgent(nn.Module):
    """
    Dual-Arm VLA agent wrapper for MAPPO (Actor-Critic).
    
    Key difference from single-arm:
    - VLA outputs 14-dim action (bimanual model)
    - Each agent extracts their 7-dim slice from the full action
    - Action head is shared between agents
    - Value heads are per-agent (V^(0), V^(1))
    
    Architecture:
        VLA Backbone ─┬─────────→ Action Head → 14-dim Action
                      └─────────→ Value Heads → Per-agent Values
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
        num_actions_chunk: int = 2,
        action_std_init: float = 0.5,
        min_action_std: float = 0.01,
        device: torch.device = torch.device("cuda"),
        freeze_vla_backbone: bool = True,
        train_proprio_projector: bool = False,
        train_action_head: bool = True,
        train_value_head: bool = True,
        value_head_hidden_dim: int = 512,
        value_head_num_layers: int = 2,
        num_agents: int = 2,
    ):
        """
        Initialize Dual-Arm VLA agent with both Action Head and per-agent Value Heads.
        
        Args:
            vla_model: Pretrained OpenVLA bimanual model
            action_head: Action prediction head (outputs 14-dim)
            proprio_projector: Proprioceptive state projector
            noisy_action_projector: Noisy action projector for diffusion
            processor: VLA processor
            model_action_dim: Full bimanual action dimension (14)
            agent_action_dim: Per-agent action dimension (7)
            num_actions_chunk: Number of actions to predict
            action_std_init: Initial standard deviation for action distribution
            min_action_std: Minimum standard deviation
            device: Target device
            freeze_vla_backbone: If True, freeze VLA backbone
            train_proprio_projector: If True, train proprio projector
            train_action_head: If True, train action head MLP
            train_value_head: If True, train value head MLP
            value_head_hidden_dim: Hidden dimension for value head
            value_head_num_layers: Number of layers in value head
            num_agents: Number of agents (creates separate value heads for each)
        """
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
        
        # Store training flags
        self.freeze_vla_backbone = freeze_vla_backbone
        self.train_proprio_projector = train_proprio_projector
        self.train_action_head = train_action_head
        self.train_value_head = train_value_head
        
        # Get VLA hidden dimension for Value Heads
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
        
        print_rank0(f"\nCreated {num_agents} per-agent Value Heads for Dual-Arm VLA:")
        print_rank0(f"  Input dim: {llm_dim}")
        print_rank0(f"  Hidden dim: {value_head_hidden_dim}")
        print_rank0(f"  Num layers: {value_head_num_layers}")
        print_rank0(f"  Model action dim: {model_action_dim} (14 for bimanual)")
        print_rank0(f"  Per-agent action dim: {agent_action_dim} (7 per arm)")
        
        # Apply freeze settings
        self._setup_trainable_parameters()
        
        # Learnable action standard deviation (for 14-dim bimanual action)
        self.log_std = nn.Parameter(
            torch.ones(self.total_action_dim, device=device, dtype=torch.bfloat16) * np.log(action_std_init)
        )
        self.min_log_std = np.log(min_action_std)
        
        # Get number of patches for hidden state extraction
        self._num_patches = None
        
        # Print trainable parameter summary
        self._print_trainable_summary()
    
    def _setup_trainable_parameters(self):
        """Setup which parameters are trainable based on config."""
        # Freeze/unfreeze VLA backbone
        if self.freeze_vla_backbone:
            print_rank0("Freezing VLA backbone...")
            for param in self.vla.parameters():
                param.requires_grad = False
            frozen_params = sum(p.numel() for p in self.vla.parameters())
            print_rank0(f"  Frozen VLA parameters: {frozen_params:,}")
        else:
            print_rank0("VLA backbone is trainable")
        
        # Freeze/unfreeze proprio projector
        if self.proprio_projector is not None:
            if self.train_proprio_projector:
                print_rank0("Proprio projector is trainable")
                for param in self.proprio_projector.parameters():
                    param.requires_grad = True
            else:
                print_rank0("Freezing proprio projector...")
                for param in self.proprio_projector.parameters():
                    param.requires_grad = False
        
        # Freeze/unfreeze action head
        if self.action_head is not None:
            if self.train_action_head:
                print_rank0("Action head is trainable")
                for param in self.action_head.parameters():
                    param.requires_grad = True
            else:
                print_rank0("Freezing action head...")
                for param in self.action_head.parameters():
                    param.requires_grad = False
        
        # Freeze/unfreeze per-agent value heads
        if self.value_heads is not None:
            for value_head in self.value_heads:
                for param in value_head.parameters():
                    param.requires_grad = self.train_value_head
            print_rank0(f"{self.num_agents} Value heads {'trainable' if self.train_value_head else 'frozen'}")
    
    def _print_trainable_summary(self):
        """Print summary of trainable parameters."""
        total_params = 0
        trainable_params = 0
        
        print_rank0("\n" + "-"*50)
        print_rank0("Trainable Parameters Summary (Dual-Arm VLA):")
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
        if self.value_heads is not None:
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
        
        # log_std (always trainable for RL exploration)
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
        
        Args:
            inputs: Processed VLA inputs (input_ids, attention_mask, pixel_values)
            proprio: Proprioceptive state tensor (14-dim for bimanual)
            use_proprio: Whether to use proprio input
            use_film: Whether to use FiLM
            
        Returns:
            Tuple of (action_hidden_states, num_patches)
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
    
    def get_action_distribution(
        self,
        action_hidden_states: torch.Tensor,
    ) -> Normal:
        """
        Get action distribution from hidden states.
        
        Returns Normal distribution over 14-dim bimanual action.
        
        Args:
            action_hidden_states: Hidden states for action prediction
            
        Returns:
            Normal distribution over 14-dim actions
        """
        batch_size = action_hidden_states.shape[0]
        
        # Get action mean from action head (14-dim per timestep)
        if hasattr(self.action_head, 'module'):
            action_mean = self.action_head.module.predict_action(action_hidden_states)
        else:
            action_mean = self.action_head.predict_action(action_hidden_states)
        
        # action_mean: (B, chunk_len, 14) -> (B, chunk_len * 14)
        action_mean = action_mean.reshape(batch_size, -1)
        
        # Get standard deviation (clamped)
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
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state (14-dim)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action [B, chunk, 14], log_prob [B], entropy [B])
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
        
        # Reshape action to (B, chunk_len, 14)
        action = action.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
        
        return action, log_prob, entropy
    
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
        
        Args:
            inputs: Processed VLA inputs
            actions: 14-dim bimanual actions to evaluate (B, chunk_len, 14)
            proprio: Proprioceptive state (14-dim)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            
        Returns:
            Tuple of (log_prob, entropy)
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
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state (14-dim)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Value tensor (B,)
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
        
        This is efficient because both Action Head and Value Head share
        the same VLA hidden states - we only run VLA backbone once!
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state (14-dim)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            deterministic: If True, return mean action
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Tuple of (action [B, chunk, 14], log_prob [B], entropy [B], value [B])
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
        
        # === Action Head (14-dim) ===
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = torch.clamp(action, -1.0, 1.0)
        action = action.reshape(batch_size, self.num_actions_chunk, self.model_action_dim)
        
        # === Per-agent Value Head ===
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
        
        Args:
            inputs: Processed VLA inputs
            actions: 14-dim bimanual actions to evaluate (B, chunk_len, 14)
            proprio: Proprioceptive state (14-dim)
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Tuple of (log_prob, entropy, value)
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
        
        # === Action evaluation ===
        dist = self.get_action_distribution(action_hidden_states)
        actions_flat = actions.reshape(batch_size, -1)
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # === Per-agent value computation ===
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return log_prob, entropy, value


class DualArmMultiAgentVLAPolicy(nn.Module):
    """
    Multi-Agent VLA Policy for Dual-Arm MAPPO.
    
    Manages the bimanual VLA model with optional parameter sharing.
    Key difference from single-arm: VLA outputs 14-dim action, split per agent.
    
    Training modes:
    - freeze_vla_backbone=True: Only train action head MLP (fast, memory efficient)
    - freeze_vla_backbone=False: Train entire VLA (slow, requires more memory)
    """
    
    def __init__(
        self,
        cfg: DualArmMAPPOConfig,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize multi-agent policy.
        
        Args:
            cfg: Dual-Arm MAPPO configuration
            vla_model: Pretrained OpenVLA bimanual model
            action_head: Action prediction head (outputs 14-dim)
            proprio_projector: Proprioceptive state projector
            noisy_action_projector: Noisy action projector for diffusion
            processor: VLA processor
            device: Target device
        """
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = DUALARM_MAPPO_CONSTANTS["NUM_AGENTS"]
        self.share_policy = cfg.share_policy
        self.device = device
        self.model_action_dim = cfg.model_action_dim  # 14
        self.agent_action_dim = cfg.agent_action_dim  # 7
        
        print_rank0("\n" + "="*60)
        print_rank0("Initializing Dual-Arm Multi-Agent VLA Policy for MAPPO")
        print_rank0("="*60)
        print_rank0(f"  VLA backbone frozen: {cfg.freeze_vla_backbone}")
        print_rank0(f"  Train action head: {cfg.train_action_head}")
        print_rank0(f"  Train proprio projector: {cfg.train_proprio_projector}")
        print_rank0(f"  Shared policy: {cfg.share_policy}")
        print_rank0(f"  Per-agent Value Heads: {self.num_agents}")
        print_rank0(f"  Model action dim: {cfg.model_action_dim} (bimanual)")
        print_rank0(f"  Agent action dim: {cfg.agent_action_dim} (per arm)")
        print_rank0("="*60 + "\n")
        
        if self.share_policy:
            # Create single shared agent with per-agent Value Heads
            self.shared_agent = DualArmVLAAgent(
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
                value_head_hidden_dim=cfg.value_hidden_dim,
                value_head_num_layers=cfg.value_num_layers,
                num_agents=self.num_agents,
            )
            self.agents = [self.shared_agent] * self.num_agents
        else:
            # Create separate agents
            self.agents = nn.ModuleList([
                DualArmVLAAgent(
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
                    value_head_hidden_dim=cfg.value_hidden_dim,
                    value_head_num_layers=cfg.value_num_layers,
                    num_agents=self.num_agents,
                )
                for _ in range(self.num_agents)
            ])
    
    def _split_action_for_agent(self, full_action: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """
        Split 14-dim bimanual action for specific agent.
        
        Args:
            full_action: Full bimanual action (B, chunk, 14)
            agent_idx: Agent index (0 for left arm, 1 for right arm)
            
        Returns:
            Per-agent action (B, chunk, 7)
        """
        if agent_idx == 0:
            return full_action[..., :self.agent_action_dim]
        else:
            return full_action[..., self.agent_action_dim:]
    
    def get_actions(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents.
        
        The VLA outputs 14-dim action, which is split per agent.
        Each agent gets their 7-dim slice.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of padded proprio states per agent (14-dim each)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions [7-dim each], log_probs, entropies) lists for all agents
        """
        actions = []
        log_probs = []
        entropies = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            # Get 14-dim action from VLA
            full_action, log_prob, entropy = agent.get_action_and_log_prob(
                inputs=inputs,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
            )
            
            # Split to get 7-dim per-agent action
            agent_action = self._split_action_for_agent(full_action, agent_idx)
            
            actions.append(agent_action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return actions, log_probs, entropies
    
    def evaluate_actions(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_actions: List[torch.Tensor],
        agent_proprios: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Evaluate log probabilities and entropies for given actions.
        
        Note: agent_actions are expected to be 14-dim full bimanual actions.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_actions: List of 14-dim actions per agent (reconstructed full action)
            agent_proprios: List of padded proprio states per agent
            
        Returns:
            Tuple of (log_probs, entropies) lists for all agents
        """
        log_probs = []
        entropies = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            actions = agent_actions[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            log_prob, entropy = agent.evaluate_actions(
                inputs=inputs,
                actions=actions,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
            )
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return log_probs, entropies
    
    def get_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Get state values for all agents using per-agent Value Heads.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of padded proprio states per agent
            
        Returns:
            List of value tensors for all agents [V^(0), V^(1)]
        """
        values = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            text_hidden_states, _ = agent.forward(
                inputs=inputs,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
            )
            value = agent.value_heads[agent_idx](text_hidden_states)
            
            values.append(value)
        
        return values
    
    def get_actions_and_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions AND values for all agents in one efficient forward pass.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of padded proprio states per agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions [7-dim], log_probs, entropies, values) lists for all agents
        """
        actions = []
        log_probs = []
        entropies = []
        values = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            # === Single VLA forward pass ===
            text_hidden_states, num_patches = agent.forward(
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
            
            # === Action computation (14-dim) ===
            dist = agent.get_action_distribution(action_hidden_states)
            
            if deterministic:
                full_action = dist.mean
            else:
                full_action = dist.rsample()
            
            log_prob = dist.log_prob(full_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            full_action = torch.clamp(full_action, -1.0, 1.0)
            full_action = full_action.reshape(batch_size, agent.num_actions_chunk, agent.model_action_dim)
            
            # Split to get 7-dim per-agent action
            agent_action = self._split_action_for_agent(full_action, agent_idx)
            
            # === Per-agent value computation ===
            value = agent.value_heads[agent_idx](text_hidden_states)
            
            actions.append(agent_action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        
        return actions, log_probs, entropies, values
    
    def evaluate_actions_and_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_actions: List[torch.Tensor],
        agent_proprios: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Evaluate actions AND get values in one forward pass (for PPO update).
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_actions: List of 14-dim actions per agent (reconstructed full action)
            agent_proprios: List of padded proprio states per agent
            
        Returns:
            Tuple of (log_probs, entropies, values) lists for all agents
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
        params = []
        
        if self.share_policy:
            params.extend(self.shared_agent.get_trainable_parameters())
        else:
            for agent in self.agents:
                params.extend(agent.get_trainable_parameters())
        
        return params
    
    def get_actor_parameters(self) -> List[nn.Parameter]:
        """Get actor (policy) parameters."""
        params = []
        
        if self.share_policy:
            params.extend(self.shared_agent.get_actor_parameters())
        else:
            for agent in self.agents:
                params.extend(agent.get_actor_parameters())
        
        return params
    
    def get_critic_parameters(self) -> List[nn.Parameter]:
        """Get critic (value function) parameters."""
        params = []
        
        if self.share_policy:
            params.extend(self.shared_agent.get_critic_parameters())
        else:
            for agent in self.agents:
                params.extend(agent.get_critic_parameters())
        
        return params
    
    def forward_evaluate_agent(
        self,
        agent_idx: int,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for evaluating a single agent's actions.
        
        Args:
            agent_idx: Index of the agent to evaluate
            inputs: Processed VLA inputs
            actions: 14-dim actions to evaluate (B, chunk_len, 14)
            proprio: Proprioceptive state (14-dim padded)
            
        Returns:
            Tuple of (log_prob, entropy, value)
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
        
        Args:
            mode: One of 'evaluate_agent', etc.
            **kwargs: Arguments passed to the sub-method
        
        Returns:
            Output from the sub-method
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
    """
    Check if the checkpoint is a bimanual (ALOHA-style) model.
    """
    import os
    
    # Check for ALOHA-related names
    aloha_keywords = ["aloha", "bimanual", "dual", "dualarm"]
    
    checkpoint_lower = checkpoint_path.lower()
    for keyword in aloha_keywords:
        if keyword in checkpoint_lower:
            return True
    
    return False


def load_vla_for_dualarm_mappo(
    cfg: DualArmMAPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module], Any, Optional[Dict]]:
    """
    Load VLA model and components for Dual-Arm MAPPO training.
    
    This loads a bimanual VLA model (ALOHA-style) that outputs 14-dim actions.
    
    Args:
        cfg: Dual-Arm MAPPO configuration
        device: Target device
        
    Returns:
        Tuple of (vla_model, action_head, proprio_projector, noisy_action_projector, processor, norm_stats)
    """
    import sys
    sys.path.append("../../..")
    
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
    
    print_rank0("Loading VLA bimanual model for Dual-Arm MAPPO training...")
    
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
