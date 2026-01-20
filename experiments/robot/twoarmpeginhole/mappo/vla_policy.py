"""
Multi-Agent VLA Policy wrapper for MAPPO training.

This module provides the actor policy using OpenVLA-OFT models for
multi-agent reinforcement learning in the TwoArmPegInHole environment.

Architecture:
    VLA Backbone ─┬─────────→ Action Head → Action (policy)
                  └─────────→ Value Head  → Value (critic)
                  
Both heads share the same VLA hidden states, making training efficient.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import MAPPOConfig, TWOARM_MAPPO_CONSTANTS


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


class VLAAgent(nn.Module):
    """
    Single VLA agent wrapper for MAPPO (Actor-Critic).
    
    Architecture:
        VLA Backbone ─┬─────────→ Action Head → Action (policy/actor)
                      └─────────→ Value Heads → Per-agent Values (critic)
    
    All Value Heads share the same VLA hidden states representation.
    
    Key difference from Fine-tuning:
    - VLA backbone can be frozen (only heads train)
    - Action output becomes stochastic (mean + std for exploration)
    - Multiple Value Heads for per-agent value estimation
    - Uses PPO loss instead of L1 loss
    """
    
    def __init__(
        self,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        action_dim: int = 6,
        num_actions_chunk: int = 4,
        action_std_init: float = 0.5,
        min_action_std: float = 0.01,
        device: torch.device = torch.device("cuda"),
        freeze_vla_backbone: bool = True,
        train_proprio_projector: bool = False,
        train_action_head: bool = True,
        train_value_head: bool = True,
        value_head_hidden_dim: int = 512,
        value_head_num_layers: int = 2,
        num_agents: int = 2,  # Per-agent value heads
    ):
        """
        Initialize VLA agent with both Action Head and per-agent Value Heads.
        
        Args:
            vla_model: Pretrained OpenVLA model
            action_head: Action prediction head (L1 or Diffusion)
            proprio_projector: Proprioceptive state projector
            noisy_action_projector: Noisy action projector for diffusion
            processor: VLA processor for inputs
            action_dim: Action dimensionality per timestep
            num_actions_chunk: Number of actions to predict
            action_std_init: Initial standard deviation for action distribution
            min_action_std: Minimum standard deviation
            device: Target device
            freeze_vla_backbone: If True, freeze VLA backbone (don't train it)
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
        
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.total_action_dim = action_dim * num_actions_chunk
        self.device = device
        self.num_agents = num_agents
        
        # Store training flags
        self.freeze_vla_backbone = freeze_vla_backbone
        self.train_proprio_projector = train_proprio_projector
        self.train_action_head = train_action_head
        self.train_value_head = train_value_head
        
        # Get VLA hidden dimension for Value Heads
        llm_dim = vla_model.llm_dim if hasattr(vla_model, 'llm_dim') else 4096
        
        # Create per-agent Value Heads (each shares VLA hidden states with Action Head)
        # This enables separate value estimation per agent
        self.value_heads = nn.ModuleList([
            ValueHead(
                input_dim=llm_dim,
                hidden_dim=value_head_hidden_dim,
                num_layers=value_head_num_layers,
            ).to(torch.bfloat16).to(device)
            for _ in range(num_agents)
        ])
        # For backward compatibility
        self.value_head = self.value_heads[0]
        
        print_rank0(f"\nCreated {num_agents} per-agent Value Heads on top of VLA backbone:")
        print_rank0(f"  Input dim: {llm_dim}")
        print_rank0(f"  Hidden dim: {value_head_hidden_dim}")
        print_rank0(f"  Num layers: {value_head_num_layers}")
        
        # Apply freeze settings
        self._setup_trainable_parameters()
        
        # Learnable action standard deviation (always trainable for RL)
        # Use bfloat16 to match VLA output dtype
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
            # Count frozen params
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
        print_rank0("Trainable Parameters Summary:")
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
            print_rank0(f"  Action head: {ah_trainable:,} / {ah_total:,} trainable")
        
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
            print_rank0(f"  Proprio projector: {pp_trainable:,} / {pp_total:,} trainable")
        
        # log_std (always trainable for RL exploration)
        log_std_params = self.total_action_dim
        trainable_params += log_std_params
        total_params += log_std_params
        print_rank0(f"  Log std: {log_std_params:,} trainable")
        
        print_rank0("-"*50)
        print_rank0(f"  TOTAL: {trainable_params:,} / {total_params:,} trainable parameters")
        print_rank0(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print_rank0("-"*50 + "\n")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get only the trainable parameters for optimizer.
        
        Returns:
            List of trainable parameters (action head + value head + log_std)
        """
        params = []
        
        # VLA backbone (if not frozen)
        if not self.freeze_vla_backbone:
            params.extend([p for p in self.vla.parameters() if p.requires_grad])
        
        # Action head (if trainable)
        if self.action_head is not None and self.train_action_head:
            params.extend([p for p in self.action_head.parameters() if p.requires_grad])
        
        # Per-agent value heads (if trainable)
        if self.value_heads is not None and self.train_value_head:
            for vh in self.value_heads:
                params.extend([p for p in vh.parameters() if p.requires_grad])
        
        # Proprio projector (if trainable)
        if self.proprio_projector is not None and self.train_proprio_projector:
            params.extend([p for p in self.proprio_projector.parameters() if p.requires_grad])
        
        # log_std is always trainable for RL exploration
        params.append(self.log_std)
        
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
            proprio: Proprioceptive state tensor
            use_proprio: Whether to use proprio input
            use_film: Whether to use FiLM
            
        Returns:
            Tuple of (action_hidden_states, num_patches)
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Create dummy labels filled with IGNORE_INDEX (-100) for inference
            # This makes action mask all False (no action tokens to mask)
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
        
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        num_patches = self._get_num_patches(use_proprio)
        
        # Extract action-related hidden states
        # The text hidden states start after vision patches
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        
        return text_hidden_states, num_patches
    
    def get_action_distribution(
        self,
        action_hidden_states: torch.Tensor,
    ) -> Normal:
        """
        Get action distribution from hidden states.
        
        Args:
            action_hidden_states: Hidden states for action prediction
            
        Returns:
            Normal distribution over actions
        """
        # Predict mean action from action head
        batch_size = action_hidden_states.shape[0]
        
        # Reshape for action head
        # action_hidden_states: (B, seq_len, hidden_dim)
        # We need: (B, chunk_len * action_dim, hidden_dim)
        
        # Get action mean from action head
        if hasattr(self.action_head, 'module'):
            action_mean = self.action_head.module.predict_action(action_hidden_states)
        else:
            action_mean = self.action_head.predict_action(action_hidden_states)
        
        # action_mean: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim)
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
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob, entropy)
        """
        # Get hidden states
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        # For action prediction, we need the hidden states corresponding to action tokens
        # This depends on how the action tokenizer works
        # For now, use a simplified approach - take all text hidden states
        batch_size = inputs["input_ids"].shape[0]
        
        # Reshape hidden states for action head
        # We'll use all available hidden states up to chunk_len * action_dim
        action_hidden_dim = self.num_actions_chunk * self.action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            # Pad if necessary
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        # Get distribution
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # Reparameterized sample for gradient flow
        
        # Compute log probability (before clipping for correct gradient)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        # Clip action to [-1, 1] range to ensure valid normalized actions
        # This prevents out-of-bound actions when unnormalizing
        action = torch.clamp(action, -1.0, 1.0)
        
        # Reshape action to (B, chunk_len, action_dim)
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        
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
            actions: Actions to evaluate (B, chunk_len, action_dim)
            proprio: Proprioceptive state
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        # Get hidden states
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.action_dim
        
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        # Get distribution
        dist = self.get_action_distribution(action_hidden_states)
        
        # Flatten actions for log_prob computation
        actions_flat = actions.reshape(batch_size, -1)
        
        # Compute log probability
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def get_value(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        agent_idx: int = 0,  # Use per-agent value head
    ) -> torch.Tensor:
        """
        Get state value from per-agent Value Head.
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Value tensor (B,)
        """
        # Get hidden states
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        # Use per-agent Value Head to compute value from hidden states
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return value
    
    def get_action_and_value(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        deterministic: bool = False,
        agent_idx: int = 0,  # Use per-agent value head
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, AND value in one forward pass.
        
        This is efficient because both Action Head and Value Head share
        the same VLA hidden states - we only run VLA backbone once!
        Uses per-agent value head V^(agent_idx).
        
        Args:
            inputs: Processed VLA inputs
            proprio: Proprioceptive state
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            deterministic: If True, return mean action
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        # Get hidden states ONCE (shared between action and value)
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.action_dim
        
        # Extract action hidden states
        if text_hidden_states.shape[1] >= action_hidden_dim:
            action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
        else:
            pad_len = action_hidden_dim - text_hidden_states.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, text_hidden_states.shape[-1],
                device=text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
        
        # === Action Head ===
        dist = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        # Compute log probability (before clipping for correct gradient)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Clip action to [-1, 1] range to ensure valid normalized actions
        action = torch.clamp(action, -1.0, 1.0)
        
        # Reshape action
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        
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
        agent_idx: int = 0,  # Use per-agent value head
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions AND get value in one forward pass.
        
        Used during PPO update to compute both policy loss and value loss.
        Uses per-agent value head V^(agent_idx).
        
        Args:
            inputs: Processed VLA inputs
            actions: Actions to evaluate (B, chunk_len, action_dim)
            proprio: Proprioceptive state
            use_proprio: Whether to use proprio
            use_film: Whether to use FiLM
            agent_idx: Index of agent to use corresponding value head
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        # Get hidden states ONCE
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        action_hidden_dim = self.num_actions_chunk * self.action_dim
        
        # Extract action hidden states
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


class MultiAgentVLAPolicy(nn.Module):
    """
    Multi-Agent VLA Policy for MAPPO.
    
    Manages multiple VLA agents with optional parameter sharing.
    
    Training modes:
    - freeze_vla_backbone=True: Only train action head MLP (fast, memory efficient)
    - freeze_vla_backbone=False: Train entire VLA (slow, requires more memory)
    """
    
    def __init__(
        self,
        cfg: MAPPOConfig,
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
            cfg: MAPPO configuration
            vla_model: Pretrained OpenVLA model
            action_head: Action prediction head
            proprio_projector: Proprioceptive state projector
            noisy_action_projector: Noisy action projector for diffusion
            processor: VLA processor
            device: Target device
        """
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]
        self.share_policy = cfg.share_policy
        self.device = device
        
        print_rank0("\n" + "="*60)
        print_rank0("Initializing Multi-Agent VLA Policy for MAPPO")
        print_rank0("="*60)
        print_rank0(f"  VLA backbone frozen: {cfg.freeze_vla_backbone}")
        print_rank0(f"  Train action head: {cfg.train_action_head}")
        print_rank0(f"  Train proprio projector: {cfg.train_proprio_projector}")
        print_rank0(f"  Shared policy: {cfg.share_policy}")
        print_rank0(f"  Per-agent Value Heads: {self.num_agents}")
        print_rank0("="*60 + "\n")
        
        if self.share_policy:
            # Create single shared agent with per-agent Value Heads
            # VLAAgent.value_heads contains V^(0), V^(1), ..., V^(N-1)
            self.shared_agent = VLAAgent(
                vla_model=vla_model,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                processor=processor,
                action_dim=cfg.action_dim,
                num_actions_chunk=cfg.num_actions_chunk,
                device=device,
                freeze_vla_backbone=cfg.freeze_vla_backbone,
                train_proprio_projector=cfg.train_proprio_projector,
                train_action_head=cfg.train_action_head,
                train_value_head=cfg.train_value_head,
                value_head_hidden_dim=cfg.value_hidden_dim,
                value_head_num_layers=cfg.value_num_layers,
                num_agents=self.num_agents,  # Creates per-agent value heads
            )
            self.agents = [self.shared_agent] * self.num_agents
        else:
            # Create separate agents (would need separate model copies)
            # Each agent has its own value heads
            self.agents = nn.ModuleList([
                VLAAgent(
                    vla_model=vla_model,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    processor=processor,
                    action_dim=cfg.action_dim,
                    num_actions_chunk=cfg.num_actions_chunk,
                    device=device,
                    freeze_vla_backbone=cfg.freeze_vla_backbone,
                    train_proprio_projector=cfg.train_proprio_projector,
                    train_action_head=cfg.train_action_head,
                    train_value_head=cfg.train_value_head,
                    value_head_hidden_dim=cfg.value_hidden_dim,
                    value_head_num_layers=cfg.value_num_layers,
                    num_agents=self.num_agents,  # Creates per-agent value heads
                )
                for _ in range(self.num_agents)
            ])
    
    def get_actions(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of proprio states per agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, entropies) lists for all agents
        """
        actions = []
        log_probs = []
        entropies = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            action, log_prob, entropy = agent.get_action_and_log_prob(
                inputs=inputs,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
            )
            
            actions.append(action)
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
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_actions: List of actions per agent
            agent_proprios: List of proprio states per agent
            
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
        
        Each agent uses its own Value Head V^(i) for value estimation.
        VLA forward is called ONCE per agent, hidden states are reused.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of proprio states per agent
            
        Returns:
            List of value tensors for all agents [V^(0), V^(1), ..., V^(N-1)]
        """
        values = []
        
        for agent_idx, agent in enumerate(self.agents[:self.num_agents]):
            inputs = agent_inputs[agent_idx]
            proprio = agent_proprios[agent_idx] if agent_proprios else None
            
            # Get hidden states from agent (single VLA forward)
            text_hidden_states, _ = agent.forward(
                inputs=inputs,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
            )
            # Use per-agent value head from VLAAgent.value_heads
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
        
        Uses per-agent Value Heads V^(i) for value estimation.
        VLA forward is called ONCE per agent - hidden states are reused for action and value.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_proprios: List of proprio states per agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, entropies, values) lists for all agents
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
            action_hidden_dim = agent.num_actions_chunk * agent.action_dim
            
            # Extract action hidden states
            if text_hidden_states.shape[1] >= action_hidden_dim:
                action_hidden_states = text_hidden_states[:, :action_hidden_dim, :]
            else:
                pad_len = action_hidden_dim - text_hidden_states.shape[1]
                padding = torch.zeros(
                    batch_size, pad_len, text_hidden_states.shape[-1],
                    device=text_hidden_states.device, dtype=text_hidden_states.dtype
                )
                action_hidden_states = torch.cat([text_hidden_states, padding], dim=1)
            
            # === Action computation from shared hidden states ===
            dist = agent.get_action_distribution(action_hidden_states)
            
            if deterministic:
                action = dist.mean
            else:
                action = dist.rsample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Clip action and reshape
            action = torch.clamp(action, -1.0, 1.0)
            action = action.reshape(batch_size, agent.num_actions_chunk, agent.action_dim)
            
            # === Per-agent value computation from shared hidden states ===
            value = agent.value_heads[agent_idx](text_hidden_states)
            
            actions.append(action)
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
        
        Uses per-agent Value Heads V^(i) for value estimation.
        VLA forward is called ONCE per agent - hidden states are reused.
        
        Args:
            agent_inputs: List of processed VLA inputs per agent
            agent_actions: List of actions per agent
            agent_proprios: List of proprio states per agent
            
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
            
            # Use per-agent value head with agent_idx
            # VLA forward is called ONCE inside evaluate_actions_and_value
            log_prob, entropy, value = agent.evaluate_actions_and_value(
                inputs=inputs,
                actions=actions,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                agent_idx=agent_idx,  # Use per-agent value head
            )
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        
        return log_probs, entropies, values
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get only the trainable parameters for optimizer.
        
        When freeze_vla_backbone=True, this returns only:
        - Action head parameters
        - Value head parameters
        - Proprio projector parameters (if train_proprio_projector=True)
        - log_std parameters (always trainable for RL)
        
        Returns:
            List of trainable parameters
        """
        params = []
        
        if self.share_policy:
            # Use agent's method to get trainable params
            params.extend(self.shared_agent.get_trainable_parameters())
        else:
            # All agents' trainable parameters
            for agent in self.agents:
                params.extend(agent.get_trainable_parameters())
        
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
        
        Uses per-agent Value Head V^(agent_idx) for value computation.
        VLA forward is called ONCE - hidden states are reused for action eval and value.
        
        IMPORTANT: This method should be called through the DDP wrapper to ensure
        proper gradient synchronization across GPUs. Do NOT call this on .module directly.
        
        Args:
            agent_idx: Index of the agent to evaluate
            inputs: Processed VLA inputs
            actions: Actions to evaluate (B, chunk_len, action_dim)
            proprio: Proprioceptive state
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        agent = self.agents[agent_idx]
        
        # Use per-agent value head with agent_idx
        # VLA forward is called ONCE inside evaluate_actions_and_value
        return agent.evaluate_actions_and_value(
            inputs=inputs,
            actions=actions,
            proprio=proprio,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
            agent_idx=agent_idx,  # Use per-agent value head
        )
    
    def forward(
        self,
        mode: str,
        **kwargs,
    ):
        """
        Generic forward method for DDP compatibility.
        
        DDP requires all forward passes to go through the wrapper for proper
        gradient synchronization. This method routes to appropriate sub-methods.
        
        Args:
            mode: One of 'evaluate_agent', 'get_actions', 'get_values', etc.
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


def _is_finetuned_checkpoint(checkpoint_path: str) -> bool:
    """
    Check if the checkpoint is a fine-tuned OpenVLA-OFT model (has proprio_projector).
    
    Returns True for:
    - HuggingFace fine-tuned models (moojink/openvla-7b-oft-finetuned-*)
    - Local checkpoints that contain proprio_projector files
    """
    import os
    
    # Known HF fine-tuned models
    hf_finetuned_models = [
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "moojink/openvla-7b-oft-finetuned-libero-10",
        "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    ]
    
    if checkpoint_path in hf_finetuned_models:
        return True
    
    # Check local directory for proprio_projector checkpoint
    if os.path.isdir(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if "proprio_projector" in filename and filename.endswith(".pt"):
                return True
    
    return False


def load_vla_for_mappo(
    cfg: MAPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module], Any, Optional[Dict]]:
    """
    Load VLA model and components for MAPPO training.
    
    Supports two modes:
    1. Fine-tuned checkpoint (e.g., moojink/openvla-7b-oft-finetuned-libero-*):
       - Load proprio_projector and action_head from checkpoint
       - Transfer learning from LIBERO to TwoArmPegInHole
    
    2. Base VLA (e.g., openvla/openvla-7b):
       - Initialize proprio_projector and action_head from scratch
       - RL training from scratch
    
    Args:
        cfg: MAPPO configuration
        device: Target device
        
    Returns:
        Tuple of (vla_model, action_head, proprio_projector, noisy_action_projector, processor, norm_stats)
        
        norm_stats contains action/proprio normalization statistics for unnormalizing
        VLA outputs before sending to environment.
    """
    import sys
    sys.path.append("../../..")
    
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
    
    print_rank0("Loading VLA model for MAPPO training...")
    
    # Check if this is a fine-tuned checkpoint
    is_finetuned = _is_finetuned_checkpoint(cfg.pretrained_checkpoint)
    
    if is_finetuned:
        print_rank0(f"  Detected FINE-TUNED checkpoint: {cfg.pretrained_checkpoint}")
        print_rank0("  Will LOAD proprio_projector and action_head from checkpoint")
    else:
        print_rank0(f"  Detected BASE VLA checkpoint: {cfg.pretrained_checkpoint}")
        print_rank0("  Will INITIALIZE proprio_projector and action_head from scratch")
    
    # Load base VLA model
    # Note: get_vla() now uses get_current_device() which respects torch.cuda.set_device()
    # so the model will be loaded to the correct GPU for each DDP rank
    vla = get_model(cfg)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    
    # Ensure model is on the correct device (should already be, but explicit is better)
    vla = vla.to(device)
    
    # Clear any temporary memory
    torch.cuda.empty_cache()
    
    # Synchronize all processes after model loading
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # Get processor
    processor = get_processor(cfg)
    
    # Get normalization statistics from VLA model (if available)
    # These are needed to unnormalize action outputs before sending to environment
    norm_stats = getattr(vla, 'norm_stats', None)
    if norm_stats is not None:
        print_rank0("Loaded action normalization statistics from VLA model")
        # Get the first key if multiple datasets
        if len(norm_stats) > 0:
            first_key = list(norm_stats.keys())[0]
            print_rank0(f"  Using normalization key: {first_key}")
    else:
        print_rank0("WARNING: No normalization statistics found in VLA model!")
        print_rank0("  Actions will NOT be unnormalized before environment execution.")
    
    # Load or initialize proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        if is_finetuned:
            # Load from fine-tuned checkpoint
            from experiments.robot.openvla_utils import get_proprio_projector
            print_rank0("Loading proprio projector from fine-tuned checkpoint...")
            proprio_projector = get_proprio_projector(
                cfg,
                vla.llm_dim,
                proprio_dim=TWOARM_MAPPO_CONSTANTS["PROPRIO_DIM"],
            )
            print_rank0(f"  Loaded proprio projector: input_dim={TWOARM_MAPPO_CONSTANTS['PROPRIO_DIM']}, output_dim={vla.llm_dim}")
        else:
            # Initialize fresh for base VLA
            print_rank0("Initializing NEW proprio projector (base VLA has no proprio_projector)")
            proprio_projector = ProprioProjector(
                llm_dim=vla.llm_dim,
                proprio_dim=TWOARM_MAPPO_CONSTANTS["PROPRIO_DIM"],
            )
            proprio_projector = proprio_projector.to(torch.bfloat16).to(device)
            print_rank0(f"  Initialized proprio projector: input_dim={TWOARM_MAPPO_CONSTANTS['PROPRIO_DIM']}, output_dim={vla.llm_dim}")
    
    # Initialize action head (ALWAYS new - ACTION_DIM differs: LIBERO=7, TwoArm=6)
    # Even with fine-tuned checkpoint, we cannot load action_head due to dimension mismatch
    action_head = None
    if cfg.use_l1_regression:
        print_rank0(f"Initializing L1 regression action head (action_dim={cfg.action_dim}, num_actions_chunk={cfg.num_actions_chunk})")
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=cfg.action_dim,
            num_actions_chunk=cfg.num_actions_chunk,
        )
        action_head = action_head.to(torch.bfloat16).to(device)
    elif cfg.use_diffusion:
        print_rank0(f"Initializing diffusion action head (action_dim={cfg.action_dim}, num_actions_chunk={cfg.num_actions_chunk})")
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=cfg.action_dim,
            num_diffusion_steps_train=cfg.num_diffusion_steps_train,
            num_actions_chunk=cfg.num_actions_chunk,
        )
        action_head = action_head.to(torch.bfloat16).to(device)
    
    # Noisy action projector (only for diffusion, always initialize new)
    noisy_action_projector = None
    if cfg.use_diffusion:
        from prismatic.models.projectors import NoisyActionProjector
        print_rank0("Initializing NEW noisy action projector")
        noisy_action_projector = NoisyActionProjector(llm_dim=vla.llm_dim)
        noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(device)
    
    return vla, action_head, proprio_projector, noisy_action_projector, processor, norm_stats
