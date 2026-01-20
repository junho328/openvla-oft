"""
Multi-Agent VLA Policy wrapper for ACPPO training.

ACPPO (Agent-Chained PPO) key changes from MAPPO:
1. All agents act SIMULTANEOUSLY - they all decide and execute actions at the same time
2. Agent 0: Uses only its own observation
3. Agent 1+: Additionally uses ESTIMATED action distribution from previous agents
4. Action distribution estimation uses:
   - Front view only (wrist image is padded with zeros)
   - Previous agent's text instruction (e.g., "You are robot 0...")
   - Shared VLM + action head
5. No gradient flows through the estimation process (detached)
6. All components (value head, proprio_projector, action_head) are shared

IMPORTANT: Agents do NOT act sequentially! They all act at the same time.
The "chaining" refers to the information flow where agent i has access to
estimated action distributions from agents 0, 1, ..., i-1.

Architecture (Simultaneous Execution):
    Agent 0 (parallel):
        [front_view, wrist_0] + proprio_0 → VLA → Action Head → Action_0 (mu_0, sigma_0)
        
    Agent 1 (parallel, with estimated action dist from Agent 0):
        Step 1 (Estimation, no grad):
            [front_view, zero_pad] + proprio (as robot 0) → VLA → Action Head → (mu_0_est, sigma_0_est)
        
        Step 2 (Forward with estimated action dist):
            [front_view, wrist_1] + [proprio_1; mu_0_est; sigma_0_est] → VLA → Action Head → Action_1
    
    Both agents' actions are applied to the environment simultaneously.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import ACPPOConfig, TWOARM_ACPPO_CONSTANTS
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


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
    
    For ACPPO, we have per-agent value functions:
    - V^(1)(s_t) for agent 0
    - V^(2)([s_t, b_t^(2)]) for agent 1, where b_t^(2) includes estimated action dist
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
    """
    
    def __init__(
        self,
        # action_dist_dim = ACTION_DIM * NUM_ACTIONS_CHUNK * 2 (mu + sigma)
        action_dist_dim: int = ACTION_DIM * NUM_ACTIONS_CHUNK * 2,
        output_dim: int = ACTION_DIM * NUM_ACTIONS_CHUNK * 2,
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


class VLAAgentACPPO(nn.Module):
    """
    Single VLA agent wrapper for ACPPO (Actor-Critic).
    
    Key difference from MAPPO VLAAgent:
    - Supports extended proprio input (proprio + action_dist for agent 1)
    - Can estimate action distribution with front-view only input
    - Shared components across agents
    - Per-agent value heads for proper ACPPO credit assignment
    """
    
    def __init__(
        self,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        action_dim: int = ACTION_DIM,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        action_std_init: float = 0.5,
        min_action_std: float = 0.01,
        device: torch.device = torch.device("cuda"),
        freeze_vla_backbone: bool = True,
        train_proprio_projector: bool = False,
        train_action_head: bool = True,
        train_value_head: bool = True,
        value_head_hidden_dim: int = 512,
        value_head_num_layers: int = 2,
        num_agents: int = 2,  # ACPPO: per-agent value heads
    ):
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
        
        self.freeze_vla_backbone = freeze_vla_backbone
        self.train_proprio_projector = train_proprio_projector
        self.train_action_head = train_action_head
        self.train_value_head = train_value_head
        
        llm_dim = vla_model.llm_dim if hasattr(vla_model, 'llm_dim') else 4096
        
        # ACPPO: Create per-agent value heads for proper credit assignment
        # V^(0)(s_t) and V^(1)([s_t, b_t^(1)]) need different value functions
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
        
        self._setup_trainable_parameters()
        
        self.log_std = nn.Parameter(
            torch.ones(self.total_action_dim, device=device, dtype=torch.bfloat16) * np.log(action_std_init)
        )
        self.min_log_std = np.log(min_action_std)
        
        self._num_patches = None
        self._print_trainable_summary()
    
    def _setup_trainable_parameters(self):
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
        
        # ACPPO: Set trainable for all per-agent value heads
        for value_head in self.value_heads:
            for param in value_head.parameters():
                param.requires_grad = self.train_value_head
    
    def _print_trainable_summary(self):
        total_params = 0
        trainable_params = 0
        
        print_rank0("\n" + "-"*50)
        print_rank0("ACPPO Trainable Parameters Summary:")
        print_rank0("-"*50)
        
        vla_total = sum(p.numel() for p in self.vla.parameters())
        vla_trainable = sum(p.numel() for p in self.vla.parameters() if p.requires_grad)
        total_params += vla_total
        trainable_params += vla_trainable
        print_rank0(f"  VLA backbone: {vla_trainable:,} / {vla_total:,} trainable")
        
        if self.action_head is not None:
            ah_total = sum(p.numel() for p in self.action_head.parameters())
            ah_trainable = sum(p.numel() for p in self.action_head.parameters() if p.requires_grad)
            total_params += ah_total
            trainable_params += ah_trainable
            print_rank0(f"  Action head: {ah_trainable:,} / {ah_total:,} trainable")
        
        # ACPPO: Count all per-agent value heads
        if self.value_heads is not None:
            for i, vh in enumerate(self.value_heads):
                vh_total = sum(p.numel() for p in vh.parameters())
                vh_trainable = sum(p.numel() for p in vh.parameters() if p.requires_grad)
                total_params += vh_total
                trainable_params += vh_trainable
                print_rank0(f"  Value head (agent {i}): {vh_trainable:,} / {vh_total:,} trainable")
        
        if self.proprio_projector is not None:
            pp_total = sum(p.numel() for p in self.proprio_projector.parameters())
            pp_trainable = sum(p.numel() for p in self.proprio_projector.parameters() if p.requires_grad)
            total_params += pp_total
            trainable_params += pp_trainable
            print_rank0(f"  Proprio projector: {pp_trainable:,} / {pp_total:,} trainable")
        
        log_std_params = self.total_action_dim
        trainable_params += log_std_params
        total_params += log_std_params
        print_rank0(f"  Log std: {log_std_params:,} trainable")
        
        print_rank0("-"*50)
        print_rank0(f"  TOTAL: {trainable_params:,} / {total_params:,} trainable")
        print_rank0("-"*50 + "\n")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        
        if not self.freeze_vla_backbone:
            params.extend([p for p in self.vla.parameters() if p.requires_grad])
        
        if self.action_head is not None and self.train_action_head:
            params.extend([p for p in self.action_head.parameters() if p.requires_grad])
        
        # ACPPO: Include all per-agent value heads
        if self.value_heads is not None and self.train_value_head:
            for vh in self.value_heads:
                params.extend([p for p in vh.parameters() if p.requires_grad])
        
        if self.proprio_projector is not None and self.train_proprio_projector:
            params.extend([p for p in self.proprio_projector.parameters() if p.requires_grad])
        
        params.append(self.log_std)
        
        return params
    
    def _get_num_patches(self, use_proprio: bool = False, use_diffusion: bool = False) -> int:
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
    ) -> Tuple[torch.Tensor, int]:
        """Forward pass through VLA to get hidden states."""
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
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        Get action distribution from hidden states.
        
        Returns:
            Tuple of (Normal distribution, mu, sigma)
        """
        batch_size = action_hidden_states.shape[0]
        
        if hasattr(self.action_head, 'module'):
            action_mean = self.action_head.module.predict_action(action_hidden_states)
        else:
            action_mean = self.action_head.predict_action(action_hidden_states)
        
        action_mean = action_mean.reshape(batch_size, -1)
        
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        action_std = log_std.exp().expand_as(action_mean)
        
        return Normal(action_mean, action_std), action_mean, action_std
    
    def get_action_and_log_prob(
        self,
        inputs: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Returns:
            Tuple of (action, log_prob, entropy, action_mean, action_std)
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        # Use constants to ensure consistency with action_heads.py
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
        
        dist, action_mean, action_std = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        
        return action, log_prob, entropy, action_mean, action_std
    
    def evaluate_actions(
        self,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given actions."""
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        # Use constants to ensure consistency with action_heads.py
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
        
        dist, _, _ = self.get_action_distribution(action_hidden_states)
        
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
        agent_idx: int = 0,  # ACPPO: select per-agent value head
    ) -> torch.Tensor:
        """Get state value from per-agent Value Head."""
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
        agent_idx: int = 0,  # ACPPO: select per-agent value head
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, value, action_mean, and action_std.
        
        Returns:
            Tuple of (action, log_prob, entropy, value, action_mean, action_std)
        """
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        # Use constants to ensure consistency with action_heads.py
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
        
        # Action Head
        dist, action_mean, action_std = self.get_action_distribution(action_hidden_states)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        
        # Value Head (per-agent for ACPPO)
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return action, log_prob, entropy, value, action_mean, action_std
    
    def evaluate_actions_and_value(
        self,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        use_proprio: bool = True,
        use_film: bool = False,
        agent_idx: int = 0,  # ACPPO: select per-agent value head
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions AND get value in one forward pass."""
        text_hidden_states, num_patches = self.forward(
            inputs, proprio, use_proprio, use_film
        )
        
        batch_size = inputs["input_ids"].shape[0]
        # Use constants to ensure consistency with action_heads.py
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
        
        dist, _, _ = self.get_action_distribution(action_hidden_states)
        actions_flat = actions.reshape(batch_size, -1)
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        value = self.value_heads[agent_idx](text_hidden_states)
        
        return log_prob, entropy, value


class MultiAgentVLAPolicyACPPO(nn.Module):
    """
    Multi-Agent VLA Policy for ACPPO with Agent-Chaining.
    
    IMPORTANT: All agents act SIMULTANEOUSLY! The "chaining" refers to the 
    information flow, not the execution order.
    
    Key ACPPO features:
    1. All agents decide and execute their actions at the SAME TIME
    2. Agent 0: Standard forward pass with its own observations
    3. Agent 1+: Additionally estimates previous agents' action distributions
       and uses them as extra input (but still acts simultaneously)
    4. All components (VLA, action_head, value_head, proprio_projector) are shared
    
    Simultaneous Execution Process:
        Agent 0 (runs in parallel):
            - Input: [front_view, wrist_0] + proprio_0
            - Output: action_0, value_0, (mu_0, sigma_0)
            
        Agent 1 (runs in parallel with Agent 0):
            Step 1: Estimate Agent 0's action distribution (no gradient)
                - Input: front_view only (wrist image padded with zeros)
                - Text: "You are robot 0. What action..."
                - Output: estimated (mu_0, sigma_0) - detached
                
            Step 2: Forward pass with estimated action dist
                - Input: [front_view, wrist_1] + [proprio_1; mu_0_est; sigma_0_est]
                - Text: "You are robot 1. What action..."
                - Output: action_1, value_1
        
        Both actions are applied to the environment at the same time step.
    """
    
    def __init__(
        self,
        cfg: ACPPOConfig,
        vla_model: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module] = None,
        proprio_projector_extended: Optional[nn.Module] = None,  # For agent 1 with action dist
        noisy_action_projector: Optional[nn.Module] = None,
        processor: Any = None,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        
        self.cfg = cfg
        self.num_agents = TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]
        self.device = device
        
        print_rank0("\n" + "="*60)
        print_rank0("Initializing Multi-Agent VLA Policy for ACPPO")
        print_rank0("="*60)
        print_rank0(f"  VLA backbone frozen: {cfg.freeze_vla_backbone}")
        print_rank0(f"  Train action head: {cfg.train_action_head}")
        print_rank0(f"  Train proprio projector: {cfg.train_proprio_projector}")
        print_rank0(f"  Use action dist input for agent 1: {cfg.use_action_dist_input}")
        print_rank0(f"  Action dist dim: {cfg.action_dist_dim}")
        print_rank0("="*60 + "\n")
        
        # Shared VLA agent (used by both agents)
        self.shared_agent = VLAAgentACPPO(
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
        )
        
        # Extended proprio projector for agent 1 (proprio + action_dist)
        # This projects the extended proprio (8 + action_dist_dim) to the same space
        if cfg.use_action_dist_input and proprio_projector_extended is not None:
            self.proprio_projector_extended = proprio_projector_extended
        else:
            # Create extended proprio projector if not provided
            llm_dim = vla_model.llm_dim if hasattr(vla_model, 'llm_dim') else 4096
            from prismatic.models.projectors import ProprioProjector
            self.proprio_projector_extended = ProprioProjector(
                llm_dim=llm_dim,
                proprio_dim=cfg.proprio_dim_agent1,  # 8 + action_dist_dim
            ).to(torch.bfloat16).to(device)
            print_rank0(f"Created extended proprio projector: input={cfg.proprio_dim_agent1}, output={llm_dim}")
        
        # Cache for storing estimated action distribution
        self._cached_action_dist: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def estimate_agent0_action_dist(
        self,
        front_view_input: Dict[str, torch.Tensor],
        proprio_agent1_as_robot0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate Agent 0's action distribution for Agent 1's input.
        
        This is run WITHOUT gradient to avoid double computation.
        
        Args:
            front_view_input: VLA inputs with front view only (wrist padded with zeros)
            proprio_agent1_as_robot0: Agent 1's proprio formatted as robot 0
            
        Returns:
            Tuple of (mu_0_est, sigma_0_est) - detached tensors
        """
        with torch.no_grad():
            _, _, _, _, mu_est, sigma_est = self.shared_agent.get_action_and_value(
                inputs=front_view_input,
                proprio=proprio_agent1_as_robot0,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=True,  # Use mean for estimation
            )
            
            # Detach to ensure no gradient flows
            mu_est = mu_est.detach()
            sigma_est = sigma_est.detach()
        
        return mu_est, sigma_est
    
    def get_actions(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        front_view_only_input: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], 
               List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents with ACPPO chaining.
        
        Args:
            agent_inputs: List of VLA inputs per agent
            agent_proprios: List of proprio states per agent
            front_view_only_input: Input for estimating agent 0's action dist (front view only)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, entropies, action_means, action_stds)
        """
        actions = []
        log_probs = []
        entropies = []
        action_means = []
        action_stds = []
        
        # === Agent 0: Standard forward pass ===
        # Note: get_action_and_log_prob doesn't need agent_idx as it doesn't use value head
        action_0, log_prob_0, entropy_0, mu_0, sigma_0 = self.shared_agent.get_action_and_log_prob(
            inputs=agent_inputs[0],
            proprio=agent_proprios[0] if agent_proprios else None,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
            deterministic=deterministic,
        )
        
        actions.append(action_0)
        log_probs.append(log_prob_0)
        entropies.append(entropy_0)
        action_means.append(mu_0)
        action_stds.append(sigma_0)
        
        # === Agent 1: Forward pass with estimated action distribution ===
        if self.cfg.use_action_dist_input and front_view_only_input is not None:
            # Step 1: Estimate Agent 0's action distribution (no gradient)
            mu_0_est, sigma_0_est = self.estimate_agent0_action_dist(
                front_view_input=front_view_only_input,
                proprio_agent1_as_robot0=agent_proprios[1] if agent_proprios else None,
            )
            
            # Step 2: Concatenate estimated action dist with Agent 1's proprio
            # proprio_1_extended = [proprio_1; mu_0_est; sigma_0_est]
            if agent_proprios is not None:
                proprio_1 = agent_proprios[1]
                batch_size = proprio_1.shape[0]
                
                # Flatten mu and sigma if needed
                mu_flat = mu_0_est.reshape(batch_size, -1)
                sigma_flat = sigma_0_est.reshape(batch_size, -1)
                
                # Concatenate: (B, proprio_dim) + (B, action_dim*chunks) + (B, action_dim*chunks)
                proprio_1_extended = torch.cat([proprio_1, mu_flat, sigma_flat], dim=-1)
            else:
                proprio_1_extended = None
            
            # Use extended proprio projector for agent 1
            # Temporarily swap proprio projector
            original_proprio_projector = self.shared_agent.proprio_projector
            self.shared_agent.proprio_projector = self.proprio_projector_extended
            
            action_1, log_prob_1, entropy_1, mu_1, sigma_1 = self.shared_agent.get_action_and_log_prob(
                inputs=agent_inputs[1],
                proprio=proprio_1_extended,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
            )
            
            # Restore original proprio projector
            self.shared_agent.proprio_projector = original_proprio_projector
        else:
            # Fallback: standard forward pass without action dist
            action_1, log_prob_1, entropy_1, mu_1, sigma_1 = self.shared_agent.get_action_and_log_prob(
                inputs=agent_inputs[1],
                proprio=agent_proprios[1] if agent_proprios else None,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
            )
        
        actions.append(action_1)
        log_probs.append(log_prob_1)
        entropies.append(entropy_1)
        action_means.append(mu_1)
        action_stds.append(sigma_1)
        
        return actions, log_probs, entropies, action_means, action_stds
    
    def get_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        front_view_only_input: Optional[Dict[str, torch.Tensor]] = None,
        estimated_action_dist: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Get state values for all agents.
        
        For ACPPO:
        - V^(0)(s_t) for agent 0 - uses value_heads[0]
        - V^(1)([s_t, b_t^(1)]) for agent 1 - uses value_heads[1]
        """
        values = []
        
        # Agent 0: V^(0)(s_t) - uses dedicated value head for agent 0
        value_0 = self.shared_agent.get_value(
            inputs=agent_inputs[0],
            proprio=agent_proprios[0] if agent_proprios else None,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
            agent_idx=0,  # ACPPO: use agent 0's value head
        )
        values.append(value_0)
        
        # Agent 1: V^(1)([s_t, b_t^(1)]) - uses dedicated value head for agent 1
        if self.cfg.use_action_dist_input:
            # Get or estimate action distribution
            if estimated_action_dist is not None:
                mu_0_est, sigma_0_est = estimated_action_dist
            elif front_view_only_input is not None:
                mu_0_est, sigma_0_est = self.estimate_agent0_action_dist(
                    front_view_input=front_view_only_input,
                    proprio_agent1_as_robot0=agent_proprios[1] if agent_proprios else None,
                )
            else:
                # Fallback: use zero action distribution
                batch_size = agent_proprios[1].shape[0] if agent_proprios else 1
                action_dist_dim = self.cfg.action_dim * self.cfg.num_actions_chunk
                mu_0_est = torch.zeros(batch_size, action_dist_dim, device=self.device)
                sigma_0_est = torch.ones(batch_size, action_dist_dim, device=self.device)
            
            # Create extended proprio for agent 1
            if agent_proprios is not None:
                proprio_1 = agent_proprios[1]
                batch_size = proprio_1.shape[0]
                mu_flat = mu_0_est.reshape(batch_size, -1)
                sigma_flat = sigma_0_est.reshape(batch_size, -1)
                proprio_1_extended = torch.cat([proprio_1, mu_flat, sigma_flat], dim=-1)
            else:
                proprio_1_extended = None
            
            # Use extended proprio projector
            original_proprio_projector = self.shared_agent.proprio_projector
            self.shared_agent.proprio_projector = self.proprio_projector_extended
            
            value_1 = self.shared_agent.get_value(
                inputs=agent_inputs[1],
                proprio=proprio_1_extended,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                agent_idx=1,  # ACPPO: use agent 1's value head
            )
            
            self.shared_agent.proprio_projector = original_proprio_projector
        else:
            value_1 = self.shared_agent.get_value(
                inputs=agent_inputs[1],
                proprio=agent_proprios[1] if agent_proprios else None,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                agent_idx=1,  # ACPPO: use agent 1's value head
            )
        
        values.append(value_1)
        
        return values
    
    def get_actions_and_values(
        self,
        agent_inputs: List[Dict[str, torch.Tensor]],
        agent_proprios: Optional[List[torch.Tensor]] = None,
        front_view_only_input: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], 
               List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions AND values for all agents in efficient forward passes.
        
        Returns:
            Tuple of (actions, log_probs, entropies, values, action_means, action_stds)
        """
        actions = []
        log_probs = []
        entropies = []
        values = []
        action_means = []
        action_stds = []
        
        # === Agent 0 - uses value_heads[0] ===
        action_0, log_prob_0, entropy_0, value_0, mu_0, sigma_0 = self.shared_agent.get_action_and_value(
            inputs=agent_inputs[0],
            proprio=agent_proprios[0] if agent_proprios else None,
            use_proprio=self.cfg.use_proprio,
            use_film=self.cfg.use_film,
            deterministic=deterministic,
            agent_idx=0,  # ACPPO: use agent 0's value head
        )
        
        actions.append(action_0)
        log_probs.append(log_prob_0)
        entropies.append(entropy_0)
        values.append(value_0)
        action_means.append(mu_0)
        action_stds.append(sigma_0)
        
        # === Agent 1 with chaining - uses value_heads[1] ===
        if self.cfg.use_action_dist_input and front_view_only_input is not None:
            # Estimate Agent 0's action distribution
            mu_0_est, sigma_0_est = self.estimate_agent0_action_dist(
                front_view_input=front_view_only_input,
                proprio_agent1_as_robot0=agent_proprios[1] if agent_proprios else None,
            )
            
            # Create extended proprio
            if agent_proprios is not None:
                proprio_1 = agent_proprios[1]
                batch_size = proprio_1.shape[0]
                mu_flat = mu_0_est.reshape(batch_size, -1)
                sigma_flat = sigma_0_est.reshape(batch_size, -1)
                proprio_1_extended = torch.cat([proprio_1, mu_flat, sigma_flat], dim=-1)
            else:
                proprio_1_extended = None
            
            # Use extended proprio projector
            original_proprio_projector = self.shared_agent.proprio_projector
            self.shared_agent.proprio_projector = self.proprio_projector_extended
            
            action_1, log_prob_1, entropy_1, value_1, mu_1, sigma_1 = self.shared_agent.get_action_and_value(
                inputs=agent_inputs[1],
                proprio=proprio_1_extended,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
                agent_idx=1,  # ACPPO: use agent 1's value head
            )
            
            self.shared_agent.proprio_projector = original_proprio_projector
        else:
            action_1, log_prob_1, entropy_1, value_1, mu_1, sigma_1 = self.shared_agent.get_action_and_value(
                inputs=agent_inputs[1],
                proprio=agent_proprios[1] if agent_proprios else None,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                deterministic=deterministic,
                agent_idx=1,  # ACPPO: use agent 1's value head
            )
        
        actions.append(action_1)
        log_probs.append(log_prob_1)
        entropies.append(entropy_1)
        values.append(value_1)
        action_means.append(mu_1)
        action_stds.append(sigma_1)
        
        return actions, log_probs, entropies, values, action_means, action_stds
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters including extended proprio projector."""
        params = self.shared_agent.get_trainable_parameters()
        
        # Add extended proprio projector parameters
        if hasattr(self, 'proprio_projector_extended') and self.proprio_projector_extended is not None:
            params.extend([p for p in self.proprio_projector_extended.parameters() if p.requires_grad])
        
        return params
    
    def forward_evaluate_agent(
        self,
        agent_idx: int,
        inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        estimated_action_dist: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for evaluating a single agent's actions.
        Uses per-agent value heads for proper ACPPO credit assignment.
        """
        if agent_idx == 0:
            return self.shared_agent.evaluate_actions_and_value(
                inputs=inputs,
                actions=actions,
                proprio=proprio,
                use_proprio=self.cfg.use_proprio,
                use_film=self.cfg.use_film,
                agent_idx=0,  # ACPPO: use agent 0's value head
            )
        else:
            # Agent 1 with extended proprio
            if self.cfg.use_action_dist_input and estimated_action_dist is not None:
                mu_0_est, sigma_0_est = estimated_action_dist
                
                if proprio is not None:
                    batch_size = proprio.shape[0]
                    mu_flat = mu_0_est.reshape(batch_size, -1)
                    sigma_flat = sigma_0_est.reshape(batch_size, -1)
                    proprio_extended = torch.cat([proprio, mu_flat, sigma_flat], dim=-1)
                else:
                    proprio_extended = None
                
                original_proprio_projector = self.shared_agent.proprio_projector
                self.shared_agent.proprio_projector = self.proprio_projector_extended
                
                result = self.shared_agent.evaluate_actions_and_value(
                    inputs=inputs,
                    actions=actions,
                    proprio=proprio_extended,
                    use_proprio=self.cfg.use_proprio,
                    use_film=self.cfg.use_film,
                    agent_idx=1,  # ACPPO: use agent 1's value head
                )
                
                self.shared_agent.proprio_projector = original_proprio_projector
                return result
            else:
                return self.shared_agent.evaluate_actions_and_value(
                    inputs=inputs,
                    actions=actions,
                    proprio=proprio,
                    use_proprio=self.cfg.use_proprio,
                    use_film=self.cfg.use_film,
                    agent_idx=1,  # ACPPO: use agent 1's value head
                )
    
    def forward(
        self,
        mode: str,
        **kwargs,
    ):
        """Generic forward method for DDP compatibility."""
        if mode == 'evaluate_agent':
            return self.forward_evaluate_agent(
                agent_idx=kwargs['agent_idx'],
                inputs=kwargs['inputs'],
                actions=kwargs['actions'],
                proprio=kwargs.get('proprio'),
                estimated_action_dist=kwargs.get('estimated_action_dist'),
            )
        else:
            raise ValueError(f"Unknown forward mode: {mode}")


def _is_finetuned_checkpoint(checkpoint_path: str) -> bool:
    """Check if the checkpoint is a fine-tuned OpenVLA-OFT model."""
    import os
    
    hf_finetuned_models = [
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "moojink/openvla-7b-oft-finetuned-libero-10",
        "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    ]
    
    if checkpoint_path in hf_finetuned_models:
        return True
    
    if os.path.isdir(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if "proprio_projector" in filename and filename.endswith(".pt"):
                return True
    
    return False


def load_vla_for_acppo(
    cfg: ACPPOConfig,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module], 
           Optional[nn.Module], Any, Optional[Dict]]:
    """
    Load VLA model and components for ACPPO training.
    
    Returns:
        Tuple of (vla_model, action_head, proprio_projector, proprio_projector_extended,
                  noisy_action_projector, processor, norm_stats)
    """
    import sys
    sys.path.append("../../..")
    
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
    
    print_rank0("Loading VLA model for ACPPO training...")
    
    is_finetuned = _is_finetuned_checkpoint(cfg.pretrained_checkpoint)
    
    if is_finetuned:
        print_rank0(f"  Detected FINE-TUNED checkpoint: {cfg.pretrained_checkpoint}")
    else:
        print_rank0(f"  Detected BASE VLA checkpoint: {cfg.pretrained_checkpoint}")
    
    vla = get_model(cfg)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla = vla.to(device)
    torch.cuda.empty_cache()
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    processor = get_processor(cfg)
    norm_stats = getattr(vla, 'norm_stats', None)
    
    # Standard proprio projector (for agent 0)
    proprio_projector = None
    if cfg.use_proprio:
        if is_finetuned:
            from experiments.robot.openvla_utils import get_proprio_projector
            proprio_projector = get_proprio_projector(
                cfg,
                vla.llm_dim,
                proprio_dim=TWOARM_ACPPO_CONSTANTS["PROPRIO_DIM"],
            )
        else:
            proprio_projector = ProprioProjector(
                llm_dim=vla.llm_dim,
                proprio_dim=TWOARM_ACPPO_CONSTANTS["PROPRIO_DIM"],
            )
            proprio_projector = proprio_projector.to(torch.bfloat16).to(device)
    
    # Extended proprio projector (for agent 1 with action dist)
    proprio_projector_extended = None
    if cfg.use_proprio and cfg.use_action_dist_input:
        print_rank0(f"Creating extended proprio projector for agent 1:")
        print_rank0(f"  Input dim: {cfg.proprio_dim_agent1} (proprio={cfg.proprio_dim_agent0} + action_dist={cfg.action_dist_dim})")
        proprio_projector_extended = ProprioProjector(
            llm_dim=vla.llm_dim,
            proprio_dim=cfg.proprio_dim_agent1,
        )
        proprio_projector_extended = proprio_projector_extended.to(torch.bfloat16).to(device)
    
    # Action head
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
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=cfg.action_dim,
            num_diffusion_steps_train=cfg.num_diffusion_steps_train,
            num_actions_chunk=cfg.num_actions_chunk,
        )
        action_head = action_head.to(torch.bfloat16).to(device)
    
    # Noisy action projector (for diffusion)
    noisy_action_projector = None
    if cfg.use_diffusion:
        from prismatic.models.projectors import NoisyActionProjector
        noisy_action_projector = NoisyActionProjector(llm_dim=vla.llm_dim)
        noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(device)
    
    return vla, action_head, proprio_projector, proprio_projector_extended, noisy_action_projector, processor, norm_stats
