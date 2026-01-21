"""
Multi-Agent Rollout Buffer for Dual-Arm ACPPO training with per-agent value heads.

Extended from Dual-Arm MAPPO buffer to support:
1. Per-agent value storage (for microstep advantage computation)
2. Action distribution storage (mu, sigma) for agent chaining
3. Microstep-based GAE computation following ACPPO formulation

Key differences from single-arm ACPPO:
- Stores 14-dim bimanual actions and 14-dim padded proprio per agent
- Each agent stores their 7-dim action slice for environment execution
- Full 14-dim actions stored for VLA evaluation
- 3 images per agent (agentview + left_wrist + right_wrist, with one wrist padded)
"""

import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass

from .config import DUALARM_ACPPO_CONSTANTS


@dataclass
class DualArmRolloutBufferSamplesACPPO:
    """Container for batched rollout samples with per-agent data for dual-arm ACPPO setup."""
    # Per-agent observations (3 images per agent: agentview + left_wrist + right_wrist, with padding)
    agent_agentview_images: List[torch.Tensor]    # List of (B, T, H, W, C) per agent
    agent_left_wrist_images: List[torch.Tensor]   # List of (B, T, H, W, C) per agent (one padded)
    agent_right_wrist_images: List[torch.Tensor]  # List of (B, T, H, W, C) per agent (one padded)
    agent_proprios: List[torch.Tensor]            # List of (B, T, 14) per agent (padded)
    
    # Per-agent action data (14-dim full bimanual actions for VLA)
    agent_actions: List[torch.Tensor]             # List of (B, chunk_len, 14) per agent
    agent_old_log_probs: List[torch.Tensor]       # List of (B,) per agent
    
    # Per-agent action distributions (for ACPPO chaining)
    agent_action_means: List[torch.Tensor]        # List of (B, 7 * chunk) per agent (agent 0's estimated dist)
    agent_action_stds: List[torch.Tensor]         # List of (B, 7 * chunk) per agent
    
    # Per-agent values and advantages
    agent_old_values: List[torch.Tensor]          # List of (B,) per agent - V^(i)(s_t)
    agent_advantages: List[torch.Tensor]          # List of (B,) per agent - A_t^(i)
    agent_returns: List[torch.Tensor]             # List of (B,) per agent
    
    # Shared reward data
    rewards: torch.Tensor                          # (B,) team reward
    dones: torch.Tensor                            # (B,) episode done flags
    
    # Legacy fields for backward compatibility
    old_values: torch.Tensor                       # (B,) averaged value
    advantages: torch.Tensor                       # (B,) averaged advantage
    returns: torch.Tensor                          # (B,) averaged return
    
    # Global state for critic
    global_proprio_states: torch.Tensor            # (B, T, 14) full proprio


class DualArmRolloutBufferACPPO:
    """
    Buffer for storing dual-arm multi-agent rollout experience with ACPPO support.
    
    Key differences from Dual-Arm MAPPO buffer:
    1. Stores per-agent values V^(i) instead of single centralized value
    2. Stores action distributions (mu, sigma) for agent chaining
    3. Computes per-agent advantages using microstep-based GAE or shared reward GAE
    
    Key differences from single-arm ACPPO:
    - Stores 14-dim proprio (padded per agent) and 14-dim actions
    - 3 images per agent (agentview + left_wrist + right_wrist, with one wrist padded)
    - Full bimanual actions stored for VLA evaluation
    
    GAE Modes:
    1. "acppo_microstep" - Original ACPPO microstep TD residuals
    2. "shared_reward" - Standard GAE where all agents receive the same reward (recommended)
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_agents: int = 2,
        num_envs: int = 1,
        model_action_dim: int = 14,      # Full bimanual action dim
        agent_action_dim: int = 7,        # Per-agent action dim  
        action_chunk_size: int = 25,
        model_proprio_dim: int = 14,     # Full bimanual proprio dim
        history_length: int = 2,
        image_size: Tuple[int, int] = (224, 224),
        gamma: float = 0.99,
        gamma_prime: float = 0.99,
        gae_lambda: float = 0.95,
        lambda_prime: float = 0.95,
        device: torch.device = torch.device("cpu"),
        store_images: bool = False,
        gae_mode: str = "shared_reward",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.model_action_dim = model_action_dim
        self.agent_action_dim = agent_action_dim
        self.action_chunk_size = action_chunk_size
        self.total_action_dim = model_action_dim * action_chunk_size
        self.agent_total_action_dim = agent_action_dim * action_chunk_size
        self.model_proprio_dim = model_proprio_dim
        self.history_length = history_length
        self.image_size = image_size
        
        self.gamma = gamma
        self.gamma_prime = gamma_prime
        self.gae_lambda = gae_lambda
        self.lambda_prime = lambda_prime
        
        self.device = device
        self.store_images = store_images
        self.gae_mode = gae_mode
        self.pos = 0
        self.full = False
        self._init_storage()

    def _init_storage(self):
        B, N, A, T = self.buffer_size, self.num_envs, self.num_agents, self.history_length
        H, W = self.image_size
        
        # Image storage (3 images per agent: agentview + left_wrist + right_wrist)
        if self.store_images:
            self.agentview_images = np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
            self.left_wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
            self.right_wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
        
        # Proprio storage (14-dim padded per agent)
        self.proprio_states = [np.zeros((B, N, T, self.model_proprio_dim), dtype=np.float32) for _ in range(A)]
        
        # Global proprio (full 14-dim without padding)
        self.global_proprio_states = np.zeros((B, N, T, self.model_proprio_dim), dtype=np.float32)
        
        # Action storage (7-dim per-agent action)
        # Each agent stores its own 7-dim action slice
        self.actions = [np.zeros((B, N, self.action_chunk_size, self.agent_action_dim), dtype=np.float32) for _ in range(A)]
        self.log_probs = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        
        # Also store the 7-dim per-agent actions for actual env execution (same as self.actions for consistency)
        self.agent_actions_7dim = [np.zeros((B, N, self.action_chunk_size, self.agent_action_dim), dtype=np.float32) for _ in range(A)]
        
        # ACPPO: Action distribution storage (for agent chaining)
        # Agent 0's action dist: (B, N, 7 * chunk)
        self.action_means = [np.zeros((B, N, self.agent_total_action_dim), dtype=np.float32) for _ in range(A)]
        self.action_stds = [np.zeros((B, N, self.agent_total_action_dim), dtype=np.float32) for _ in range(A)]
        
        # Per-agent value storage
        self.values = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        
        # Reward and done storage
        self.rewards = np.zeros((B, N), dtype=np.float32)
        self.dones = np.zeros((B, N), dtype=np.float32)
        
        # Per-agent advantage and return storage
        self.advantages = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        self.returns = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]

    def reset(self):
        self.pos = 0
        self.full = False

    def add(
        self,
        agentview_images=None,
        left_wrist_images=None,
        right_wrist_images=None,
        proprio_states=None,
        global_proprio=None,
        actions=None,
        agent_actions_7dim=None,
        log_probs=None,
        action_means=None,
        action_stds=None,
        values=None,
        reward=None,
        done=None,
    ):
        """
        Add a transition to the buffer.
        """
        if self.store_images and agentview_images is not None:
            self.agentview_images[self.pos] = agentview_images
            if left_wrist_images is not None:
                for agent_idx in range(self.num_agents):
                    self.left_wrist_images[agent_idx][self.pos] = left_wrist_images[agent_idx]
            if right_wrist_images is not None:
                for agent_idx in range(self.num_agents):
                    self.right_wrist_images[agent_idx][self.pos] = right_wrist_images[agent_idx]
        
        for agent_idx in range(self.num_agents):
            if proprio_states is not None:
                self.proprio_states[agent_idx][self.pos] = proprio_states[agent_idx]
            if actions is not None:
                self.actions[agent_idx][self.pos] = actions[agent_idx]
            if agent_actions_7dim is not None:
                self.agent_actions_7dim[agent_idx][self.pos] = agent_actions_7dim[agent_idx]
            if log_probs is not None:
                self.log_probs[agent_idx][self.pos] = log_probs[agent_idx]
            if action_means is not None:
                self.action_means[agent_idx][self.pos] = action_means[agent_idx]
            if action_stds is not None:
                self.action_stds[agent_idx][self.pos] = action_stds[agent_idx]
            if values is not None:
                self.values[agent_idx][self.pos] = values[agent_idx]
        
        if global_proprio is not None:
            self.global_proprio_states[self.pos] = global_proprio
        if reward is not None:
            self.rewards[self.pos] = reward
        if done is not None:
            self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_values: List[np.ndarray],
        last_dones: np.ndarray,
    ) -> None:
        """
        Compute returns and advantages using per-agent GAE.
        """
        if self.gae_mode == "shared_reward":
            self._compute_shared_reward_gae(last_values, last_dones)
        elif self.gae_mode in ["acppo_microstep", "microstep"]:
            self._compute_microstep_gae(last_values, last_dones)
        else:
            raise ValueError(f"Unknown GAE mode: {self.gae_mode}. Use 'shared_reward' or 'microstep'/'acppo_microstep'.")
    
    def _compute_shared_reward_gae(
        self,
        last_values: List[np.ndarray],
        last_dones: np.ndarray,
    ) -> None:
        """
        Standard GAE where all agents receive the same reward.
        """
        buffer_len = self.pos if not self.full else self.buffer_size
        
        for agent_idx in range(self.num_agents):
            last_gae_lam = 0
            
            for t in reversed(range(buffer_len)):
                if t == buffer_len - 1:
                    next_non_terminal = 1.0 - last_dones
                    next_values = last_values[agent_idx]
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_values = self.values[agent_idx][t + 1]
                
                delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[agent_idx][t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[agent_idx][t] = last_gae_lam
            
            self.returns[agent_idx] = self.advantages[agent_idx] + self.values[agent_idx]
    
    def _compute_microstep_gae(
        self,
        last_values: List[np.ndarray],
        last_dones: np.ndarray,
    ) -> None:
        """
        ACPPO microstep-based GAE computation.
        
        For agent i < N:
            ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])
            
        For agent N (last):
            ζ_t^(N) = r_t + γ' V^(1)(s_{t+1}) - V^(N)([s_t, b_t^(N)])
        """
        buffer_len = self.pos if not self.full else self.buffer_size
        
        for t in reversed(range(buffer_len)):
            for agent_idx in range(self.num_agents):
                if agent_idx < self.num_agents - 1:
                    # Non-terminal agent: microstep TD
                    delta = self.gamma_prime * self.values[agent_idx + 1][t] - self.values[agent_idx][t]
                else:
                    # Terminal agent: receives actual reward
                    if t == buffer_len - 1:
                        next_non_terminal = 1.0 - last_dones
                        next_value = last_values[0]  # V^(0) at next step
                    else:
                        next_non_terminal = 1.0 - self.dones[t + 1]
                        next_value = self.values[0][t + 1]  # V^(0) at next step
                    
                    delta = self.rewards[t] + self.gamma_prime * next_value * next_non_terminal - self.values[agent_idx][t]
                
                # GAE accumulation
                if t == buffer_len - 1:
                    self.advantages[agent_idx][t] = delta
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    # For microstep, we use lambda_prime
                    self.advantages[agent_idx][t] = delta + self.gamma_prime * self.lambda_prime * next_non_terminal * self.advantages[agent_idx][t + 1]
            
        for agent_idx in range(self.num_agents):
            self.returns[agent_idx] = self.advantages[agent_idx] + self.values[agent_idx]

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DualArmRolloutBufferSamplesACPPO, None, None]:
        """
        Generate batches of experience.
        """
        buffer_len = self.pos if not self.full else self.buffer_size
        indices = np.random.permutation(buffer_len * self.num_envs)
        
        if batch_size is None:
            batch_size = buffer_len * self.num_envs
        
        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self._get_samples(batch_indices)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_indices: np.ndarray,
    ) -> DualArmRolloutBufferSamplesACPPO:
        """
        Convert batch indices to actual samples.
        """
        buffer_len = self.pos if not self.full else self.buffer_size
        env_indices = batch_indices % self.num_envs
        time_indices = batch_indices // self.num_envs
        
        # Per-agent data
        agent_agentview_images = []
        agent_left_wrist_images = []
        agent_right_wrist_images = []
        agent_proprios = []
        agent_actions = []
        agent_old_log_probs = []
        agent_action_means = []
        agent_action_stds = []
        agent_old_values = []
        agent_advantages = []
        agent_returns = []
        
        for agent_idx in range(self.num_agents):
            if self.store_images:
                agent_agentview_images.append(
                    torch.as_tensor(self.agentview_images[time_indices, env_indices], device=self.device)
                )
                agent_left_wrist_images.append(
                    torch.as_tensor(self.left_wrist_images[agent_idx][time_indices, env_indices], device=self.device)
                )
                agent_right_wrist_images.append(
                    torch.as_tensor(self.right_wrist_images[agent_idx][time_indices, env_indices], device=self.device)
                )
            else:
                agent_agentview_images.append(None)
                agent_left_wrist_images.append(None)
                agent_right_wrist_images.append(None)
            
            agent_proprios.append(
                torch.as_tensor(self.proprio_states[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_actions.append(
                torch.as_tensor(self.actions[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_old_log_probs.append(
                torch.as_tensor(self.log_probs[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_action_means.append(
                torch.as_tensor(self.action_means[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_action_stds.append(
                torch.as_tensor(self.action_stds[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_old_values.append(
                torch.as_tensor(self.values[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_advantages.append(
                torch.as_tensor(self.advantages[agent_idx][time_indices, env_indices], device=self.device)
            )
            agent_returns.append(
                torch.as_tensor(self.returns[agent_idx][time_indices, env_indices], device=self.device)
            )
        
        # Shared data
        rewards = torch.as_tensor(self.rewards[time_indices, env_indices], device=self.device)
        dones = torch.as_tensor(self.dones[time_indices, env_indices], device=self.device)
        global_proprio = torch.as_tensor(self.global_proprio_states[time_indices, env_indices], device=self.device)
        
        # Legacy averaged values
        old_values = torch.stack(agent_old_values).mean(dim=0)
        advantages = torch.stack(agent_advantages).mean(dim=0)
        returns = torch.stack(agent_returns).mean(dim=0)
        
        return DualArmRolloutBufferSamplesACPPO(
            agent_agentview_images=agent_agentview_images,
            agent_left_wrist_images=agent_left_wrist_images,
            agent_right_wrist_images=agent_right_wrist_images,
            agent_proprios=agent_proprios,
            agent_actions=agent_actions,
            agent_old_log_probs=agent_old_log_probs,
            agent_action_means=agent_action_means,
            agent_action_stds=agent_action_stds,
            agent_old_values=agent_old_values,
            agent_advantages=agent_advantages,
            agent_returns=agent_returns,
            rewards=rewards,
            dones=dones,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            global_proprio_states=global_proprio,
        )

    @property
    def size(self) -> int:
        return self.pos if not self.full else self.buffer_size


class SharedRewardWrapper:
    """
    Wrapper to compute shared team reward from environment reward.
    """
    
    def __init__(
        self,
        reward_type: str = "shared",
        num_agents: int = 2,
        reward_scale: float = 1.0,
    ):
        self.reward_type = reward_type
        self.num_agents = num_agents
        self.reward_scale = reward_scale
    
    def __call__(self, reward: float) -> float:
        """
        Process environment reward into team reward.
        """
        if self.reward_type == "shared":
            return reward * self.reward_scale
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")


class RewardNormalizer:
    """
    Running mean/std normalizer for rewards.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip_value: float = 10.0,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_value = clip_value
        
        self.return_rms_mean = 0.0
        self.return_rms_var = 1.0
        self.return_rms_count = 0
        
        self.returns = 0.0
    
    def normalize(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize rewards using running statistics.
        """
        # Update return estimate
        self.returns = rewards + self.gamma * self.returns * (1 - dones)
        
        # Update running statistics
        batch_mean = np.mean(self.returns)
        batch_var = np.var(self.returns)
        batch_count = len(self.returns) if hasattr(self.returns, '__len__') else 1
        
        delta = batch_mean - self.return_rms_mean
        tot_count = self.return_rms_count + batch_count
        
        new_mean = self.return_rms_mean + delta * batch_count / tot_count
        m_a = self.return_rms_var * self.return_rms_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.return_rms_count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.return_rms_mean = new_mean
        self.return_rms_var = new_var
        self.return_rms_count = tot_count
        
        # Normalize
        std = np.sqrt(self.return_rms_var + self.epsilon)
        normalized = rewards / std
        
        # Clip
        normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        
        return normalized
