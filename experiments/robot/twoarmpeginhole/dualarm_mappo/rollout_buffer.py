"""
Multi-Agent Rollout Buffer for Dual-Arm MAPPO training with per-agent value heads.

Stores experience tuples (observations, actions, rewards, etc.) for
multiple agents and provides batched sampling for PPO updates.

Key differences from single-arm MAPPO:
- Stores 14-dim bimanual actions and 14-dim padded proprio per agent
- Each agent stores their 7-dim action slice for environment execution
- Full 14-dim actions stored for VLA evaluation

Per-agent Value Heads:
- Each agent has its own value head V^(i)
- Separate value/advantage/return storage per agent
"""

import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass

from .config import DUALARM_MAPPO_CONSTANTS


@dataclass
class DualArmRolloutBufferSamples:
    """Container for batched rollout samples with per-agent data for dual-arm setup."""
    # Per-agent observations (3 images per agent: agentview + left_wrist + right_wrist, with padding)
    agent_agentview_images: List[torch.Tensor]    # List of (B, T, H, W, C) per agent
    agent_left_wrist_images: List[torch.Tensor]   # List of (B, T, H, W, C) per agent (one padded)
    agent_right_wrist_images: List[torch.Tensor]  # List of (B, T, H, W, C) per agent (one padded)
    agent_proprios: List[torch.Tensor]            # List of (B, T, 14) per agent (padded)
    
    # Per-agent action data (14-dim full bimanual actions for VLA)
    agent_actions: List[torch.Tensor]             # List of (B, chunk_len, 14) per agent
    agent_old_log_probs: List[torch.Tensor]       # List of (B,) per agent
    
    # Per-agent values and advantages
    agent_old_values: List[torch.Tensor]          # List of (B,) per agent - V^(i)(s_t)
    agent_advantages: List[torch.Tensor]          # List of (B,) per agent - A_t^(i)
    agent_returns: List[torch.Tensor]             # List of (B,) per agent
    
    # Shared reward data
    rewards: torch.Tensor                          # (B,) team reward
    
    # Legacy fields for backward compatibility
    old_values: torch.Tensor                       # (B,) averaged value
    advantages: torch.Tensor                       # (B,) averaged advantage
    returns: torch.Tensor                          # (B,) averaged return
    
    # Global state for critic
    global_proprio_states: torch.Tensor            # (B, T, 14) full proprio


class DualArmRolloutBuffer:
    """
    Buffer for storing dual-arm multi-agent rollout experience with per-agent value heads.
    
    Key differences from single-arm:
    - Stores 14-dim proprio (padded per agent) and 14-dim actions
    - 3 images per agent (agentview + left_wrist + right_wrist, with one wrist padded)
    - Full bimanual actions stored for VLA evaluation
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_agents: int = 2,
        num_envs: int = 1,
        model_action_dim: int = 14,      # Full bimanual action dim
        agent_action_dim: int = 7,        # Per-agent action dim  
        action_chunk_size: int = 2,
        model_proprio_dim: int = 14,     # Full bimanual proprio dim
        history_length: int = 2,
        image_size: Tuple[int, int] = (224, 224),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
        store_images: bool = False,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.model_action_dim = model_action_dim
        self.agent_action_dim = agent_action_dim
        self.action_chunk_size = action_chunk_size
        self.model_proprio_dim = model_proprio_dim
        self.history_length = history_length
        self.image_size = image_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.store_images = store_images
        self.pos = 0
        self.full = False
        self._init_storage()

    def _init_storage(self):
        B, N, A, T = self.buffer_size, self.num_envs, self.num_agents, self.history_length
        H, W = self.image_size
        
        # Image storage (3 images per agent: agentview + left_wrist + right_wrist)
        if self.store_images:
            # Shared agentview (same for all agents)
            self.agentview_images = np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
            # Per-agent wrist images (one will be padded per agent)
            self.left_wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
            self.right_wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
        
        # Proprio storage (14-dim padded per agent)
        self.proprio_states = [np.zeros((B, N, T, self.model_proprio_dim), dtype=np.float32) for _ in range(A)]
        
        # Global proprio (full 14-dim without padding)
        self.global_proprio_states = np.zeros((B, N, T, self.model_proprio_dim), dtype=np.float32)
        
        # Action storage (14-dim full bimanual action per agent for VLA)
        self.actions = [np.zeros((B, N, self.action_chunk_size, self.model_action_dim), dtype=np.float32) for _ in range(A)]
        self.log_probs = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        
        # Also store the 7-dim per-agent actions for actual env execution
        self.agent_actions_7dim = [np.zeros((B, N, self.action_chunk_size, self.agent_action_dim), dtype=np.float32) for _ in range(A)]
        
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
        values=None,
        reward=None,
        done=None,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            agentview_images: Agent view images (N, T, H, W, C)
            left_wrist_images: List of left wrist images per agent (padded for agent 1)
            right_wrist_images: List of right wrist images per agent (padded for agent 0)
            proprio_states: List of padded 14-dim proprio states per agent
            global_proprio: Full 14-dim proprio without padding (N, T, 14)
            actions: List of 14-dim full bimanual actions per agent
            agent_actions_7dim: List of 7-dim per-agent actions (actual env actions)
            log_probs: List of log probs per agent
            values: List of values per agent V^(i)
            reward: Team reward
            done: Episode done flag
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
    ):
        """
        Compute per-agent returns and advantages using standard GAE.
        
        Each agent computes its own GAE using its value head V^(i).
        
        Args:
            last_values: List of last values per agent [V^(0)(s_T+1), V^(1)(s_T+1)]
            last_dones: Done flags for last step
        """
        N = self.num_agents
        
        for agent_idx in range(N):
            last_gae_lam = np.zeros_like(last_dones)
            
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - last_dones
                    next_values = last_values[agent_idx]
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_values = self.values[agent_idx][step + 1]
                
                delta = (
                    self.rewards[step] + 
                    self.gamma * next_values * next_non_terminal - 
                    self.values[agent_idx][step]
                )
                self.advantages[agent_idx][step] = last_gae_lam = (
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )
            
            self.returns[agent_idx] = self.advantages[agent_idx] + self.values[agent_idx]

    def get(self, batch_size: Optional[int] = None) -> Generator[DualArmRolloutBufferSamples, None, None]:
        """
        Generate batched samples from the buffer.
        
        Args:
            batch_size: Size of each batch. If None, returns full buffer.
            
        Yields:
            DualArmRolloutBufferSamples for each minibatch
        """
        assert self.full, "Buffer must be full before sampling"
        
        total_size = self.buffer_size * self.num_envs
        if batch_size is None:
            batch_size = total_size
        
        indices = np.random.permutation(total_size)
        
        def flatten(arr):
            return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
        
        # Flatten per-agent data
        flat_proprio_states = [flatten(ps) for ps in self.proprio_states]
        flat_actions = [flatten(a) for a in self.actions]
        flat_log_probs = [flatten(lp) for lp in self.log_probs]
        flat_values = [flatten(v) for v in self.values]
        flat_advantages = [flatten(a) for a in self.advantages]
        flat_returns = [flatten(r) for r in self.returns]
        
        flat_rewards = flatten(self.rewards)
        flat_global_proprio = flatten(self.global_proprio_states)
        
        if self.store_images:
            flat_agentview = flatten(self.agentview_images)
            flat_left_wrist = [flatten(lw) for lw in self.left_wrist_images]
            flat_right_wrist = [flatten(rw) for rw in self.right_wrist_images]
        
        for start_idx in range(0, total_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            agent_proprios = [
                torch.as_tensor(flat_proprio_states[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_actions = [
                torch.as_tensor(flat_actions[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_old_log_probs = [
                torch.as_tensor(flat_log_probs[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_old_values = [
                torch.as_tensor(flat_values[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_advantages = [
                torch.as_tensor(flat_advantages[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_returns = [
                torch.as_tensor(flat_returns[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            
            if self.store_images:
                agent_agentview = [
                    torch.as_tensor(flat_agentview[batch_indices], device=self.device) 
                    for _ in range(self.num_agents)
                ]
                agent_left_wrist = [
                    torch.as_tensor(flat_left_wrist[i][batch_indices], device=self.device) 
                    for i in range(self.num_agents)
                ]
                agent_right_wrist = [
                    torch.as_tensor(flat_right_wrist[i][batch_indices], device=self.device) 
                    for i in range(self.num_agents)
                ]
            else:
                agent_agentview = [None] * self.num_agents
                agent_left_wrist = [None] * self.num_agents
                agent_right_wrist = [None] * self.num_agents
            
            avg_old_values = torch.stack(agent_old_values).mean(dim=0)
            avg_advantages = torch.stack(agent_advantages).mean(dim=0)
            avg_returns = torch.stack(agent_returns).mean(dim=0)
            
            yield DualArmRolloutBufferSamples(
                agent_agentview_images=agent_agentview,
                agent_left_wrist_images=agent_left_wrist,
                agent_right_wrist_images=agent_right_wrist,
                agent_proprios=agent_proprios,
                agent_actions=agent_actions,
                agent_old_log_probs=agent_old_log_probs,
                agent_old_values=agent_old_values,
                agent_advantages=agent_advantages,
                agent_returns=agent_returns,
                rewards=torch.as_tensor(flat_rewards[batch_indices], device=self.device),
                old_values=avg_old_values,
                advantages=avg_advantages,
                returns=avg_returns,
                global_proprio_states=torch.as_tensor(flat_global_proprio[batch_indices], device=self.device),
            )


class SharedRewardWrapper:
    """Wrapper to handle shared team reward for multi-agent setting."""
    def __init__(self, reward_type="shared", num_agents=2):
        self.reward_type = reward_type
        self.num_agents = num_agents
    
    def __call__(self, env_reward, agent_rewards=None):
        if self.reward_type == "shared": 
            return env_reward
        elif self.reward_type == "mean": 
            return np.mean(agent_rewards) if agent_rewards else env_reward
        elif self.reward_type == "sum": 
            return np.sum(agent_rewards) if agent_rewards else env_reward
        return env_reward


class RunningMeanStd:
    """
    Running mean and standard deviation for normalization.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)
        
    def sync(self, device):
        """Synchronize statistics across all processes in DDP."""
        if not dist.is_available() or not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        
        mean_shape = np.asarray(self.mean).shape
        
        mean_flat = np.asarray(self.mean).flatten()
        var_flat = np.asarray(self.var).flatten()
        
        mean_t = torch.tensor(mean_flat, device=device, dtype=torch.float64)
        var_t = torch.tensor(var_flat, device=device, dtype=torch.float64)
        count_t = torch.tensor([self.count], device=device, dtype=torch.float64)
        
        all_counts = [torch.zeros_like(count_t) for _ in range(world_size)]
        dist.all_gather(all_counts, count_t)
        all_counts = torch.stack(all_counts)
        total_count = all_counts.sum()
        
        all_means = [torch.zeros_like(mean_t) for _ in range(world_size)]
        dist.all_gather(all_means, mean_t)
        all_means = torch.stack(all_means)
        
        all_vars = [torch.zeros_like(var_t) for _ in range(world_size)]
        dist.all_gather(all_vars, var_t)
        all_vars = torch.stack(all_vars)
        
        weights = all_counts / total_count
        combined_mean = (all_means * weights).sum(dim=0)
        
        delta_sq = (all_means - combined_mean.unsqueeze(0)) ** 2
        combined_var = ((all_vars + delta_sq) * all_counts).sum(dim=0) / total_count
        
        if mean_shape == ():
            self.mean = combined_mean.item()
            self.var = combined_var.item()
        else:
            self.mean = combined_mean.cpu().numpy().reshape(mean_shape)
            self.var = combined_var.cpu().numpy().reshape(mean_shape)
        
        self.count = total_count.item()


class RewardNormalizer:
    """Normalizes rewards using running statistics."""
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        per_env_normalization: bool = False,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env_normalization = per_env_normalization
        
        self.return_rms = RunningMeanStd()
        self.returns = None
    
    def normalize(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Normalize rewards."""
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
        
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.return_rms.update(self.returns.flatten())
        
        normalized = rewards / (self.return_rms.std + self.epsilon)
        
        return normalized

    def sync(self, device):
        """Sync running statistics."""
        self.return_rms.sync(device)
