"""
Multi-Agent Rollout Buffer for MAPPO training.

Stores experience tuples (observations, actions, rewards, etc.) for
multiple agents and provides batched sampling for PPO updates.
"""

import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass


@dataclass
class RolloutBufferSamples:
    """Container for batched rollout samples."""
    # Per-agent observations
    agent_front_images: List[torch.Tensor]      # List of (B, T, H, W, C) per agent
    agent_wrist_images: List[torch.Tensor]      # List of (B, T, H, W, C) per agent
    agent_proprios: List[torch.Tensor]          # List of (B, T, proprio_dim) per agent
    
    # Per-agent action data
    agent_actions: List[torch.Tensor]           # List of (B, chunk_len, action_dim) per agent
    agent_old_log_probs: List[torch.Tensor]     # List of (B,) per agent
    
    # Shared reward and value data
    rewards: torch.Tensor                        # (B,) team reward
    old_values: torch.Tensor                     # (B,) centralized value
    advantages: torch.Tensor                     # (B,) advantages
    returns: torch.Tensor                        # (B,) returns
    
    # Global state for critic
    global_proprio_states: List[torch.Tensor]   # List of (B, T, proprio_dim) per agent


class MultiAgentRolloutBuffer:
    """
    Buffer for storing multi-agent rollout experience.
    
    Stores observations, actions, rewards, and values for all agents,
    computes advantages using GAE, and provides minibatch sampling.
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_agents: int = 2,
        num_envs: int = 1,
        action_dim: int = 6,
        action_chunk_size: int = 4,
        proprio_dim: int = 8,
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
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.proprio_dim = proprio_dim
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
        if self.store_images:
            self.front_images = np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
            self.wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
        self.proprio_states = [np.zeros((B, N, T, self.proprio_dim), dtype=np.float32) for _ in range(A)]
        self.actions = [np.zeros((B, N, self.action_chunk_size, self.action_dim), dtype=np.float32) for _ in range(A)]
        self.log_probs = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        self.rewards = np.zeros((B, N), dtype=np.float32)
        self.values = np.zeros((B, N), dtype=np.float32)
        self.dones = np.zeros((B, N), dtype=np.float32)
        self.advantages = np.zeros((B, N), dtype=np.float32)
        self.returns = np.zeros((B, N), dtype=np.float32)

    def reset(self):
        self.pos = 0
        self.full = False

    def add(self, front_images=None, wrist_images=None, proprio_states=None, actions=None, log_probs=None, reward=None, value=None, done=None):
        if self.store_images and front_images is not None:
            self.front_images[self.pos] = front_images
            if wrist_images is not None:
                for agent_idx in range(self.num_agents):
                    self.wrist_images[agent_idx][self.pos] = wrist_images[agent_idx]
        
        for agent_idx in range(self.num_agents):
            if proprio_states is not None: self.proprio_states[agent_idx][self.pos] = proprio_states[agent_idx]
            if actions is not None: self.actions[agent_idx][self.pos] = actions[agent_idx]
            if log_probs is not None: self.log_probs[agent_idx][self.pos] = log_probs[agent_idx]
        
        if reward is not None: self.rewards[self.pos] = reward
        if value is not None: self.values[self.pos] = value
        if done is not None: self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size: self.full = True

    def compute_returns_and_advantages(self, last_values, last_dones):
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        assert self.full
        total_size = self.buffer_size * self.num_envs
        if batch_size is None: batch_size = total_size
        indices = np.random.permutation(total_size)
        
        def flatten(arr): return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
        
        flat_proprio_states = [flatten(ps) for ps in self.proprio_states]
        flat_actions = [flatten(a) for a in self.actions]
        flat_log_probs = [flatten(lp) for lp in self.log_probs]
        flat_rewards = flatten(self.rewards)
        flat_values = flatten(self.values)
        flat_advantages = flatten(self.advantages)
        flat_returns = flatten(self.returns)
        
        if self.store_images:
            flat_front_images = flatten(self.front_images)
            flat_wrist_images = [flatten(wi) for wi in self.wrist_images]
        
        for start_idx in range(0, total_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            agent_proprios = [torch.as_tensor(flat_proprio_states[i][batch_indices], device=self.device) for i in range(self.num_agents)]
            agent_actions = [torch.as_tensor(flat_actions[i][batch_indices], device=self.device) for i in range(self.num_agents)]
            agent_old_log_probs = [torch.as_tensor(flat_log_probs[i][batch_indices], device=self.device) for i in range(self.num_agents)]
            
            if self.store_images:
                agent_front_images = [torch.as_tensor(flat_front_images[batch_indices], device=self.device) for _ in range(self.num_agents)]
                agent_wrist_images = [torch.as_tensor(flat_wrist_images[i][batch_indices], device=self.device) for i in range(self.num_agents)]
            else:
                agent_front_images = [None] * self.num_agents
                agent_wrist_images = [None] * self.num_agents
            
            yield RolloutBufferSamples(
                agent_front_images=agent_front_images, agent_wrist_images=agent_wrist_images,
                agent_proprios=agent_proprios, agent_actions=agent_actions, agent_old_log_probs=agent_old_log_probs,
                rewards=torch.as_tensor(flat_rewards[batch_indices], device=self.device),
                old_values=torch.as_tensor(flat_values[batch_indices], device=self.device),
                advantages=torch.as_tensor(flat_advantages[batch_indices], device=self.device),
                returns=torch.as_tensor(flat_returns[batch_indices], device=self.device),
                global_proprio_states=agent_proprios,
            )


class SharedRewardWrapper:
    def __init__(self, reward_type="shared", num_agents=2):
        self.reward_type = reward_type
        self.num_agents = num_agents
    
    def __call__(self, env_reward, agent_rewards=None):
        if self.reward_type == "shared": return env_reward
        elif self.reward_type == "mean": return np.mean(agent_rewards) if agent_rewards else env_reward
        elif self.reward_type == "sum": return np.sum(agent_rewards) if agent_rewards else env_reward
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
        """
        Synchronize statistics across all processes in DDP.
        
        Uses proper Welford's algorithm for combining running statistics
        from multiple processes.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        
        # Handle both scalar and array cases
        mean_shape = np.asarray(self.mean).shape
        
        # Flatten for all_reduce (handles both scalar and array)
        mean_flat = np.asarray(self.mean).flatten()
        var_flat = np.asarray(self.var).flatten()
        
        mean_t = torch.tensor(mean_flat, device=device, dtype=torch.float64)
        var_t = torch.tensor(var_flat, device=device, dtype=torch.float64)
        count_t = torch.tensor([self.count], device=device, dtype=torch.float64)
        
        # Gather all counts first
        all_counts = [torch.zeros_like(count_t) for _ in range(world_size)]
        dist.all_gather(all_counts, count_t)
        all_counts = torch.stack(all_counts)
        total_count = all_counts.sum()
        
        # Gather all means
        all_means = [torch.zeros_like(mean_t) for _ in range(world_size)]
        dist.all_gather(all_means, mean_t)
        all_means = torch.stack(all_means)  # (world_size, dim)
        
        # Gather all vars
        all_vars = [torch.zeros_like(var_t) for _ in range(world_size)]
        dist.all_gather(all_vars, var_t)
        all_vars = torch.stack(all_vars)  # (world_size, dim)
        
        # Compute combined mean (weighted by count)
        weights = all_counts / total_count  # (world_size, 1)
        combined_mean = (all_means * weights).sum(dim=0)
        
        # Compute combined variance using parallel algorithm
        # Var_combined = sum(n_i * (var_i + (mean_i - mean_combined)^2)) / n_total
        delta_sq = (all_means - combined_mean.unsqueeze(0)) ** 2
        combined_var = ((all_vars + delta_sq) * all_counts).sum(dim=0) / total_count
        
        # Update statistics
        if mean_shape == ():
            # Scalar case
            self.mean = combined_mean.item()
            self.var = combined_var.item()
        else:
            # Array case
            self.mean = combined_mean.cpu().numpy().reshape(mean_shape)
            self.var = combined_var.cpu().numpy().reshape(mean_shape)
        
        self.count = total_count.item()


class RewardNormalizer:
    """
    Normalizes rewards using running statistics.
    """
    
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
        """
        Normalize rewards.
        """
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
        
        # Update running return
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.return_rms.update(self.returns.flatten())
        
        # Normalize
        normalized = rewards / (self.return_rms.std + self.epsilon)
        
        return normalized

    def sync(self, device):
        """Sync running statistics."""
        self.return_rms.sync(device)