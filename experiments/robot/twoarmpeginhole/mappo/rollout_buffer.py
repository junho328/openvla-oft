"""
Multi-Agent Rollout Buffer for MAPPO training.

Stores experience tuples (observations, actions, rewards, etc.) for
multiple agents and provides batched sampling for PPO updates.
"""

import numpy as np
import torch
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
        store_images: bool = False,  # Images can be large, optionally skip
    ):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size: Number of steps to store per rollout
            num_agents: Number of agents
            num_envs: Number of parallel environments
            action_dim: Action dimension per timestep
            action_chunk_size: Number of actions in a chunk
            proprio_dim: Proprioceptive state dimension
            history_length: Number of historical frames
            image_size: Image dimensions (H, W)
            gamma: Discount factor
            gae_lambda: GAE lambda
            device: Target device
            store_images: Whether to store image observations
        """
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
        
        # Position in buffer
        self.pos = 0
        self.full = False
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage arrays."""
        B = self.buffer_size
        N = self.num_envs
        A = self.num_agents
        T = self.history_length
        H, W = self.image_size
        
        # Per-agent observations (store as numpy for memory efficiency)
        if self.store_images:
            self.front_images = np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
            self.wrist_images = [
                np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
                for _ in range(A)
            ]
        
        # Proprio states per agent: (buffer_size, num_envs, history_length, proprio_dim)
        self.proprio_states = [
            np.zeros((B, N, T, self.proprio_dim), dtype=np.float32)
            for _ in range(A)
        ]
        
        # Actions per agent: (buffer_size, num_envs, chunk_size, action_dim)
        self.actions = [
            np.zeros((B, N, self.action_chunk_size, self.action_dim), dtype=np.float32)
            for _ in range(A)
        ]
        
        # Log probs per agent: (buffer_size, num_envs)
        self.log_probs = [
            np.zeros((B, N), dtype=np.float32)
            for _ in range(A)
        ]
        
        # Shared team reward: (buffer_size, num_envs)
        self.rewards = np.zeros((B, N), dtype=np.float32)
        
        # Values from centralized critic: (buffer_size, num_envs)
        self.values = np.zeros((B, N), dtype=np.float32)
        
        # Episode termination: (buffer_size, num_envs)
        self.dones = np.zeros((B, N), dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros((B, N), dtype=np.float32)
        self.returns = np.zeros((B, N), dtype=np.float32)
    
    def reset(self):
        """Reset buffer position."""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        front_images: Optional[np.ndarray] = None,
        wrist_images: Optional[List[np.ndarray]] = None,
        proprio_states: List[np.ndarray] = None,
        actions: List[np.ndarray] = None,
        log_probs: List[np.ndarray] = None,
        reward: np.ndarray = None,
        value: np.ndarray = None,
        done: np.ndarray = None,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            front_images: Front camera images (num_envs, T, H, W, 3)
            wrist_images: List of wrist images per agent
            proprio_states: List of proprio states per agent (num_envs, T, proprio_dim)
            actions: List of actions per agent (num_envs, chunk_size, action_dim)
            log_probs: List of log probs per agent (num_envs,)
            reward: Team reward (num_envs,)
            value: Value estimate (num_envs,)
            done: Episode termination flags (num_envs,)
        """
        if self.store_images and front_images is not None:
            self.front_images[self.pos] = front_images
            if wrist_images is not None:
                for agent_idx in range(self.num_agents):
                    self.wrist_images[agent_idx][self.pos] = wrist_images[agent_idx]
        
        for agent_idx in range(self.num_agents):
            if proprio_states is not None:
                self.proprio_states[agent_idx][self.pos] = proprio_states[agent_idx]
            if actions is not None:
                self.actions[agent_idx][self.pos] = actions[agent_idx]
            if log_probs is not None:
                self.log_probs[agent_idx][self.pos] = log_probs[agent_idx]
        
        if reward is not None:
            self.rewards[self.pos] = reward
        if value is not None:
            self.values[self.pos] = value
        if done is not None:
            self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for the last state (num_envs,)
            last_dones: Done flags for the last state (num_envs,)
        """
        # GAE computation
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            
            self.advantages[step] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
    
    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generate minibatches from the buffer.
        
        Args:
            batch_size: Size of each minibatch (None = full buffer)
            
        Yields:
            RolloutBufferSamples for each minibatch
        """
        assert self.full, "Buffer must be full before sampling!"
        
        # Flatten buffer: (buffer_size, num_envs, ...) -> (buffer_size * num_envs, ...)
        total_size = self.buffer_size * self.num_envs
        
        if batch_size is None:
            batch_size = total_size
        
        # Generate random indices
        indices = np.random.permutation(total_size)
        
        # Reshape data for sampling
        def flatten(arr):
            shape = arr.shape
            return arr.reshape(shape[0] * shape[1], *shape[2:])
        
        # Flatten all arrays
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
        
        # Generate minibatches
        for start_idx in range(0, total_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Extract batch
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
            
            if self.store_images:
                agent_front_images = [
                    torch.as_tensor(flat_front_images[batch_indices], device=self.device)
                    for _ in range(self.num_agents)  # Same front image for all agents
                ]
                agent_wrist_images = [
                    torch.as_tensor(flat_wrist_images[i][batch_indices], device=self.device)
                    for i in range(self.num_agents)
                ]
            else:
                agent_front_images = [None] * self.num_agents
                agent_wrist_images = [None] * self.num_agents
            
            yield RolloutBufferSamples(
                agent_front_images=agent_front_images,
                agent_wrist_images=agent_wrist_images,
                agent_proprios=agent_proprios,
                agent_actions=agent_actions,
                agent_old_log_probs=agent_old_log_probs,
                rewards=torch.as_tensor(flat_rewards[batch_indices], device=self.device),
                old_values=torch.as_tensor(flat_values[batch_indices], device=self.device),
                advantages=torch.as_tensor(flat_advantages[batch_indices], device=self.device),
                returns=torch.as_tensor(flat_returns[batch_indices], device=self.device),
                global_proprio_states=agent_proprios,  # Use same for global
            )


class SharedRewardWrapper:
    """
    Wrapper to compute shared/team rewards from individual agent rewards.
    
    In cooperative MARL, agents typically share a common reward.
    This wrapper handles different reward sharing schemes.
    """
    
    def __init__(
        self,
        reward_type: str = "shared",
        num_agents: int = 2,
    ):
        """
        Initialize reward wrapper.
        
        Args:
            reward_type: Type of reward sharing ("shared", "mean", "sum")
            num_agents: Number of agents
        """
        self.reward_type = reward_type
        self.num_agents = num_agents
    
    def __call__(
        self,
        env_reward: float,
        agent_rewards: Optional[List[float]] = None,
    ) -> float:
        """
        Compute team reward.
        
        Args:
            env_reward: Reward from environment
            agent_rewards: Optional per-agent rewards
            
        Returns:
            Team reward
        """
        if self.reward_type == "shared":
            # All agents get the same environment reward
            return env_reward
        
        elif self.reward_type == "mean":
            if agent_rewards is not None:
                return np.mean(agent_rewards)
            return env_reward
        
        elif self.reward_type == "sum":
            if agent_rewards is not None:
                return np.sum(agent_rewards)
            return env_reward
        
        else:
            return env_reward


class RunningMeanStd:
    """
    Running mean and standard deviation for normalization.
    
    Used for reward normalization in RL training.
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
        
        Args:
            rewards: Raw rewards
            dones: Episode termination flags
            
        Returns:
            Normalized rewards
        """
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
        
        # Update running return
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.return_rms.update(self.returns.flatten())
        
        # Normalize
        normalized = rewards / (self.return_rms.std + self.epsilon)
        
        return normalized
