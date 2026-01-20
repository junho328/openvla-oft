"""
Multi-Agent Rollout Buffer for ACPPO training.

Extended from MAPPO buffer to support:
1. Per-agent value storage (for microstep advantage computation)
2. Action distribution storage (mu, sigma) for agent chaining
3. Microstep-based GAE computation following ACPPO formulation
"""

import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass


@dataclass
class RolloutBufferSamplesACPPO:
    """Container for batched rollout samples in ACPPO."""
    # Per-agent observations
    agent_front_images: List[torch.Tensor]      # List of (B, T, H, W, C) per agent
    agent_wrist_images: List[torch.Tensor]      # List of (B, T, H, W, C) per agent
    agent_proprios: List[torch.Tensor]          # List of (B, T, proprio_dim) per agent
    
    # Per-agent action data
    agent_actions: List[torch.Tensor]           # List of (B, chunk_len, action_dim) per agent
    agent_old_log_probs: List[torch.Tensor]     # List of (B,) per agent
    
    # Per-agent action distributions (for chaining)
    agent_action_means: List[torch.Tensor]      # List of (B, total_action_dim) per agent
    agent_action_stds: List[torch.Tensor]       # List of (B, total_action_dim) per agent
    
    # Per-agent values and advantages (for ACPPO microstep computation)
    agent_old_values: List[torch.Tensor]        # List of (B,) per agent - V^(i)(s_t, b_t^(i))
    agent_advantages: List[torch.Tensor]        # List of (B,) per agent - A_t^(i)
    agent_returns: List[torch.Tensor]           # List of (B,) per agent
    
    # Shared reward data
    rewards: torch.Tensor                        # (B,) team reward (only at last agent's step)
    dones: torch.Tensor                          # (B,) episode done flags


class MultiAgentRolloutBufferACPPO:
    """
    Buffer for storing multi-agent rollout experience with ACPPO support.
    
    Key differences from MAPPO buffer:
    1. Stores per-agent values V^(i) instead of single centralized value
    2. Stores action distributions (mu, sigma) for agent chaining
    3. Computes per-agent advantages using microstep-based GAE or shared reward GAE
    
    GAE Modes:
    1. "acppo_microstep" - Original ACPPO microstep TD residuals:
        ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])  for i < N
        ζ_t^(N) = r_t + γ' V^(1)(s_{t+1}) - V^(N)([s_t, b_t^(N)])        for i = N
        
        WARNING: May cause value collapse if V^(0) ≈ V^(1) (no reward signal for agent 0)
    
    2. "shared_reward" - Standard GAE where all agents receive the same reward:
        δ_t^(i) = r_t + γ V^(i)(s_{t+1}) - V^(i)(s_t)
        A_t^(i) = Σ (γλ)^l δ_{t+l}^(i)
        
        This is more stable and recommended as default.
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
        gamma_prime: float = 0.99,
        gae_lambda: float = 0.95,
        lambda_prime: float = 0.95,
        device: torch.device = torch.device("cpu"),
        store_images: bool = False,
        gae_mode: str = "shared_reward",  # "acppo_microstep" or "shared_reward"
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.total_action_dim = action_dim * action_chunk_size
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.image_size = image_size
        
        # Discount factors
        self.gamma = gamma
        self.gamma_prime = gamma_prime  # Microstep discount
        self.gae_lambda = gae_lambda
        self.lambda_prime = lambda_prime  # Microstep GAE lambda
        
        self.device = device
        self.store_images = store_images
        self.gae_mode = gae_mode
        self.pos = 0
        self.full = False
        
        self._init_storage()

    def _init_storage(self):
        B, N, A, T = self.buffer_size, self.num_envs, self.num_agents, self.history_length
        H, W = self.image_size
        
        # Image storage
        if self.store_images:
            self.front_images = np.zeros((B, N, T, H, W, 3), dtype=np.uint8)
            self.wrist_images = [np.zeros((B, N, T, H, W, 3), dtype=np.uint8) for _ in range(A)]
        
        # Proprio storage
        self.proprio_states = [np.zeros((B, N, T, self.proprio_dim), dtype=np.float32) for _ in range(A)]
        
        # Action storage
        self.actions = [np.zeros((B, N, self.action_chunk_size, self.action_dim), dtype=np.float32) for _ in range(A)]
        self.log_probs = [np.zeros((B, N), dtype=np.float32) for _ in range(A)]
        
        # Action distribution storage (for agent chaining)
        self.action_means = [np.zeros((B, N, self.total_action_dim), dtype=np.float32) for _ in range(A)]
        self.action_stds = [np.zeros((B, N, self.total_action_dim), dtype=np.float32) for _ in range(A)]
        
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
        front_images: Optional[np.ndarray] = None,
        wrist_images: Optional[List[np.ndarray]] = None,
        proprio_states: Optional[List[np.ndarray]] = None,
        actions: Optional[List[np.ndarray]] = None,
        log_probs: Optional[List[np.ndarray]] = None,
        action_means: Optional[List[np.ndarray]] = None,
        action_stds: Optional[List[np.ndarray]] = None,
        values: Optional[List[np.ndarray]] = None,
        reward: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            front_images: Front view images (N, T, H, W, C)
            wrist_images: List of wrist images per agent
            proprio_states: List of proprio states per agent
            actions: List of actions per agent
            log_probs: List of log probs per agent
            action_means: List of action means per agent
            action_stds: List of action stds per agent
            values: List of values per agent V^(i)
            reward: Team reward
            done: Episode done flag
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
            if action_means is not None:
                self.action_means[agent_idx][self.pos] = action_means[agent_idx]
            if action_stds is not None:
                self.action_stds[agent_idx][self.pos] = action_stds[agent_idx]
            if values is not None:
                self.values[agent_idx][self.pos] = values[agent_idx]
        
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
        Compute per-agent returns and advantages.
        
        Supports two modes via self.gae_mode:
        
        1. "shared_reward" (default, recommended):
            Standard GAE where all agents receive the same reward.
            δ_t^(i) = r_t + γ V^(i)(s_{t+1}) - V^(i)(s_t)
            A_t^(i) = Σ (γλ)^l δ_{t+l}^(i)
        
        2. "acppo_microstep":
            Original ACPPO microstep TD residuals.
            WARNING: May cause value collapse if V^(0) ≈ V^(1)
        
        Args:
            last_values: List of last values per agent [V^(0)(s_T+1), V^(1)(...)]
            last_dones: Done flags for last step
        """
        if self.gae_mode == "shared_reward":
            self._compute_shared_reward_gae(last_values, last_dones)
        else:
            self._compute_acppo_microstep_gae(last_values, last_dones)
    
    def _compute_shared_reward_gae(
        self,
        last_values: List[np.ndarray],
        last_dones: np.ndarray,
    ):
        """
        Compute GAE with shared reward for all agents.
        
        Each agent uses standard GAE:
        δ_t^(i) = r_t + γ V^(i)(s_{t+1}) - V^(i)(s_t)
        A_t^(i) = Σ (γλ)^l δ_{t+l}^(i)
        
        This is more stable than ACPPO microstep GAE.
        """
        N = self.num_agents
        gamma = self.gamma
        gae_lambda = self.gae_lambda
        
        for agent_idx in range(N):
            last_gae_lam = np.zeros_like(last_dones)
            
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - last_dones
                    next_value = last_values[agent_idx]
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.values[agent_idx][step + 1]
                
                # Standard TD residual: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
                delta = (
                    self.rewards[step] + 
                    gamma * next_value * next_non_terminal - 
                    self.values[agent_idx][step]
                )
                
                # GAE: A_t = δ_t + (γλ) A_{t+1}
                self.advantages[agent_idx][step] = (
                    delta + 
                    gamma * gae_lambda * next_non_terminal * last_gae_lam
                )
                last_gae_lam = self.advantages[agent_idx][step]
            
            # Compute returns: R_t = A_t + V(s_t)
            self.returns[agent_idx] = self.advantages[agent_idx] + self.values[agent_idx]
    
    def _compute_acppo_microstep_gae(
        self,
        last_values: List[np.ndarray],
        last_dones: np.ndarray,
    ):
        """
        Compute ACPPO microstep GAE (original formulation).
        
        WARNING: This may cause value collapse if V^(0) ≈ V^(1) because
        Agent 0's TD residual doesn't include reward: ζ_t^(0) = γ' V^(1) - V^(0)
        
        ACPPO TD Residuals:
            For agent i < N:
                ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])
            
            For agent N (last agent):
                ζ_t^(N) = r_t + γ' V^(0)(s_{t+1}) - V^(N)([s_t, b_t^(N)])
        """
        N = self.num_agents
        gamma_prime = self.gamma_prime
        lambda_prime = self.lambda_prime
        
        # Initialize GAE accumulators per agent
        last_gae_lam = [np.zeros_like(last_dones) for _ in range(N)]
        
        # Process in reverse time order
        for step in reversed(range(self.buffer_size)):
            # Determine next step values and done flags
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = [self.values[i][step + 1] for i in range(N)]
            
            # Compute TD residuals for each agent at this timestep
            # Process agents in reverse order (N to 1)
            td_residuals = [None] * N
            
            for i in reversed(range(N)):
                if i == N - 1:
                    # Last agent: ζ_t^(N) = r_t + γ' V^(0)(s_{t+1}) - V^(N)([s_t, b_t^(N)])
                    # Note: reward only comes at the last agent's step
                    td_residuals[i] = (
                        self.rewards[step] + 
                        gamma_prime * next_values[0] * next_non_terminal - 
                        self.values[i][step]
                    )
                else:
                    # Intermediate agents: ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])
                    # Note: V^(i+1) at same timestep (not next timestep)
                    td_residuals[i] = (
                        gamma_prime * self.values[i + 1][step] - 
                        self.values[i][step]
                    )
            
            # Compute GAE advantages per agent
            # Process agents in reverse order for proper credit assignment
            for i in reversed(range(N)):
                if i == N - 1:
                    # Last agent: continues to next timestep's first agent
                    # A_t^(N) = ζ_t^(N) + (γ'λ') * A_{t+1}^(0) * (1 - done)
                    if step == self.buffer_size - 1:
                        # Bootstrap from stored GAE
                        next_gae = last_gae_lam[0]
                    else:
                        next_gae = self.advantages[0][step + 1]
                    
                    self.advantages[i][step] = (
                        td_residuals[i] + 
                        gamma_prime * lambda_prime * next_non_terminal * next_gae
                    )
                else:
                    # Intermediate agents: chain to next agent at same timestep
                    # A_t^(i) = ζ_t^(i) + (γ'λ') * A_t^(i+1)
                    self.advantages[i][step] = (
                        td_residuals[i] + 
                        gamma_prime * lambda_prime * self.advantages[i + 1][step]
                    )
            
            # Update last_gae_lam for bootstrapping
            for i in range(N):
                last_gae_lam[i] = self.advantages[i][step]
        
        # Compute returns: R_t^(i) = A_t^(i) + V^(i)
        for i in range(N):
            self.returns[i] = self.advantages[i] + self.values[i]

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesACPPO, None, None]:
        """
        Generate batched samples from the buffer.
        
        Args:
            batch_size: Size of each batch. If None, returns full buffer.
            
        Yields:
            RolloutBufferSamplesACPPO for each minibatch
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
        flat_action_means = [flatten(am) for am in self.action_means]
        flat_action_stds = [flatten(ast) for ast in self.action_stds]
        flat_values = [flatten(v) for v in self.values]
        flat_advantages = [flatten(a) for a in self.advantages]
        flat_returns = [flatten(r) for r in self.returns]
        
        flat_rewards = flatten(self.rewards)
        flat_dones = flatten(self.dones)
        
        if self.store_images:
            flat_front_images = flatten(self.front_images)
            flat_wrist_images = [flatten(wi) for wi in self.wrist_images]
        
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
            agent_action_means = [
                torch.as_tensor(flat_action_means[i][batch_indices], device=self.device) 
                for i in range(self.num_agents)
            ]
            agent_action_stds = [
                torch.as_tensor(flat_action_stds[i][batch_indices], device=self.device) 
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
                agent_front_images = [
                    torch.as_tensor(flat_front_images[batch_indices], device=self.device) 
                    for _ in range(self.num_agents)
                ]
                agent_wrist_images = [
                    torch.as_tensor(flat_wrist_images[i][batch_indices], device=self.device) 
                    for i in range(self.num_agents)
                ]
            else:
                agent_front_images = [None] * self.num_agents
                agent_wrist_images = [None] * self.num_agents
            
            yield RolloutBufferSamplesACPPO(
                agent_front_images=agent_front_images,
                agent_wrist_images=agent_wrist_images,
                agent_proprios=agent_proprios,
                agent_actions=agent_actions,
                agent_old_log_probs=agent_old_log_probs,
                agent_action_means=agent_action_means,
                agent_action_stds=agent_action_stds,
                agent_old_values=agent_old_values,
                agent_advantages=agent_advantages,
                agent_returns=agent_returns,
                rewards=torch.as_tensor(flat_rewards[batch_indices], device=self.device),
                dones=torch.as_tensor(flat_dones[batch_indices], device=self.device),
            )


class SharedRewardWrapper:
    """Wrapper for handling team rewards."""
    
    def __init__(self, reward_type: str = "shared", num_agents: int = 2):
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
    """Running mean and standard deviation for normalization."""
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
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
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
        
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.return_rms.update(self.returns.flatten())
        
        normalized = rewards / (self.return_rms.std + self.epsilon)
        
        return normalized

    def sync(self, device):
        """Sync running statistics."""
        self.return_rms.sync(device)
