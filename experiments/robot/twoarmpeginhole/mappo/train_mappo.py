"""
MAPPO Training Script for Multi-Agent VLA on TwoArmPegInHole Environment.
Modified for Multi-GPU (DDP) training.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import draccus
import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import wandb

from .config import MAPPOConfig, TWOARM_MAPPO_CONSTANTS
from .observation_utils import (
    ObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    unnormalize_action,
)
from .vla_policy import MultiAgentVLAPolicy, load_vla_for_mappo
from .rollout_buffer import (
    MultiAgentRolloutBuffer,
    SharedRewardWrapper,
    RewardNormalizer,
)

# Import environment utilities
from experiments.robot.twoarmpeginhole.twoarm_utils import (
    get_twoarm_env,
    get_twoarm_task_descriptions,
    get_twoarm_dummy_action,
    get_twoarm_video_frame,
    save_rollout_video,
)
from experiments.robot.robot_utils import set_seed_everywhere

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # CRITICAL: Set CUDA device BEFORE any CUDA operations or model loading
        # This prevents models from being loaded to cuda:0 first
        torch.cuda.set_device(local_rank)
        
        # Check if already initialized (e.g., by torchrun)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Only print from rank 0 for clean logs
        if dist.get_rank() == 0:
            print(f"Distributed init: World Size: {dist.get_world_size()}")
        return True
    return False


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class MAPPOTrainer:
    """
    MAPPO Trainer for Multi-Agent VLA training on TwoArmPegInHole.
    Supports Multi-GPU via DDP.
    """
    
    def __init__(self, cfg: MAPPOConfig):
        """
        Initialize MAPPO trainer.
        """
        self.cfg = cfg
        self.distributed = dist.is_initialized()
        
        if self.distributed:
            self.device = torch.device(f"cuda:{cfg.local_rank}")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.world_size = 1
            self.rank = 0
            
        if self.rank == 0:
            logger.info(f"Using device: {self.device}, Distributed: {self.distributed}, World Size: {self.world_size}")
        
        # Set seed (offset by rank to ensure diverse environments)
        set_seed_everywhere(cfg.seed + self.rank)
        
        # Setup directories (only on main process)
        self.run_id = self._create_run_id()
        self.run_dir = cfg.run_root_dir / self.run_id
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize environment
        # Note: num_envs in config is treated as "per GPU" in this implementation
        if self.rank == 0:
            logger.info("Initializing TwoArmPegInHole environment...")
            
        self.env, _ = get_twoarm_env(
            cfg.model_family,
            resolution=cfg.env_img_res,
            robot1=cfg.robot1,
            robot2=cfg.robot2,
            controller=cfg.controller,
            env_configuration=cfg.env_configuration,
            reward_shaping=cfg.reward_shaping,
            # Dense reward component weights
            reaching_weight=cfg.reaching_weight,
            perpendicular_weight=cfg.perpendicular_weight,
            parallel_weight=cfg.parallel_weight,
            alignment_weight=cfg.alignment_weight,
        )
        
        # Get task descriptions
        self.task_desc_robot0, self.task_desc_robot1, self.combined_description = \
            get_twoarm_task_descriptions(mode=cfg.instruction_mode)
        
        # Initialize observation history manager
        self.obs_history = ObservationHistoryManager(
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            history_length=cfg.history_length,
        )
        
        # Initialize VLA policy with Action Head + Value Head
        if self.rank == 0:
            logger.info("Loading VLA model with Action Head + Value Head...")
        self._init_policy()
        
        # Initialize rollout buffer (Local to each GPU)
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=cfg.num_steps_per_rollout,
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            num_envs=cfg.num_envs,
            action_dim=cfg.action_dim,
            action_chunk_size=cfg.num_actions_chunk,
            proprio_dim=TWOARM_MAPPO_CONSTANTS["PROPRIO_DIM"],
            history_length=cfg.history_length,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=self.device,
            store_images=True,
        )
        
        # Initialize reward handling
        self.reward_wrapper = SharedRewardWrapper(
            reward_type="shared",
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
        )
        
        if cfg.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(gamma=cfg.gamma)
        else:
            self.reward_normalizer = None
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_success_rate = 0.0
        
        # Log configuration
        if self.rank == 0:
            self._log_config()
    
    def _create_run_id(self) -> str:
        """Create unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"mappo_twoarm_{timestamp}"
        if self.cfg.run_id_note:
            run_id += f"_{self.cfg.run_id_note}"
        return run_id
    
    def _setup_logging(self):
        """Setup logging to file and wandb."""
        if self.rank != 0:
            return

        # File logging
        log_file = self.run_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(file_handler)
        
        # Tensorboard
        self.tb_writer = SummaryWriter(self.run_dir / "tensorboard")
        
        # Wandb
        if self.cfg.use_wandb:
            wandb.init(
                entity=self.cfg.wandb_entity,
                project=self.cfg.wandb_project,
                name=self.run_id,
                config=vars(self.cfg),
                mode=self.cfg.wandb_mode,
            )
    
    def _init_policy(self):
        """Initialize VLA policy and wrap with DDP if distributed."""
        # Load VLA components
        # Note: Models are now loaded directly to the correct device via get_current_device()
        vla, action_head, proprio_projector, noisy_action_projector, processor, norm_stats = \
            load_vla_for_mappo(self.cfg, self.device)
        
        # Create multi-agent policy
        self.raw_policy = MultiAgentVLAPolicy(
            cfg=self.cfg,
            vla_model=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            processor=processor,
            device=self.device,
        )
        
        if self.distributed:
            # Wrap policy with DDP
            # Note: find_unused_parameters=False is safe because:
            # - Frozen VLA backbone params have requires_grad=False (not tracked by DDP)
            # - All trainable params (action head, value head, log_std) are used in forward
            self.policy = DDP(
                self.raw_policy, 
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=False  # All trainable params are used
            )
        else:
            self.policy = self.raw_policy
            
        self.processor = processor
        self.norm_stats = norm_stats
        self.action_norm_stats = None
        
        # Load normalization stats setup (same as before)
        if norm_stats is not None and len(norm_stats) > 0:
            unnorm_key = getattr(self.cfg, 'unnorm_key', None)
            if unnorm_key is None:
                unnorm_key = list(norm_stats.keys())[0]
            if unnorm_key in norm_stats and "action" in norm_stats[unnorm_key]:
                self.action_norm_stats = norm_stats[unnorm_key]["action"]
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        # Note: When using DDP, we optimize the DDP wrapper parameters or original parameters.
        # DDP handles gradient synchronization.
        if self.distributed:
            trainable_params = self.raw_policy.get_trainable_parameters()
        else:
            trainable_params = self.policy.get_trainable_parameters()
        
        self._trainable_params = trainable_params
        
        if self.rank == 0:
            total_trainable = sum(p.numel() for p in trainable_params)
            logger.info(f"Total trainable parameters: {total_trainable:,}")
        
        self.optimizer = optim.Adam(trainable_params, lr=self.cfg.learning_rate)

    def _log_config(self):
        """Log configuration to file."""
        config_path = self.run_dir / "config.json"
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in vars(self.cfg).items()}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_path}")
        
    def _average_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Average statistics across all GPUs."""
        if not self.distributed:
            return stats
            
        keys = sorted(stats.keys())
        values = torch.tensor([stats[k] for k in keys], device=self.device, dtype=torch.float32)
        
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        
        return {k: v.item() for k, v in zip(keys, values)}
    
    def _normalize_advantages_global(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages across ALL GPUs for consistent training.
        
        This ensures all GPUs use the same global mean and std for advantage normalization,
        which is important for stable multi-GPU training.
        
        Args:
            advantages: Local advantages tensor
            
        Returns:
            Globally normalized advantages
        """
        if not self.distributed:
            # Single GPU: just normalize locally
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute local statistics
        local_sum = advantages.sum()
        local_sq_sum = (advantages ** 2).sum()
        local_count = torch.tensor(advantages.numel(), device=self.device, dtype=torch.float32)
        
        # Gather global statistics
        stats = torch.stack([local_sum, local_sq_sum, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
        
        # Compute global mean and std
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean ** 2)
        global_std = torch.sqrt(global_var + 1e-8)
        
        # Normalize using global statistics
        return (advantages - global_mean) / global_std

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout experience.
        This runs on every GPU independently.
        """
        self.policy.eval()
        self.buffer.reset()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Step-wise reward tracking for dense reward logging
        step_rewards = []
        step_reward_components = {
            "reaching": [],
            "perpendicular": [],
            "parallel": [],
            "alignment": [],
            "peg_hole_dist": [],
            "perpendicular_dist": [],
            "parallel_dist": [],
            "alignment_cos": [],
        }
        
        obs = self.env.reset()
        self.obs_history.reset()
        
        for _ in range(self.cfg.num_steps_wait):
            obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
        
        front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
        self.obs_history.update(front_img, wrist_imgs, proprio_states)
        
        # Access underlying policy methods via .module if DDP
        policy_module = self.policy.module if self.distributed else self.policy
        
        for step in range(self.cfg.num_steps_per_rollout):
            with torch.no_grad():
                agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                agent_inputs = []
                agent_proprios = []
                task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                
                for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                    inputs = prepare_vla_input(
                        images=agent_obs[agent_idx]['images'],
                        proprio=agent_obs[agent_idx]['proprio'],
                        task_description=task_descriptions[agent_idx],
                        processor=self.processor,
                        device=self.device,
                        center_crop=self.cfg.center_crop,
                        agent_idx=agent_idx,
                    )
                    agent_inputs.append(inputs)
                    
                    proprio_tensor = torch.as_tensor(
                        agent_obs[agent_idx]['proprio'],
                        device=self.device,
                        dtype=torch.bfloat16,
                    ).unsqueeze(0)
                    agent_proprios.append(proprio_tensor)
                
                # Forward pass
                actions, log_probs, entropies, values = policy_module.get_actions_and_values(
                    agent_inputs=agent_inputs,
                    agent_proprios=agent_proprios,
                    deterministic=False,
                )
                
                value = torch.stack(values).mean().float().cpu().numpy()
            
            # Action processing
            action_0_normalized = actions[0][0, 0].float().cpu().numpy()
            action_1_normalized = actions[1][0, 0].float().cpu().numpy()
            
            if self.action_norm_stats is not None:
                action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats)
                action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats)
            else:
                action_0 = action_0_normalized
                action_1 = action_1_normalized
            
            full_action = np.concatenate([action_0, action_1])
            next_obs, reward, done, info = self.env.step(full_action.tolist())
            
            # Track step-wise rewards for logging
            step_rewards.append(reward)
            
            # Track dense reward components from info
            if "reward/reaching" in info:
                step_reward_components["reaching"].append(info.get("reward/reaching", 0))
                step_reward_components["perpendicular"].append(info.get("reward/perpendicular", 0))
                step_reward_components["parallel"].append(info.get("reward/parallel", 0))
                step_reward_components["alignment"].append(info.get("reward/alignment", 0))
                step_reward_components["peg_hole_dist"].append(info.get("reward/peg_hole_dist", 0))
                step_reward_components["perpendicular_dist"].append(info.get("reward/perpendicular_dist", 0))
                step_reward_components["parallel_dist"].append(info.get("reward/parallel_dist", 0))
                step_reward_components["alignment_cos"].append(info.get("reward/alignment_cos", 0))
            
            team_reward = self.reward_wrapper(reward)
            
            # Normalizer update and sync later
            if self.reward_normalizer is not None:
                team_reward = self.reward_normalizer.normalize(
                    np.array([team_reward]),
                    np.array([float(done)]),
                )[0]
            
            # Store transition
            proprio_states_np = [agent_obs[i]['proprio_history'] for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])]
            actions_np = [actions[i][0].float().cpu().numpy() for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])]
            log_probs_np = [log_probs[i][0].float().cpu().numpy() for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])]
            
            # Store images for PPO update
            front_images_list = []
            for t in range(self.cfg.history_length):
                front_images_list.append(agent_obs[0]['images'][t * 2])
            front_images_np = np.stack(front_images_list, axis=0)[np.newaxis, ...]
            
            wrist_images_np = []
            for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                wrist_list = []
                for t in range(self.cfg.history_length):
                    wrist_list.append(agent_obs[agent_idx]['images'][t * 2 + 1])
                wrist_images_np.append(np.stack(wrist_list, axis=0)[np.newaxis, ...])
            
            self.buffer.add(
                front_images=front_images_np,
                wrist_images=wrist_images_np,
                proprio_states=[np.expand_dims(ps, 0) for ps in proprio_states_np],
                actions=[np.expand_dims(a, 0) for a in actions_np],
                log_probs=[np.expand_dims(lp, 0) for lp in log_probs_np],
                reward=np.array([team_reward]),
                value=np.array([value]),
                done=np.array([float(done)]),
            )
            
            current_episode_reward += reward
            current_episode_length += 1
            success = self.env._check_success()
            
            if done or success or current_episode_length >= self.cfg.max_episode_steps:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_successes.append(float(success))
                self.episode_count += 1
                
                obs = self.env.reset()
                self.obs_history.reset()
                for _ in range(self.cfg.num_steps_wait):
                    obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
                current_episode_reward = 0
                current_episode_length = 0
            else:
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(next_obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
                obs = next_obs
            
            self.global_step += 1
        
        # Compute final value
        with torch.no_grad():
            agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
            agent_inputs = []
            agent_proprios = []
            for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                inputs = prepare_vla_input(
                    images=agent_obs[agent_idx]['images'],
                    proprio=agent_obs[agent_idx]['proprio'],
                    task_description=task_descriptions[agent_idx],
                    processor=self.processor,
                    device=self.device,
                    center_crop=self.cfg.center_crop,
                    agent_idx=agent_idx,
                )
                agent_inputs.append(inputs)
                proprio_tensor = torch.as_tensor(agent_obs[agent_idx]['proprio'], device=self.device, dtype=torch.bfloat16).unsqueeze(0)
                agent_proprios.append(proprio_tensor)
            
            last_values = policy_module.get_values(agent_inputs=agent_inputs, agent_proprios=agent_proprios)
            last_value = torch.stack(last_values).mean().float().cpu().numpy()
        
        self.buffer.compute_returns_and_advantages(
            last_values=np.array([last_value]),
            last_dones=np.array([float(done)]),
        )
        
        # Sync reward normalizer if used
        if self.reward_normalizer is not None and self.distributed:
            self.reward_normalizer.sync(self.device)
        
        stats = {
            "rollout/mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "rollout/mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "rollout/success_rate": np.mean(episode_successes) if episode_successes else 0,
            "rollout/num_episodes": len(episode_rewards),
        }
        
        # Add step-wise dense reward statistics (these are always available regardless of episode completion)
        if step_rewards:
            stats["rollout/mean_step_reward"] = np.mean(step_rewards)
            stats["rollout/sum_step_reward"] = np.sum(step_rewards)
            stats["rollout/min_step_reward"] = np.min(step_rewards)
            stats["rollout/max_step_reward"] = np.max(step_rewards)
        
        # Add dense reward component breakdowns
        for key, values in step_reward_components.items():
            if values:
                stats[f"dense_reward/{key}_mean"] = np.mean(values)
                if key.endswith("_dist") or key.endswith("_cos"):
                    # For raw metrics, also show min/max
                    stats[f"dense_reward/{key}_min"] = np.min(values)
                    stats[f"dense_reward/{key}_max"] = np.max(values)
        
        # Aggregate stats across GPUs
        return self._average_stats(stats)
    
    def update(self) -> Dict[str, float]:
        """Perform MAPPO update using collected rollouts with DDP support."""
        self.policy.train()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        # Each GPU has its own buffer
        batch_size = (self.cfg.num_steps_per_rollout * self.cfg.num_envs) // self.cfg.num_minibatches
        
        task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
        policy_module = self.policy.module if self.distributed else self.policy
        
        for epoch in range(self.cfg.num_epochs):
            for batch in self.buffer.get(batch_size):
                batch_advantages = batch.advantages
                batch_returns = batch.returns
                batch_old_values = batch.old_values
                
                if self.cfg.normalize_advantages:
                    # Use global normalization across all GPUs for consistent training
                    batch_advantages = self._normalize_advantages_global(batch_advantages)
                
                all_new_log_probs = []
                all_entropies = []
                all_new_values = []
                
                current_batch_size = batch.agent_actions[0].shape[0]
                
                for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_actions = batch.agent_actions[agent_idx]
                    agent_old_log_probs = batch.agent_old_log_probs[agent_idx]
                    agent_proprios = batch.agent_proprios[agent_idx]
                    
                    agent_front_images = batch.agent_front_images[agent_idx]
                    agent_wrist_images = batch.agent_wrist_images[agent_idx]
                    
                    if agent_front_images is not None and agent_wrist_images is not None:
                        batch_new_log_probs = []
                        batch_entropies = []
                        batch_new_values = []
                        
                        sub_batch_size = min(4, current_batch_size)
                        
                        for sub_start in range(0, current_batch_size, sub_batch_size):
                            sub_end = min(sub_start + sub_batch_size, current_batch_size)
                            sub_actions = agent_actions[sub_start:sub_end]
                            sub_front = agent_front_images[sub_start:sub_end]
                            sub_wrist = agent_wrist_images[sub_start:sub_end]
                            sub_proprio = agent_proprios[sub_start:sub_end, -1, :]
                            
                            sub_B = sub_actions.shape[0]
                            sub_log_probs = []
                            sub_entropies = []
                            sub_values = []
                            
                            for b in range(sub_B):
                                images = []
                                for t in range(self.cfg.history_length):
                                    images.append(sub_front[b, t].cpu().numpy().astype(np.uint8))
                                    images.append(sub_wrist[b, t].cpu().numpy().astype(np.uint8))
                                
                                vla_input = prepare_vla_input(
                                    images=images,
                                    proprio=sub_proprio[b].cpu().numpy(),
                                    task_description=task_descriptions[agent_idx],
                                    processor=self.processor,
                                    device=self.device,
                                    center_crop=self.cfg.center_crop,
                                    agent_idx=agent_idx,
                                )
                                
                                proprio_tensor = sub_proprio[b:b+1].to(dtype=torch.bfloat16, device=self.device)
                                action_to_eval = sub_actions[b:b+1]
                                
                                # CRITICAL: Call through DDP wrapper for proper gradient sync
                                # Do NOT use policy_module.agents[...] directly as it bypasses DDP
                                log_prob, entropy, value = self.policy(
                                    mode='evaluate_agent',
                                    agent_idx=agent_idx,
                                    inputs=vla_input,
                                    actions=action_to_eval,
                                    proprio=proprio_tensor,
                                )
                                
                                sub_log_probs.append(log_prob)
                                sub_entropies.append(entropy)
                                sub_values.append(value)
                            
                            batch_new_log_probs.append(torch.cat(sub_log_probs, dim=0))
                            batch_entropies.append(torch.cat(sub_entropies, dim=0))
                            batch_new_values.append(torch.cat(sub_values, dim=0))
                        
                        new_log_probs = torch.cat(batch_new_log_probs, dim=0)
                        entropies = torch.cat(batch_entropies, dim=0)
                        new_values = torch.cat(batch_new_values, dim=0)
                    else:
                        new_log_probs = agent_old_log_probs
                        entropies = torch.zeros_like(agent_old_log_probs)
                        new_values = batch_old_values
                    
                    all_new_log_probs.append(new_log_probs)
                    all_entropies.append(entropies)
                    all_new_values.append(new_values)
                
                all_policy_losses = []
                all_clip_fracs = []
                all_approx_kl = []
                
                for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_old_log_probs = batch.agent_old_log_probs[agent_idx]
                    new_log_probs = all_new_log_probs[agent_idx]
                    
                    log_ratio = new_log_probs - agent_old_log_probs
                    ratio = torch.exp(log_ratio)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    all_policy_losses.append(policy_loss)
                    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        all_approx_kl.append(approx_kl)
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
                        all_clip_fracs.append(clip_frac)
                
                policy_loss = torch.stack(all_policy_losses).mean().float()
                entropy = torch.stack(all_entropies).mean().float()
                new_values_mean = torch.stack(all_new_values).mean(dim=0).float()
                
                if self.cfg.clip_value_loss:
                    values_clipped = batch_old_values + torch.clamp(new_values_mean - batch_old_values, -self.cfg.clip_epsilon, self.cfg.clip_epsilon)
                    value_loss_unclipped = (new_values_mean - batch_returns) ** 2
                    value_loss_clipped = (values_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values_mean - batch_returns) ** 2).mean()
                
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.cfg.value_loss_coef * value_loss + self.cfg.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # DDP automatically averages gradients here
                
                if self.cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.max_grad_norm)
                
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                approx_kls.append(torch.stack(all_approx_kl).mean().item())
                clip_fractions.append(torch.stack(all_clip_fracs).mean().item())
        
        stats = {
            "update/policy_loss": np.mean(policy_losses),
            "update/value_loss": np.mean(value_losses),
            "update/entropy_loss": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kls),
            "update/clip_fraction": np.mean(clip_fractions),
        }
        
        return self._average_stats(stats)
    
    def evaluate(self, num_episodes: int = 10, save_video: bool = True, num_videos: int = 3) -> Dict[str, float]:
        """Evaluate policy (distributed)."""
        self.policy.eval()
        
        # Divide episodes among workers
        my_num_episodes = num_episodes // self.world_size
        if self.rank < num_episodes % self.world_size:
            my_num_episodes += 1
            
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        
        # Only rank 0 saves video for simplicity
        save_video = save_video and (self.rank == 0)
        if save_video:
            video_dir = self.run_dir / "eval_videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            
        policy_module = self.policy.module if self.distributed else self.policy
        
        for ep in range(my_num_episodes):
            obs = self.env.reset()
            self.obs_history.reset()
            should_save_video = save_video and (ep < num_videos)
            replay_images = [] if should_save_video else None
            
            for _ in range(self.cfg.num_steps_wait):
                obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
            
            front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
            self.obs_history.update(front_img, wrist_imgs, proprio_states)
            
            episode_reward = 0
            episode_length = 0
            success = False
            
            for step in range(self.cfg.max_episode_steps):
                if should_save_video: replay_images.append(get_twoarm_video_frame(obs))
                
                with torch.no_grad():
                    agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                    agent_inputs = []
                    agent_proprios = []
                    task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                    
                    for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                        inputs = prepare_vla_input(
                            images=agent_obs[agent_idx]['images'],
                            proprio=agent_obs[agent_idx]['proprio'],
                            task_description=task_descriptions[agent_idx],
                            processor=self.processor,
                            device=self.device,
                            center_crop=self.cfg.center_crop,
                            agent_idx=agent_idx,
                        )
                        agent_inputs.append(inputs)
                        proprio_tensor = torch.as_tensor(agent_obs[agent_idx]['proprio'], device=self.device, dtype=torch.bfloat16).unsqueeze(0)
                        agent_proprios.append(proprio_tensor)
                    
                    actions, _, _ = policy_module.get_actions(agent_inputs=agent_inputs, agent_proprios=agent_proprios, deterministic=True)
                
                action_0_normalized = actions[0][0, 0].float().cpu().numpy()
                action_1_normalized = actions[1][0, 0].float().cpu().numpy()
                
                if self.action_norm_stats is not None:
                    action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats)
                    action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats)
                else:
                    action_0, action_1 = action_0_normalized, action_1_normalized
                
                full_action = np.concatenate([action_0, action_1])
                next_obs, reward, done, info = self.env.step(full_action.tolist())
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                success = self.env._check_success()
                
                if done or success:
                    if should_save_video: replay_images.append(get_twoarm_video_frame(obs))
                    break
                
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(next_obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(float(success))
            
            if should_save_video and replay_images:
                import imageio
                video_filename = f"step_{self.global_step}_ep_{ep}_success_{success}.mp4"
                video_writer = imageio.get_writer(str(video_dir / video_filename), fps=20)
                for frame in replay_images: video_writer.append_data(frame)
                video_writer.close()
                logger.info(f"Saved eval video: {video_filename}")

        # Aggregate metrics from all workers
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_success = np.mean(episode_successes) if episode_successes else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        # We need to weight the average by number of episodes if unbalanced (simplification here: assume balanced)
        stats = {
            "eval/mean_reward": mean_reward,
            "eval/success_rate": mean_success,
            "eval/mean_episode_length": mean_length,
        }
        return self._average_stats(stats)

    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint (Rank 0 only)."""
        if self.rank != 0: return
        
        checkpoint_dir = self.run_dir / f"checkpoint_{self.global_step}{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        policy_module = self.policy.module if self.distributed else self.policy
        torch.save(policy_module.state_dict(), checkpoint_dir / "policy.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "best_success_rate": self.best_success_rate,
        }, checkpoint_dir / "training_state.pt")
        
        # Save reward normalizer state if used
        if self.reward_normalizer is not None:
            torch.save({
                "mean": self.reward_normalizer.return_rms.mean,
                "var": self.reward_normalizer.return_rms.var,
                "count": self.reward_normalizer.return_rms.count,
                "returns": self.reward_normalizer.returns,
            }, checkpoint_dir / "reward_normalizer.pt")
        
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.cfg).items()}
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load training checkpoint for resuming training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if self.rank == 0:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load policy state dict
        policy_state_path = checkpoint_path / "policy.pt"
        if not policy_state_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_state_path}")
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        # Load with map_location to handle multi-GPU
        policy_state = torch.load(policy_state_path, map_location=self.device)
        policy_module.load_state_dict(policy_state)
        
        if self.rank == 0:
            logger.info("Loaded policy state dict")
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state["optimizer"])
            self.global_step = training_state["global_step"]
            self.episode_count = training_state["episode_count"]
            self.best_success_rate = training_state.get("best_success_rate", 0.0)
            
            if self.rank == 0:
                logger.info(f"Resumed from global_step={self.global_step}, "
                           f"episode_count={self.episode_count}, "
                           f"best_success_rate={self.best_success_rate:.2%}")
        
        # Load reward normalizer state if exists
        normalizer_path = checkpoint_path / "reward_normalizer.pt"
        if self.reward_normalizer is not None and normalizer_path.exists():
            normalizer_state = torch.load(normalizer_path, map_location="cpu")
            self.reward_normalizer.return_rms.mean = normalizer_state["mean"]
            self.reward_normalizer.return_rms.var = normalizer_state["var"]
            self.reward_normalizer.return_rms.count = normalizer_state["count"]
            self.reward_normalizer.returns = normalizer_state["returns"]
            
            if self.rank == 0:
                logger.info("Loaded reward normalizer state")
        
        # Synchronize all processes after loading
        if self.distributed:
            dist.barrier()
            if self.rank == 0:
                logger.info("All processes synchronized after checkpoint load")

    def train(self):
        """Main training loop."""
        # Resume from checkpoint if specified
        if self.cfg.resume_checkpoint is not None:
            self.load_checkpoint(Path(self.cfg.resume_checkpoint))
        
        if self.rank == 0:
            logger.info(f"Starting MAPPO training for {self.cfg.total_timesteps} timesteps")
            logger.info(f"Run directory: {self.run_dir}")
            if self.cfg.resume_checkpoint:
                logger.info(f"Resumed from checkpoint: {self.cfg.resume_checkpoint}")
            
        num_updates = self.cfg.total_timesteps // (self.cfg.num_steps_per_rollout * self.cfg.num_envs)
        
        # Calculate starting update number if resuming
        start_update = self.global_step // (self.cfg.num_steps_per_rollout * self.cfg.num_envs)
        
        start_time = time.time()
        
        for update in tqdm.tqdm(range(start_update, num_updates), desc="Training", disable=self.rank!=0):
            rollout_stats = self.collect_rollouts()
            update_stats = self.update()
            
            stats = {**rollout_stats, **update_stats}
            stats["train/global_step"] = self.global_step
            stats["train/episodes"] = self.episode_count
            stats["train/fps"] = self.global_step / (time.time() - start_time)
            
            if self.rank == 0:
                for key, value in stats.items():
                    self.tb_writer.add_scalar(key, value, self.global_step)
                if self.cfg.use_wandb and self.global_step % self.cfg.wandb_log_freq == 0:
                    wandb.log(stats, step=self.global_step)
                
                if update % 10 == 0:
                    # Include dense reward info in log
                    step_reward_str = f"StepRwd: {stats.get('rollout/mean_step_reward', 0):.3f}" if 'rollout/mean_step_reward' in stats else ""
                    dist_str = f"Dist: {stats.get('dense_reward/peg_hole_dist_mean', 0):.3f}" if 'dense_reward/peg_hole_dist_mean' in stats else ""
                    
                    logger.info(
                        f"Step {self.global_step} | EpRwd: {stats['rollout/mean_episode_reward']:.2f} | "
                        f"{step_reward_str} | {dist_str} | "
                        f"Success: {stats['rollout/success_rate']:.2%} | Loss: {stats['update/total_loss']:.4f}"
                    )
                    
                    # Log detailed dense reward breakdown every 50 updates
                    if update % 50 == 0 and 'dense_reward/reaching_mean' in stats:
                        logger.info(
                            f"  Dense Rewards | Reach: {stats.get('dense_reward/reaching_mean', 0):.3f} | "
                            f"Perp: {stats.get('dense_reward/perpendicular_mean', 0):.3f} | "
                            f"Para: {stats.get('dense_reward/parallel_mean', 0):.3f} | "
                            f"Align: {stats.get('dense_reward/alignment_mean', 0):.3f}"
                        )
                        logger.info(
                            f"  Raw Metrics   | Dist: {stats.get('dense_reward/peg_hole_dist_mean', 0):.4f} | "
                            f"Cos: {stats.get('dense_reward/alignment_cos_mean', 0):.4f}"
                        )

            if self.global_step % self.cfg.eval_freq == 0:
                eval_stats = self.evaluate(
                    num_episodes=self.cfg.num_eval_episodes,
                    save_video=self.cfg.save_eval_videos,
                    num_videos=self.cfg.num_eval_videos,
                )
                
                if self.rank == 0:
                    for key, value in eval_stats.items():
                        self.tb_writer.add_scalar(key, value, self.global_step)
                    if self.cfg.use_wandb:
                        wandb.log(eval_stats, step=self.global_step)
                    logger.info(f"Evaluation | Reward: {eval_stats['eval/mean_reward']:.2f} | Success: {eval_stats['eval/success_rate']:.2%}")
                    
                    if eval_stats['eval/success_rate'] > self.best_success_rate:
                        self.best_success_rate = eval_stats['eval/success_rate']
                        self.save_checkpoint(suffix="_best")

            if self.global_step % self.cfg.save_freq == 0:
                self.save_checkpoint()
        
        self.save_checkpoint(suffix="_final")
        if self.rank == 0:
            logger.info("Training completed!")
            self.tb_writer.close()
            if self.cfg.use_wandb: wandb.finish()
        
        if self.distributed:
            dist.destroy_process_group()


@draccus.wrap()
def main(cfg: MAPPOConfig):
    setup_distributed()
    trainer = MAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()