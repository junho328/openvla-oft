"""
Dual-Arm ACPPO Training Script for Multi-Agent VLA on TwoArmPegInHole Environment.

This trains a bimanual (ALOHA-style) VLA model with ACPPO where:
- VLA outputs 14-dim action (7 per arm)
- Agent 0 uses action[:7], Agent 1 uses action[7:]
- Each agent sees agentview + own wrist + padded other wrist
- Each agent uses padded 14-dim proprio (own 7-dim + zeros)
- Agent 1 additionally uses estimated action distribution from Agent 0

ACPPO-specific features:
1. All agents act SIMULTANEOUSLY
2. Agent 0: Uses only its own observation
3. Agent 1: Additionally uses ESTIMATED action distribution from Agent 0
4. Microstep-based advantage calculation (optional)

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

from .config import DualArmACPPOConfig, DUALARM_ACPPO_CONSTANTS
from .observation_utils import (
    DualArmObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    prepare_vla_input_for_action_dist_estimation,
    unnormalize_action,
    split_bimanual_action,
    combine_agent_actions,
)
from .vla_policy import DualArmMultiAgentVLAPolicyACPPO, load_vla_for_dualarm_acppo
from .rollout_buffer import (
    DualArmRolloutBufferACPPO,
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
        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        if dist.get_rank() == 0:
            print(f"Distributed init: World Size: {dist.get_world_size()}")
        return True
    return False


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class DualArmACPPOTrainer:
    """
    Dual-Arm ACPPO Trainer for Multi-Agent VLA training on TwoArmPegInHole.
    
    Uses a bimanual VLA model (ALOHA-style) that outputs 14-dim actions.
    Agent 1 receives estimated action distribution from Agent 0.
    """
    
    def __init__(self, cfg: DualArmACPPOConfig):
        """Initialize Dual-Arm ACPPO trainer."""
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
            logger.info(f"Dual-Arm ACPPO: model_action_dim={cfg.model_action_dim}, agent_action_dim={cfg.agent_action_dim}")
            logger.info(f"Action dist input for Agent 1: {cfg.use_action_dist_input}")
        
        set_seed_everywhere(cfg.seed + self.rank)
        
        self.run_id = self._create_run_id()
        self.run_dir = cfg.run_root_dir / self.run_id
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        if self.distributed:
            dist.barrier()
        
        self._setup_logging()
        
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
            reaching_weight=cfg.reaching_weight,
            perpendicular_weight=cfg.perpendicular_weight,
            parallel_weight=cfg.parallel_weight,
            alignment_weight=cfg.alignment_weight,
        )
        
        self.task_desc_robot0, self.task_desc_robot1, self.combined_description = \
            get_twoarm_task_descriptions(mode=cfg.instruction_mode)
        
        self.obs_history = DualArmObservationHistoryManager(
            num_agents=DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            history_length=cfg.history_length,
        )
        
        if self.rank == 0:
            logger.info("Loading bimanual VLA model for ACPPO...")
        self._init_policy()
        
        self.buffer = DualArmRolloutBufferACPPO(
            buffer_size=cfg.num_steps_per_rollout,
            num_agents=DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            num_envs=cfg.num_envs,
            model_action_dim=cfg.model_action_dim,
            agent_action_dim=cfg.agent_action_dim,
            action_chunk_size=cfg.num_actions_chunk,
            model_proprio_dim=cfg.model_proprio_dim,
            history_length=cfg.history_length,
            gamma=cfg.gamma,
            gamma_prime=cfg.gamma_prime,
            gae_lambda=cfg.gae_lambda,
            lambda_prime=cfg.lambda_prime,
            device=self.device,
            store_images=True,
            gae_mode=cfg.gae_mode,
        )
        
        self.reward_wrapper = SharedRewardWrapper(
            reward_type="shared",
            num_agents=DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"],
        )
        
        if cfg.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(gamma=cfg.gamma)
        else:
            self.reward_normalizer = None
        
        self._init_optimizer()
        
        self.global_step = 0
        self.episode_count = 0
        self.best_success_rate = 0.0
        
        if self.rank == 0:
            self._log_config()
    
    def _create_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"dualarm_acppo_twoarm_{timestamp}"
        if self.cfg.run_id_note:
            run_id += f"_{self.cfg.run_id_note}"
        return run_id
    
    def _setup_logging(self):
        if self.rank != 0:
            return

        log_file = self.run_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(file_handler)
        
        self.tb_writer = SummaryWriter(self.run_dir / "tensorboard")
        
        if self.cfg.use_wandb:
            wandb.init(
                entity=self.cfg.wandb_entity,
                project=self.cfg.wandb_project,
                name=self.run_id,
                config=vars(self.cfg),
                mode=self.cfg.wandb_mode,
            )
    
    def _init_policy(self):
        vla, action_head, proprio_projector, noisy_action_projector, processor, norm_stats = \
            load_vla_for_dualarm_acppo(self.cfg, self.device)
        
        self.raw_policy = DualArmMultiAgentVLAPolicyACPPO(
            cfg=self.cfg,
            vla_model=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            processor=processor,
            device=self.device,
        )
        
        if self.distributed:
            self.policy = DDP(
                self.raw_policy, 
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=True,  # ACPPO may have unused parameters
            )
        else:
            self.policy = self.raw_policy
            
        self.processor = processor
        self.norm_stats = norm_stats
        self.action_norm_stats = None
        
        if norm_stats is not None and len(norm_stats) > 0:
            unnorm_key = getattr(self.cfg, 'unnorm_key', None)
            if unnorm_key is None:
                unnorm_key = list(norm_stats.keys())[0]
            if unnorm_key in norm_stats and "action" in norm_stats[unnorm_key]:
                self.action_norm_stats = norm_stats[unnorm_key]["action"]
                if is_main_process():
                    logger.info(f">> Using normalization key: {unnorm_key}")
            else:
                if is_main_process():
                    logger.warning(f">> Normalization key '{unnorm_key}' not found in norm_stats. Available keys: {list(norm_stats.keys())}")
    
    def _init_optimizer(self):
        if self.distributed:
            policy_module = self.raw_policy
        else:
            policy_module = self.policy
        
        actor_params = policy_module.get_actor_parameters()
        critic_params = policy_module.get_critic_parameters()
        
        self._trainable_params = actor_params + critic_params
        
        if self.rank == 0:
            actor_count = sum(p.numel() for p in actor_params)
            critic_count = sum(p.numel() for p in critic_params)
            total_trainable = actor_count + critic_count
            logger.info(f"Actor parameters: {actor_count:,} (lr={self.cfg.actor_lr})")
            logger.info(f"Critic parameters: {critic_count:,} (lr={self.cfg.critic_lr})")
            logger.info(f"Total trainable parameters: {total_trainable:,}")
        
        param_groups = [
            {'params': actor_params, 'lr': self.cfg.actor_lr},
            {'params': critic_params, 'lr': self.cfg.critic_lr},
        ]
        self.optimizer = optim.Adam(param_groups)

    def _log_config(self):
        config_path = self.run_dir / "config.json"
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in vars(self.cfg).items()}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_path}")
        
    def _average_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        if not self.distributed:
            return stats
            
        keys = sorted(stats.keys())
        values = torch.tensor([stats[k] for k in keys], device=self.device, dtype=torch.float32)
        
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        
        return {k: v.item() for k, v in zip(keys, values)}
    
    def _normalize_advantages_global(self, advantages: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        local_sum = advantages.sum()
        local_sq_sum = (advantages ** 2).sum()
        local_count = torch.tensor(advantages.numel(), device=self.device, dtype=torch.float32)
        
        stats = torch.stack([local_sum, local_sq_sum, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
        
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean ** 2)
        global_std = torch.sqrt(global_var + 1e-8)
        
        return (advantages - global_mean) / global_std

    def _reconstruct_full_action(self, agent_action_7dim: np.ndarray, agent_idx: int) -> np.ndarray:
        """Reconstruct 14-dim full action from 7-dim agent action."""
        agent_action_dim = DUALARM_ACPPO_CONSTANTS["AGENT_ACTION_DIM"]
        model_action_dim = DUALARM_ACPPO_CONSTANTS["MODEL_ACTION_DIM"]
        
        full_action = np.zeros(model_action_dim, dtype=np.float32)
        
        if agent_idx == 0:
            full_action[:agent_action_dim] = agent_action_7dim
        else:
            full_action[agent_action_dim:] = agent_action_7dim
        
        return full_action

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout experience using open-loop action chunk execution.
        """
        self.policy.eval()
        self.buffer.reset()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        current_episode_reward = 0
        current_episode_length = 0
        
        obs = self.env.reset()
        self.obs_history.reset()
        
        for _ in range(self.cfg.num_steps_wait):
            obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
        
        agentview_img, left_wrist_img, right_wrist_img, proprio_state = extract_observations_from_env(obs)
        self.obs_history.update(agentview_img, left_wrist_img, right_wrist_img, proprio_state)
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        for step_in_rollout in range(self.cfg.num_steps_per_rollout // self.cfg.num_actions_chunk):
            with torch.no_grad():
                agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                agent_inputs = []
                agent_proprios = []
                task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                
                for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
                
                # ACPPO: Prepare front-view only input for Agent 0's action distribution estimation
                front_view_only_obs = self.obs_history.get_front_view_only_observation(include_history=True)
                front_images_list = [front_view_only_obs['images'][t * 3] 
                                    for t in range(self.cfg.history_length)]
                front_images_list = front_images_list[::-1]  # Reverse to get most recent first
                
                front_view_only_input = prepare_vla_input_for_action_dist_estimation(
                    front_images=front_images_list,
                    task_description=self.task_desc_robot0,
                    processor=self.processor,
                    device=self.device,
                    center_crop=self.cfg.center_crop,
                )
                
                # ACPPO: Get actions with action distribution chaining
                actions, log_probs, entropies, values, action_dist = policy_module.get_actions_and_values(
                    agent_inputs=agent_inputs,
                    agent_proprios=agent_proprios,
                    front_view_only_input=front_view_only_input,
                    deterministic=False,
                )
                
                per_agent_values = [v[0].float().cpu().numpy() for v in values]
            
            # Store initial observations for this chunk
            initial_agentview_images_np = []
            for t in range(self.cfg.history_length):
                initial_agentview_images_np.append(agent_obs[0]['images'][t * 3])
            initial_agentview_images_np = np.stack(initial_agentview_images_np, axis=0)[np.newaxis, ...]
            
            initial_left_wrist_images_np = []
            initial_right_wrist_images_np = []
            for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                left_list = []
                right_list = []
                for t in range(self.cfg.history_length):
                    left_list.append(agent_obs[agent_idx]['images'][t * 3 + 1])
                    right_list.append(agent_obs[agent_idx]['images'][t * 3 + 2])
                initial_left_wrist_images_np.append(np.stack(left_list, axis=0)[np.newaxis, ...])
                initial_right_wrist_images_np.append(np.stack(right_list, axis=0)[np.newaxis, ...])
            
            initial_proprio_states_padded = [
                np.expand_dims(agent_obs[i]['proprio_history'], 0)
                for i in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"])
            ]
            initial_global_proprio_history = np.stack([
                list(self.obs_history.proprio_history)[-(t+1)] 
                for t in range(self.cfg.history_length)
            ][::-1], axis=0)[np.newaxis, ...]

            chunk_reward_sum = 0
            chunk_done = False
            
            for action_in_chunk_idx in range(self.cfg.num_actions_chunk):
                action_0_normalized = actions[0][0, action_in_chunk_idx].float().cpu().numpy()
                action_1_normalized = actions[1][0, action_in_chunk_idx].float().cpu().numpy()
                
                if self.action_norm_stats is not None:
                    action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats, agent_idx=0)
                    action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats, agent_idx=1)
                else:
                    action_0 = action_0_normalized
                    action_1 = action_1_normalized
                
                env_action_dim = DUALARM_ACPPO_CONSTANTS["ENV_ACTION_DIM"]
                full_action_for_env = np.concatenate([action_0[:env_action_dim], action_1[:env_action_dim]])
                
                next_obs, reward, done, info = self.env.step(full_action_for_env.tolist())
                
                chunk_reward_sum += reward
                
                peg_hole_dist = info.get("reward/peg_hole_dist", float('inf'))
                if peg_hole_dist > self.cfg.max_peg_hole_distance:
                    done = True
                
                if done or self.env._check_success() or (current_episode_length + action_in_chunk_idx + 1) >= self.cfg.max_episode_steps:
                    chunk_done = True
                
                agentview_img, left_wrist_img, right_wrist_img, proprio_state = extract_observations_from_env(next_obs)
                self.obs_history.update(agentview_img, left_wrist_img, right_wrist_img, proprio_state)
                
                if chunk_done:
                    break
            
            team_reward = self.reward_wrapper(chunk_reward_sum)
            if self.reward_normalizer is not None:
                team_reward = self.reward_normalizer.normalize(
                    np.array([team_reward]),
                    np.array([float(chunk_done)]),
                )[0]
            
            # Store action distribution for ACPPO
            action_means = []
            action_stds = []
            if action_dist is not None:
                action_mu, action_sigma = action_dist
                action_means = [
                    action_mu[0].float().cpu().numpy().flatten(),
                    action_mu[0].float().cpu().numpy().flatten(),  # Same for both (from Agent 0)
                ]
                action_stds = [
                    action_sigma[0].float().cpu().numpy().flatten(),
                    action_sigma[0].float().cpu().numpy().flatten(),
                ]
            else:
                for _ in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    action_means.append(np.zeros(self.cfg.agent_action_dim * self.cfg.num_actions_chunk))
                    action_stds.append(np.ones(self.cfg.agent_action_dim * self.cfg.num_actions_chunk))
            
            actions_14dim = []
            actions_7dim_for_buffer = []
            for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                actions_14dim.append(actions[agent_idx].float().cpu().numpy())
                actions_7dim_for_buffer.append(actions[agent_idx].float().cpu().numpy())
            
            log_probs_np = [lp[0].float().cpu().numpy() for lp in log_probs]
            
            self.buffer.add(
                agentview_images=initial_agentview_images_np,
                left_wrist_images=initial_left_wrist_images_np,
                right_wrist_images=initial_right_wrist_images_np,
                proprio_states=[ps for ps in initial_proprio_states_padded],
                global_proprio=initial_global_proprio_history,
                actions=[a for a in actions_14dim],
                agent_actions_7dim=[a for a in actions_7dim_for_buffer],
                log_probs=[np.expand_dims(lp, 0) for lp in log_probs_np],
                action_means=[np.expand_dims(am, 0) for am in action_means],
                action_stds=[np.expand_dims(ast, 0) for ast in action_stds],
                values=[np.expand_dims(v, 0) for v in per_agent_values],
                reward=np.array([team_reward]),
                done=np.array([float(chunk_done)]),
            )
            
            current_episode_reward += chunk_reward_sum
            current_episode_length += (action_in_chunk_idx + 1)
            success = self.env._check_success()
            
            if chunk_done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_successes.append(float(success))
                self.episode_count += 1
                
                obs = self.env.reset()
                self.obs_history.reset()
                for _ in range(self.cfg.num_steps_wait):
                    obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
                agentview_img, left_wrist_img, right_wrist_img, proprio_state = extract_observations_from_env(obs)
                self.obs_history.update(agentview_img, left_wrist_img, right_wrist_img, proprio_state)
                current_episode_reward = 0
                current_episode_length = 0
            
            self.global_step += (action_in_chunk_idx + 1)
        
        # Compute GAE
        with torch.no_grad():
            agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
            agent_inputs = []
            agent_proprios = []
            task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
            
            for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
            
            _, _, _, last_values, _ = policy_module.get_actions_and_values(
                agent_inputs=agent_inputs,
                agent_proprios=agent_proprios,
                deterministic=True,
            )
            
            last_values_np = [v[0].float().cpu().numpy() for v in last_values]
        
        last_dones = np.array([float(chunk_done)])
        self.buffer.compute_returns_and_advantages(last_values_np, last_dones)
        
        stats = {
            "rollout/ep_reward_mean": np.mean(episode_rewards) if episode_rewards else 0,
            "rollout/ep_length_mean": np.mean(episode_lengths) if episode_lengths else 0,
            "rollout/success_rate": np.mean(episode_successes) if episode_successes else 0,
            "rollout/num_episodes": len(episode_rewards),
        }
        
        return stats

    def update(self) -> Dict[str, float]:
        """Perform PPO update on collected rollouts."""
        self.policy.train()
        
        batch_size = self.buffer.size * self.cfg.num_envs // self.cfg.num_minibatches
        
        epoch_stats = {
            "loss/total": [],
            "loss/policy": [],
            "loss/value": [],
            "loss/entropy": [],
            "training/approx_kl": [],
            "training/clip_fraction": [],
        }
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        for epoch in range(self.cfg.num_epochs):
            for batch in self.buffer.get(batch_size):
                agent_proprios = batch.agent_proprios
                agent_actions = batch.agent_actions
                agent_old_log_probs = batch.agent_old_log_probs
                agent_advantages = batch.agent_advantages
                agent_returns = batch.agent_returns
                
                if self.cfg.normalize_advantages:
                    for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                        agent_advantages[agent_idx] = self._normalize_advantages_global(agent_advantages[agent_idx])
                
                # Prepare inputs for each agent
                agent_inputs = []
                task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                
                for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_proprio_np = agent_proprios[agent_idx][:, -1, :].float().cpu().numpy()
                    
                    # Create single dummy images (3D: H, W, C), not batched
                    dummy_images = [
                        np.zeros((224, 224, 3), dtype=np.uint8)
                        for _ in range(3 * self.cfg.history_length)
                    ]
                    
                    inputs = prepare_vla_input(
                        images=dummy_images,
                        proprio=agent_proprio_np[0],
                        task_description=task_descriptions[agent_idx],
                        processor=self.processor,
                        device=self.device,
                        center_crop=self.cfg.center_crop,
                        agent_idx=agent_idx,
                    )
                    
                    batch_inputs = {k: v.repeat(agent_proprio_np.shape[0], *([1] * (v.dim() - 1))) 
                                   for k, v in inputs.items()}
                    agent_inputs.append(batch_inputs)
                
                # Reconstruct full 14-dim actions for VLA evaluation
                agent_actions_14dim = []
                for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    action_7dim = agent_actions[agent_idx]
                    batch_size_local = action_7dim.shape[0]
                    chunk_size = action_7dim.shape[1]
                    
                    full_actions = torch.zeros(
                        batch_size_local, chunk_size, DUALARM_ACPPO_CONSTANTS["MODEL_ACTION_DIM"],
                        device=self.device, dtype=action_7dim.dtype
                    )
                    
                    if agent_idx == 0:
                        full_actions[..., :DUALARM_ACPPO_CONSTANTS["AGENT_ACTION_DIM"]] = action_7dim
                    else:
                        full_actions[..., DUALARM_ACPPO_CONSTANTS["AGENT_ACTION_DIM"]:] = action_7dim
                    
                    agent_actions_14dim.append(full_actions)
                
                # Evaluate actions
                agent_proprio_tensors = [
                    proprio[:, -1, :].to(dtype=torch.bfloat16)
                    for proprio in agent_proprios
                ]
                
                new_log_probs, new_entropies, new_values = policy_module.evaluate_actions_and_values(
                    agent_inputs=agent_inputs,
                    agent_actions=agent_actions_14dim,
                    agent_proprios=agent_proprio_tensors,
                )
                
                # Compute losses for each agent and aggregate
                total_policy_loss = 0
                total_value_loss = 0
                total_entropy_loss = 0
                total_approx_kl = 0
                total_clip_frac = 0
                
                for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    log_ratio = new_log_probs[agent_idx].float() - agent_old_log_probs[agent_idx].float()
                    ratio = log_ratio.exp()
                    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
                    
                    advantages = agent_advantages[agent_idx].float()
                    
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    returns = agent_returns[agent_idx].float()
                    values_pred = new_values[agent_idx].float()
                    
                    if self.cfg.clip_value_loss:
                        old_values = batch.agent_old_values[agent_idx].float()
                        values_clipped = old_values + torch.clamp(
                            values_pred - old_values,
                            -self.cfg.clip_epsilon,
                            self.cfg.clip_epsilon
                        )
                        value_loss1 = (values_pred - returns) ** 2
                        value_loss2 = (values_clipped - returns) ** 2
                        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                    else:
                        value_loss = 0.5 * ((values_pred - returns) ** 2).mean()
                    
                    entropy_loss = -new_entropies[agent_idx].float().mean()
                    
                    total_policy_loss += pg_loss
                    total_value_loss += value_loss
                    total_entropy_loss += entropy_loss
                    total_approx_kl += approx_kl
                    total_clip_frac += clip_frac
                
                num_agents = DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]
                total_policy_loss /= num_agents
                total_value_loss /= num_agents
                total_entropy_loss /= num_agents
                total_approx_kl /= num_agents
                total_clip_frac /= num_agents
                
                loss = (
                    total_policy_loss +
                    self.cfg.value_loss_coef * total_value_loss +
                    self.cfg.entropy_coef * total_entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.max_grad_norm)
                
                self.optimizer.step()
                
                epoch_stats["loss/total"].append(loss.item())
                epoch_stats["loss/policy"].append(total_policy_loss.item())
                epoch_stats["loss/value"].append(total_value_loss.item())
                epoch_stats["loss/entropy"].append(-total_entropy_loss.item())
                epoch_stats["training/approx_kl"].append(total_approx_kl.item())
                epoch_stats["training/clip_fraction"].append(total_clip_frac.item())
        
        stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        
        return stats

    def evaluate(self, num_episodes: int = 10, save_video: bool = True, num_videos: int = 3) -> Dict[str, float]:
        """Run evaluation episodes with optional video saving."""
        self.policy.eval()
        
        # Only rank 0 saves videos in distributed setting
        my_num_episodes = num_episodes // self.world_size
        if self.rank < num_episodes % self.world_size:
            my_num_episodes += 1
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        # Video saving setup (only rank 0)
        save_video = save_video and (self.rank == 0)
        if save_video:
            video_dir = self.run_dir / "eval_videos"
            video_dir.mkdir(parents=True, exist_ok=True)
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        for ep in range(my_num_episodes):
            obs = self.env.reset()
            self.obs_history.reset()
            
            # Determine if we should save video for this episode
            should_save_video = save_video and (ep < num_videos)
            replay_images = [] if should_save_video else None
            
            for _ in range(self.cfg.num_steps_wait):
                obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
            
            agentview_img, left_wrist_img, right_wrist_img, proprio_state = extract_observations_from_env(obs)
            self.obs_history.update(agentview_img, left_wrist_img, right_wrist_img, proprio_state)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            from collections import deque
            action_queue = deque()
            
            while not done and episode_length < self.cfg.max_episode_steps:
                # Capture video frame before action
                if should_save_video:
                    replay_images.append(get_twoarm_video_frame(obs))
                
                if len(action_queue) == 0:
                    with torch.no_grad():
                        agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                        agent_inputs = []
                        agent_proprios = []
                        task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                        
                        for agent_idx in range(DUALARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
                        
                        # ACPPO: Prepare front-view only input for Agent 0's action distribution estimation
                        front_view_only_obs = self.obs_history.get_front_view_only_observation(include_history=True)
                        front_images_list = [front_view_only_obs['images'][t * 3] 
                                            for t in range(self.cfg.history_length)]
                        front_images_list = front_images_list[::-1]  # Reverse to get most recent first
                        
                        front_view_only_input = prepare_vla_input_for_action_dist_estimation(
                            front_images=front_images_list,
                            task_description=self.task_desc_robot0,
                            processor=self.processor,
                            device=self.device,
                            center_crop=self.cfg.center_crop,
                        )
                        
                        actions, _, _, _, _ = policy_module.get_actions_and_values(
                            agent_inputs=agent_inputs,
                            agent_proprios=agent_proprios,
                            front_view_only_input=front_view_only_input,
                            deterministic=True,
                        )
                        
                        for action_idx in range(self.cfg.num_actions_chunk):
                            action_0 = actions[0][0, action_idx].float().cpu().numpy()
                            action_1 = actions[1][0, action_idx].float().cpu().numpy()
                            action_queue.append((action_0, action_1))
                
                action_0_normalized, action_1_normalized = action_queue.popleft()
                
                if self.action_norm_stats is not None:
                    action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats, agent_idx=0)
                    action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats, agent_idx=1)
                else:
                    action_0 = action_0_normalized
                    action_1 = action_1_normalized
                
                env_action_dim = DUALARM_ACPPO_CONSTANTS["ENV_ACTION_DIM"]
                full_action = np.concatenate([action_0[:env_action_dim], action_1[:env_action_dim]])
                
                obs, reward, done, info = self.env.step(full_action.tolist())
                
                episode_reward += reward
                episode_length += 1
                
                agentview_img, left_wrist_img, right_wrist_img, proprio_state = extract_observations_from_env(obs)
                self.obs_history.update(agentview_img, left_wrist_img, right_wrist_img, proprio_state)
                
                if self.env._check_success():
                    done = True
            
            success = self.env._check_success()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(float(success))
            
            # Save video if requested
            if should_save_video and replay_images:
                video_path = video_dir / f"eval_step{self.global_step}_ep{ep}_{'success' if success else 'fail'}.mp4"
                try:
                    save_rollout_video(
                        replay_images,
                        str(video_path),
                        fps=30,
                    )
                    logger.info(f"Saved eval video: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to save video: {e}")
        
        # Aggregate stats across GPUs if distributed
        if self.distributed:
            all_rewards = [None for _ in range(self.world_size)]
            all_lengths = [None for _ in range(self.world_size)]
            all_successes = [None for _ in range(self.world_size)]
            
            dist.all_gather_object(all_rewards, episode_rewards)
            dist.all_gather_object(all_lengths, episode_lengths)
            dist.all_gather_object(all_successes, episode_successes)
            
            episode_rewards = [r for rewards in all_rewards for r in rewards]
            episode_lengths = [l for lengths in all_lengths for l in lengths]
            episode_successes = [s for successes in all_successes for s in successes]
        
        stats = {
            "eval/ep_reward_mean": np.mean(episode_rewards) if episode_rewards else 0,
            "eval/ep_reward_std": np.std(episode_rewards) if episode_rewards else 0,
            "eval/ep_length_mean": np.mean(episode_lengths) if episode_lengths else 0,
            "eval/success_rate": np.mean(episode_successes) if episode_successes else 0,
        }
        
        return stats

    def save_checkpoint(self, path: Optional[Path] = None):
        """Save training checkpoint."""
        if self.rank != 0:
            return
        
        if path is None:
            path = self.run_dir / f"checkpoint_{self.global_step}.pt"
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        checkpoint = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "best_success_rate": self.best_success_rate,
            "policy_state_dict": policy_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.cfg),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def train(self):
        """Main training loop."""
        if self.rank == 0:
            logger.info("="*60)
            logger.info("Starting Dual-Arm ACPPO Training")
            logger.info("="*60)
        
        num_updates = self.cfg.total_timesteps // (self.cfg.num_steps_per_rollout * self.world_size)
        
        for update in tqdm.tqdm(range(num_updates), desc="Training", disable=self.rank != 0):
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            rollout_stats = self._average_stats(rollout_stats)
            
            # Update policy
            update_stats = self.update()
            update_stats = self._average_stats(update_stats)
            
            # Logging
            if self.rank == 0 and update % self.cfg.wandb_log_freq == 0:
                all_stats = {**rollout_stats, **update_stats}
                all_stats["training/global_step"] = self.global_step
                all_stats["training/episode_count"] = self.episode_count
                
                if self.cfg.use_wandb:
                    wandb.log(all_stats, step=self.global_step)
                
                for key, value in all_stats.items():
                    self.tb_writer.add_scalar(key, value, self.global_step)
            
            # Evaluation
            if self.global_step % self.cfg.eval_freq < self.cfg.num_steps_per_rollout:
                eval_stats = self.evaluate(
                    num_episodes=self.cfg.num_eval_episodes,
                    save_video=self.cfg.save_eval_videos,
                    num_videos=self.cfg.num_eval_videos,
                )
                
                if self.rank == 0:
                    if self.cfg.use_wandb:
                        wandb.log(eval_stats, step=self.global_step)
                    
                    for key, value in eval_stats.items():
                        self.tb_writer.add_scalar(key, value, self.global_step)
                    
                    if eval_stats["eval/success_rate"] > self.best_success_rate:
                        self.best_success_rate = eval_stats["eval/success_rate"]
                        self.save_checkpoint(self.run_dir / "best_checkpoint.pt")
                    
                    logger.info(f"Step {self.global_step}: Eval success rate = {eval_stats['eval/success_rate']:.3f}")
            
            # Save checkpoint
            if self.global_step % self.cfg.save_freq < self.cfg.num_steps_per_rollout:
                self.save_checkpoint()
        
        self.save_checkpoint(self.run_dir / "final_checkpoint.pt")
        
        if self.rank == 0:
            logger.info("="*60)
            logger.info("Training Complete!")
            logger.info(f"Best success rate: {self.best_success_rate:.3f}")
            logger.info("="*60)


@draccus.wrap()
def main(cfg: DualArmACPPOConfig):
    """Main entry point for Dual-Arm ACPPO training."""
    setup_distributed()
    
    trainer = DualArmACPPOTrainer(cfg)
    trainer.train()
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
