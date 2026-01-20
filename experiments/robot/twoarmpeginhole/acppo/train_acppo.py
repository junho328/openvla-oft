"""
ACPPO Training Script for Multi-Agent VLA on TwoArmPegInHole Environment.

ACPPO (Agent-Chained PPO) key features:
1. All agents act SIMULTANEOUSLY - they decide and execute at the same time
2. Agent 0: Uses only its own observations
3. Agent 1+: Additionally uses ESTIMATED action distributions from previous agents
   (but still acts at the same time as other agents)
4. Microstep-based advantage calculation for proper credit assignment
5. Shared policy components across agents

IMPORTANT: The "chaining" refers to information flow, NOT execution order.
All agents act at the same time step.

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

from .config import ACPPOConfig, TWOARM_ACPPO_CONSTANTS
from .observation_utils import (
    ObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    prepare_vla_input_for_action_dist_estimation,
    unnormalize_action,
)
from .vla_policy import MultiAgentVLAPolicyACPPO, load_vla_for_acppo
from .rollout_buffer import (
    MultiAgentRolloutBufferACPPO,
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


class ACPPOTrainer:
    """
    ACPPO Trainer for Multi-Agent VLA training on TwoArmPegInHole.
    
    Key differences from MAPPO:
    1. All agents act SIMULTANEOUSLY (not sequentially)
    2. Agent 1+ estimates previous agents' action distributions and uses them as input
       (but still acts at the same time as other agents)
    3. Per-agent value functions and advantages
    4. Microstep-based GAE for proper credit assignment
    """
    
    def __init__(self, cfg: ACPPOConfig):
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
        
        set_seed_everywhere(cfg.seed + self.rank)
        
        self.run_id = self._create_run_id()
        self.run_dir = cfg.run_root_dir / self.run_id
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure directory is created before other ranks proceed
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
        
        self.obs_history = ObservationHistoryManager(
            num_agents=TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            history_length=cfg.history_length,
        )
        
        if self.rank == 0:
            logger.info("Loading VLA model for ACPPO...")
        self._init_policy()
        
        # ACPPO buffer with per-agent values and action distributions
        self.buffer = MultiAgentRolloutBufferACPPO(
            buffer_size=cfg.num_steps_per_rollout,
            num_agents=TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"],
            num_envs=cfg.num_envs,
            action_dim=cfg.action_dim,
            action_chunk_size=cfg.num_actions_chunk,
            proprio_dim=TWOARM_ACPPO_CONSTANTS["PROPRIO_DIM"],
            history_length=cfg.history_length,
            gamma=cfg.gamma,
            gamma_prime=cfg.gamma_prime,
            gae_lambda=cfg.gae_lambda,
            lambda_prime=cfg.lambda_prime,
            device=self.device,
            store_images=True,
        )
        
        self.reward_wrapper = SharedRewardWrapper(
            reward_type="shared",
            num_agents=TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"],
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
        run_id = f"acppo_twoarm_{timestamp}"
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
        """Initialize VLA policy with ACPPO extensions."""
        vla, action_head, proprio_projector, proprio_projector_extended, \
            noisy_action_projector, processor, norm_stats = load_vla_for_acppo(self.cfg, self.device)
        
        self.raw_policy = MultiAgentVLAPolicyACPPO(
            cfg=self.cfg,
            vla_model=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            proprio_projector_extended=proprio_projector_extended,
            noisy_action_projector=noisy_action_projector,
            processor=processor,
            device=self.device,
        )
        
        # Synchronize all processes before DDP wrapping
        if self.distributed:
            dist.barrier()
        
        if self.distributed:
            # _set_static_graph() is needed because in ACPPO:
            # - estimate_agent0_action_dist() uses shared_agent (with no_grad)
            # - then main forward pass also uses shared_agent
            # This causes proprio_projector to be "used" twice per iteration
            # Note: _set_static_graph() automatically detects unused parameters,
            # so find_unused_parameters=True is not needed
            self.policy = DDP(
                self.raw_policy, 
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
            )
            # Enable static graph to allow same parameters to be used multiple times
            self.policy._set_static_graph()
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
    
    def _init_optimizer(self):
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
        """Normalize advantages across ALL GPUs."""
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

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout experience with ACPPO agent chaining.
        
        Key differences from MAPPO:
        1. Prepare front-view only input for action distribution estimation
        2. Store per-agent values
        3. Store action distributions for agent chaining
        """
        self.policy.eval()
        self.buffer.reset()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        current_episode_reward = 0
        current_episode_length = 0
        
        step_rewards = []
        
        obs = self.env.reset()
        self.obs_history.reset()
        
        for _ in range(self.cfg.num_steps_wait):
            obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
        
        front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
        self.obs_history.update(front_img, wrist_imgs, proprio_states)
        
        policy_module = self.policy.module if self.distributed else self.policy
        
        for step in range(self.cfg.num_steps_per_rollout):
            with torch.no_grad():
                agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                
                agent_inputs = []
                agent_proprios = []
                task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                
                for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
                
                # Prepare front-view only input for action distribution estimation
                front_images_list = [list(self.obs_history.front_image_history)[-(i+1)] 
                                    for i in range(self.cfg.history_length)]
                front_images_list = front_images_list[::-1]  # Chronological order
                
                front_view_only_input = prepare_vla_input_for_action_dist_estimation(
                    front_images=front_images_list,
                    task_description=self.task_desc_robot0,  # Use robot 0's task
                    processor=self.processor,
                    device=self.device,
                    center_crop=self.cfg.center_crop,
                )
                
                # Forward pass with ACPPO chaining
                actions, log_probs, entropies, values, action_means, action_stds = \
                    policy_module.get_actions_and_values(
                        agent_inputs=agent_inputs,
                        agent_proprios=agent_proprios,
                        front_view_only_input=front_view_only_input,
                        deterministic=False,
                    )
            
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
            
            step_rewards.append(reward)
            
            team_reward = self.reward_wrapper(reward)
            
            if self.reward_normalizer is not None:
                team_reward = self.reward_normalizer.normalize(
                    np.array([team_reward]),
                    np.array([float(done)]),
                )[0]
            
            # Store transition with per-agent data
            proprio_states_np = [agent_obs[i]['proprio_history'] for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            actions_np = [actions[i][0].float().cpu().numpy() for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            log_probs_np = [log_probs[i][0].float().cpu().numpy() for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            action_means_np = [action_means[i][0].float().cpu().numpy() for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            action_stds_np = [action_stds[i][0].float().cpu().numpy() for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            values_np = [values[i][0].float().cpu().numpy() for i in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
            
            # Store images
            front_images_list = []
            for t in range(self.cfg.history_length):
                front_images_list.append(agent_obs[0]['images'][t * 2])
            front_images_np = np.stack(front_images_list, axis=0)[np.newaxis, ...]
            
            wrist_images_np = []
            for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
                action_means=[np.expand_dims(am, 0) for am in action_means_np],
                action_stds=[np.expand_dims(ast, 0) for ast in action_stds_np],
                values=[np.expand_dims(v, 0) for v in values_np],
                reward=np.array([team_reward]),
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
        
        # Compute final values for all agents
        with torch.no_grad():
            agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
            agent_inputs = []
            agent_proprios = []
            for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
            
            # Prepare front-view only input for last step
            front_images_list = [list(self.obs_history.front_image_history)[-(i+1)] 
                                for i in range(self.cfg.history_length)]
            front_images_list = front_images_list[::-1]
            
            front_view_only_input = prepare_vla_input_for_action_dist_estimation(
                front_images=front_images_list,
                task_description=self.task_desc_robot0,
                processor=self.processor,
                device=self.device,
                center_crop=self.cfg.center_crop,
            )
            
            last_values = policy_module.get_values(
                agent_inputs=agent_inputs, 
                agent_proprios=agent_proprios,
                front_view_only_input=front_view_only_input,
            )
            last_values_np = [v.float().cpu().numpy() for v in last_values]
        
        # Compute per-agent returns and advantages using ACPPO microstep GAE
        self.buffer.compute_returns_and_advantages(
            last_values=last_values_np,
            last_dones=np.array([float(done)]),
        )
        
        if self.reward_normalizer is not None and self.distributed:
            self.reward_normalizer.sync(self.device)
        
        stats = {
            "rollout/mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "rollout/mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "rollout/success_rate": np.mean(episode_successes) if episode_successes else 0,
            "rollout/num_episodes": len(episode_rewards),
        }
        
        if step_rewards:
            stats["rollout/mean_step_reward"] = np.mean(step_rewards)
            stats["rollout/sum_step_reward"] = np.sum(step_rewards)
        
        return self._average_stats(stats)
    
    def update(self) -> Dict[str, float]:
        """Perform ACPPO update with per-agent losses."""
        self.policy.train()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        # Per-agent stats
        agent_policy_losses = [[] for _ in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
        agent_value_losses = [[] for _ in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"])]
        
        batch_size = (self.cfg.num_steps_per_rollout * self.cfg.num_envs) // self.cfg.num_minibatches
        
        task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
        policy_module = self.policy.module if self.distributed else self.policy
        
        for epoch in range(self.cfg.num_epochs):
            for batch in self.buffer.get(batch_size):
                # Per-agent advantages (normalized)
                batch_advantages = []
                for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    adv = batch.agent_advantages[agent_idx]
                    if self.cfg.normalize_advantages:
                        adv = self._normalize_advantages_global(adv)
                    batch_advantages.append(adv)
                
                all_new_log_probs = []
                all_entropies = []
                all_new_values = []
                
                current_batch_size = batch.agent_actions[0].shape[0]
                
                for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_actions = batch.agent_actions[agent_idx]
                    agent_proprios = batch.agent_proprios[agent_idx]
                    
                    agent_front_images = batch.agent_front_images[agent_idx]
                    agent_wrist_images = batch.agent_wrist_images[agent_idx]
                    
                    # Get estimated action distribution for agent 1
                    estimated_action_dist = None
                    if agent_idx == 1 and self.cfg.use_action_dist_input:
                        # Use stored action distributions from buffer
                        estimated_action_dist = (
                            batch.agent_action_means[0],  # Agent 0's mu
                            batch.agent_action_stds[0],   # Agent 0's sigma
                        )
                    
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
                            
                            # Get estimated action dist for this sub-batch
                            sub_estimated_action_dist = None
                            if estimated_action_dist is not None:
                                sub_estimated_action_dist = (
                                    estimated_action_dist[0][sub_start:sub_end],
                                    estimated_action_dist[1][sub_start:sub_end],
                                )
                            
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
                                
                                # Get estimated action dist for this sample
                                sample_estimated_action_dist = None
                                if sub_estimated_action_dist is not None:
                                    sample_estimated_action_dist = (
                                        sub_estimated_action_dist[0][b:b+1],
                                        sub_estimated_action_dist[1][b:b+1],
                                    )
                                
                                # Call through DDP wrapper
                                log_prob, entropy, value = self.policy(
                                    mode='evaluate_agent',
                                    agent_idx=agent_idx,
                                    inputs=vla_input,
                                    actions=action_to_eval,
                                    proprio=proprio_tensor,
                                    estimated_action_dist=sample_estimated_action_dist,
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
                        agent_old_log_probs = batch.agent_old_log_probs[agent_idx]
                        new_log_probs = agent_old_log_probs
                        entropies = torch.zeros_like(agent_old_log_probs)
                        new_values = batch.agent_old_values[agent_idx]
                    
                    all_new_log_probs.append(new_log_probs)
                    all_entropies.append(entropies)
                    all_new_values.append(new_values)
                
                # Compute per-agent losses
                all_policy_losses = []
                all_value_losses = []
                all_clip_fracs = []
                all_approx_kl = []
                
                for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_old_log_probs = batch.agent_old_log_probs[agent_idx]
                    new_log_probs = all_new_log_probs[agent_idx]
                    agent_advantages = batch_advantages[agent_idx]
                    agent_returns = batch.agent_returns[agent_idx]
                    agent_old_values = batch.agent_old_values[agent_idx]
                    new_values = all_new_values[agent_idx]
                    
                    # Policy loss
                    log_ratio = new_log_probs - agent_old_log_probs
                    ratio = torch.exp(log_ratio)
                    
                    surr1 = ratio * agent_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * agent_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    all_policy_losses.append(policy_loss)
                    agent_policy_losses[agent_idx].append(policy_loss.item())
                    
                    # Value loss (per-agent)
                    if self.cfg.clip_value_loss:
                        values_clipped = agent_old_values + torch.clamp(
                            new_values - agent_old_values, 
                            -self.cfg.clip_epsilon, 
                            self.cfg.clip_epsilon
                        )
                        value_loss_unclipped = (new_values - agent_returns) ** 2
                        value_loss_clipped = (values_clipped - agent_returns) ** 2
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * ((new_values - agent_returns) ** 2).mean()
                    
                    all_value_losses.append(value_loss)
                    agent_value_losses[agent_idx].append(value_loss.item())
                    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        all_approx_kl.append(approx_kl)
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
                        all_clip_fracs.append(clip_frac)
                
                # Aggregate losses
                policy_loss = torch.stack(all_policy_losses).mean().float()
                value_loss = torch.stack(all_value_losses).mean().float()
                entropy = torch.stack(all_entropies).mean().float()
                
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.cfg.value_loss_coef * value_loss + self.cfg.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
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
        
        # Per-agent stats
        for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
            stats[f"update/agent{agent_idx}_policy_loss"] = np.mean(agent_policy_losses[agent_idx])
            stats[f"update/agent{agent_idx}_value_loss"] = np.mean(agent_value_losses[agent_idx])
        
        return self._average_stats(stats)
    
    def evaluate(self, num_episodes: int = 10, save_video: bool = True, num_videos: int = 3) -> Dict[str, float]:
        """Evaluate policy."""
        self.policy.eval()
        
        my_num_episodes = num_episodes // self.world_size
        if self.rank < num_episodes % self.world_size:
            my_num_episodes += 1
            
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        
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
                if should_save_video:
                    replay_images.append(get_twoarm_video_frame(obs))
                
                with torch.no_grad():
                    agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                    agent_inputs = []
                    agent_proprios = []
                    task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                    
                    for agent_idx in range(TWOARM_ACPPO_CONSTANTS["NUM_AGENTS"]):
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
                            dtype=torch.bfloat16
                        ).unsqueeze(0)
                        agent_proprios.append(proprio_tensor)
                    
                    # Prepare front-view only input
                    front_images_list = [list(self.obs_history.front_image_history)[-(i+1)] 
                                        for i in range(self.cfg.history_length)]
                    front_images_list = front_images_list[::-1]
                    
                    front_view_only_input = prepare_vla_input_for_action_dist_estimation(
                        front_images=front_images_list,
                        task_description=self.task_desc_robot0,
                        processor=self.processor,
                        device=self.device,
                        center_crop=self.cfg.center_crop,
                    )
                    
                    actions, _, _, _, _ = policy_module.get_actions(
                        agent_inputs=agent_inputs, 
                        agent_proprios=agent_proprios,
                        front_view_only_input=front_view_only_input,
                        deterministic=True,
                    )
                
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
                    if should_save_video:
                        replay_images.append(get_twoarm_video_frame(obs))
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
                for frame in replay_images:
                    video_writer.append_data(frame)
                video_writer.close()
                logger.info(f"Saved eval video: {video_filename}")

        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_success = np.mean(episode_successes) if episode_successes else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        stats = {
            "eval/mean_reward": mean_reward,
            "eval/success_rate": mean_success,
            "eval/mean_episode_length": mean_length,
        }
        return self._average_stats(stats)

    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint."""
        if self.rank != 0:
            return
        
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
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if self.rank == 0:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        policy_state_path = checkpoint_path / "policy.pt"
        if not policy_state_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_state_path}")
        
        policy_module = self.policy.module if self.distributed else self.policy
        policy_state = torch.load(policy_state_path, map_location=self.device)
        policy_module.load_state_dict(policy_state)
        
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state["optimizer"])
            self.global_step = training_state["global_step"]
            self.episode_count = training_state["episode_count"]
            self.best_success_rate = training_state.get("best_success_rate", 0.0)
        
        normalizer_path = checkpoint_path / "reward_normalizer.pt"
        if self.reward_normalizer is not None and normalizer_path.exists():
            normalizer_state = torch.load(normalizer_path, map_location="cpu")
            self.reward_normalizer.return_rms.mean = normalizer_state["mean"]
            self.reward_normalizer.return_rms.var = normalizer_state["var"]
            self.reward_normalizer.return_rms.count = normalizer_state["count"]
            self.reward_normalizer.returns = normalizer_state["returns"]
        
        if self.distributed:
            dist.barrier()

    def train(self):
        """Main training loop."""
        if self.cfg.resume_checkpoint is not None:
            self.load_checkpoint(Path(self.cfg.resume_checkpoint))
        
        if self.rank == 0:
            logger.info(f"Starting ACPPO training for {self.cfg.total_timesteps} timesteps")
            logger.info(f"Run directory: {self.run_dir}")
            
        num_updates = self.cfg.total_timesteps // (self.cfg.num_steps_per_rollout * self.cfg.num_envs)
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
                    step_reward_str = f"StepRwd: {stats.get('rollout/mean_step_reward', 0):.3f}" if 'rollout/mean_step_reward' in stats else ""
                    
                    logger.info(
                        f"Step {self.global_step} | EpRwd: {stats['rollout/mean_episode_reward']:.2f} | "
                        f"{step_reward_str} | "
                        f"Success: {stats['rollout/success_rate']:.2%} | Loss: {stats['update/total_loss']:.4f}"
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
        
        # Ensure all processes finish training before final operations
        if self.distributed:
            dist.barrier()
        
        self.save_checkpoint(suffix="_final")
        if self.rank == 0:
            logger.info("Training completed!")
            self.tb_writer.close()
            if self.cfg.use_wandb:
                wandb.finish()
        
        if self.distributed:
            dist.barrier()  # Ensure all processes finished before cleanup
            dist.destroy_process_group()


@draccus.wrap()
def main(cfg: ACPPOConfig):
    setup_distributed()
    trainer = ACPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
