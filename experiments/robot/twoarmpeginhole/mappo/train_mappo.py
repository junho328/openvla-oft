"""
MAPPO Training Script for Multi-Agent VLA on TwoArmPegInHole Environment.

This script trains two OpenVLA-OFT agents using Multi-Agent Proximal Policy
Optimization (MAPPO) in the TwoArmPegInHole robosuite environment.

Architecture:
    VLA Backbone (frozen) ─┬─────────→ Action Head (trainable) → Action
                           └─────────→ Value Head (trainable)  → Value
    
    Both heads share the same VLA hidden states for efficient training.

Key Features:
- Two separate VLA models for two robot arms (or shared policy)
- Value Head on top of VLA backbone (separate from Action Head)
- VLA backbone frozen, only train: Action Head + Value Head + Proprio Projector
- Image input: agentview + wrist view with history
- Action chunking with parallel decoding
- Reward from environment (success-based)
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


class MAPPOTrainer:
    """
    MAPPO Trainer for Multi-Agent VLA training on TwoArmPegInHole.
    
    Architecture:
        VLA Backbone (frozen) ─┬─→ Action Head (trainable) → Action
                               └─→ Value Head (trainable)  → Value
    
    Training:
        - VLA backbone: FROZEN (no gradient)
        - Proprio Projector: TRAINABLE (if enabled)
        - Action Head MLP: TRAINABLE
        - Value Head MLP: TRAINABLE
        - log_std: TRAINABLE (for exploration)
    """
    
    def __init__(self, cfg: MAPPOConfig):
        """
        Initialize MAPPO trainer.
        
        Args:
            cfg: MAPPO configuration
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        set_seed_everywhere(cfg.seed)
        
        # Setup directories
        self.run_id = self._create_run_id()
        self.run_dir = cfg.run_root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize environment
        logger.info("Initializing TwoArmPegInHole environment...")
        self.env, _ = get_twoarm_env(
            cfg.model_family,
            resolution=cfg.env_img_res,
            robot1=cfg.robot1,
            robot2=cfg.robot2,
            controller=cfg.controller,
            env_configuration=cfg.env_configuration,
            reward_shaping=cfg.reward_shaping,
        )
        
        # Get task descriptions for each robot
        self.task_desc_robot0, self.task_desc_robot1, self.combined_description = \
            get_twoarm_task_descriptions(mode=cfg.instruction_mode)
        
        logger.info(f"Robot 0 task: {self.task_desc_robot0}")
        logger.info(f"Robot 1 task: {self.task_desc_robot1}")
        
        # Initialize observation history manager
        self.obs_history = ObservationHistoryManager(
            num_agents=TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"],
            history_length=cfg.history_length,
        )
        
        # Initialize VLA policy with Action Head + Value Head
        logger.info("Loading VLA model with Action Head + Value Head...")
        self._init_policy()
        
        # Initialize rollout buffer
        # store_images=True for full PPO update (re-evaluation during update)
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
            store_images=True,  # Store images for full PPO update
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
            )
    
    def _init_policy(self):
        """
        Initialize VLA policy with Action Head and Value Head.
        
        Both heads are on top of frozen VLA backbone.
        Also loads action normalization statistics for unnormalizing VLA outputs.
        """
        # Load VLA components (including norm_stats for action unnormalization)
        vla, action_head, proprio_projector, noisy_action_projector, processor, norm_stats = \
            load_vla_for_mappo(self.cfg, self.device)
        
        # Create multi-agent policy (includes both Action Head and Value Head)
        self.policy = MultiAgentVLAPolicy(
            cfg=self.cfg,
            vla_model=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            processor=processor,
            device=self.device,
        )
        
        self.processor = processor
        
        # Store normalization statistics for action unnormalization
        # VLA outputs normalized actions in [-1, 1], need to unnormalize for environment
        self.norm_stats = norm_stats
        self.action_norm_stats = None
        
        if norm_stats is not None and len(norm_stats) > 0:
            # Get the first dataset's action stats (or use unnorm_key if specified)
            unnorm_key = getattr(self.cfg, 'unnorm_key', None)
            if unnorm_key is None:
                unnorm_key = list(norm_stats.keys())[0]
            
            if unnorm_key in norm_stats and "action" in norm_stats[unnorm_key]:
                self.action_norm_stats = norm_stats[unnorm_key]["action"]
                logger.info(f"Action normalization stats loaded for key: {unnorm_key}")
                logger.info(f"  Action dim: {len(self.action_norm_stats.get('min', []))}")
            else:
                logger.warning(f"No action stats found for key: {unnorm_key}")
        else:
            logger.warning("No normalization statistics available - actions will NOT be unnormalized!")
        
        logger.info("Policy initialized with:")
        logger.info(f"  - VLA backbone frozen: {self.cfg.freeze_vla_backbone}")
        logger.info(f"  - Train proprio projector: {self.cfg.train_proprio_projector}")
        logger.info(f"  - Train action head: {self.cfg.train_action_head}")
        logger.info(f"  - Train value head: {self.cfg.train_value_head}")
        logger.info(f"  - Action unnormalization: {'enabled' if self.action_norm_stats else 'disabled'}")
    
    def _init_optimizer(self):
        """
        Initialize optimizer for trainable parameters.
        
        Only trains: Action Head + Value Head + Proprio Projector (if enabled) + log_std
        """
        trainable_params = self.policy.get_trainable_parameters()
        
        # Cache trainable params for gradient clipping (avoid repeated calls)
        self._trainable_params = trainable_params
        
        # Log total trainable parameters (only once during init)
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable parameters: {total_trainable:,}")
        
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.cfg.learning_rate,
        )
        
        logger.info(f"Optimizer initialized with {len(trainable_params)} parameter groups")
        logger.info(f"  Learning rate: {self.cfg.learning_rate}")
    
    def _log_config(self):
        """Log configuration to file."""
        config_path = self.run_dir / "config.json"
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in vars(self.cfg).items()}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_path}")
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout experience from the environment.
        
        Uses integrated Action Head + Value Head for efficient single-pass inference.
        
        Returns:
            Dictionary of rollout statistics
        """
        self.policy.eval()
        
        # Reset buffer
        self.buffer.reset()
        
        # Statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Reset environment and observation history
        obs = self.env.reset()
        self.obs_history.reset()
        
        # Wait for objects to stabilize
        for _ in range(self.cfg.num_steps_wait):
            obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
        
        # Extract initial observations
        front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
        self.obs_history.update(front_img, wrist_imgs, proprio_states)
        
        for step in range(self.cfg.num_steps_per_rollout):
            with torch.no_grad():
                # Get observations for each agent (with history)
                agent_obs = self.obs_history.get_all_agent_observations(include_history=True)
                
                # Prepare VLA inputs for each agent
                agent_inputs = []
                agent_proprios = []
                task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
                
                for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                    # Images: [front_t, wrist_t, front_{t-1}, wrist_{t-1}, ...]
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
                    
                    # Proprio tensor: (1, proprio_dim)
                    # Use bfloat16 to match proprio_projector dtype
                    proprio_tensor = torch.as_tensor(
                        agent_obs[agent_idx]['proprio'],
                        device=self.device,
                        dtype=torch.bfloat16,
                    ).unsqueeze(0)
                    agent_proprios.append(proprio_tensor)
                
                # Get actions AND values in ONE forward pass (efficient!)
                actions, log_probs, entropies, values = self.policy.get_actions_and_values(
                    agent_inputs=agent_inputs,
                    agent_proprios=agent_proprios,
                    deterministic=False,
                )
                
                # Use mean value across agents for team value
                # Convert bfloat16 to float32 before numpy (numpy doesn't support bfloat16)
                value = torch.stack(values).mean().float().cpu().numpy()
            
            # Execute first action from each agent's action chunk
            # actions[i]: (1, chunk_len, action_dim) - these are NORMALIZED actions in [-1, 1]
            # Convert bfloat16 to float32 before numpy
            action_0_normalized = actions[0][0, 0].float().cpu().numpy()  # (action_dim,)
            action_1_normalized = actions[1][0, 0].float().cpu().numpy()  # (action_dim,)
            
            # Unnormalize actions before sending to environment
            # VLA outputs are in [-1, 1] range, need to convert to actual action range
            if self.action_norm_stats is not None:
                action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats)
                action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats)
            else:
                # No unnormalization - use normalized actions directly
                action_0 = action_0_normalized
                action_1 = action_1_normalized
            
            # Concatenate to form full action for environment
            full_action = np.concatenate([action_0, action_1])
            
            # Step environment
            next_obs, reward, done, info = self.env.step(full_action.tolist())
            
            # Handle reward
            team_reward = self.reward_wrapper(reward)
            
            if self.reward_normalizer is not None:
                team_reward = self.reward_normalizer.normalize(
                    np.array([team_reward]),
                    np.array([float(done)]),
                )[0]
            
            # Store transition
            proprio_states_np = [
                agent_obs[i]['proprio_history']
                for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])
            ]
            actions_np = [
                actions[i][0].float().cpu().numpy()
                for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])
            ]
            log_probs_np = [
                log_probs[i][0].float().cpu().numpy()
                for i in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"])
            ]
            
            # Store images for PPO update
            # Extract front images with history: (1, T, H, W, C)
            front_images_list = []
            for t in range(self.cfg.history_length):
                front_images_list.append(agent_obs[0]['images'][t * 2])
            front_images_np = np.stack(front_images_list, axis=0)[np.newaxis, ...]
            
            # Extract wrist images per agent: List[(1, T, H, W, C)]
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
            
            # Update statistics
            current_episode_reward += reward
            current_episode_length += 1
            
            # Check for episode end - use env._check_success() directly
            # (robosuite doesn't include 'success' in info dict)
            success = self.env._check_success()
            
            if done or success or current_episode_length >= self.cfg.max_episode_steps:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_successes.append(float(success))
                
                self.episode_count += 1
                
                # Reset
                obs = self.env.reset()
                self.obs_history.reset()
                
                for _ in range(self.cfg.num_steps_wait):
                    obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
                
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
                
                current_episode_reward = 0
                current_episode_length = 0
            else:
                # Update observation history
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(next_obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
                obs = next_obs
            
            self.global_step += 1
        
        # Compute final value for GAE
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
            
            # Get final values
            last_values = self.policy.get_values(
                agent_inputs=agent_inputs,
                agent_proprios=agent_proprios,
            )
            last_value = torch.stack(last_values).mean().float().cpu().numpy()
        
        # Compute returns and advantages using GAE
        self.buffer.compute_returns_and_advantages(
            last_values=np.array([last_value]),
            last_dones=np.array([float(done)]),
        )
        
        # Return statistics
        stats = {
            "rollout/mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "rollout/mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "rollout/success_rate": np.mean(episode_successes) if episode_successes else 0,
            "rollout/num_episodes": len(episode_rewards),
        }
        
        return stats
    
    def update(self) -> Dict[str, float]:
        """
        Perform MAPPO update using collected rollouts.
        
        Full PPO Update:
        1. Re-evaluate stored actions through VLA to get new log_probs, values
        2. Compute importance sampling ratio: r = exp(new_log_prob - old_log_prob)
        3. Compute PPO clipped objective
        4. Compute value loss with new values
        5. Backward pass and optimize
        
        Returns:
            Dictionary of update statistics
        """
        self.policy.train()
        
        # Statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        # Calculate batch size
        batch_size = (
            self.cfg.num_steps_per_rollout * self.cfg.num_envs
        ) // self.cfg.num_minibatches
        
        task_descriptions = [self.task_desc_robot0, self.task_desc_robot1]
        
        # PPO update epochs
        for epoch in range(self.cfg.num_epochs):
            for batch in self.buffer.get(batch_size):
                # Get batch data
                batch_advantages = batch.advantages
                batch_returns = batch.returns
                batch_old_values = batch.old_values
                
                # Normalize advantages
                if self.cfg.normalize_advantages:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # Re-evaluate actions through VLA
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
                        # Full PPO: Re-evaluate through VLA
                        batch_new_log_probs = []
                        batch_entropies = []
                        batch_new_values = []
                        
                        # Process in sub-batches to fit in GPU memory
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
                                # Reconstruct images: [front_t, wrist_t, front_{t-1}, wrist_{t-1}]
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
                                
                                # Use agents list (works for both shared and non-shared policy)
                                log_prob, entropy, value = self.policy.agents[agent_idx].evaluate_actions_and_value(
                                    inputs=vla_input,
                                    actions=action_to_eval,
                                    proprio=proprio_tensor,
                                    use_proprio=self.cfg.use_proprio,
                                    use_film=self.cfg.use_film,
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
                        # Fallback if images not stored
                        new_log_probs = agent_old_log_probs
                        entropies = torch.zeros_like(agent_old_log_probs)
                        new_values = batch_old_values
                    
                    all_new_log_probs.append(new_log_probs)
                    all_entropies.append(entropies)
                    all_new_values.append(new_values)
                
                # Compute PPO loss for each agent
                all_policy_losses = []
                all_clip_fracs = []
                all_approx_kl = []
                
                for agent_idx in range(TWOARM_MAPPO_CONSTANTS["NUM_AGENTS"]):
                    agent_old_log_probs = batch.agent_old_log_probs[agent_idx]
                    new_log_probs = all_new_log_probs[agent_idx]
                    
                    # Importance sampling ratio
                    log_ratio = new_log_probs - agent_old_log_probs
                    ratio = torch.exp(log_ratio)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_epsilon,
                        1.0 + self.cfg.clip_epsilon
                    ) * batch_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    all_policy_losses.append(policy_loss)
                    
                    # Statistics
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        all_approx_kl.append(approx_kl)
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
                        all_clip_fracs.append(clip_frac)
                
                # Average across agents (convert to float32 for stable loss computation)
                policy_loss = torch.stack(all_policy_losses).mean().float()
                entropy = torch.stack(all_entropies).mean().float()
                new_values_mean = torch.stack(all_new_values).mean(dim=0).float()
                
                # Value loss
                if self.cfg.clip_value_loss:
                    values_clipped = batch_old_values + torch.clamp(
                        new_values_mean - batch_old_values,
                        -self.cfg.clip_epsilon,
                        self.cfg.clip_epsilon,
                    )
                    value_loss_unclipped = (new_values_mean - batch_returns) ** 2
                    value_loss_clipped = (values_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values_mean - batch_returns) ** 2).mean()
                
                # Total loss
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (use cached params to avoid repeated function calls)
                if self.cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self._trainable_params,
                        self.cfg.max_grad_norm,
                    )
                
                self.optimizer.step()
                
                # Record statistics
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
        
        return stats
    
    def evaluate(
        self, 
        num_episodes: int = 10, 
        save_video: bool = True,
        num_videos: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            save_video: Whether to save evaluation videos
            num_videos: Number of videos to save (first N episodes)
            
        Returns:
            Dictionary of evaluation statistics
        """
        self.policy.eval()
        
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        
        # Create video directory
        if save_video:
            video_dir = self.run_dir / "eval_videos"
            video_dir.mkdir(parents=True, exist_ok=True)
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            self.obs_history.reset()
            
            # Only save video for first N episodes
            should_save_video = save_video and (ep < num_videos)
            
            # Collect video frames
            replay_images = [] if should_save_video else None
            
            for _ in range(self.cfg.num_steps_wait):
                obs, _, _, _ = self.env.step(get_twoarm_dummy_action(self.cfg.model_family))
            
            front_img, wrist_imgs, proprio_states = extract_observations_from_env(obs)
            self.obs_history.update(front_img, wrist_imgs, proprio_states)
            
            episode_reward = 0
            episode_length = 0
            success = False
            
            for step in range(self.cfg.max_episode_steps):
                # Save video frame at the beginning of each step
                if should_save_video:
                    replay_images.append(get_twoarm_video_frame(obs))
                
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
                    
                    # Deterministic actions for evaluation
                    actions, _, _ = self.policy.get_actions(
                        agent_inputs=agent_inputs,
                        agent_proprios=agent_proprios,
                        deterministic=True,
                    )
                
                # Get normalized actions from policy (convert bfloat16 to float32)
                action_0_normalized = actions[0][0, 0].float().cpu().numpy()
                action_1_normalized = actions[1][0, 0].float().cpu().numpy()
                
                # Unnormalize actions before sending to environment
                if self.action_norm_stats is not None:
                    action_0 = unnormalize_action(action_0_normalized, self.action_norm_stats)
                    action_1 = unnormalize_action(action_1_normalized, self.action_norm_stats)
                else:
                    action_0 = action_0_normalized
                    action_1 = action_1_normalized
                
                full_action = np.concatenate([action_0, action_1])
                
                next_obs, reward, done, info = self.env.step(full_action.tolist())
                obs = next_obs  # Update obs for next video frame
                
                episode_reward += reward
                episode_length += 1
                
                # Check success using env._check_success() directly
                # (robosuite doesn't include 'success' in info dict)
                success = self.env._check_success()
                
                if done or success:
                    # Save final frame
                    if should_save_video:
                        replay_images.append(get_twoarm_video_frame(obs))
                    break
                
                front_img, wrist_imgs, proprio_states = extract_observations_from_env(next_obs)
                self.obs_history.update(front_img, wrist_imgs, proprio_states)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(float(success))
            
            # Save video for this episode
            if should_save_video and replay_images:
                # Save to run_dir/eval_videos with descriptive name
                import imageio
                video_filename = f"step_{self.global_step}_ep_{ep}_success_{success}.mp4"
                video_path = video_dir / video_filename
                
                video_writer = imageio.get_writer(str(video_path), fps=20)
                for frame in replay_images:
                    video_writer.append_data(frame)
                video_writer.close()
                
                logger.info(f"Saved eval video: {video_filename} ({len(replay_images)} frames)")
        
        stats = {
            "eval/mean_reward": np.mean(episode_rewards),
            "eval/std_reward": np.std(episode_rewards),
            "eval/success_rate": np.mean(episode_successes),
            "eval/mean_episode_length": np.mean(episode_lengths),
        }
        
        return stats
    
    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint."""
        checkpoint_dir = self.run_dir / f"checkpoint_{self.global_step}{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy
        torch.save(
            self.policy.state_dict(),
            checkpoint_dir / "policy.pt",
        )
        
        # Save optimizer and training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "best_success_rate": self.best_success_rate,
        }, checkpoint_dir / "training_state.pt")
        
        # Save config
        config_dict = {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(self.cfg).items()}
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting MAPPO training for {self.cfg.total_timesteps} timesteps")
        logger.info(f"Run directory: {self.run_dir}")
        
        num_updates = self.cfg.total_timesteps // (
            self.cfg.num_steps_per_rollout * self.cfg.num_envs
        )
        
        start_time = time.time()
        
        for update in tqdm.tqdm(range(num_updates), desc="Training"):
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # Perform PPO update
            update_stats = self.update()
            
            # Combine stats
            stats = {**rollout_stats, **update_stats}
            stats["train/global_step"] = self.global_step
            stats["train/episodes"] = self.episode_count
            stats["train/fps"] = self.global_step / (time.time() - start_time)
            
            # Log to tensorboard
            for key, value in stats.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
            
            # Log to wandb
            if self.cfg.use_wandb and self.global_step % self.cfg.wandb_log_freq == 0:
                wandb.log(stats, step=self.global_step)
            
            # Periodic logging
            if update % 10 == 0:
                logger.info(
                    f"Step {self.global_step} | "
                    f"Reward: {stats['rollout/mean_episode_reward']:.2f} | "
                    f"Success: {stats['rollout/success_rate']:.2%} | "
                    f"Policy Loss: {stats['update/policy_loss']:.4f} | "
                    f"Value Loss: {stats['update/value_loss']:.4f}"
                )
            
            # Evaluation
            if self.global_step % self.cfg.eval_freq == 0:
                eval_stats = self.evaluate(
                    num_episodes=self.cfg.num_eval_episodes,
                    save_video=self.cfg.save_eval_videos,
                    num_videos=self.cfg.num_eval_videos,
                )
                
                for key, value in eval_stats.items():
                    self.tb_writer.add_scalar(key, value, self.global_step)
                
                if self.cfg.use_wandb:
                    wandb.log(eval_stats, step=self.global_step)
                
                logger.info(
                    f"Evaluation | "
                    f"Reward: {eval_stats['eval/mean_reward']:.2f} | "
                    f"Success: {eval_stats['eval/success_rate']:.2%}"
                )
                
                # Save best model
                if eval_stats['eval/success_rate'] > self.best_success_rate:
                    self.best_success_rate = eval_stats['eval/success_rate']
                    self.save_checkpoint(suffix="_best")
            
            # Periodic checkpoint
            if self.global_step % self.cfg.save_freq == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint(suffix="_final")
        
        logger.info("Training completed!")
        logger.info(f"Best success rate: {self.best_success_rate:.2%}")
        
        # Cleanup
        self.tb_writer.close()
        if self.cfg.use_wandb:
            wandb.finish()


@draccus.wrap()
def main(cfg: MAPPOConfig):
    """Main entry point for MAPPO training."""
    trainer = MAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
