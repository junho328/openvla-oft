"""
Configuration and constants for ACPPO training on TwoArmPegInHole environment.

ACPPO (Agent-Chained Proximal Policy Optimization) extends MAPPO by:
1. Second agent receives estimated action distribution from first agent
2. Microstep-based advantage calculation for better credit assignment
3. Shared value/action heads across agents
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from enum import Enum
import os


class NormalizationType(str, Enum):
    """Normalization types for actions and proprioceptive states."""
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


# TwoArmPegInHole ACPPO specific constants
TWOARM_ACPPO_CONSTANTS = {
    "NUM_AGENTS": 2,                          # Two agents (one per arm)
    "NUM_ACTIONS_CHUNK": 1,                   # Action chunk size
    "ACTION_DIM": 6,                          # 6-DoF per arm (no gripper)
    "PROPRIO_DIM": 8,                         # EEF pos (3) + axis-angle (3) + padding (2)
    "HISTORY_LENGTH": 2,                      # Number of historical frames
    "NUM_IMAGES_PER_AGENT": 2,                # Front image + wrist image
    "TOTAL_IMAGES_WITH_HISTORY": 4,           # (front + wrist) * history_length
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
    # Action distribution dimensions for chaining
    "ACTION_DIST_DIM": 12,                    # 6 action * 1 chunk * 2 (mu + sigma) = 12
}


@dataclass
class ACPPOConfig:
    """Configuration for ACPPO training."""
    
    # fmt: off
    
    #################################################################################################################
    # Distributed Training
    #################################################################################################################
    local_rank: int = int(os.getenv("LOCAL_RANK", "0"))
    world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    rank: int = int(os.getenv("RANK", "0"))
    dist_backend: str = "nccl"
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    
    # VLA Architecture
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 10
    use_film: bool = False
    use_proprio: bool = True
    
    # Image inputs (with history)
    num_images_in_input: int = 4                      # (front + wrist) * 2 (current + history)
    history_length: int = 2                           # Number of frames in history (current + previous)
    
    # Action space
    num_actions_chunk: int = 1                        # Action chunk size
    action_dim: int = 6                               # 6-DoF per arm (no gripper)
    
    # LoRA
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    center_crop: bool = True
    
    # Freeze options for RL training
    freeze_vla_backbone: bool = True
    train_proprio_projector: bool = True
    train_action_head: bool = True
    train_value_head: bool = True
    train_action_dist_projector: bool = True  # Whether to train action distribution projector for agent 1
    
    #################################################################################################################
    # ACPPO Specific Parameters
    #################################################################################################################
    
    # Action distribution for chaining
    action_dist_dim: int = 12                         # 6 * 1 * 2 = 12 (mu + sigma) - auto-adjusted in __post_init__ based on num_actions_chunk
    use_action_dist_input: bool = True                # Whether second agent uses estimated action dist
    detach_action_dist_grad: bool = True              # No gradient through action dist estimation
    
    # Extended proprio dimension for second agent (proprio + action_dist)
    # Agent 0: proprio_dim = 8
    # Agent 1: proprio_dim = 8 + 12 = 20 (proprio + estimated action dist)
    proprio_dim_agent0: int = 8
    proprio_dim_agent1: int = 20                      # 8 + 12 (auto-adjusted in __post_init__)
    
    #################################################################################################################
    # ACPPO/PPO Hyperparameters
    #################################################################################################################
    
    # PPO Core
    gamma: float = 0.99                               # Discount factor (γ)
    gamma_prime: float = 0.99                         # Discount factor for microsteps (γ')
    gae_lambda: float = 0.95                          # GAE lambda for advantage estimation
    lambda_prime: float = 0.95                        # Lambda for microstep advantage (λ')
    clip_epsilon: float = 0.2                         # PPO clipping parameter
    clip_value_loss: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    # Note: learning_rate is deprecated. Use actor_lr and critic_lr instead.
    learning_rate: float = 3e-4  # Kept for backward compatibility (not used)
    actor_lr: float = 5e-5       # Conservative lr for VLA-based policy (actor)
    critic_lr: float = 1e-4      # Slightly higher lr for value function (critic)
    
    num_envs: int = 1                                 # Number of parallel environments PER GPU
    num_steps_per_rollout: int = 256                  # Steps per rollout before update
    num_minibatches: int = 4
    num_epochs: int = 4
    total_timesteps: int = 1_000_000
    
    # Normalization
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    
    # Shared Policy (always True for ACPPO - agents share components)
    share_policy: bool = True
    
    #################################################################################################################
    # Environment-specific parameters
    #################################################################################################################
    robot1: str = "Panda"
    robot2: str = "Panda"
    controller: str = "BASIC"
    env_configuration: str = "opposed"
    reward_shaping: bool = True
    
    # Dense reward component weights
    reaching_weight: float = 0.4                      # Weight for distance-based reward (reduced to prevent collision)
    perpendicular_weight: float = 1.2                 # Weight for perpendicular alignment reward
    parallel_weight: float = 0.6                     # Weight for parallel (depth) positioning reward
    alignment_weight: float = 1.2                     # Weight for angular alignment reward (increased for better alignment)
    
    env_img_res: int = 256
    max_episode_steps: int = 300
    num_steps_wait: int = 10
    
    # Early termination
    max_peg_hole_distance: float = 50  # Early terminate if peg-hole distance exceeds this (in meters)
                                         # Note: Initial distance in "opposed" config is ~0.18m (18cm)
                                         # Setting to 0.4m (~2.2x initial) allows tolerance before early termination
    
    # Task description
    instruction_mode: str = "shared"
    custom_instructions: Optional[str] = None
    
    #################################################################################################################
    # Per-Agent Value Network (for ACPPO microstep advantage)
    #################################################################################################################
    value_hidden_dim: int = 512
    value_num_layers: int = 3
    use_global_state: bool = True
    
    # Per-agent value functions for ACPPO
    # V^(0)(s_t) for agent 0
    # V^(1)([s_t, b_t^(1)]) for agent 1, where b_t^(1) includes action dist from agent 0
    use_per_agent_value: bool = True
    
    # GAE computation mode:
    # - "acppo_microstep": Original ACPPO with microstep TD residuals (may cause value collapse)
    # - "shared_reward": Standard GAE where both agents receive the same reward
    # Recommendation: Use "shared_reward" if value_loss converges to near-zero quickly
    gae_mode: str = "acppo_microstep"
    
    #################################################################################################################
    # Logging and Checkpointing
    #################################################################################################################
    run_root_dir: Path = Path("runs/acppo")
    run_id_note: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    
    use_wandb: bool = True
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "acppo-twoarm"
    wandb_log_freq: int = 10
    wandb_mode: str = "online"
    
    save_freq: int = 10_000
    eval_freq: int = 5_000
    num_eval_episodes: int = 10
    save_eval_videos: bool = True
    num_eval_videos: int = 3
    
    log_dir: str = "./experiments/logs/acppo"
    video_dir: str = "./rollouts/acppo"
    
    seed: int = 42
    
    # fmt: on
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        assert not (self.use_l1_regression and self.use_diffusion), \
            "Cannot use both L1 regression and diffusion!"
        
        assert self.history_length >= 1, "History length must be at least 1 (current frame)"
        
        # Calculate expected number of images
        expected_images = 2 * self.history_length  # (front + wrist) * history
        if self.num_images_in_input != expected_images:
            if self.rank == 0:
                print(f"Warning: Adjusting num_images_in_input from {self.num_images_in_input} "
                      f"to {expected_images} based on history_length={self.history_length}")
            self.num_images_in_input = expected_images
        
        # Validate action distribution dimension
        expected_action_dist_dim = self.action_dim * self.num_actions_chunk * 2  # mu + sigma
        if self.action_dist_dim != expected_action_dist_dim:
            if self.rank == 0:
                print(f"Warning: Adjusting action_dist_dim from {self.action_dist_dim} "
                      f"to {expected_action_dist_dim}")
            self.action_dist_dim = expected_action_dist_dim
        
        # Update agent 1's proprio dim
        self.proprio_dim_agent1 = self.proprio_dim_agent0 + self.action_dist_dim
        
        # Create directories (only rank 0)
        if self.rank == 0:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.video_dir).mkdir(parents=True, exist_ok=True)
            self.run_root_dir.mkdir(parents=True, exist_ok=True)


def get_config_from_args(args=None):
    """Create ACPPOConfig from command line arguments."""
    import draccus
    
    @draccus.wrap()
    def _create_config(cfg: ACPPOConfig) -> ACPPOConfig:
        return cfg
    
    return _create_config()
