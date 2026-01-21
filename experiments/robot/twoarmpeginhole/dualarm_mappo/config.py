"""
Configuration and constants for Dual-Arm MAPPO training on TwoArmPegInHole environment.

This uses a bimanual model (ALOHA-style) where:
- Total action dim is 14 (7 per arm)
- Total proprio dim is 14 (7 per arm)
- Agent 0 uses the first 7-dim action, Agent 1 uses the last 7-dim action
- Each agent sees agentview + own wrist view, with the other agent's wrist padded
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


# Dual-Arm MAPPO specific constants for bimanual model
# Note: Environment action is 12-dim (6 per arm, no gripper), but model outputs 14-dim (7 per arm).
#       We take the first 6-dim of each 7-dim action for the environment.
# Note: Environment proprio (joint_pos) is 14-dim (7 per arm), matching model input.
DUALARM_MAPPO_CONSTANTS = {
    "NUM_AGENTS": 2,                          # Two agents (one per arm)
    "NUM_ACTIONS_CHUNK": 25,                   # Reduced chunk size for RL
    "MODEL_ACTION_DIM": 14,                   # 14-dim total action from bimanual model
    "AGENT_ACTION_DIM": 7,                    # 7-dim per agent (first 7 for agent0, last 7 for agent1)
    "ENV_ACTION_DIM": 6,                      # 6-dim per agent for TwoArm env (no gripper)
    "MODEL_PROPRIO_DIM": 14,                  # 14-dim total proprio for bimanual model
    "AGENT_PROPRIO_DIM": 7,                   # 7-dim per agent (7 joint positions, Panda robot has 7 DoF)
    "HISTORY_LENGTH": 2,                      # Number of historical frames
    "NUM_IMAGES_PER_AGENT": 3,                # agentview + left_wrist + right_wrist (one padded)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,  # ALOHA uses BOUNDS
}


@dataclass
class DualArmMAPPOConfig:
    """Configuration for Dual-Arm MAPPO training with bimanual model."""
    
    # fmt: off
    
    #################################################################################################################
    # Distributed Training
    #################################################################################################################
    # These are usually set automatically by torchrun
    local_rank: int = int(os.getenv("LOCAL_RANK", "0"))
    world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    rank: int = int(os.getenv("RANK", "0"))
    dist_backend: str = "nccl"
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    # Fine-tuned ALOHA bimanual checkpoint
    pretrained_checkpoint: str = "/path/to/your/aloha/bimanual/checkpoint"
    
    # VLA Architecture
    use_l1_regression: bool = True                    # Use continuous action head with L1
    use_diffusion: bool = False                       # Use diffusion-based action head
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 10
    use_film: bool = True                             # ALOHA typically uses FiLM
    use_proprio: bool = True
    
    # Image inputs (bimanual model uses 3 images: agentview + 2 wrist views)
    # With history: 3 * history_length
    num_images_in_input: int = 6                      # 3 images * 2 (history)
    history_length: int = 2                           # Number of frames in history (current + previous)
    
    # Action space (bimanual model outputs 14-dim action)
    num_actions_chunk: int = 2                        # Reduced chunk size for RL
    model_action_dim: int = 14                        # Full bimanual action dimension
    agent_action_dim: int = 7                         # Per-agent action dimension
    
    # Proprio dimensions
    model_proprio_dim: int = 14                       # Full bimanual proprio dimension  
    agent_proprio_dim: int = 7                        # Per-agent proprio dimension
    
    # LoRA
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    center_crop: bool = True
    
    # Normalization key for action/proprio (from VLA's norm_stats)
    # If None, uses the first available key; otherwise uses the specified key
    unnorm_key: Optional[str] = "aloha_velroc_dataset"
    
    # Gradient checkpointing to reduce GPU memory usage (trades compute for memory)
    use_gradient_checkpointing: bool = True
    
    # Freeze options for RL training
    freeze_vla_backbone: bool = True              # Freeze VLA backbone (only train heads)
    train_proprio_projector: bool = True          # Whether to train proprio projector
    train_action_head: bool = True                # Whether to train action head MLP
    train_value_head: bool = True                 # Whether to train value head MLP
    
    #################################################################################################################
    # MAPPO Hyperparameters
    #################################################################################################################
    
    # PPO Core
    gamma: float = 0.99                               # Discount factor
    gae_lambda: float = 0.95                          # GAE lambda for advantage estimation
    clip_epsilon: float = 0.2                         # PPO clipping parameter
    clip_value_loss: bool = True                      # Whether to clip value loss
    value_loss_coef: float = 0.5                      # Value loss coefficient
    entropy_coef: float = 0.01                        # Entropy bonus coefficient
    max_grad_norm: float = 0.5                        # Gradient clipping norm
    
    # Training
    actor_lr: float = 5e-5                            # Conservative lr for VLA-based policy (actor)
    critic_lr: float = 1e-4                           # Slightly higher lr for value function (critic)
    
    num_envs: int = 1                                 # Number of parallel environments PER GPU
    num_steps_per_rollout: int = 256                  # Steps per rollout before update
    num_minibatches: int = 4                          # Number of minibatches per update
    num_epochs: int = 4                               # Number of epochs per update
    total_timesteps: int = 1_000_000                  # Total training timesteps
    
    # Normalization
    normalize_advantages: bool = True                 # Normalize advantages
    normalize_rewards: bool = True                    # Normalize rewards
    
    # Shared Policy
    share_policy: bool = True                         # Share policy weights between agents
    
    #################################################################################################################
    # Environment-specific parameters
    #################################################################################################################
    robot1: str = "Panda"
    robot2: str = "Panda"
    controller: str = "BASIC"
    env_configuration: str = "opposed"
    reward_shaping: bool = True                       # Use dense reward shaping for RL
    
    # Dense reward component weights (only used when reward_shaping=True)
    reaching_weight: float = 0.4
    perpendicular_weight: float = 1.2
    parallel_weight: float = 0.6
    alignment_weight: float = 1.2
    
    env_img_res: int = 256
    max_episode_steps: int = 300
    num_steps_wait: int = 10
    
    # Early termination
    max_peg_hole_distance: float = 1.4
    
    # Task description
    instruction_mode: str = "shared"
    custom_instructions: Optional[str] = None
    
    #################################################################################################################
    # Centralized Value Network
    #################################################################################################################
    value_hidden_dim: int = 512                       # Hidden dimension for value network
    value_num_layers: int = 3                         # Number of layers in value network
    use_global_state: bool = True                     # Use global state for value network
    
    #################################################################################################################
    # Logging and Checkpointing
    #################################################################################################################
    run_root_dir: Path = Path("runs/dualarm_mappo")
    run_id_note: Optional[str] = None
    resume_checkpoint: Optional[str] = None           # Path to checkpoint directory to resume from
    
    use_wandb: bool = True
    wandb_entity: str = "acpo"
    wandb_project: str = "dualarm-mappo-twoarm"
    wandb_log_freq: int = 10
    wandb_mode: str = "online"
    
    save_freq: int = 10_000                           # Checkpoint save frequency
    eval_freq: int = 5_000                            # Evaluation frequency
    num_eval_episodes: int = 10                       # Number of evaluation episodes
    save_eval_videos: bool = True                     # Save videos during evaluation
    num_eval_videos: int = 3                          # Number of evaluation videos to save
    
    log_dir: str = "./experiments/logs/dualarm_mappo"
    video_dir: str = "./rollouts/dualarm_mappo"
    
    seed: int = 42
    
    # fmt: on
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        assert not (self.use_l1_regression and self.use_diffusion), \
            "Cannot use both L1 regression and diffusion!"
        
        assert self.history_length >= 1, "History length must be at least 1 (current frame)"
        
        # Calculate expected number of images (3 images * history)
        expected_images = 3 * self.history_length  # (agentview + left_wrist + right_wrist) * history
        if self.num_images_in_input != expected_images:
            if self.rank == 0:
                print(f"Warning: Adjusting num_images_in_input from {self.num_images_in_input} "
                      f"to {expected_images} based on history_length={self.history_length}")
            self.num_images_in_input = expected_images
        
        # Validate action dimensions
        assert self.model_action_dim == 14, "Bimanual model must have 14-dim action"
        assert self.agent_action_dim == 7, "Each agent action must be 7-dim"
        
        # Create directories (only rank 0 needs to ensure existence, but safe for all)
        if self.rank == 0:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.video_dir).mkdir(parents=True, exist_ok=True)
            self.run_root_dir.mkdir(parents=True, exist_ok=True)


def get_config_from_args(args=None):
    """Create DualArmMAPPOConfig from command line arguments."""
    import draccus
    
    @draccus.wrap()
    def _create_config(cfg: DualArmMAPPOConfig) -> DualArmMAPPOConfig:
        return cfg
    
    return _create_config()
