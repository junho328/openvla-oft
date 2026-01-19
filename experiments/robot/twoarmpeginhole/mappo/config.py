"""
Configuration and constants for MAPPO training on TwoArmPegInHole environment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from enum import Enum


class NormalizationType(str, Enum):
    """Normalization types for actions and proprioceptive states."""
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


# TwoArmPegInHole MAPPO specific constants
TWOARM_MAPPO_CONSTANTS = {
    "NUM_AGENTS": 2,                          # Two agents (one per arm)
    "NUM_ACTIONS_CHUNK": 2,                   # Reduced chunk size for RL
    "ACTION_DIM": 6,                          # 6-DoF per arm (no gripper)
    "PROPRIO_DIM": 8,                         # EEF pos (3) + axis-angle (3) + padding (2)
    "HISTORY_LENGTH": 2,                      # Number of historical frames
    "NUM_IMAGES_PER_AGENT": 2,                # Front image + wrist image
    "TOTAL_IMAGES_WITH_HISTORY": 4,           # (front + wrist) * history_length
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training."""
    
    # fmt: off
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    # Fine-tuned checkpoint (recommended) - has trained proprio_projector and action_head
    # Options: moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}
    # Or base VLA: openvla/openvla-7b (will initialize heads from scratch)
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    
    # VLA Architecture
    use_l1_regression: bool = True                    # Use continuous action head with L1
    use_diffusion: bool = False                       # Use diffusion-based action head
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 10
    use_film: bool = False
    use_proprio: bool = True
    
    # Image inputs (with history)
    num_images_in_input: int = 4                      # (front + wrist) * 2 (current + history)
    history_length: int = 2                           # Number of frames in history (current + previous)
    
    # Action space
    num_actions_chunk: int = 2                        # Reduced chunk size for RL
    action_dim: int = 6                               # 6-DoF per arm (no gripper)
    
    # LoRA
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    center_crop: bool = True
    
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
    learning_rate: float = 3e-4                       # Learning rate for both actor and critic
    actor_lr: float = 1e-4                            # Actor-specific learning rate
    critic_lr: float = 5e-4                           # Critic-specific learning rate
    
    num_envs: int = 1                                 # Number of parallel environments
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
    env_img_res: int = 256
    max_episode_steps: int = 300
    num_steps_wait: int = 10
    
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
    run_root_dir: Path = Path("runs/mappo")
    run_id_note: Optional[str] = None
    
    use_wandb: bool = True
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "mappo-twoarm"
    wandb_log_freq: int = 10
    
    save_freq: int = 10_000                           # Checkpoint save frequency
    eval_freq: int = 5_000                            # Evaluation frequency
    num_eval_episodes: int = 10                       # Number of evaluation episodes
    save_eval_videos: bool = True                     # Save videos during evaluation
    num_eval_videos: int = 3                          # Number of evaluation videos to save (subset of eval episodes)
    
    log_dir: str = "./experiments/logs/mappo"
    video_dir: str = "./rollouts/mappo"
    
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
            print(f"Warning: Adjusting num_images_in_input from {self.num_images_in_input} "
                  f"to {expected_images} based on history_length={self.history_length}")
            self.num_images_in_input = expected_images
        
        # Create directories
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        self.run_root_dir.mkdir(parents=True, exist_ok=True)


def get_config_from_args(args=None):
    """Create MAPPOConfig from command line arguments."""
    import draccus
    
    @draccus.wrap()
    def _create_config(cfg: MAPPOConfig) -> MAPPOConfig:
        return cfg
    
    return _create_config()
