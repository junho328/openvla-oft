"""
run_twoarm_eval.py

Evaluates a trained policy in the TwoArmPegInHole robosuite task.
"""

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import wandb

sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from experiments.robot.twoarmpeginhole.twoarm_utils import (
    get_twoarm_dummy_action,
    get_twoarm_env,
    get_twoarm_image,
    get_twoarm_video_frame,
    get_twoarm_wrist_image,
    get_twoarm_task_descriptions,
    quat2axisangle,
    save_rollout_video,
    RolloutVideoWriter,
    INSTRUCTION_MODE_SHARED,
    INSTRUCTION_MODE_SPLIT,
    INSTRUCTION_MODE_SPLIT_DETAILED,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

TASK_MAX_STEPS = 300

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # TwoArmPegInHole environment-specific parameters
    #################################################################################################################
    robot1: str = "Panda"                             # Robot type for arm 0
    robot2: str = "Panda"                             # Robot type for arm 1
    controller: str = "BASIC"                         # Controller type
    env_configuration: str = "opposed"                # Robot configuration in the environment
    reward_shaping: bool = False                      # Whether to use dense reward shaping
    num_steps_wait: int = 10                          # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                     # Number of rollouts per task
    env_img_res: int = 256                            # Resolution for environment images (not policy input resolution)

    # Instruction mode: "shared" (same instruction for both), "split" (different instructions), "split_detailed"
    instruction_mode: str = "shared"
    # Custom instructions (optional) - if provided, overrides instruction_mode
    # Format: "robot0_instruction|robot1_instruction" (pipe-separated)
    custom_instructions: Optional[str] = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                 # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"         # Local directory for eval logs

    use_wandb: bool = False                           # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"           # Name of WandB entity
    wandb_project: str = "your-wandb-project"         # Name of WandB project

    seed: int = 7                                     # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.unnorm_key or "libero_spatial"

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key




def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-twoarm_peginhole-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def prepare_robot_observation(obs, resize_size, robot_index: int):
    """Prepare per-robot observation for policy input."""
    img = get_twoarm_image(obs)
    wrist_img0, wrist_img1 = get_twoarm_wrist_image(obs)
    wrist_img = wrist_img0 if robot_index == 0 else wrist_img1

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    eef_pos = obs[f"robot{robot_index}_eef_pos"]
    eef_quat = obs[f"robot{robot_index}_eef_quat"]

    state = np.concatenate((eef_pos, quat2axisangle(eef_quat)))
    if state.shape[0] < 8:
        pad = np.zeros(8 - state.shape[0], dtype=state.dtype)
        state = np.concatenate((state, pad))

    observation = {"full_image": img_resized, "wrist_image": wrist_img_resized, "state": state}

    return observation, img


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description_robot0: str,
    task_description_robot1: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run a single episode in the environment.
    
    Args:
        task_description_robot0: Task instruction for robot 0 (peg robot)
        task_description_robot1: Task instruction for robot 1 (hole robot)
    """
    # Reset environment
    obs = env.reset()

    # Initialize action queues
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk."
        )
    action_queue_0 = deque(maxlen=cfg.num_open_loop_steps)
    action_queue_1 = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []

    # Run episode
    success = False
    try:
        while t < TASK_MAX_STEPS + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_twoarm_dummy_action(cfg.model_family))
                t += 1
                continue

            # Save frame for video
            replay_images.append(get_twoarm_video_frame(obs))

            # Prepare per-robot observations
            obs_robot0, _ = prepare_robot_observation(obs, resize_size, robot_index=0)
            obs_robot1, _ = prepare_robot_observation(obs, resize_size, robot_index=1)

            # If action queue is empty, requery model for each robot with their respective instructions
            if len(action_queue_0) == 0:
                actions_0 = get_action(
                    cfg,
                    model,
                    obs_robot0,
                    task_description_robot0,  # Robot 0 specific instruction
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue_0.extend(actions_0)

            if len(action_queue_1) == 0:
                actions_1 = get_action(
                    cfg,
                    model,
                    obs_robot1,
                    task_description_robot1,  # Robot 1 specific instruction
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue_1.extend(actions_1)

            # Get action from queues
            action_0 = np.asarray(action_queue_0.popleft())
            action_1 = np.asarray(action_queue_1.popleft())

            # Drop gripper or extra dims (use first 6 dims per arm)
            if action_0.shape[-1] > 6:
                action_0 = action_0[..., :6]
            if action_1.shape[-1] > 6:
                action_1 = action_1[..., :6]
            if action_0.shape[-1] < 6:
                action_0 = np.pad(action_0, (0, 6 - action_0.shape[-1]))
            if action_1.shape[-1] < 6:
                action_1 = np.pad(action_1, (0, 6 - action_1.shape[-1]))

            # Concatenate two 6D actions into 12D action
            action = np.concatenate((action_0, action_1), axis=-1)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if success or done:
                # Save final frame
                replay_images.append(get_twoarm_video_frame(obs))
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Initialize environment
    env, _ = get_twoarm_env(
        cfg.model_family,
        resolution=cfg.env_img_res,
        robot1=cfg.robot1,
        robot2=cfg.robot2,
        controller=cfg.controller,
        env_configuration=cfg.env_configuration,
        reward_shaping=cfg.reward_shaping,
    )

    # Get task descriptions based on instruction mode
    custom_descriptions = None
    if cfg.custom_instructions is not None:
        # Parse custom instructions: "robot0_instruction|robot1_instruction"
        parts = cfg.custom_instructions.split("|")
        if len(parts) == 2:
            custom_descriptions = {"robot0": parts[0].strip(), "robot1": parts[1].strip()}
        else:
            logger.warning(f"Invalid custom_instructions format: {cfg.custom_instructions}. "
                          "Expected 'robot0_instruction|robot1_instruction'. Using instruction_mode instead.")
    
    task_desc_robot0, task_desc_robot1, combined_description = get_twoarm_task_descriptions(
        mode=cfg.instruction_mode,
        custom_descriptions=custom_descriptions,
    )
    
    # Log instruction setup
    log_message(f"\nInstruction mode: {cfg.instruction_mode}", log_file)
    log_message(f"Robot 0 instruction: {task_desc_robot0}", log_file)
    log_message(f"Robot 1 instruction: {task_desc_robot1}", log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for _ in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {combined_description}", log_file)

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode with separate instructions for each robot
        success, replay_images = run_episode(
            cfg,
            env,
            task_desc_robot0,
            task_desc_robot1,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes + 1, success=success, task_description=combined_description, log_file=log_file
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    return total_episodes, total_successes


@draccus.wrap()
def eval_twoarm(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on TwoArmPegInHole."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_episodes, total_successes = run_task(
        cfg,
        model,
        resize_size,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        total_episodes,
        total_successes,
        log_file,
    )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_twoarm()
