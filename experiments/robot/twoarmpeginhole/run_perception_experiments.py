"""
run_perception_experiments.py

Runs two experiments to test perception confusion hypothesis:
1. Baseline: Two robots with individual prompts (no masking)
2. Masked: Same setup but with robot body masking enabled

Robot A (robot0): Carries the peg - "pegger"
Robot B (robot1): Carries the square with hole - "receiver"
"""

import os
import sys

# CRITICAL: Force LIBERO constants BEFORE any prismatic imports
# The model was trained on LIBERO, so we need LIBERO dimensions (ACTION_DIM=7)
# This must be done before importing prismatic.vla.constants
sys.argv = [arg for arg in sys.argv if 'twoarm' not in arg.lower()]
sys.argv.insert(0, 'libero_eval')  # Trick the auto-detection

# Set headless rendering BEFORE importing anything else
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import imageio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import torch

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.twoarmpeginhole.perception_masking import (
    mask_other_robot,
    create_comparison_frame,
    SEGMENTATION_IDS,
)
from experiments.robot.twoarmpeginhole.twoarm_utils import quat2axisangle

# Force LIBERO constants since we're using a LIBERO-trained model
# The model expects 7D actions and 8-action chunks
LIBERO_NUM_ACTIONS_CHUNK = 8
LIBERO_ACTION_DIM = 7
LIBERO_PROPRIO_DIM = 8


# Experiment configuration
@dataclass
class ExperimentConfig:
    # Model settings
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    unnorm_key: str = "libero_spatial_no_noops"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Task prompts
    robot0_prompt: str = "insert the peg into the hole"
    robot1_prompt: str = "align the hole to receive the peg"
    
    # Execution settings - use LIBERO settings since model was trained on LIBERO
    num_open_loop_steps: int = LIBERO_NUM_ACTIONS_CHUNK
    max_steps: int = 400
    num_episodes: int = 3
    
    # Environment settings
    env_img_res: int = 256
    
    # Output settings
    output_dir: str = "./experiment_outputs"
    save_video: bool = True
    video_fps: int = 20


def create_environment(config: ExperimentConfig, enable_segmentation: bool = False):
    """Create the TwoArmPegInHole environment."""
    controller_config = load_composite_controller_config(controller='BASIC')
    body_parts = controller_config.get('body_parts', {})
    if 'right' in body_parts:
        controller_config['body_parts'] = {'right': body_parts['right']}
    elif 'left' in body_parts:
        controller_config['body_parts'] = {'left': body_parts['left']}
    
    camera_names = ['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand']
    
    env_kwargs = dict(
        env_name='TwoArmPegInHole',
        robots=['Panda', 'Panda'],
        controller_configs=controller_config,
        env_configuration='opposed',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=config.env_img_res,
        camera_widths=config.env_img_res,
    )
    
    if enable_segmentation:
        # Use element-level segmentation for complete robot masking
        # This provides per-geom IDs including robot bases/pedestals
        env_kwargs['camera_segmentations'] = 'element'
    
    env = suite.make(**env_kwargs)
    return env


def load_model(config: ExperimentConfig):
    """Load the VLA model and components."""
    print(f"Loading model: {config.pretrained_checkpoint}")
    
    # Verify we're using LIBERO constants (should be set by sys.argv manipulation at top of file)
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    print(f"Verified constants: ACTION_DIM={ACTION_DIM}, NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, PROPRIO_DIM={PROPRIO_DIM}")
    
    if ACTION_DIM != 7:
        raise RuntimeError(f"Expected LIBERO ACTION_DIM=7, got {ACTION_DIM}. Check sys.argv manipulation.")
    
    # Create a simple config object for the utilities
    class ModelConfig:
        pass
    
    cfg = ModelConfig()
    cfg.pretrained_checkpoint = config.pretrained_checkpoint
    cfg.use_l1_regression = config.use_l1_regression
    cfg.use_diffusion = config.use_diffusion
    cfg.use_film = config.use_film
    cfg.num_images_in_input = config.num_images_in_input
    cfg.use_proprio = config.use_proprio
    cfg.center_crop = config.center_crop
    cfg.load_in_8bit = config.load_in_8bit
    cfg.load_in_4bit = config.load_in_4bit
    cfg.unnorm_key = config.unnorm_key
    cfg.lora_rank = 32
    cfg.num_diffusion_steps_train = 50
    cfg.num_diffusion_steps_inference = 50
    cfg.num_open_loop_steps = config.num_open_loop_steps
    
    # Load VLA
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    
    # Load action head (will use LIBERO dimensions from constants)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    
    # Load proprio projector (will use LIBERO dimensions from constants)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
    
    return vla, processor, action_head, proprio_projector, cfg


def prepare_observation(obs, resize_size: int, robot_index: int, apply_mask: bool = False, 
                       segmentation: Optional[np.ndarray] = None):
    """Prepare observation for a specific robot."""
    # Get main image
    img = obs['agentview_image'][::-1].copy()
    
    # Get wrist image for this robot
    wrist_key = f'robot{robot_index}_eye_in_hand_image'
    wrist_img = obs[wrist_key][::-1].copy()
    
    # Apply masking if enabled (using element-level segmentation for complete coverage)
    if apply_mask and segmentation is not None:
        seg = segmentation[::-1].copy()
        viewer_robot = f'robot{robot_index}'
        img = mask_other_robot(img, seg, viewer_robot, fill_method='inpaint', use_element_segmentation=True)
        
        # Also mask wrist camera if segmentation available (element-level)
        wrist_seg_key = f'robot{robot_index}_eye_in_hand_segmentation_element'
        if wrist_seg_key in obs:
            wrist_seg = obs[wrist_seg_key][::-1].copy()
            wrist_img = mask_other_robot(wrist_img, wrist_seg, viewer_robot, fill_method='inpaint', use_element_segmentation=True)
    
    # Ensure uint8 format
    for arr in [img, wrist_img]:
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    img = img.astype(np.uint8) if img.dtype != np.uint8 else img
    wrist_img = wrist_img.astype(np.uint8) if wrist_img.dtype != np.uint8 else wrist_img
    
    # Resize images
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    
    # Get robot state
    eef_pos = obs[f'robot{robot_index}_eef_pos']
    eef_quat = obs[f'robot{robot_index}_eef_quat']
    state = np.concatenate((eef_pos, quat2axisangle(eef_quat)))
    
    # Pad state to 8 dimensions if needed
    if state.shape[0] < 8:
        pad = np.zeros(8 - state.shape[0], dtype=state.dtype)
        state = np.concatenate((state, pad))
    
    return {
        'full_image': img_resized,
        'wrist_image': wrist_img_resized,
        'state': state,
    }


def run_episode(env, vla, processor, action_head, proprio_projector, cfg, config: ExperimentConfig,
                use_masking: bool = False) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
    """Run a single episode."""
    obs = env.reset()
    
    # Initialize action queues for both robots
    action_queue_0 = deque(maxlen=config.num_open_loop_steps)
    action_queue_1 = deque(maxlen=config.num_open_loop_steps)
    
    frames = []  # For video
    masked_frames = []  # Comparison frames showing masking effect
    
    success = False
    
    for t in range(config.max_steps):
        # Get segmentation if using masking (element-level for complete robot coverage)
        segmentation = None
        if use_masking and 'agentview_segmentation_element' in obs:
            segmentation = obs['agentview_segmentation_element']
        
        # Save frame for video
        frame = obs['agentview_image'][::-1].copy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        frames.append(frame)
        
        # If masking, create comparison frame
        if use_masking and segmentation is not None:
            seg = segmentation[::-1].copy()
            masked_r0 = mask_other_robot(frame, seg, 'robot0', fill_method='inpaint')
            masked_r1 = mask_other_robot(frame, seg, 'robot1', fill_method='inpaint')
            comparison = create_comparison_frame(frame, masked_r0, masked_r1, seg)
            masked_frames.append(comparison)
        
        # Prepare observations for each robot
        obs_robot0 = prepare_observation(obs, 224, 0, apply_mask=use_masking, segmentation=segmentation)
        obs_robot1 = prepare_observation(obs, 224, 1, apply_mask=use_masking, segmentation=segmentation)
        
        # Get actions for robot 0 (pegger)
        if len(action_queue_0) == 0:
            actions_0 = get_vla_action(
                cfg, vla, processor, obs_robot0, config.robot0_prompt,
                action_head=action_head, proprio_projector=proprio_projector
            )
            action_queue_0.extend(actions_0)
        
        # Get actions for robot 1 (receiver)
        if len(action_queue_1) == 0:
            actions_1 = get_vla_action(
                cfg, vla, processor, obs_robot1, config.robot1_prompt,
                action_head=action_head, proprio_projector=proprio_projector
            )
            action_queue_1.extend(actions_1)
        
        # Pop actions from queues
        action_0 = np.asarray(action_queue_0.popleft())
        action_1 = np.asarray(action_queue_1.popleft())
        
        # The model outputs 7D actions (6D pose + 1D gripper) from LIBERO training
        # TwoArmPegInHole needs 12D actions (6D for each robot, no gripper)
        # Take only the first 6 dimensions (xyz position + xyz rotation delta)
        action_0 = action_0[:6] if action_0.shape[-1] >= 6 else np.pad(action_0, (0, 6 - action_0.shape[-1]))
        action_1 = action_1[:6] if action_1.shape[-1] >= 6 else np.pad(action_1, (0, 6 - action_1.shape[-1]))
        
        # Scale actions down slightly as LIBERO actions may be larger than TwoArm expects
        action_scale = 0.5
        action_0 = action_0 * action_scale
        action_1 = action_1 * action_scale
        
        # Combine actions (12D total)
        action = np.concatenate([action_0, action_1])
        
        # Step environment
        obs, reward, done, info = env.step(action.tolist())
        
        # Check success
        success = bool(info.get('success', False)) if isinstance(info, dict) else False
        if success or done:
            # Save final frame
            final_frame = obs['agentview_image'][::-1].copy()
            if final_frame.dtype != np.uint8:
                final_frame = (final_frame * 255).astype(np.uint8) if final_frame.max() <= 1.0 else final_frame.astype(np.uint8)
            frames.append(final_frame)
            break
        
        if (t + 1) % 50 == 0:
            print(f"  Step {t + 1}/{config.max_steps}")
    
    return success, frames, masked_frames


def save_video(frames: List[np.ndarray], path: str, fps: int = 20):
    """Save frames as video."""
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {path}")


def run_experiment(config: ExperimentConfig, use_masking: bool, experiment_name: str):
    """Run a full experiment (multiple episodes)."""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Masking: {'ENABLED' if use_masking else 'DISABLED'}")
    print(f"Robot 0 (Pegger) prompt: {config.robot0_prompt}")
    print(f"Robot 1 (Receiver) prompt: {config.robot1_prompt}")
    print("="*70)
    
    # Create environment
    print("\nCreating environment...")
    env = create_environment(config, enable_segmentation=use_masking)
    
    # Load model
    print("Loading model...")
    vla, processor, action_head, proprio_projector, cfg = load_model(config)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run episodes
    results = []
    for ep in range(config.num_episodes):
        print(f"\n--- Episode {ep + 1}/{config.num_episodes} ---")
        
        success, frames, masked_frames = run_episode(
            env, vla, processor, action_head, proprio_projector, cfg, config,
            use_masking=use_masking
        )
        
        results.append(success)
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'}")
        print(f"  Frames: {len(frames)}")
        
        # Save video
        if config.save_video and len(frames) > 0:
            video_path = os.path.join(exp_dir, f"episode_{ep + 1}_{'success' if success else 'fail'}.mp4")
            save_video(frames, video_path, config.video_fps)
            
            # Save masked comparison video if available
            if len(masked_frames) > 0:
                masked_path = os.path.join(exp_dir, f"episode_{ep + 1}_masked_comparison.mp4")
                save_video(masked_frames, masked_path, config.video_fps)
    
    # Summary
    success_rate = sum(results) / len(results) * 100
    print(f"\n--- {experiment_name} Summary ---")
    print(f"Success rate: {sum(results)}/{len(results)} ({success_rate:.1f}%)")
    print(f"Results: {results}")
    print(f"Output directory: {exp_dir}")
    
    # Cleanup
    env.close()
    
    return results, exp_dir


def main():
    print("="*70)
    print("PERCEPTION CONFUSION EXPERIMENT - MASKED ONLY")
    print("Testing robot cooperation with other robot masked from view")
    print("="*70)
    
    # Configuration for FULL EVALUATION
    config = ExperimentConfig(
        # Prompts for each robot
        robot0_prompt="insert the peg into the hole",
        robot1_prompt="align the hole to receive the peg",
        
        # Run settings - full evaluation with 20 episodes
        num_episodes=20,
        max_steps=400,
        
        # Output
        output_dir="./experiment_outputs",
        save_video=True,
    )
    
    # Run only masked experiment
    print("\n" + "#"*70)
    print("# EXPERIMENT: WITH ROBOT BODY MASKING (ELEMENT-LEVEL)")
    print("# Robot0 sees: itself + peg + hole (Robot1 body fully masked)")
    print("# Robot1 sees: itself + peg + hole (Robot0 body fully masked)")
    print("#"*70)
    masked_results, masked_dir = run_experiment(
        config,
        use_masking=True,
        experiment_name="masked_perception"
    )
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    masked_rate = sum(masked_results) / len(masked_results) * 100
    
    print(f"With masking: {sum(masked_results)}/{len(masked_results)} ({masked_rate:.1f}%)")
    print(f"\nOutput directory: {masked_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
