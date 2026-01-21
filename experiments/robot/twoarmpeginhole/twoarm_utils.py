"""Utils for evaluating policies in the TwoArmPegInHole robosuite environment."""

import math
import os
import sys
from pathlib import Path

import imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROBOSUITE_DIR = _REPO_ROOT / "robosuite"
if _ROBOSUITE_DIR.exists():
    sys.path.insert(0, str(_ROBOSUITE_DIR))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from experiments.robot.robot_utils import DATE, DATE_TIME


# =============================================================================
# Task Description Constants for TwoArmPegInHole
# =============================================================================

# Instruction modes
INSTRUCTION_MODE_SHARED = "shared"          # Both robots get the same instruction
INSTRUCTION_MODE_SPLIT = "split"            # Each robot gets a different instruction (peg/hole)
INSTRUCTION_MODE_SPLIT_DETAILED = "split_detailed"  # More detailed separate instructions

# Shared instruction (default)
TASK_DESCRIPTION_SHARED = "insert the green peg into the hole of the red square"

# Split instructions - Robot 0 handles peg, Robot 1 handles hole
TASK_DESCRIPTIONS_SPLIT = {
    "robot0": "insert the peg",
    "robot1": "align the hole",
}

# More detailed split instructions
# TASK_DESCRIPTIONS_SPLIT_DETAILED = {
#     "robot0": "You are the left robot. Move the peg and insert it into the hole.",
#     "robot1": "You are the right robot. Position the hole so it can receive the peg.",
# }
TASK_DESCRIPTIONS_SPLIT_DETAILED = {
    "robot0": "You are the left robot. Your goal is to insert the green peg inside the hole of the red square.",
    "robot1": "You are the right robot. Your goal is to align with the green peg so that it can be inserted into the hole of the red square.",
}
# TASK_DESCRIPTIONS_SPLIT_DETAILED = {
#     "robot0": "You are the left robot from the frontview. Lay down the green peg.",
#     "robot1": "You are the right robot from the frontview. Raise up the red square.",
# }

def get_twoarm_task_descriptions(mode: str = INSTRUCTION_MODE_SHARED, custom_descriptions: dict = None):
    """Get task descriptions for each robot based on the instruction mode.
    
    Args:
        mode: Instruction mode - "shared", "split", or "split_detailed"
        custom_descriptions: Optional dict with "robot0" and "robot1" keys for custom instructions
        
    Returns:
        tuple: (task_description_robot0, task_description_robot1, combined_description)
            - task_description_robot0: Instruction for robot 0
            - task_description_robot1: Instruction for robot 1  
            - combined_description: Combined description for logging/video naming
    """
    if custom_descriptions is not None:
        desc_robot0 = custom_descriptions.get("robot0", TASK_DESCRIPTION_SHARED)
        desc_robot1 = custom_descriptions.get("robot1", TASK_DESCRIPTION_SHARED)
        combined = f"{desc_robot0} | {desc_robot1}"
        return desc_robot0, desc_robot1, combined
    
    if mode == INSTRUCTION_MODE_SHARED:
        return TASK_DESCRIPTION_SHARED, TASK_DESCRIPTION_SHARED, TASK_DESCRIPTION_SHARED
    
    elif mode == INSTRUCTION_MODE_SPLIT:
        desc_robot0 = TASK_DESCRIPTIONS_SPLIT["robot0"]
        desc_robot1 = TASK_DESCRIPTIONS_SPLIT["robot1"]
        combined = "insert_peg_in_hole_split"
        return desc_robot0, desc_robot1, combined
    
    elif mode == INSTRUCTION_MODE_SPLIT_DETAILED:
        desc_robot0 = TASK_DESCRIPTIONS_SPLIT_DETAILED["robot0"]
        desc_robot1 = TASK_DESCRIPTIONS_SPLIT_DETAILED["robot1"]
        combined = "insert_peg_in_hole_detailed"
        return desc_robot0, desc_robot1, combined
    
    else:
        raise ValueError(f"Unknown instruction mode: {mode}. "
                        f"Use one of: {INSTRUCTION_MODE_SHARED}, {INSTRUCTION_MODE_SPLIT}, {INSTRUCTION_MODE_SPLIT_DETAILED}")


def get_twoarm_env(
    model_family,
    resolution=256,
    robot1="Panda",
    robot2="Panda",
    controller="BASIC",
    env_configuration="opposed",
    reward_shaping=False,
    # Dense reward component weights (only used when reward_shaping=True)
    reaching_weight=1.0,
    perpendicular_weight=1.0,
    parallel_weight=1.0,
    alignment_weight=1.0,
):
    """Initializes and returns the TwoArmPegInHole environment, along with the task description.
    
    Dense Reward Components (when reward_shaping=True):
        - reaching_weight: Weight for distance-based reward (peg-hole distance)
                          WARNING: High values may cause robots to collide without alignment
        - perpendicular_weight: Weight for perpendicular alignment reward
        - parallel_weight: Weight for parallel (depth) positioning reward  
        - alignment_weight: Weight for angular alignment reward (cos similarity)
    """
    task_description = "insert peg in hole"
    controller_config = load_composite_controller_config(controller=controller)
    body_parts = controller_config.get("body_parts", {})
    if "right" in body_parts:
        controller_config["body_parts"] = {"right": body_parts["right"]}
    elif "left" in body_parts:
        controller_config["body_parts"] = {"left": body_parts["left"]}
    camera_names = ["frontview", "agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]
    env = suite.make(
        "TwoArmPegInHole",
        robots=[robot1, robot2],
        gripper_types=None,
        controller_configs=controller_config,
        env_configuration=env_configuration,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        reward_shaping=reward_shaping,
        reaching_weight=reaching_weight,
        perpendicular_weight=perpendicular_weight,
        parallel_weight=parallel_weight,
        alignment_weight=alignment_weight,
    )
    
    # Warm up the offscreen renderer to avoid noise in first episode
    # MuJoCo's OpenGL context needs a few render cycles to fully initialize
    env.reset()
    dummy_action = [0.0] * 12
    for _ in range(5):  # Do a few dummy steps to warm up renderer
        env.step(dummy_action)
    env.reset()  # Reset again to start fresh
    
    return env, task_description


def get_twoarm_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0.0] * 12


def get_twoarm_image(obs):
    """Extracts third-person image from observations and preprocesses it.
    
    Note: robosuite reuses observation buffers, so we must make deep copies.
    Following libero's pattern: flip then copy.
    """
    raw_img = obs["agentview_image"]
    # Flip vertically and create a new independent array
    img = raw_img[::-1].copy()
    
    # Ensure uint8 [0, 255]
    if img.dtype != np.uint8:
        if img.max() <= 1.05:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_twoarm_video_frame(obs):
    """Extracts image for video logging (prefer agentview).
    
    Note: robosuite reuses observation buffers, so we must make deep copies
    to avoid all frames showing the same (last) image.
    """
    if "agentview_image" in obs:
        # Get the raw image and make a deep copy immediately
        raw_img = obs["agentview_image"]
        # Flip vertically and create a new contiguous array (robosuite images are upside down)
        # Using [::-1] like libero, then .copy() to ensure independent array
        img = raw_img[::-1].copy()
        
        # Ensure uint8 format [0, 255]
        if img.dtype != np.uint8:
            if img.max() <= 1.05:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    # Fallback to frontview with explicit copy
    if "frontview_image" in obs:
        raw_img = obs["frontview_image"]
        img = raw_img[::-1].copy()
        if img.dtype != np.uint8:
            if img.max() <= 1.05:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    # Final fallback using get_twoarm_image
    return get_twoarm_image(obs).copy()


def get_twoarm_wrist_image(obs):
    """Extracts wrist camera images for both robots and preprocesses them.
    
    Note: robosuite reuses observation buffers, so we must make deep copies.
    Following libero's pattern: flip then copy.
    """
    # Flip vertically and make independent copies
    wrist_img0 = obs["robot0_eye_in_hand_image"][::-1].copy()
    wrist_img1 = obs["robot1_eye_in_hand_image"][::-1].copy()
    
    # Process both images
    processed_imgs = []
    for img in [wrist_img0, wrist_img1]:
        if img.dtype != np.uint8:
            if img.max() <= 1.05:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        processed_imgs.append(img)
        
    return processed_imgs[0], processed_imgs[1]


class RolloutVideoWriter:
    """Video writer that saves frames immediately to avoid buffer reuse issues."""

    def __init__(self, idx, task_description, fps=20):
        rollout_dir = f"./rollouts/{DATE}"
        os.makedirs(rollout_dir, exist_ok=True)
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        self.mp4_path = (
            f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--task={processed_task_description}.mp4"
        )
        self.fps = fps
        self.frames = []
        self.frame_count = 0

    def add_frame(self, obs):
        """Add a frame from observation dict. Call this immediately after env.step().
        
        Note: robosuite reuses observation buffers, so we must make deep copies
        to avoid all frames showing the same (last) image.
        """
        # Flip vertically and make a deep copy (robosuite images are upside down)
        # Following libero's pattern: slice then copy for independence
        frame = obs["frontview_image"][::-1].copy()

        # Ensure frame is uint8 (robosuite may return float or other types)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Store frame (imageio expects RGB format)
        self.frames.append(frame)
        self.frame_count += 1

    def close(self, success, log_file=None):
        """Close the video writer and rename file with success status."""
        # Rename file to include success status
        final_path = self.mp4_path.replace(".mp4", f"--success={success}.mp4")

        # Save video using imageio (more reliable than cv2)
        if len(self.frames) > 0:
            video_writer = imageio.get_writer(final_path, fps=self.fps)
            for frame in self.frames:
                video_writer.append_data(frame)
            video_writer.close()

            print(f"Saved rollout MP4 at path {final_path} ({self.frame_count} frames)")
            if log_file is not None:
                log_file.write(f"Saved rollout MP4 at path {final_path}\n")
        else:
            print(f"No frames to save for episode")
            final_path = None

        return final_path


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"

    if len(rollout_images) == 0:
        print(f"No frames to save for episode {idx}")
        return None

    video_writer = imageio.get_writer(mp4_path, fps=20)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
