"""
Observation history management utilities for ACPPO VLA training.

Extended from MAPPO observation_utils to support:
1. Preparing front-view only input for action distribution estimation
2. Extended proprio with action distribution for agent 1
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch


class ObservationHistoryManager:
    """
    Manages observation history for multi-agent VLA policy.
    
    Same as MAPPO version - maintains separate history buffers for:
    - Front view images (shared between agents)
    - Wrist camera images (per-agent)
    - Proprioceptive states (per-agent)
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        history_length: int = 2,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.num_agents = num_agents
        self.history_length = history_length
        self.image_size = image_size
        
        self.front_image_history: deque = deque(maxlen=history_length)
        self.wrist_image_history: List[deque] = [
            deque(maxlen=history_length) for _ in range(num_agents)
        ]
        self.proprio_history: List[deque] = [
            deque(maxlen=history_length) for _ in range(num_agents)
        ]
        
        self._initialize_history()
    
    def _initialize_history(self):
        zero_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        zero_proprio = np.zeros(8, dtype=np.float32)
        
        for _ in range(self.history_length):
            self.front_image_history.append(zero_image.copy())
            for agent_idx in range(self.num_agents):
                self.wrist_image_history[agent_idx].append(zero_image.copy())
                self.proprio_history[agent_idx].append(zero_proprio.copy())
    
    def reset(self):
        self.front_image_history.clear()
        for agent_idx in range(self.num_agents):
            self.wrist_image_history[agent_idx].clear()
            self.proprio_history[agent_idx].clear()
        self._initialize_history()
    
    def update(
        self,
        front_image: np.ndarray,
        wrist_images: List[np.ndarray],
        proprio_states: List[np.ndarray],
    ):
        self.front_image_history.append(front_image.copy())
        
        for agent_idx in range(self.num_agents):
            self.wrist_image_history[agent_idx].append(wrist_images[agent_idx].copy())
            self.proprio_history[agent_idx].append(proprio_states[agent_idx].copy())
    
    def get_agent_observation(
        self,
        agent_idx: int,
        include_history: bool = True,
    ) -> Dict[str, np.ndarray]:
        if include_history and self.history_length > 1:
            images = []
            front_list = list(self.front_image_history)
            wrist_list = list(self.wrist_image_history[agent_idx])
            
            for t in range(len(front_list) - 1, -1, -1):
                images.append(front_list[t])
                images.append(wrist_list[t])
            
            proprio_list = list(self.proprio_history[agent_idx])
            proprio_history = np.stack(proprio_list[::-1], axis=0)
            
            return {
                'images': images,
                'proprio': proprio_list[-1],
                'proprio_history': proprio_history,
            }
        else:
            return {
                'images': [
                    list(self.front_image_history)[-1],
                    list(self.wrist_image_history[agent_idx])[-1],
                ],
                'proprio': list(self.proprio_history[agent_idx])[-1],
            }
    
    def get_front_view_only_observation(
        self,
        include_history: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get front view only observation with wrist images padded to zeros.
        
        Used for estimating agent 0's action distribution for agent 1.
        
        Returns:
            Dictionary with:
            - 'images': List of [front_t, zero_pad, front_{t-1}, zero_pad, ...]
        """
        zero_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        if include_history and self.history_length > 1:
            images = []
            front_list = list(self.front_image_history)
            
            for t in range(len(front_list) - 1, -1, -1):
                images.append(front_list[t])
                images.append(zero_image.copy())  # Pad wrist with zeros
            
            return {'images': images}
        else:
            return {
                'images': [
                    list(self.front_image_history)[-1],
                    zero_image.copy(),
                ],
            }
    
    def get_global_state(self) -> np.ndarray:
        proprio_states = []
        for agent_idx in range(self.num_agents):
            proprio_states.append(list(self.proprio_history[agent_idx])[-1])
        
        global_state = np.concatenate(proprio_states, axis=0)
        return global_state
    
    def get_all_agent_observations(
        self,
        include_history: bool = True,
    ) -> List[Dict[str, np.ndarray]]:
        return [
            self.get_agent_observation(agent_idx, include_history)
            for agent_idx in range(self.num_agents)
        ]


def extract_observations_from_env(
    obs: Dict,
    resize_size: int = 224,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Extract and preprocess observations from robosuite environment.
    """
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.twoarmpeginhole.twoarm_utils import (
        get_twoarm_image,
        get_twoarm_wrist_image,
        quat2axisangle,
    )
    
    front_image = get_twoarm_image(obs)
    front_image = resize_image_for_policy(front_image, resize_size)
    
    wrist_img0, wrist_img1 = get_twoarm_wrist_image(obs)
    wrist_img0 = resize_image_for_policy(wrist_img0, resize_size)
    wrist_img1 = resize_image_for_policy(wrist_img1, resize_size)
    wrist_images = [wrist_img0, wrist_img1]
    
    proprio_states = []
    for robot_idx in range(2):
        eef_pos = obs[f"robot{robot_idx}_eef_pos"]
        eef_quat = obs[f"robot{robot_idx}_eef_quat"]
        state = np.concatenate((eef_pos, quat2axisangle(eef_quat)))
        
        if state.shape[0] < 8:
            pad = np.zeros(8 - state.shape[0], dtype=state.dtype)
            state = np.concatenate((state, pad))
        
        proprio_states.append(state.astype(np.float32))
    
    return front_image, wrist_images, proprio_states


def prepare_vla_input(
    images: List[np.ndarray],
    proprio: np.ndarray,
    task_description: str,
    processor,
    device: torch.device,
    center_crop: bool = True,
    agent_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Prepare observation for VLA model input.
    """
    from experiments.robot.openvla_utils import (
        center_crop_image,
        check_image_format,
        OPENVLA_IMAGE_SIZE,
    )
    
    processed_images = []
    for img in images:
        check_image_format(img)
        
        if img.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
            from experiments.robot.openvla_utils import resize_image_for_policy
            img = resize_image_for_policy(img, OPENVLA_IMAGE_SIZE)
        
        pil_image = Image.fromarray(img).convert("RGB")
        
        if center_crop:
            pil_image = center_crop_image(pil_image)
        
        processed_images.append(pil_image)
    
    if agent_idx is not None:
        prompt = f"In: You are robot {agent_idx}. What action should the robot take to {task_description.lower()}?\nOut:"
    else:
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    
    inputs = processor(prompt, processed_images[0]).to(device, dtype=torch.bfloat16)
    
    if len(processed_images) > 1:
        additional_pixel_values = []
        for img in processed_images[1:]:
            additional_inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
            additional_pixel_values.append(additional_inputs["pixel_values"])
        
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"]] + additional_pixel_values,
            dim=1,
        )
    
    return inputs


def prepare_vla_input_for_action_dist_estimation(
    front_images: List[np.ndarray],
    task_description: str,
    processor,
    device: torch.device,
    center_crop: bool = True,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, torch.Tensor]:
    """
    Prepare VLA input for estimating agent 0's action distribution.
    
    This is used by agent 1 to estimate what agent 0 would do.
    - Uses front view images only
    - Wrist images are padded with zeros
    - Text instruction is for robot 0 (not robot 1)
    
    Args:
        front_images: List of front view images (with history)
        task_description: Task description string
        processor: VLA processor
        device: Target device
        center_crop: Whether to apply center cropping
        image_size: Image size for zero padding
        
    Returns:
        Dictionary of tensors ready for VLA forward pass
    """
    from experiments.robot.openvla_utils import (
        center_crop_image,
        check_image_format,
        OPENVLA_IMAGE_SIZE,
    )
    
    # Create zero-padded wrist images
    zero_image = np.zeros((*image_size, 3), dtype=np.uint8)
    
    # Interleave front images with zero-padded wrist images
    # [front_t, zero, front_{t-1}, zero, ...]
    images_with_padding = []
    for front_img in front_images:
        images_with_padding.append(front_img)
        images_with_padding.append(zero_image.copy())
    
    processed_images = []
    for img in images_with_padding:
        check_image_format(img)
        
        if img.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
            from experiments.robot.openvla_utils import resize_image_for_policy
            img = resize_image_for_policy(img, OPENVLA_IMAGE_SIZE)
        
        pil_image = Image.fromarray(img).convert("RGB")
        
        if center_crop:
            pil_image = center_crop_image(pil_image)
        
        processed_images.append(pil_image)
    
    # Use robot 0's instruction for estimating its action distribution
    prompt = f"In: You are robot 0. What action should the robot take to {task_description.lower()}?\nOut:"
    
    inputs = processor(prompt, processed_images[0]).to(device, dtype=torch.bfloat16)
    
    if len(processed_images) > 1:
        additional_pixel_values = []
        for img in processed_images[1:]:
            additional_inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
            additional_pixel_values.append(additional_inputs["pixel_values"])
        
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"]] + additional_pixel_values,
            dim=1,
        )
    
    return inputs


def normalize_proprio(
    proprio: np.ndarray,
    norm_stats: Dict,
    normalization_type: str = "bounds_q99",
) -> np.ndarray:
    """Normalize proprioceptive state using dataset statistics."""
    if normalization_type == "bounds_q99":
        mask = norm_stats.get("mask", np.ones_like(norm_stats.get("q01", proprio), dtype=bool))
        proprio_high = np.array(norm_stats.get("q99", norm_stats.get("max", np.ones_like(proprio))))
        proprio_low = np.array(norm_stats.get("q01", norm_stats.get("min", -np.ones_like(proprio))))
    elif normalization_type == "bounds":
        mask = norm_stats.get("mask", np.ones_like(norm_stats.get("min", proprio), dtype=bool))
        proprio_high = np.array(norm_stats.get("max", np.ones_like(proprio)))
        proprio_low = np.array(norm_stats.get("min", -np.ones_like(proprio)))
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    normalized = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    
    return normalized.astype(np.float32)


def unnormalize_action(
    normalized_action: np.ndarray,
    norm_stats: Dict,
    normalization_type: str = "bounds_q99",
) -> np.ndarray:
    """Unnormalize action from [-1, 1] to actual action range."""
    action_dim = len(normalized_action)
    
    if normalization_type == "bounds_q99":
        mask_full = norm_stats.get("mask", np.ones_like(norm_stats.get("q01", normalized_action), dtype=bool))
        action_high_full = np.array(norm_stats.get("q99", norm_stats.get("max", np.ones_like(normalized_action))))
        action_low_full = np.array(norm_stats.get("q01", norm_stats.get("min", -np.ones_like(normalized_action))))
    elif normalization_type == "bounds":
        mask_full = norm_stats.get("mask", np.ones_like(norm_stats.get("min", normalized_action), dtype=bool))
        action_high_full = np.array(norm_stats.get("max", np.ones_like(normalized_action)))
        action_low_full = np.array(norm_stats.get("min", -np.ones_like(normalized_action)))
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    if len(mask_full) > action_dim:
        mask = mask_full[:action_dim]
        action_high = action_high_full[:action_dim]
        action_low = action_low_full[:action_dim]
    else:
        mask = mask_full
        action_high = action_high_full
        action_low = action_low_full
    
    unnormalized = np.where(
        mask,
        0.5 * (normalized_action + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_action,
    )
    
    return unnormalized.astype(np.float32)


def normalize_action(
    action: np.ndarray,
    norm_stats: Dict,
    normalization_type: str = "bounds_q99",
) -> np.ndarray:
    """Normalize action from actual range to [-1, 1]."""
    if normalization_type == "bounds_q99":
        mask = norm_stats.get("mask", np.ones_like(norm_stats.get("q01", action), dtype=bool))
        action_high = np.array(norm_stats.get("q99", norm_stats.get("max", np.ones_like(action))))
        action_low = np.array(norm_stats.get("q01", norm_stats.get("min", -np.ones_like(action))))
    elif normalization_type == "bounds":
        mask = norm_stats.get("mask", np.ones_like(norm_stats.get("min", action), dtype=bool))
        action_high = np.array(norm_stats.get("max", np.ones_like(action)))
        action_low = np.array(norm_stats.get("min", -np.ones_like(action)))
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    normalized = np.clip(
        np.where(
            mask,
            2 * (action - action_low) / (action_high - action_low + 1e-8) - 1,
            action,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    
    return normalized.astype(np.float32)


def create_extended_proprio(
    proprio: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    """
    Create extended proprioceptive state by concatenating action distribution.
    
    Used for agent 1 to include estimated agent 0's action distribution.
    
    Args:
        proprio: Base proprioceptive state (8,)
        action_mean: Action distribution mean (action_dim * chunk_size,)
        action_std: Action distribution std (action_dim * chunk_size,)
        
    Returns:
        Extended proprio (8 + 2 * action_dim * chunk_size,)
    """
    return np.concatenate([proprio, action_mean.flatten(), action_std.flatten()], axis=0).astype(np.float32)
