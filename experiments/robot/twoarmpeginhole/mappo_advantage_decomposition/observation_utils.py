"""
Observation history management utilities for MAPPO VLA training.

Handles the accumulation and management of image history for VLA policy inputs,
including front-view images and wrist camera images for each agent.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch


class ObservationHistoryManager:
    """
    Manages observation history for multi-agent VLA policy.
    
    Maintains separate history buffers for:
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
        """
        Initialize the observation history manager.
        
        Args:
            num_agents: Number of agents (default: 2 for two-arm task)
            history_length: Number of frames to keep in history (including current)
            image_size: Size of images (H, W)
        """
        self.num_agents = num_agents
        self.history_length = history_length
        self.image_size = image_size
        
        # History buffers for front view (shared)
        self.front_image_history: deque = deque(maxlen=history_length)
        
        # History buffers per agent
        self.wrist_image_history: List[deque] = [
            deque(maxlen=history_length) for _ in range(num_agents)
        ]
        self.proprio_history: List[deque] = [
            deque(maxlen=history_length) for _ in range(num_agents)
        ]
        
        # Initialize with zero images
        self._initialize_history()
    
    def _initialize_history(self):
        """Initialize history buffers with zero images."""
        zero_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        zero_proprio = np.zeros(8, dtype=np.float32)  # 8-dim proprio state
        
        for _ in range(self.history_length):
            self.front_image_history.append(zero_image.copy())
            for agent_idx in range(self.num_agents):
                self.wrist_image_history[agent_idx].append(zero_image.copy())
                self.proprio_history[agent_idx].append(zero_proprio.copy())
    
    def reset(self):
        """Reset all history buffers (e.g., at episode start)."""
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
        """
        Update history with new observations.
        
        Args:
            front_image: Front view image (H, W, 3)
            wrist_images: List of wrist images per agent
            proprio_states: List of proprioceptive states per agent
        """
        self.front_image_history.append(front_image.copy())
        
        for agent_idx in range(self.num_agents):
            self.wrist_image_history[agent_idx].append(wrist_images[agent_idx].copy())
            self.proprio_history[agent_idx].append(proprio_states[agent_idx].copy())
    
    def get_agent_observation(
        self,
        agent_idx: int,
        include_history: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get observation for a specific agent with history.
        
        Args:
            agent_idx: Agent index (0 or 1)
            include_history: Whether to include historical frames
            
        Returns:
            Dictionary containing:
            - 'images': List of images [front_t, wrist_t, front_{t-1}, wrist_{t-1}, ...]
            - 'proprio': Current proprioceptive state
            - 'proprio_history': Historical proprio states (if include_history)
        """
        if include_history and self.history_length > 1:
            # Collect images with history (most recent first)
            images = []
            front_list = list(self.front_image_history)
            wrist_list = list(self.wrist_image_history[agent_idx])
            
            # Reverse to get most recent first
            for t in range(len(front_list) - 1, -1, -1):
                images.append(front_list[t])
                images.append(wrist_list[t])
            
            # Get proprio history
            proprio_list = list(self.proprio_history[agent_idx])
            proprio_history = np.stack(proprio_list[::-1], axis=0)  # Most recent first
            
            return {
                'images': images,
                'proprio': proprio_list[-1],  # Current proprio
                'proprio_history': proprio_history,
            }
        else:
            # Just current observation
            return {
                'images': [
                    list(self.front_image_history)[-1],
                    list(self.wrist_image_history[agent_idx])[-1],
                ],
                'proprio': list(self.proprio_history[agent_idx])[-1],
            }
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state for centralized value function.
        
        Returns:
            Global state vector combining both agents' proprioceptive states
            and other relevant global information.
        """
        # Combine latest proprio states from both agents
        proprio_states = []
        for agent_idx in range(self.num_agents):
            proprio_states.append(list(self.proprio_history[agent_idx])[-1])
        
        # Concatenate all proprio states
        global_state = np.concatenate(proprio_states, axis=0)
        
        return global_state
    
    def get_all_agent_observations(
        self,
        include_history: bool = True,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Get observations for all agents.
        
        Args:
            include_history: Whether to include historical frames
            
        Returns:
            List of observation dictionaries for each agent
        """
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
    
    Args:
        obs: Raw observation dictionary from environment
        resize_size: Target image size
        
    Returns:
        Tuple of (front_image, wrist_images, proprio_states)
    """
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.twoarmpeginhole.twoarm_utils import (
        get_twoarm_image,
        get_twoarm_wrist_image,
        quat2axisangle,
    )
    
    # Get front view image
    front_image = get_twoarm_image(obs)
    front_image = resize_image_for_policy(front_image, resize_size)
    
    # Get wrist images
    wrist_img0, wrist_img1 = get_twoarm_wrist_image(obs)
    wrist_img0 = resize_image_for_policy(wrist_img0, resize_size)
    wrist_img1 = resize_image_for_policy(wrist_img1, resize_size)
    wrist_images = [wrist_img0, wrist_img1]
    
    # Get proprioceptive states
    proprio_states = []
    for robot_idx in range(2):
        eef_pos = obs[f"robot{robot_idx}_eef_pos"]
        eef_quat = obs[f"robot{robot_idx}_eef_quat"]
        state = np.concatenate((eef_pos, quat2axisangle(eef_quat)))
        
        # Pad to 8 dimensions if needed
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
    
    Args:
        images: List of images (with history)
        proprio: Proprioceptive state
        task_description: Task instruction string
        processor: VLA processor
        device: Target device
        center_crop: Whether to apply center cropping
        agent_idx: Agent index (0 or 1) for multi-agent prompt prefix
        
    Returns:
        Dictionary of tensors ready for VLA forward pass
    """
    from experiments.robot.openvla_utils import (
        center_crop_image,
        check_image_format,
        OPENVLA_IMAGE_SIZE,
    )
    
    # Process images
    processed_images = []
    for img in images:
        check_image_format(img)
        
        # Resize if needed
        if img.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
            from experiments.robot.openvla_utils import resize_image_for_policy
            img = resize_image_for_policy(img, OPENVLA_IMAGE_SIZE)
        
        # Convert to PIL
        pil_image = Image.fromarray(img).convert("RGB")
        
        # Center crop if specified
        if center_crop:
            pil_image = center_crop_image(pil_image)
        
        processed_images.append(pil_image)
    
    # Build prompt with optional agent identity prefix
    if agent_idx is not None:
        prompt = f"In: You are robot {agent_idx}. You need to cooperate with the other robot to complete {task_description.lower()} task. What action should you take to {task_description.lower()}?\nOut:"
    else:
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    
    # Process primary image
    inputs = processor(prompt, processed_images[0]).to(device, dtype=torch.bfloat16)
    
    # Add additional images if any
    if len(processed_images) > 1:
        additional_pixel_values = []
        for img in processed_images[1:]:
            additional_inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
            additional_pixel_values.append(additional_inputs["pixel_values"])
        
        # Concatenate all pixel values
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
    """
    Normalize proprioceptive state using dataset statistics.
    
    Args:
        proprio: Raw proprioceptive state
        norm_stats: Normalization statistics dictionary
        normalization_type: Type of normalization to apply
        
    Returns:
        Normalized proprioceptive state
    """
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
    
    # Normalize to [-1, 1]
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
    """
    Unnormalize action from [-1, 1] to actual action range.
    
    VLA model outputs normalized actions in [-1, 1] range.
    This function converts them to the actual action range for the environment.
    
    NOTE: Handles dimension mismatch between norm_stats and action.
    E.g., LIBERO stats are 7-dim but TwoArm action is 6-dim.
    Only uses the first N dimensions of stats where N = action dimension.
    
    Args:
        normalized_action: Normalized action in [-1, 1] range
        norm_stats: Normalization statistics dictionary with action bounds
        normalization_type: Type of normalization ("bounds" or "bounds_q99")
        
    Returns:
        Unnormalized action in actual action range
    """
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
    
    # Handle dimension mismatch: only use first action_dim elements
    if len(mask_full) > action_dim:
        mask = mask_full[:action_dim]
        action_high = action_high_full[:action_dim]
        action_low = action_low_full[:action_dim]
    else:
        mask = mask_full
        action_high = action_high_full
        action_low = action_low_full
    
    # Unnormalize: [-1, 1] -> [action_low, action_high]
    # Formula: action = 0.5 * (normalized + 1) * (high - low) + low
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
    """
    Normalize action from actual range to [-1, 1].
    
    This is the inverse of unnormalize_action.
    Used when storing actions in buffer or for RL training.
    
    Args:
        action: Action in actual action range
        norm_stats: Normalization statistics dictionary with action bounds
        normalization_type: Type of normalization ("bounds" or "bounds_q99")
        
    Returns:
        Normalized action in [-1, 1] range
    """
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
    
    # Normalize: [action_low, action_high] -> [-1, 1]
    # Formula: normalized = 2 * (action - low) / (high - low) - 1
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
