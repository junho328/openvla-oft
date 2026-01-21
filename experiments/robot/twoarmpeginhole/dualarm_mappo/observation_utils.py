"""
Observation history management utilities for Dual-Arm MAPPO VLA training.

Handles the accumulation and management of image history for VLA policy inputs,
with per-agent image and proprio padding for bimanual model.

Key differences from single-arm MAPPO:
- Each agent sees agentview + own wrist view + padded other agent's wrist view
- Proprio is 14-dim total, each agent uses their own 7-dim and pads the other 7-dim
- Agent 0: uses action[:7], proprio[:7] with pad, left wrist real + right wrist padded
- Agent 1: uses action[7:], proprio[7:] with pad, left wrist padded + right wrist real
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch

from .config import DUALARM_MAPPO_CONSTANTS


class DualArmObservationHistoryManager:
    """
    Manages observation history for dual-arm VLA policy with bimanual model.
    
    For each agent, prepares observations as:
    - Agent 0: agentview + left_wrist (real) + right_wrist (padded to zeros)
    - Agent 1: agentview + left_wrist (padded to zeros) + right_wrist (real)
    
    Proprio handling:
    - Full proprio is 14-dim (7-dim per arm from ALOHA bimanual model)
    - Agent 0: uses proprio[:7] padded to 14-dim as [proprio[:7], zeros(7)]
    - Agent 1: uses proprio[7:] padded to 14-dim as [zeros(7), proprio[7:]]
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        history_length: int = 2,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize the dual-arm observation history manager.
        
        Args:
            num_agents: Number of agents (default: 2 for two-arm task)
            history_length: Number of frames to keep in history (including current)
            image_size: Size of images (H, W)
        """
        self.num_agents = num_agents
        self.history_length = history_length
        self.image_size = image_size
        
        # History buffers for agentview (shared between agents)
        self.agentview_history: deque = deque(maxlen=history_length)
        
        # History buffers for wrist images (separate for left and right)
        self.left_wrist_history: deque = deque(maxlen=history_length)
        self.right_wrist_history: deque = deque(maxlen=history_length)
        
        # Full proprio history (14-dim: 7 for each arm)
        self.proprio_history: deque = deque(maxlen=history_length)
        
        # Initialize with zero images/proprio
        self._initialize_history()
    
    def _initialize_history(self):
        """Initialize history buffers with zero images/proprio."""
        zero_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        zero_proprio = np.zeros(DUALARM_MAPPO_CONSTANTS["MODEL_PROPRIO_DIM"], dtype=np.float32)
        
        for _ in range(self.history_length):
            self.agentview_history.append(zero_image.copy())
            self.left_wrist_history.append(zero_image.copy())
            self.right_wrist_history.append(zero_image.copy())
            self.proprio_history.append(zero_proprio.copy())
    
    def reset(self):
        """Reset all history buffers (e.g., at episode start)."""
        self.agentview_history.clear()
        self.left_wrist_history.clear()
        self.right_wrist_history.clear()
        self.proprio_history.clear()
        self._initialize_history()
    
    def update(
        self,
        agentview_image: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        proprio_state: np.ndarray,
    ):
        """
        Update history with new observations.
        
        Args:
            agentview_image: Agent view image (H, W, 3)
            left_wrist_image: Left wrist camera image (H, W, 3)
            right_wrist_image: Right wrist camera image (H, W, 3)
            proprio_state: Full 14-dim proprioceptive state
        """
        self.agentview_history.append(agentview_image.copy())
        self.left_wrist_history.append(left_wrist_image.copy())
        self.right_wrist_history.append(right_wrist_image.copy())
        self.proprio_history.append(proprio_state.copy())
    
    def get_agent_observation(
        self,
        agent_idx: int,
        include_history: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get observation for a specific agent with history, including per-agent padding.
        
        For bimanual model:
        - Agent 0: agentview + left_wrist (real) + right_wrist (zero-padded)
        - Agent 1: agentview + left_wrist (zero-padded) + right_wrist (real)
        
        Proprio padding:
        - Agent 0: [proprio[:7], zeros(7)] -> 14-dim
        - Agent 1: [zeros(7), proprio[7:]] -> 14-dim
        
        Args:
            agent_idx: Agent index (0 or 1)
            include_history: Whether to include historical frames
            
        Returns:
            Dictionary containing:
            - 'images': List of images [agentview_t, left_wrist_t, right_wrist_t, 
                                       agentview_{t-1}, left_wrist_{t-1}, right_wrist_{t-1}, ...]
            - 'proprio': Current padded proprio state (14-dim)
            - 'proprio_history': Historical padded proprio states (if include_history)
        """
        zero_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        agent_proprio_dim = DUALARM_MAPPO_CONSTANTS["AGENT_PROPRIO_DIM"]
        model_proprio_dim = DUALARM_MAPPO_CONSTANTS["MODEL_PROPRIO_DIM"]
        
        if include_history and self.history_length > 1:
            # Collect images with history (most recent first)
            images = []
            agentview_list = list(self.agentview_history)
            left_wrist_list = list(self.left_wrist_history)
            right_wrist_list = list(self.right_wrist_history)
            proprio_list = list(self.proprio_history)
            
            # Reverse to get most recent first
            for t in range(len(agentview_list) - 1, -1, -1):
                images.append(agentview_list[t])
                
                # Agent 0: left wrist real, right wrist padded
                # Agent 1: left wrist padded, right wrist real
                if agent_idx == 0:
                    images.append(left_wrist_list[t])
                    images.append(zero_image.copy())  # Pad right wrist for agent 0
                else:
                    images.append(zero_image.copy())  # Pad left wrist for agent 1
                    images.append(right_wrist_list[t])
            
            # Build padded proprio history
            proprio_history_padded = []
            for t in range(len(proprio_list) - 1, -1, -1):
                full_proprio = proprio_list[t]
                if agent_idx == 0:
                    # Agent 0: use first 7-dim, pad last 7-dim
                    padded_proprio = np.concatenate([
                        full_proprio[:agent_proprio_dim],
                        np.zeros(agent_proprio_dim, dtype=np.float32)
                    ])
                else:
                    # Agent 1: pad first 7-dim, use last 7-dim
                    padded_proprio = np.concatenate([
                        np.zeros(agent_proprio_dim, dtype=np.float32),
                        full_proprio[agent_proprio_dim:]
                    ])
                proprio_history_padded.append(padded_proprio)
            
            proprio_history = np.stack(proprio_history_padded, axis=0)
            
            return {
                'images': images,
                'proprio': proprio_history_padded[0],  # Current padded proprio (most recent)
                'proprio_history': proprio_history,
            }
        else:
            # Just current observation
            current_agentview = list(self.agentview_history)[-1]
            current_left_wrist = list(self.left_wrist_history)[-1]
            current_right_wrist = list(self.right_wrist_history)[-1]
            current_proprio = list(self.proprio_history)[-1]
            
            # Build padded images
            if agent_idx == 0:
                images = [current_agentview, current_left_wrist, zero_image.copy()]
                padded_proprio = np.concatenate([
                    current_proprio[:agent_proprio_dim],
                    np.zeros(agent_proprio_dim, dtype=np.float32)
                ])
            else:
                images = [current_agentview, zero_image.copy(), current_right_wrist]
                padded_proprio = np.concatenate([
                    np.zeros(agent_proprio_dim, dtype=np.float32),
                    current_proprio[agent_proprio_dim:]
                ])
            
            return {
                'images': images,
                'proprio': padded_proprio,
            }
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state for centralized value function.
        
        Returns:
            Global state vector with full 14-dim proprio (no padding).
        """
        return list(self.proprio_history)[-1].copy()
    
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and preprocess observations from robosuite environment for dual-arm setup.
    
    Args:
        obs: Raw observation dictionary from environment
        resize_size: Target image size
        
    Returns:
        Tuple of (agentview_image, left_wrist_image, right_wrist_image, proprio_state)
        where proprio_state is 14-dim (7-dim per arm for bimanual model)
    """
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.twoarmpeginhole.twoarm_utils import (
        get_twoarm_image,
        get_twoarm_wrist_image,
    )
    
    # Get agentview image
    agentview_image = get_twoarm_image(obs)
    agentview_image = resize_image_for_policy(agentview_image, resize_size)
    
    # Get wrist images
    left_wrist_img, right_wrist_img = get_twoarm_wrist_image(obs)
    left_wrist_img = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img = resize_image_for_policy(right_wrist_img, resize_size)
    
    # Get proprioceptive state for bimanual model (14-dim total: 7 per arm)
    # ALOHA-style proprio: joint positions (6) + gripper (1) for each arm
    proprio_state = extract_bimanual_proprio_from_env(obs)
    
    return agentview_image, left_wrist_img, right_wrist_img, proprio_state


def extract_bimanual_proprio_from_env(obs: Dict) -> np.ndarray:
    """
    Extract 14-dim bimanual proprio state from environment observation.
    
    TwoArmPegInHole environment provides:
    - robot{i}_joint_pos: 7-dim (Panda robot has 7 DoF)
    
    We directly use the 7-dim joint positions for each arm to match
    the ALOHA bimanual proprio format (14-dim total = 7 + 7).
    
    Note: TwoArmPegInHole has no grippers, but the joint_pos is still 7-dim
    because Panda robot has 7 joints. This matches ALOHA's 7-dim per arm format.
    
    For robosuite TwoArmPegInHole, we map:
    - robot0 -> left arm (agent 0)
    - robot1 -> right arm (agent 1)
    
    Args:
        obs: Environment observation dictionary
        
    Returns:
        14-dim proprio state [left_arm(7), right_arm(7)]
    """
    agent_proprio_dim = DUALARM_MAPPO_CONSTANTS["AGENT_PROPRIO_DIM"]  # 7
    
    # Extract 7-dim joint positions directly for each robot
    # Panda robot has 7 DoF, matching ALOHA's 7-dim proprio per arm
    left_arm_proprio = obs.get("robot0_joint_pos", np.zeros(7))
    right_arm_proprio = obs.get("robot1_joint_pos", np.zeros(7))
    
    # Ensure correct dimensions (should be 7-dim)
    left_arm_proprio = np.asarray(left_arm_proprio, dtype=np.float32)
    right_arm_proprio = np.asarray(right_arm_proprio, dtype=np.float32)
    
    if len(left_arm_proprio) < agent_proprio_dim:
        left_arm_proprio = np.pad(left_arm_proprio, (0, agent_proprio_dim - len(left_arm_proprio)))
    else:
        left_arm_proprio = left_arm_proprio[:agent_proprio_dim]
        
    if len(right_arm_proprio) < agent_proprio_dim:
        right_arm_proprio = np.pad(right_arm_proprio, (0, agent_proprio_dim - len(right_arm_proprio)))
    else:
        right_arm_proprio = right_arm_proprio[:agent_proprio_dim]
    
    # Combine: [left_arm(7), right_arm(7)] = 14-dim
    bimanual_proprio = np.concatenate([left_arm_proprio, right_arm_proprio])
    
    return bimanual_proprio.astype(np.float32)


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
    Prepare observation for VLA model input (bimanual model).
    
    Args:
        images: List of images [agentview, left_wrist, right_wrist] * history
        proprio: 14-dim padded proprio state for the agent
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
    normalization_type: str = "bounds",
) -> np.ndarray:
    """
    Normalize proprioceptive state using dataset statistics.
    ALOHA uses "bounds" normalization type.
    
    Args:
        proprio: Raw proprioceptive state (14-dim)
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
    
    # Handle dimension mismatch
    if len(mask) > len(proprio):
        mask = mask[:len(proprio)]
        proprio_high = proprio_high[:len(proprio)]
        proprio_low = proprio_low[:len(proprio)]
    
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
    normalization_type: str = "bounds",
    agent_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Unnormalize action from [-1, 1] to actual action range.
    
    For bimanual model, action is 14-dim (7-dim per arm).
    ALOHA uses "bounds" normalization type.
    
    Args:
        normalized_action: Normalized action in [-1, 1] range (7 or 14 dim)
        norm_stats: Normalization statistics dictionary with action bounds (14-dim for bimanual)
        normalization_type: Type of normalization ("bounds" or "bounds_q99")
        agent_idx: Agent index (0 for left arm using stats[:7], 1 for right arm using stats[7:14]).
                   If None, uses first action_dim elements of stats.
        
    Returns:
        Unnormalized action in actual action range
    """
    action_dim = len(normalized_action)
    agent_action_dim = DUALARM_MAPPO_CONSTANTS["AGENT_ACTION_DIM"]  # 7
    
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
    
    # Handle per-agent unnormalization for bimanual model
    # norm_stats is 14-dim: [left_arm(7), right_arm(7)]
    # Agent 0 (left arm): use stats[:7]
    # Agent 1 (right arm): use stats[7:14]
    if agent_idx is not None and len(mask_full) >= 2 * agent_action_dim:
        if agent_idx == 0:
            start_idx = 0
        else:
            start_idx = agent_action_dim
        
        end_idx = start_idx + min(action_dim, agent_action_dim)
        mask = mask_full[start_idx:end_idx]
        action_high = action_high_full[start_idx:end_idx]
        action_low = action_low_full[start_idx:end_idx]
        
        # Pad if action_dim > agent_action_dim
        if action_dim > len(mask):
            pad_len = action_dim - len(mask)
            mask = np.concatenate([mask, np.ones(pad_len, dtype=bool)])
            action_high = np.concatenate([action_high, np.ones(pad_len)])
            action_low = np.concatenate([action_low, -np.ones(pad_len)])
    else:
        # Fallback: use first action_dim elements
        if len(mask_full) > action_dim:
            mask = mask_full[:action_dim]
            action_high = action_high_full[:action_dim]
            action_low = action_low_full[:action_dim]
        else:
            mask = mask_full
            action_high = action_high_full
            action_low = action_low_full
    
    # Unnormalize: [-1, 1] -> [action_low, action_high]
    unnormalized = np.where(
        mask,
        0.5 * (normalized_action + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_action,
    )
    
    return unnormalized.astype(np.float32)


def split_bimanual_action(
    full_action: np.ndarray,
    agent_idx: int,
) -> np.ndarray:
    """
    Split a 14-dim bimanual action into per-agent 7-dim action.
    
    Args:
        full_action: Full 14-dim bimanual action from VLA
        agent_idx: Agent index (0 for left arm, 1 for right arm)
        
    Returns:
        7-dim action for the specific agent
    """
    agent_action_dim = DUALARM_MAPPO_CONSTANTS["AGENT_ACTION_DIM"]
    
    if agent_idx == 0:
        return full_action[:agent_action_dim]
    else:
        return full_action[agent_action_dim:]


def combine_agent_actions(
    action_0: np.ndarray,
    action_1: np.ndarray,
) -> np.ndarray:
    """
    Combine per-agent actions into full bimanual action.
    
    Args:
        action_0: 7-dim action for agent 0 (left arm)
        action_1: 7-dim action for agent 1 (right arm)
        
    Returns:
        14-dim bimanual action
    """
    return np.concatenate([action_0, action_1])


def normalize_action(
    action: np.ndarray,
    norm_stats: Dict,
    normalization_type: str = "bounds",
    agent_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize action from actual range to [-1, 1].
    
    Args:
        action: Action in actual action range (7 or 14 dim)
        norm_stats: Normalization statistics dictionary with action bounds (14-dim for bimanual)
        normalization_type: Type of normalization ("bounds" or "bounds_q99")
        agent_idx: Agent index (0 for left arm using stats[:7], 1 for right arm using stats[7:14]).
                   If None, uses first action_dim elements of stats.
        
    Returns:
        Normalized action in [-1, 1] range
    """
    action_dim = len(action)
    agent_action_dim = DUALARM_MAPPO_CONSTANTS["AGENT_ACTION_DIM"]  # 7
    
    if normalization_type == "bounds_q99":
        mask_full = norm_stats.get("mask", np.ones_like(norm_stats.get("q01", action), dtype=bool))
        action_high_full = np.array(norm_stats.get("q99", norm_stats.get("max", np.ones_like(action))))
        action_low_full = np.array(norm_stats.get("q01", norm_stats.get("min", -np.ones_like(action))))
    elif normalization_type == "bounds":
        mask_full = norm_stats.get("mask", np.ones_like(norm_stats.get("min", action), dtype=bool))
        action_high_full = np.array(norm_stats.get("max", np.ones_like(action)))
        action_low_full = np.array(norm_stats.get("min", -np.ones_like(action)))
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    # Handle per-agent normalization for bimanual model
    # norm_stats is 14-dim: [left_arm(7), right_arm(7)]
    # Agent 0 (left arm): use stats[:7]
    # Agent 1 (right arm): use stats[7:14]
    if agent_idx is not None and len(mask_full) >= 2 * agent_action_dim:
        if agent_idx == 0:
            start_idx = 0
        else:
            start_idx = agent_action_dim
        
        end_idx = start_idx + min(action_dim, agent_action_dim)
        mask = mask_full[start_idx:end_idx]
        action_high = action_high_full[start_idx:end_idx]
        action_low = action_low_full[start_idx:end_idx]
        
        # Pad if action_dim > agent_action_dim
        if action_dim > len(mask):
            pad_len = action_dim - len(mask)
            mask = np.concatenate([mask, np.ones(pad_len, dtype=bool)])
            action_high = np.concatenate([action_high, np.ones(pad_len)])
            action_low = np.concatenate([action_low, -np.ones(pad_len)])
    else:
        # Fallback: use first action_dim elements
        if len(mask_full) > action_dim:
            mask = mask_full[:action_dim]
            action_high = action_high_full[:action_dim]
            action_low = action_low_full[:action_dim]
        else:
            mask = mask_full
            action_high = action_high_full
            action_low = action_low_full
    
    # Normalize: [action_low, action_high] -> [-1, 1]
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
