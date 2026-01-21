"""
perception_masking.py

Utilities for masking robot bodies from images to test perception confusion hypothesis.

The goal is to test whether robots can cooperate better when they don't see each other's
body (only seeing their own body and the object the other robot is carrying).

Robot A (robot0): Carries the peg
Robot B (robot1): Carries the square with hole

Masking strategy:
- For Robot A's view: Mask out Robot B's body, but keep the hole visible
- For Robot B's view: Mask out Robot A's body, but keep the peg visible
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2


# Segmentation ID mapping for TwoArmPegInHole
# Using ELEMENT-level segmentation (per-geom IDs) for complete robot masking
#
# Instance segmentation only provides high-level grouping that misses robot bases.
# Element segmentation provides per-geom IDs allowing complete masking.

# Robot0 geom IDs (arm parts + pedestal)
ROBOT0_GEOM_IDS = [
    # Arm links
    18, 19, 20, 21, 22, 24, 26, 28, 29, 30, 31, 33, 34, 35, 36, 
    38, 39, 40, 42, 43, 44, 46, 47, 51, 52, 53, 54, 56, 57, 58, 
    60, 61, 62, 63, 65, 66, 67,
    # Pedestal/mount
    73, 74,
]

# Robot1 geom IDs (arm parts + pedestal)  
ROBOT1_GEOM_IDS = [
    # Arm links
    78, 84, 85, 86, 87, 89, 91, 93, 94, 95, 96, 98, 99, 100, 101,
    103, 104, 105, 107, 109, 110, 111, 113, 114, 116, 117, 118, 119,
    122, 123, 125, 126, 127, 128, 129, 130, 132,
    # Pedestal/mount
    144, 145,
]

# Peg geom IDs
PEG_GEOM_IDS = [69, 70]

# Hole (square with hole) geom IDs
HOLE_GEOM_IDS = [134, 135, 136, 137, 138, 139, 140, 141]

# Legacy instance-level IDs (less accurate, kept for reference)
SEGMENTATION_IDS = {
    'background': 0,
    'robot0': 1,  # Peg carrier (left robot) - INCOMPLETE, misses pedestal
    'robot1': 2,  # Hole carrier (right robot) - INCOMPLETE, misses pedestal
    'peg': 4,     # Green peg
    'hole': 5,    # Red square with hole
}


def get_robot_mask(
    segmentation: np.ndarray,
    robot_id: int,
    include_attached_object: bool = True
) -> np.ndarray:
    """
    Get a binary mask for a specific robot.
    
    Args:
        segmentation: Instance segmentation array from robosuite
        robot_id: The segmentation ID of the robot to mask
        include_attached_object: If True, also mask the object attached to this robot
        
    Returns:
        Binary mask where True indicates pixels belonging to the robot
    """
    mask = segmentation.squeeze() == robot_id
    return mask.astype(np.uint8) * 255


def mask_other_robot(
    image: np.ndarray,
    segmentation: np.ndarray,
    viewer_robot: str,
    fill_method: str = 'inpaint',
    dilation_iterations: int = 2,
    use_element_segmentation: bool = True
) -> np.ndarray:
    """
    Mask out the other robot from the given robot's view.
    
    Args:
        image: RGB image from the camera (H, W, 3)
        segmentation: Segmentation array (H, W, 1) - element or instance level
        viewer_robot: Which robot is viewing ('robot0' or 'robot1')
        fill_method: How to fill masked regions ('inpaint', 'black', 'blur', 'gray')
        dilation_iterations: Number of times to dilate the mask for complete coverage
        use_element_segmentation: If True, use element-level geom IDs for complete masking
        
    Returns:
        Image with the other robot masked out
    """
    seg_squeezed = segmentation.squeeze()
    
    if use_element_segmentation:
        # Use element-level segmentation (per-geom IDs) for complete robot masking
        # This includes all arm parts AND the pedestal/base
        if viewer_robot == 'robot0':
            # Robot0 viewing - mask Robot1's complete body
            geom_ids_to_mask = ROBOT1_GEOM_IDS
        else:
            # Robot1 viewing - mask Robot0's complete body
            geom_ids_to_mask = ROBOT0_GEOM_IDS
        
        # Create mask for all geom IDs belonging to the other robot
        robot_mask = np.zeros_like(seg_squeezed, dtype=np.uint8)
        for geom_id in geom_ids_to_mask:
            robot_mask = robot_mask | (seg_squeezed == geom_id).astype(np.uint8)
    else:
        # Legacy: Use instance-level segmentation (less accurate)
        if viewer_robot == 'robot0':
            mask_robot_id = SEGMENTATION_IDS['robot1']
        else:
            mask_robot_id = SEGMENTATION_IDS['robot0']
        robot_mask = (seg_squeezed == mask_robot_id).astype(np.uint8)
    
    # Dilate the mask to ensure complete coverage and smooth edges
    kernel = np.ones((5, 5), np.uint8)
    robot_mask = cv2.dilate(robot_mask, kernel, iterations=dilation_iterations)
    
    # Apply the masking
    masked_image = apply_mask(image, robot_mask, fill_method)
    
    return masked_image


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    fill_method: str = 'inpaint'
) -> np.ndarray:
    """
    Apply a mask to an image using the specified fill method.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask where non-zero pixels should be filled (H, W)
        fill_method: How to fill masked regions
        
    Returns:
        Image with masked regions filled
    """
    result = image.copy()
    
    if fill_method == 'black':
        result[mask > 0] = 0
        
    elif fill_method == 'gray':
        result[mask > 0] = 128
        
    elif fill_method == 'blur':
        # Apply heavy blur to masked regions
        blurred = cv2.GaussianBlur(result, (31, 31), 0)
        result[mask > 0] = blurred[mask > 0]
        
    elif fill_method == 'inpaint':
        # Use OpenCV inpainting to fill masked regions naturally
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        result = cv2.inpaint(result, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
    elif fill_method == 'background':
        # Fill with estimated background color (table color)
        # This is a simple approximation - uses median color of unmasked regions
        unmasked_pixels = result[mask == 0]
        if len(unmasked_pixels) > 0:
            bg_color = np.median(unmasked_pixels, axis=0).astype(np.uint8)
            result[mask > 0] = bg_color
    
    return result


def process_observation_for_robot(
    obs: Dict,
    robot_id: str,
    fill_method: str = 'inpaint'
) -> Dict:
    """
    Process an observation dictionary to mask out the other robot.
    
    This creates a modified observation where the viewing robot cannot see
    the other robot's body, but can still see the objects (peg and hole).
    
    Args:
        obs: Observation dictionary from robosuite
        robot_id: Which robot is viewing ('robot0' or 'robot1')
        fill_method: How to fill masked regions
        
    Returns:
        Modified observation dictionary with masked images
    """
    result = obs.copy()
    
    # Process agentview image
    if 'agentview_image' in obs and 'agentview_segmentation_instance' in obs:
        result['agentview_image'] = mask_other_robot(
            obs['agentview_image'],
            obs['agentview_segmentation_instance'],
            robot_id,
            fill_method
        )
    
    # Process wrist camera images
    wrist_key = f'{robot_id}_eye_in_hand_image'
    seg_key = f'{robot_id}_eye_in_hand_segmentation_instance'
    
    if wrist_key in obs and seg_key in obs:
        result[wrist_key] = mask_other_robot(
            obs[wrist_key],
            obs[seg_key],
            robot_id,
            fill_method
        )
    
    return result


def create_masked_observations(
    obs: Dict,
    fill_method: str = 'inpaint'
) -> Tuple[Dict, Dict]:
    """
    Create separate masked observations for both robots.
    
    Args:
        obs: Original observation dictionary from robosuite
        fill_method: How to fill masked regions
        
    Returns:
        Tuple of (obs_robot0, obs_robot1) with masked images
    """
    obs_robot0 = process_observation_for_robot(obs, 'robot0', fill_method)
    obs_robot1 = process_observation_for_robot(obs, 'robot1', fill_method)
    
    return obs_robot0, obs_robot1


def visualize_segmentation(
    segmentation: np.ndarray,
    save_path: Optional[str] = None,
    use_element_segmentation: bool = True
) -> np.ndarray:
    """
    Create a color visualization of the segmentation mask.
    
    Args:
        segmentation: Segmentation array (element or instance level)
        save_path: Optional path to save the visualization
        use_element_segmentation: If True, color by element-level geom IDs
        
    Returns:
        Color visualization image
    """
    seg = segmentation.squeeze()
    
    vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
    
    if use_element_segmentation:
        # Color by element-level geom IDs
        # Background/floor - Gray
        vis[seg == 0] = [100, 100, 100]
        vis[seg == 5] = [80, 80, 80]  # wall
        
        # Robot0 (all parts including pedestal) - Red
        for geom_id in ROBOT0_GEOM_IDS:
            vis[seg == geom_id] = [255, 0, 0]
        
        # Robot1 (all parts including pedestal) - Blue
        for geom_id in ROBOT1_GEOM_IDS:
            vis[seg == geom_id] = [0, 0, 255]
        
        # Peg - Green
        for geom_id in PEG_GEOM_IDS:
            vis[seg == geom_id] = [0, 255, 0]
        
        # Hole - Yellow
        for geom_id in HOLE_GEOM_IDS:
            vis[seg == geom_id] = [255, 255, 0]
    else:
        # Legacy instance-level coloring
        unique_ids = np.unique(seg)
        colors = {
            SEGMENTATION_IDS['background']: [100, 100, 100],
            SEGMENTATION_IDS['robot0']: [255, 0, 0],
            SEGMENTATION_IDS['robot1']: [0, 0, 255],
            SEGMENTATION_IDS['peg']: [0, 255, 0],
            SEGMENTATION_IDS['hole']: [255, 255, 0],
        }
        for seg_id in unique_ids:
            color = colors.get(seg_id, [128, 128, 128])
            vis[seg == seg_id] = color
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return vis


def create_comparison_frame(
    original: np.ndarray,
    masked_robot0: np.ndarray,
    masked_robot1: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    use_element_segmentation: bool = True
) -> np.ndarray:
    """
    Create a side-by-side comparison frame for visualization.
    
    Args:
        original: Original image
        masked_robot0: Image with robot1 masked (robot0's view)
        masked_robot1: Image with robot0 masked (robot1's view)
        segmentation: Optional segmentation for visualization
        use_element_segmentation: If True, use element-level coloring
        
    Returns:
        Combined comparison image
    """
    h, w = original.shape[:2]
    
    # Add labels
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img_copy
    
    original_labeled = add_label(original, "Original")
    robot0_labeled = add_label(masked_robot0, "Robot0 View (R1 masked)")
    robot1_labeled = add_label(masked_robot1, "Robot1 View (R0 masked)")
    
    if segmentation is not None:
        seg_vis = visualize_segmentation(segmentation, use_element_segmentation=use_element_segmentation)
        seg_labeled = add_label(seg_vis, "Segmentation")
        top_row = np.hstack([original_labeled, seg_labeled])
        bottom_row = np.hstack([robot0_labeled, robot1_labeled])
        combined = np.vstack([top_row, bottom_row])
    else:
        combined = np.hstack([original_labeled, robot0_labeled, robot1_labeled])
    
    return combined
