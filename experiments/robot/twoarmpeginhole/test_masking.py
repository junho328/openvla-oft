"""
test_masking.py

Test script to demonstrate robot body masking for perception confusion testing.
Renders a video showing original view vs masked views for each robot.
"""

import os
import sys

# Set headless rendering BEFORE importing anything else
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import imageio
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from experiments.robot.twoarmpeginhole.perception_masking import (
    mask_other_robot,
    create_comparison_frame,
    visualize_segmentation,
    SEGMENTATION_IDS,
)


def create_environment():
    """Create the TwoArmPegInHole environment with segmentation enabled."""
    controller_config = load_composite_controller_config(controller='BASIC')
    body_parts = controller_config.get('body_parts', {})
    if 'right' in body_parts:
        controller_config['body_parts'] = {'right': body_parts['right']}
    elif 'left' in body_parts:
        controller_config['body_parts'] = {'left': body_parts['left']}
    
    env = suite.make(
        'TwoArmPegInHole',
        robots=['Panda', 'Panda'],
        controller_configs=controller_config,
        env_configuration='opposed',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand'],
        camera_heights=256,
        camera_widths=256,
        camera_segmentations='element',  # Element-level for complete robot masking
    )
    
    return env


def test_segmentation_mapping(env):
    """Test and display the segmentation ID mapping."""
    obs = env.reset()
    
    seg = obs['agentview_segmentation_element'].squeeze()
    unique_ids = np.unique(seg)
    
    print("\n" + "="*60)
    print("ELEMENT-LEVEL SEGMENTATION TEST")
    print("="*60)
    print(f"Total unique geom IDs visible: {len(unique_ids)}")
    
    # Import the geom ID lists
    from experiments.robot.twoarmpeginhole.perception_masking import (
        ROBOT0_GEOM_IDS, ROBOT1_GEOM_IDS, PEG_GEOM_IDS, HOLE_GEOM_IDS
    )
    
    r0_visible = len([g for g in ROBOT0_GEOM_IDS if g in unique_ids])
    r1_visible = len([g for g in ROBOT1_GEOM_IDS if g in unique_ids])
    peg_visible = len([g for g in PEG_GEOM_IDS if g in unique_ids])
    hole_visible = len([g for g in HOLE_GEOM_IDS if g in unique_ids])
    
    print(f"Robot0 parts visible: {r0_visible}/{len(ROBOT0_GEOM_IDS)}")
    print(f"Robot1 parts visible: {r1_visible}/{len(ROBOT1_GEOM_IDS)}")
    print(f"Peg parts visible: {peg_visible}/{len(PEG_GEOM_IDS)}")
    print(f"Hole parts visible: {hole_visible}/{len(HOLE_GEOM_IDS)}")
    print("="*60 + "\n")
    
    return obs


def generate_test_video(env, num_frames=100, output_dir='./test_outputs'):
    """
    Generate a test video demonstrating the masking effect.
    
    Args:
        env: Robosuite environment
        num_frames: Number of frames to record
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f'masking_demo_{timestamp}.mp4')
    
    print(f"Generating test video with {num_frames} frames...")
    print(f"Output will be saved to: {video_path}")
    
    obs = env.reset()
    frames = []
    
    # Generate random actions to create some motion
    action_dim = env.action_dim
    
    for i in range(num_frames):
        # Random small actions to create motion
        action = np.random.uniform(-0.1, 0.1, size=action_dim)
        
        # Get observation
        img = obs['agentview_image'][::-1].copy()  # Flip vertically (robosuite convention)
        seg = obs['agentview_segmentation_element'][::-1].copy()  # Element-level segmentation
        
        # Ensure uint8 format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Create masked versions (using element-level segmentation for complete coverage)
        masked_for_robot0 = mask_other_robot(img, seg, 'robot0', fill_method='inpaint', use_element_segmentation=True)
        masked_for_robot1 = mask_other_robot(img, seg, 'robot1', fill_method='inpaint', use_element_segmentation=True)
        
        # Create comparison frame
        comparison = create_comparison_frame(img, masked_for_robot0, masked_for_robot1, seg)
        
        frames.append(comparison)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()
        
        if (i + 1) % 20 == 0:
            print(f"  Frame {i + 1}/{num_frames} completed")
    
    # Save video
    print(f"Saving video to {video_path}...")
    writer = imageio.get_writer(video_path, fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"Video saved successfully!")
    print(f"Video dimensions: {frames[0].shape}")
    
    # Also save a sample frame as PNG
    sample_frame_path = os.path.join(output_dir, f'sample_frame_{timestamp}.png')
    imageio.imwrite(sample_frame_path, frames[0])
    print(f"Sample frame saved to: {sample_frame_path}")
    
    return video_path, sample_frame_path


def main():
    print("="*60)
    print("PERCEPTION MASKING TEST")
    print("Testing robot body masking for perception confusion hypothesis")
    print("="*60)
    
    # Create environment
    print("\nCreating TwoArmPegInHole environment...")
    env = create_environment()
    print("Environment created successfully!")
    
    # Test segmentation mapping
    test_segmentation_mapping(env)
    
    # Generate test video
    video_path, sample_path = generate_test_video(env, num_frames=100)
    
    # Cleanup
    env.close()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Video: {os.path.abspath(video_path)}")
    print(f"Sample frame: {os.path.abspath(sample_path)}")
    print("\nTo download the video, use the file browser or:")
    print(f"  scp <server>:{os.path.abspath(video_path)} .")
    print("="*60)


if __name__ == "__main__":
    main()
