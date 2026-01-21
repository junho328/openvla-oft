"""
Converts LeRobot v2.0 format ALOHA dataset to HDF5 format compatible with OpenVLA-OFT preprocessing pipeline.

LeRobot v2.0 format:
    - data/train-{episode_index:05d}-of-{total_episodes:05d}.parquet
    - videos/{video_key}_episode_{episode_index:06d}.mp4
    - meta/info.json

Output HDF5 format (compatible with preprocess_split_aloha_data.py):
    /PATH/TO/OUTPUT/dataset_name/
        - episode_0.hdf5
        - episode_1.hdf5
        - ...

Example usage:
    python experiments/robot/aloha/convert_lerobot_to_hdf5.py \
        --lerobot_path /path/to/lerobot/dataset/ \
        --output_path /path/to/output/hdf5/dataset/ \
        --task_name "your_task_description"

Note: Requires PyAV for AV1 video decoding: pip install av
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import av (PyAV) for better video codec support (especially AV1)
try:
    import av
    USE_PYAV = True
    print("Using PyAV for video decoding (AV1 supported)")
except ImportError:
    import cv2
    USE_PYAV = False
    print("Warning: PyAV not found, falling back to OpenCV (AV1 may not work)")
    print("Install PyAV with: pip install av")


def load_video_frames_pyav(video_path: str) -> np.ndarray:
    """Load all frames from a video file using PyAV (supports AV1)."""
    frames = []
    try:
        container = av.open(video_path)
        for frame in container.decode(video=0):
            # Convert to RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        container.close()
    except Exception as e:
        print(f"  Error loading video with PyAV: {e}")
        return np.array([])
    return np.array(frames)


def load_video_frames_opencv(video_path: str) -> np.ndarray:
    """Load all frames from a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def load_video_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file."""
    if USE_PYAV:
        return load_video_frames_pyav(video_path)
    else:
        return load_video_frames_opencv(video_path)


def get_episode_data_from_parquet(parquet_path: str) -> dict:
    """Load episode data from parquet file."""
    df = pd.read_parquet(parquet_path)
    return df


def convert_episode(
    episode_idx: int,
    parquet_path: str,
    videos_dir: str,
    output_dir: str,
    video_keys: list,
    is_sim: bool = False,
):
    """Convert a single episode from LeRobot format to HDF5 format."""
    # Load parquet data
    df = pd.read_parquet(parquet_path)
    
    # Extract state and action
    # LeRobot uses "observation.state" and "action" columns
    qpos = np.array(df["observation.state"].tolist())
    action = np.array(df["action"].tolist())
    
    episode_len = len(qpos)
    
    # Create qvel and effort as zeros (LeRobot doesn't have these)
    qvel = np.zeros_like(qpos)
    effort = np.zeros_like(qpos)
    
    # Load video frames for each camera
    image_dict = {}
    
    # Map LeRobot video keys to ALOHA camera names
    # NOTE: Only include the 3 cameras that preprocess_split_aloha_data.py expects:
    # cam_high, cam_left_wrist, cam_right_wrist (cam_low is excluded)
    camera_mapping = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist", 
        "observation.images.cam_right_wrist": "cam_right_wrist",
        # "observation.images.cam_low": "cam_low",  # Excluded - not supported by preprocess script
    }
    
    for lerobot_key in video_keys:
        if lerobot_key in camera_mapping:
            camera_name = camera_mapping[lerobot_key]
            video_path = os.path.join(
                videos_dir, 
                f"{lerobot_key}_episode_{episode_idx:06d}.mp4"
            )
            
            if os.path.exists(video_path):
                print(f"  Loading video: {video_path}")
                frames = load_video_frames(video_path)
                
                # Check if video loading failed
                if len(frames) == 0:
                    print(f"  ERROR: Failed to load video (0 frames): {video_path}")
                    print(f"  Make sure PyAV is installed: pip install av")
                    raise RuntimeError(f"Failed to decode video: {video_path}")
                
                # Ensure frame count matches episode length
                if len(frames) != episode_len:
                    print(f"  Warning: Video has {len(frames)} frames but episode has {episode_len} steps")
                    # Truncate or pad as needed
                    if len(frames) > episode_len:
                        frames = frames[:episode_len]
                    else:
                        # Repeat last frame if needed
                        padding = np.repeat(frames[-1:], episode_len - len(frames), axis=0)
                        frames = np.concatenate([frames, padding], axis=0)
                
                image_dict[camera_name] = frames
            else:
                print(f"  Warning: Video not found: {video_path}")
    
    # Save to HDF5
    output_path = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")
    
    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = is_sim
        
        # Create observations group
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        obs.create_dataset("effort", data=effort)
        
        # Create images group
        images = obs.create_group("images")
        for cam_name, frames in image_dict.items():
            H, W, C = frames[0].shape
            images.create_dataset(
                cam_name,
                data=frames,
                dtype="uint8",
                chunks=(1, H, W, C),
            )
        
        # Save action
        root.create_dataset("action", data=action)
        
        # Compute and save relative actions
        relative_actions = np.zeros_like(action)
        relative_actions[:-1] = action[1:] - action[:-1]
        relative_actions[-1] = relative_actions[-2] if episode_len > 1 else 0
        root.create_dataset("relative_action", data=relative_actions)
    
    print(f"  Saved: {output_path}")
    return output_path


def main(args):
    # Load LeRobot metadata
    meta_path = os.path.join(args.lerobot_path, "meta", "info.json")
    if not os.path.exists(meta_path):
        # Try alternative location
        meta_path = os.path.join(args.lerobot_path, "info.json")
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        total_episodes = meta.get("total_episodes", None)
        video_keys = meta.get("video_keys", [])
        print(f"Loaded metadata: {total_episodes} episodes, video keys: {video_keys}")
    else:
        print(f"Warning: Metadata file not found at {meta_path}")
        print("Attempting to auto-detect episodes...")
        total_episodes = None
        video_keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Find all parquet files
    data_dir = os.path.join(args.lerobot_path, "data")
    videos_dir = os.path.join(args.lerobot_path, "videos")
    
    # Get list of parquet files
    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    
    if total_episodes is None:
        total_episodes = len(parquet_files)
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Videos directory: {videos_dir}")
    print(f"Output directory: {args.output_path}")
    
    # Convert each episode
    for episode_idx in tqdm(range(total_episodes), desc="Converting episodes"):
        # Find the parquet file for this episode
        # LeRobot format: train-{episode_index:05d}-of-{total_episodes:05d}.parquet
        parquet_pattern = f"train-{episode_idx:05d}-of-{total_episodes:05d}.parquet"
        parquet_path = os.path.join(data_dir, parquet_pattern)
        
        if not os.path.exists(parquet_path):
            # Try alternative patterns
            possible_patterns = [
                f"episode_{episode_idx}.parquet",
                f"train-{episode_idx:05d}-*.parquet",
            ]
            found = False
            for pattern in possible_patterns:
                matches = list(Path(data_dir).glob(pattern))
                if matches:
                    parquet_path = str(matches[0])
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find parquet file for episode {episode_idx}")
                continue
        
        print(f"\nProcessing episode {episode_idx}: {parquet_path}")
        convert_episode(
            episode_idx=episode_idx,
            parquet_path=parquet_path,
            videos_dir=videos_dir,
            output_dir=args.output_path,
            video_keys=video_keys,
            is_sim=args.is_sim,
        )
    
    print(f"\nConversion complete! Output saved to: {args.output_path}")
    print(f"\nNext steps:")
    print(f"1. Run preprocess_split_aloha_data.py to resize images and split train/val:")
    print(f"   python experiments/robot/aloha/preprocess_split_aloha_data.py \\")
    print(f"     --dataset_path {args.output_path} \\")
    print(f"     --out_base_dir /path/to/preprocessed/output/ \\")
    print(f"     --percent_val 0.05")
    print(f"\n2. Convert to RLDS format using rlds_dataset_builder")
    print(f"\n3. Register dataset in configs.py, transforms.py, and mixtures.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2.0 ALOHA dataset to HDF5 format"
    )
    parser.add_argument(
        "--lerobot_path",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory (containing data/, videos/, meta/ folders)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output HDF5 dataset directory",
    )
    parser.add_argument(
        "--is_sim",
        action="store_true",
        default=False,
        help="Set to True if this is simulation data",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="aloha_task",
        help="Task name/description for the dataset",
    )
    args = parser.parse_args()
    
    main(args)
