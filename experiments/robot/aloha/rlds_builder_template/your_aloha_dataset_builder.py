"""
RLDS Dataset Builder for custom ALOHA dataset.

This is a template for converting preprocessed ALOHA HDF5 data to RLDS format.
Modify the paths and dataset name as needed.

Setup:
1. Clone rlds_dataset_builder repo:
   git clone https://github.com/moojink/rlds_dataset_builder.git
   cd rlds_dataset_builder

2. Copy this file to the repo and modify:
   - DATASET_NAME
   - DATA_PATHS
   - LANGUAGE_INSTRUCTION

3. Run the builder:
   cd rlds_dataset_builder
   tfds build --data_dir /path/to/output/rlds/
"""

import glob
import os
from typing import Any, Dict, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds


# ============== MODIFY THESE VALUES ==============
DATASET_NAME = "your_aloha_dataset"  # Change this to your dataset name
DATA_PATHS = {
    "train": "/path/to/preprocessed/your_task_name/train/",
    "val": "/path/to/preprocessed/your_task_name/val/",
}
LANGUAGE_INSTRUCTION = "your task description here"  # e.g., "pick up the red pepper and put it into the pot"
# =================================================


class YourAlohaDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for custom ALOHA dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Third-person camera (cam_high) RGB observation.",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Left wrist camera RGB observation.",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Right wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Robot joint positions (14-dim: 7 left arm + 7 right arm including grippers).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Robot action (14-dim joint positions).",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, default to 1 on last step.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original HDF5 file.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(DATA_PATHS["train"]),
            "val": self._generate_examples(DATA_PATHS["val"]),
        }

    def _generate_examples(self, data_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Generator of examples for each split."""
        # Get all HDF5 files in the directory
        hdf5_files = sorted(glob.glob(os.path.join(data_path, "*.hdf5")))
        
        for episode_idx, hdf5_path in enumerate(hdf5_files):
            with h5py.File(hdf5_path, "r") as f:
                # Load data
                qpos = f["/observations/qpos"][()]
                action = f["/action"][()]
                
                # Load images
                cam_high = f["/observations/images/cam_high"][()]
                cam_left_wrist = f["/observations/images/cam_left_wrist"][()]
                cam_right_wrist = f["/observations/images/cam_right_wrist"][()]
                
                episode_len = len(qpos)
                
                # Create episode data
                episode = []
                for i in range(episode_len):
                    step = {
                        "observation": {
                            "image": cam_high[i],
                            "left_wrist_image": cam_left_wrist[i],
                            "right_wrist_image": cam_right_wrist[i],
                            "state": qpos[i].astype(np.float32),
                        },
                        "action": action[i].astype(np.float32),
                        "discount": 1.0,
                        "reward": 1.0 if i == episode_len - 1 else 0.0,
                        "is_first": i == 0,
                        "is_last": i == episode_len - 1,
                        "is_terminal": i == episode_len - 1,
                        "language_instruction": LANGUAGE_INSTRUCTION,
                    }
                    episode.append(step)
                
                # Yield episode
                sample = {
                    "steps": episode,
                    "episode_metadata": {
                        "file_path": hdf5_path,
                    },
                }
                
                yield f"episode_{episode_idx}", sample
