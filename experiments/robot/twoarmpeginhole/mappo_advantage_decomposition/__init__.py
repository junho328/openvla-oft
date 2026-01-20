"""
MAPPO Training Module for Multi-Agent VLA on TwoArmPegInHole environment.

This module provides Multi-Agent Proximal Policy Optimization (MAPPO) training
for two OpenVLA-OFT agents in the TwoArmPegInHole robosuite environment.

Key Components:
- MAPPOConfig: Configuration for training hyperparameters
- MultiAgentVLAPolicy: Multi-agent actor using VLA models
- CentralizedValueNetwork: Centralized critic for MAPPO
- MultiAgentRolloutBuffer: Experience buffer for rollout collection
- ObservationHistoryManager: Manages observation history for temporal context
- MAPPOTrainer: Main trainer class that orchestrates training

Usage:
    python -m experiments.robot.twoarmpeginhole.mappo.train_mappo \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --total_timesteps 1000000 \\
        --use_wandb true
"""

from .config import MAPPOConfig, TWOARM_MAPPO_CONSTANTS
from .vla_policy import MultiAgentVLAPolicy, VLAAgent, ValueHead, load_vla_for_mappo
from .value_network import (
    CentralizedValueNetwork,
    LightweightCentralizedCritic,
    create_value_network,
)
from .rollout_buffer import (
    MultiAgentRolloutBuffer,
    RolloutBufferSamples,
    SharedRewardWrapper,
    RewardNormalizer,
)
from .observation_utils import (
    ObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    normalize_proprio,
    unnormalize_action,
    normalize_action,
)

__all__ = [
    # Configuration
    "MAPPOConfig",
    "TWOARM_MAPPO_CONSTANTS",
    # Policy
    "MultiAgentVLAPolicy",
    "VLAAgent",
    "ValueHead",
    "load_vla_for_mappo",
    # Value Network
    "CentralizedValueNetwork",
    "LightweightCentralizedCritic",
    "create_value_network",
    # Rollout Buffer
    "MultiAgentRolloutBuffer",
    "RolloutBufferSamples",
    "SharedRewardWrapper",
    "RewardNormalizer",
    # Observation Utils
    "ObservationHistoryManager",
    "extract_observations_from_env",
    "prepare_vla_input",
    "normalize_proprio",
    "unnormalize_action",
    "normalize_action",
]
