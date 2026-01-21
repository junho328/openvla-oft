"""
Dual-Arm MAPPO Training Module for TwoArmPegInHole Environment.

This module implements Multi-Agent PPO (MAPPO) using a bimanual VLA model
(ALOHA-style) that outputs 14-dim actions for the TwoArmPegInHole task.

Key Features:
- Bimanual VLA model with 14-dim action output (7-dim per arm)
- Agent 0 uses action[:7], Agent 1 uses action[7:]
- Each agent sees agentview + own wrist + padded other wrist (3 images)
- 14-dim proprio per agent with own 7-dim real + other 7-dim padded
- Shared action head, per-agent value heads
- Multi-GPU support via DDP

Architecture:
    VLA Backbone ─┬─────────→ Action Head → 14-dim Action → Split to 7-dim per agent
                  └─────────→ Value Heads → Per-agent Values (V^(0), V^(1))

Usage:
    # Single GPU
    python -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \\
        --pretrained_checkpoint /path/to/aloha/checkpoint
    
    # Multi-GPU
    torchrun --nproc-per-node 4 -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \\
        --pretrained_checkpoint /path/to/aloha/checkpoint
"""

from .config import DualArmMAPPOConfig, DUALARM_MAPPO_CONSTANTS
from .observation_utils import (
    DualArmObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    unnormalize_action,
    split_bimanual_action,
    combine_agent_actions,
)
from .vla_policy import (
    DualArmVLAAgent,
    DualArmMultiAgentVLAPolicy,
    load_vla_for_dualarm_mappo,
)
from .rollout_buffer import (
    DualArmRolloutBuffer,
    DualArmRolloutBufferSamples,
    SharedRewardWrapper,
    RewardNormalizer,
)
from .value_network import (
    DualArmCentralizedValueNetwork,
    LightweightDualArmCritic,
    create_value_network,
)
from .train_mappo import DualArmMAPPOTrainer

__all__ = [
    # Config
    "DualArmMAPPOConfig",
    "DUALARM_MAPPO_CONSTANTS",
    # Observation utilities
    "DualArmObservationHistoryManager",
    "extract_observations_from_env",
    "prepare_vla_input",
    "unnormalize_action",
    "split_bimanual_action",
    "combine_agent_actions",
    # VLA Policy
    "DualArmVLAAgent",
    "DualArmMultiAgentVLAPolicy",
    "load_vla_for_dualarm_mappo",
    # Rollout Buffer
    "DualArmRolloutBuffer",
    "DualArmRolloutBufferSamples",
    "SharedRewardWrapper",
    "RewardNormalizer",
    # Value Network
    "DualArmCentralizedValueNetwork",
    "LightweightDualArmCritic",
    "create_value_network",
    # Trainer
    "DualArmMAPPOTrainer",
]
