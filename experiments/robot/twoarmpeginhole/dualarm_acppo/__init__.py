"""
Dual-Arm ACPPO: Multi-Agent RL with Agent-Chained PPO using Bimanual VLA Models.

This package implements ACPPO (Agent-Chained Proximal Policy Optimization) for
dual-arm manipulation tasks using bimanual VLA models (ALOHA-style).

Key differences from single-arm ACPPO:
- VLA outputs 14-dim action (bimanual), split into 7-dim per agent
- Agent 0 uses action[:7], Agent 1 uses action[7:]
- Each agent sees agentview + own wrist view, other agent's wrist padded
- Proprio is 14-dim (7-dim per arm), padded per agent

ACPPO-specific features:
1. All agents act SIMULTANEOUSLY
2. Agent 0: Uses only its own observation
3. Agent 1: Additionally uses ESTIMATED action distribution from Agent 0
4. No gradient flows through the estimation process (detached)

Modules:
    config: Configuration and constants for Dual-Arm ACPPO
    observation_utils: Observation preprocessing utilities
    vla_policy: VLA-based multi-agent policy with action distribution chaining
    rollout_buffer: Multi-agent rollout buffer with per-agent advantages
    value_network: Centralized value network for critics
    train_acppo: Main training loop

Usage:
    from dualarm_acppo import DualArmACPPOConfig, train_acppo
    
    config = DualArmACPPOConfig(
        pretrained_checkpoint="/path/to/aloha/bimanual/checkpoint",
        ...
    )
    train_acppo(config)
"""

from .config import (
    DualArmACPPOConfig,
    DUALARM_ACPPO_CONSTANTS,
    NormalizationType,
    get_config_from_args,
)

from .observation_utils import (
    DualArmObservationHistoryManager,
    extract_observations_from_env,
    extract_bimanual_proprio_from_env,
    prepare_vla_input,
    prepare_vla_input_for_action_dist_estimation,
    unnormalize_action,
    normalize_action,
    normalize_proprio,
    split_bimanual_action,
    combine_agent_actions,
)

from .vla_policy import (
    DualArmVLAAgentACPPO,
    DualArmMultiAgentVLAPolicyACPPO,
    load_vla_for_dualarm_acppo,
    ValueHead,
    ActionDistributionProjector,
    PassThroughProjector,
)

from .rollout_buffer import (
    DualArmRolloutBufferACPPO,
    DualArmRolloutBufferSamplesACPPO,
    SharedRewardWrapper,
    RewardNormalizer,
)

from .value_network import (
    DualArmCentralizedValueNetwork,
    LightweightDualArmCritic,
    create_value_network,
)

__all__ = [
    # Config
    "DualArmACPPOConfig",
    "DUALARM_ACPPO_CONSTANTS",
    "NormalizationType",
    "get_config_from_args",
    # Observation Utils
    "DualArmObservationHistoryManager",
    "extract_observations_from_env",
    "extract_bimanual_proprio_from_env",
    "prepare_vla_input",
    "prepare_vla_input_for_action_dist_estimation",
    "unnormalize_action",
    "normalize_action",
    "normalize_proprio",
    "split_bimanual_action",
    "combine_agent_actions",
    # VLA Policy
    "DualArmVLAAgentACPPO",
    "DualArmMultiAgentVLAPolicyACPPO",
    "load_vla_for_dualarm_acppo",
    "ValueHead",
    "ActionDistributionProjector",
    "PassThroughProjector",
    # Rollout Buffer
    "DualArmRolloutBufferACPPO",
    "DualArmRolloutBufferSamplesACPPO",
    "SharedRewardWrapper",
    "RewardNormalizer",
    # Value Network
    "DualArmCentralizedValueNetwork",
    "LightweightDualArmCritic",
    "create_value_network",
]
