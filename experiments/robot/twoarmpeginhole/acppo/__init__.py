"""
ACPPO (Agent-Chained Proximal Policy Optimization) for Multi-Agent VLA training.

This module implements ACPPO where:
- Agent 0 acts first and its action distribution is estimated
- Agent 1 uses the estimated action distribution as additional input
- Advantage calculation uses microstep-based TD residuals for proper credit assignment
"""

from .config import ACPPOConfig, TWOARM_ACPPO_CONSTANTS
from .vla_policy import MultiAgentVLAPolicyACPPO, load_vla_for_acppo
from .rollout_buffer import MultiAgentRolloutBufferACPPO
from .observation_utils import (
    ObservationHistoryManager,
    extract_observations_from_env,
    prepare_vla_input,
    prepare_vla_input_for_action_dist_estimation,
    unnormalize_action,
)

__all__ = [
    "ACPPOConfig",
    "TWOARM_ACPPO_CONSTANTS",
    "MultiAgentVLAPolicyACPPO",
    "load_vla_for_acppo",
    "MultiAgentRolloutBufferACPPO",
    "ObservationHistoryManager",
    "extract_observations_from_env",
    "prepare_vla_input",
    "prepare_vla_input_for_action_dist_estimation",
    "unnormalize_action",
]
