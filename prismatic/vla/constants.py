"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,#25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# TwoArmPegInHole constants (for MAPPO training)
# ACTION_DIM=6 because TwoArm has no gripper (6-DoF only)
TWOARM_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,   # Reduced chunk size for RL (faster iteration)
    "ACTION_DIM": 6,          # 6-DoF per arm (no gripper, different from LIBERO=7)
    "PROPRIO_DIM": 8,         # EEF pos (3) + axis-angle (3) + padding (2)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# TwoArmPegInHole constants for ACPPO training (larger action chunks)
TWOARM_ACPPO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,   # Larger chunk size for ACPPO
    "ACTION_DIM": 6,          # 6-DoF per arm (no gripper)
    "PROPRIO_DIM": 8,         # EEF pos (3) + axis-angle (3) + padding (2)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    # Check ACPPO first (more specific than TWOARM)
    if "acppo" in cmd_args:
        return "TWOARM_ACPPO"
    elif "twoarm" in cmd_args or "mappo" in cmd_args or "two_arm" in cmd_args:
        return "TWOARM"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "TWOARM_ACPPO":
    constants = TWOARM_ACPPO_CONSTANTS
elif ROBOT_PLATFORM == "TWOARM":
    constants = TWOARM_CONSTANTS
elif ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
