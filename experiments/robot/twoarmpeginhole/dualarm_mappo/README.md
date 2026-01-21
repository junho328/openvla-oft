# Dual-Arm MAPPO for TwoArmPegInHole

Multi-Agent PPO (MAPPO) training using a **bimanual VLA model** (ALOHA-style) for the TwoArmPegInHole task.

## Overview

This module extends the single-arm MAPPO implementation to use a bimanual (dual-arm) VLA model where:
- **Action dimension**: 14 (7-dim per arm)
- **Proprio dimension**: 14 (7-dim per arm)
- **Agent 0** (left arm): Uses `action[:7]`, sees `proprio[:7]` (padded to 14-dim)
- **Agent 1** (right arm): Uses `action[7:]`, sees `proprio[7:]` (padded to 14-dim)

## Key Differences from Single-Arm MAPPO

| Feature | Single-Arm MAPPO | Dual-Arm MAPPO |
|---------|-----------------|----------------|
| Model Action Dim | 6 | 14 |
| Model Proprio Dim | 8 | 14 |
| Per-Agent Action | 6-dim | 7-dim (split from 14) |
| Images per Agent | 2 (front + wrist) | 3 (agentview + left_wrist + right_wrist) |
| Wrist Padding | None | Other agent's wrist padded with zeros |
| Proprio Padding | None | Other agent's 7-dim padded with zeros |
| Normalization Type | BOUNDS_Q99 | BOUNDS (ALOHA default) |

## Architecture

```
VLA Backbone (Bimanual) 
    ├── Action Head → 14-dim Action → Split
    │                                   ├── Agent 0: action[:7]
    │                                   └── Agent 1: action[7:]
    │
    └── Value Heads → Per-agent Values
                       ├── V^(0) for Agent 0
                       └── V^(1) for Agent 1
```

## Observation Structure

### Per-Agent Images (with padding)

**Agent 0 (Left Arm):**
- `agentview` (real)
- `left_wrist` (real)
- `right_wrist` (zeros - padded)

**Agent 1 (Right Arm):**
- `agentview` (real)
- `left_wrist` (zeros - padded)
- `right_wrist` (real)

### Per-Agent Proprio (14-dim with padding)

**Agent 0:**
```
[proprio_left(7), zeros(7)] → 14-dim
```

**Agent 1:**
```
[zeros(7), proprio_right(7)] → 14-dim
```

## Usage

### Single GPU Training

```bash
cd experiments/robot/twoarmpeginhole/dualarm_mappo

# Using the run script
CHECKPOINT=/path/to/aloha/checkpoint ./run_train.sh

# Or directly with Python
python -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \
    --pretrained_checkpoint /path/to/aloha/checkpoint \
    --run_id_note my_experiment \
    --total_timesteps 1000000
```

### Multi-GPU Training (DDP)

```bash
cd experiments/robot/twoarmpeginhole/dualarm_mappo

# Using the run script
NUM_GPUS=4 CHECKPOINT=/path/to/aloha/checkpoint ./run_train_multigpu.sh

# Or directly with torchrun
torchrun --nproc-per-node 4 \
    -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \
    --pretrained_checkpoint /path/to/aloha/checkpoint \
    --run_id_note my_ddp_experiment
```

## Configuration

Key configuration parameters in `config.py`:

```python
DUALARM_MAPPO_CONSTANTS = {
    "NUM_AGENTS": 2,
    "NUM_ACTIONS_CHUNK": 2,
    "MODEL_ACTION_DIM": 14,        # Full bimanual action output from model
    "AGENT_ACTION_DIM": 7,         # Per-agent action (7-dim)
    "ENV_ACTION_DIM": 6,           # TwoArm env uses 6-dim per arm (no gripper)
    "MODEL_PROPRIO_DIM": 14,       # Full bimanual proprio input to model
    "AGENT_PROPRIO_DIM": 7,        # Per-agent proprio (7 joint positions for Panda)
    "HISTORY_LENGTH": 2,
    "NUM_IMAGES_PER_AGENT": 3,     # agentview + left_wrist + right_wrist
}
```

## File Structure

```
dualarm_mappo/
├── __init__.py              # Module exports
├── config.py                # Configuration and constants
├── observation_utils.py     # Dual-arm observation handling with padding
├── vla_policy.py           # Bimanual VLA policy wrapper
├── rollout_buffer.py       # Experience buffer for dual-arm
├── value_network.py        # Centralized value network
├── train_mappo.py          # Main training script
├── run_train.sh            # Single-GPU run script
├── run_train_multigpu.sh   # Multi-GPU run script
└── README.md               # This file
```

## Action Flow (Open-Loop Execution)

This implementation uses **open-loop action chunk execution**, matching LIBERO/ALOHA evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Query Model (once per action chunk)                       │
│    - Input: observation (14-dim padded proprio, 3 images)    │
│    - Output: action chunk [num_actions_chunk × 14-dim]       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Execute Action Chunk (open-loop)                          │
│    For each action in chunk:                                 │
│    - Split per agent: action[:7] → Agent 0, action[7:] → 1   │
│    - Unnormalize using ALOHA bounds                          │
│    - Take first 6-dim (no gripper) → env.step()              │
│    - Accumulate rewards, update observation                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Store Single Transition per Chunk                         │
│    - Observation: at chunk start                             │
│    - Action: full chunk (for policy evaluation)              │
│    - Reward: aggregated over chunk                           │
│    - Done: final state after chunk execution                 │
└─────────────────────────────────────────────────────────────┘
```

### Comparison with Close-Loop

| Method | Query Frequency | Reward Handling | Temporal Consistency |
|--------|-----------------|-----------------|----------------------|
| **Open-Loop** (current) | Once per chunk | Aggregated | Preserved |
| Close-Loop | Every step | Per-step | Lost |

## Notes

- The TwoArmPegInHole environment uses **6-dim actions** per arm (no gripper)
- The bimanual model outputs **7-dim actions** per arm
- We use the **first 6 dimensions** and ignore the last dimension for environment actions
- TwoArmPegInHole provides **7-dim joint positions** per arm (`robot{i}_joint_pos`) because Panda has 7 DoF
- The 7-dim proprio from each arm matches the model's expected input format
- ALOHA uses `bounds` normalization (not `bounds_q99` like LIBERO)
- Default `num_actions_chunk = 2` (can be increased for longer open-loop execution)
