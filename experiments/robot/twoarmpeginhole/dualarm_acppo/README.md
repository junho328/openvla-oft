# Dual-Arm ACPPO: Multi-Agent RL with Agent-Chained PPO using Bimanual VLA

This module implements **Dual-Arm ACPPO** (Agent-Chained Proximal Policy Optimization) for the TwoArmPegInHole task using bimanual VLA models (ALOHA-style).

## Key Features

### Bimanual Model Adaptations
- **Action Dimension**: VLA outputs 14-dim action (7 per arm)
  - Agent 0 uses `action[:7]` (left arm)
  - Agent 1 uses `action[7:]` (right arm)
- **Proprio Dimension**: 14-dim total (7 per arm from joint positions)
  - Each agent receives padded proprio: own 7-dim + zeros
- **Image Input**: 3 images per agent
  - Agentview (shared)
  - Own wrist camera (real)
  - Other agent's wrist camera (zero-padded)

### ACPPO-Specific Features
1. **Simultaneous Action**: All agents act at the same time step
2. **Action Distribution Chaining**: Agent 1 receives estimated action distribution from Agent 0
3. **No Gradient Flow**: Action distribution estimation is detached (no backprop)
4. **Per-Agent Value Heads**: Separate value functions for each agent
5. **Microstep GAE (Optional)**: Original ACPPO advantage computation

## Architecture

```
Agent 0:
    [agentview, left_wrist, zero_pad] + padded_proprio(14) 
    → VLA → Action Head → 14-dim Action → Split → 7-dim Action_0 (mu_0, sigma_0)
    → Value Head[0] → V^(0)

Agent 1 (with estimated action dist from Agent 0):
    Step 1 (Estimation, no grad):
        [agentview, left_wrist, zero_pad] + Agent0_proprio 
        → VLA → Action Head → (mu_0_est, sigma_0_est)
    
    Step 2 (Forward with estimated action dist):
        [agentview, zero_pad, right_wrist] + [padded_proprio(14); mu_0_est; sigma_0_est]
        → VLA → Action Head → 14-dim Action → Split → 7-dim Action_1
        → Value Head[1] → V^(1)
```

## Action Flow

1. VLA model receives padded observation (14-dim proprio, 3 images)
2. VLA outputs a **chunk of 14-dim bimanual actions** (e.g., `num_actions_chunk` actions)
3. For each action in the chunk (open-loop execution):
   a. Action is split per agent: `action[:7]` for Agent 0, `action[7:]` for Agent 1
   b. Each 7-dim action is unnormalized using ALOHA bounds (agent-specific stats)
   c. First 6 dimensions are sent to TwoArm environment (no gripper)
4. Rewards within the action chunk are accumulated
5. A single transition is stored in the buffer

## GAE Modes

### 1. Shared Reward
Standard GAE where both agents receive the same team reward:
```
δ_t^(i) = r_t + γ V^(i)(s_{t+1}) - V^(i)(s_t)
A_t^(i) = Σ (γλ)^l δ_{t+l}^(i)
```

### 2. ACPPO Microstep (gae_mode="microstep" or "acppo_microstep")
Original ACPPO with microstep TD residuals:
```
ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])  for i < N
ζ_t^(N) = r_t + γ' V^(1)(s_{t+1}) - V^(N)([s_t, b_t^(N)])        for i = N
```

## Usage

### Single GPU Training
```bash
bash run_train.sh
```

### Multi-GPU Training (DDP)
```bash
bash run_train_multigpu.sh
```

### Python API
```python
from dualarm_acppo import DualArmACPPOConfig, DualArmACPPOTrainer

config = DualArmACPPOConfig(
    pretrained_checkpoint="/path/to/aloha/bimanual/checkpoint",
    use_action_dist_input=True,  # Enable ACPPO action dist chaining
    gae_mode="shared_reward",    # or "acppo_microstep"
    num_actions_chunk=25,
)

trainer = DualArmACPPOTrainer(config)
trainer.train()
```

## Configuration

Key configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_action_dim` | 14 | Full bimanual action dimension |
| `agent_action_dim` | 7 | Per-agent action dimension |
| `model_proprio_dim` | 14 | Full bimanual proprio dimension |
| `num_actions_chunk` | 25 | Action chunk size for open-loop |
| `use_action_dist_input` | True | Enable action dist chaining for Agent 1 |
| `detach_action_dist_grad` | True | No gradient through action dist estimation |
| `gae_mode` | "shared_reward" | GAE computation mode |
| `action_dist_dim` | 350 | 7 * 25 * 2 (mu + sigma) |

## Differences from Single-Arm ACPPO

| Aspect | Single-Arm ACPPO | Dual-Arm ACPPO |
|--------|------------------|----------------|
| Action dim | 6 | 14 (7 per arm) |
| Proprio dim | 8 | 14 (7 per arm) |
| Images/agent | 2 | 3 (1 padded) |
| Action dist dim | 12 | 350 (7*25*2) |
| Env action | 6-dim | 12-dim (6+6) |

## Files

- `config.py` - Configuration and constants
- `observation_utils.py` - Observation preprocessing with wrist padding
- `vla_policy.py` - VLA policy with action dist chaining
- `rollout_buffer.py` - Buffer with action dist storage
- `value_network.py` - Centralized value network
- `train_acppo.py` - Main training script
- `run_train.sh` - Single GPU training script
- `run_train_multigpu.sh` - Multi-GPU training script
