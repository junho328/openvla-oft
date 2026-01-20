# ACPPO (Agent-Chained Proximal Policy Optimization)

ACPPO는 Multi-Agent Reinforcement Learning을 위한 알고리즘으로, MAPPO를 확장하여 Agent-Chained 구조를 도입합니다.

## 핵심 개념

### 1. Agent Chaining (Simultaneous Action)
모든 에이전트가 **동시에(simultaneous)** 행동을 결정합니다. 단, agent idx가 0보다 큰 에이전트들은 이전 에이전트들의 action distribution을 추정하여 추가 입력으로 사용합니다:
- **Agent 0**: 자신의 observation만 보고 행동을 결정
- **Agent 1**: 자신의 observation + **Agent 0의 추정된 action distribution**을 추가 입력으로 받아 행동을 결정

⚠️ **중요**: 에이전트들이 순차적으로 행동하는 것이 아닙니다! 모든 에이전트가 동시에 행동을 결정하고, 환경에 동시에 적용됩니다.

### 2. Action Distribution Estimation
Agent 1이 Agent 0의 action distribution을 추정하는 과정:
1. **이미지 입력**: Front view만 사용 (wrist image는 zero padding)
2. **텍스트 입력**: Robot 0의 instruction 사용 ("You are robot 0...")
3. **출력**: Action distribution의 mu와 sigma (총 48차원: 6 action × 4 chunk × 2)
4. **역전파 차단**: 이 추정 과정에 대해서는 gradient가 흐르지 않음 (detached)

### 3. Extended Proprio for Agent 1
Agent 1의 proprioceptive 입력:
```
proprio_agent1_extended = [proprio_1; mu_0_est; sigma_0_est]
                        = (8 dim) + (24 dim) + (24 dim) = 56 dim
```

### 4. Microstep-based Advantage Calculation
ACPPO의 TD Residual 계산:

**Agent i < N (중간 에이전트):**
```
ζ_t^(i) = γ' V^(i+1)([s_t, b_t^(i+1)]) - V^(i)([s_t, b_t^(i)])
```

**Agent N (마지막 에이전트):**
```
ζ_t^(N) = r_t + γ' V^(1)(s_{t+1}) - V^(N)([s_t, b_t^(N)])
```

**Advantage 계산:**
```
A_t^(i) = Σ_{j=i}^{N} (γ'λ')^{j-i} ζ_t^(j) + Σ_{k=1}^{∞} Σ_{j=1}^{N} (γ'λ')^{kN+j-i} ζ_{t+k}^(j)
```

## 파일 구조

```
acppo/
├── __init__.py                 # 모듈 초기화
├── config.py                   # ACPPO 설정 (ACPPOConfig)
├── vla_policy.py               # VLA Policy with action distribution chaining
├── rollout_buffer.py           # Per-agent values와 action dist 저장
├── train_acppo.py              # 메인 훈련 스크립트
├── observation_utils.py        # 관측값 처리 유틸리티
├── value_network.py            # Per-agent value network
├── run_train.sh                # Single GPU 훈련 스크립트
├── run_train_multigpu.sh       # Multi-GPU 훈련 스크립트
└── README.md                   # 문서
```

## 주요 클래스

### `ACPPOConfig`
ACPPO 훈련을 위한 설정 클래스:
- `gamma_prime`: Microstep discount factor (γ')
- `lambda_prime`: Microstep GAE lambda (λ')
- `action_dist_dim`: Action distribution 차원 (48)
- `use_action_dist_input`: Agent 1에 action dist 입력 사용 여부
- `detach_action_dist_grad`: Action dist 추정 시 gradient 차단 여부

### `MultiAgentVLAPolicyACPPO`
Agent chaining을 지원하는 VLA 정책:
- `estimate_agent0_action_dist()`: Agent 0의 action distribution 추정
- `get_actions_and_values()`: Action과 value를 동시에 계산
- `proprio_projector_extended`: Agent 1을 위한 확장된 proprio projector

### `MultiAgentRolloutBufferACPPO`
Per-agent data를 저장하는 버퍼:
- Per-agent values 저장
- Action distribution (mu, sigma) 저장
- Microstep-based advantage 계산

## 사용법

### Single GPU 훈련
```bash
cd experiments/robot/twoarmpeginhole/acppo
./run_train.sh
```

### Multi-GPU 훈련
```bash
NUM_GPUS=4 ./run_train_multigpu.sh
```

### 커스텀 설정
```bash
LEARNING_RATE=1e-4 \
GAMMA_PRIME=0.99 \
LAMBDA_PRIME=0.95 \
NUM_ACTIONS_CHUNK=4 \
./run_train.sh --run_id_note "custom_exp"
```

## MAPPO와의 차이점

| Feature | MAPPO | ACPPO |
|---------|-------|-------|
| Action Execution | Simultaneous | **Simultaneous** (동일) |
| Agent Chaining (Info Flow) | ❌ | ✅ (Agent 1+가 이전 agent의 action dist 추정) |
| Action Distribution Input | ❌ | ✅ (Agent 1+) |
| Value Function | Centralized | Per-Agent |
| Advantage | Standard GAE | Microstep GAE |
| Proprio Dim (Agent 1) | 8 | 56 (8 + 48) |

> **Note**: 두 알고리즘 모두 에이전트들이 **동시에 행동**합니다. ACPPO의 "chaining"은 행동 순서가 아니라 **정보 흐름**을 의미합니다.

## 이론적 배경

ACPPO는 Agent-Chained BMDP (AC-BMDP)에서 유도된 알고리즘으로, 
multi-agent 환경에서 serialization을 통해 credit assignment 문제를 해결합니다.

자세한 이론은 다음을 참조하세요:
- Bellman operators for AC-BMDP
- TD residuals with microsteps
- Generalized Advantage Estimation (GAE) extension

## 하이퍼파라미터 가이드

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `gamma` | 0.99 | Environment discount factor |
| `gamma_prime` | 0.99 | Microstep discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `lambda_prime` | 0.95 | Microstep GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping |
| `num_actions_chunk` | 4 | Action chunk size |
| `learning_rate` | 3e-4 | Learning rate |
