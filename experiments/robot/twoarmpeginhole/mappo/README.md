# MAPPO Training for Multi-Agent VLA

이 모듈은 **TwoArmPegInHole** 환경에서 두 개의 OpenVLA-OFT 에이전트를 훈련하기 위한 **Multi-Agent Proximal Policy Optimization (MAPPO)** 알고리즘을 구현합니다.

## 개요

### Fine-tuning과의 차이점

| 특성 | Fine-tuning | MAPPO RL |
|------|-------------|----------|
| 학습 방식 | Supervised (데모 데이터) | Reinforcement Learning |
| 입력 이미지 | 현재 프레임만 | 현재 + 이전 프레임 (히스토리) |
| Action Chunk Size | 8 (기본값) | 4 (감소) |
| 보상 | 없음 (데모 따라하기) | 환경에서 자동 제공 |
| 에이전트 수 | 단일 모델 | 두 에이전트 (팔당 하나) |

### 핵심 특징

1. **Centralized Training with Decentralized Execution (CTDE)**
   - 중앙집중식 Value Network (두 에이전트 정보 모두 사용)
   - 분산 실행 Actor (각 에이전트는 자신의 관측만 사용)

2. **관측 히스토리**
   - 각 에이전트 입력: `[front_t, wrist_t, front_{t-1}, wrist_{t-1}]`
   - 시간적 맥락을 위한 이전 프레임 포함

3. **공유 정책 (선택적)**
   - 두 에이전트가 같은 VLA 모델 가중치 공유 가능
   - 메모리 효율적이고 학습 속도 향상

## 파일 구조

```
mappo/
├── __init__.py           # 모듈 초기화
├── config.py             # MAPPO 설정 및 상수
├── observation_utils.py  # 관측 히스토리 관리
├── vla_policy.py         # Multi-Agent VLA 정책
├── value_network.py      # 중앙집중식 Value Network
├── rollout_buffer.py     # 경험 버퍼
├── train_mappo.py        # 메인 훈련 스크립트
├── run_train.sh          # 훈련 실행 쉘 스크립트
└── README.md             # 이 문서
```

## 설치

기존 openvla-oft 환경에서 추가 의존성:

```bash
pip install tensorboard
```

## 사용법

### Checkpoint 선택

**두 가지 옵션이 있습니다:**

1. **Fine-tuned checkpoint (권장)**: LIBERO에서 훈련된 OpenVLA-OFT 사용
   - `proprio_projector`와 `action_head`가 이미 학습되어 있음
   - Transfer learning으로 더 빠른 수렴 기대
   ```
   moojink/openvla-7b-oft-finetuned-libero-spatial  (default)
   moojink/openvla-7b-oft-finetuned-libero-object
   moojink/openvla-7b-oft-finetuned-libero-goal
   moojink/openvla-7b-oft-finetuned-libero-10
   ```

2. **Base VLA**: 처음부터 RL 훈련
   - `proprio_projector`와 `action_head`를 새로 초기화
   ```
   openvla/openvla-7b
   ```

### 기본 훈련

```bash
cd /path/to/openvla-oft

# 방법 1: 쉘 스크립트 사용 (권장: fine-tuned checkpoint 기본 사용)
./experiments/robot/twoarmpeginhole/mappo/run_train.sh

# 또는 특정 checkpoint 지정
./experiments/robot/twoarmpeginhole/mappo/run_train.sh moojink/openvla-7b-oft-finetuned-libero-10

# 방법 2: Python 직접 실행
python -m experiments.robot.twoarmpeginhole.mappo.train_mappo \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --total_timesteps 1000000 \
    --use_wandb true
```

### 주요 설정 옵션

```bash
python -m experiments.robot.twoarmpeginhole.mappo.train_mappo \
    # 모델 설정 (fine-tuned checkpoint 권장)
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --use_l1_regression true \
    --use_proprio true \
    --history_length 2 \
    --num_actions_chunk 2 \
    
    # MAPPO 하이퍼파라미터
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_epsilon 0.2 \
    --entropy_coef 0.01 \
    --actor_lr 1e-4 \
    --critic_lr 5e-4 \
    
    # 환경 설정
    --reward_shaping true \
    --max_episode_steps 300 \
    
    # 훈련 설정
    --total_timesteps 1000000 \
    --num_steps_per_rollout 256 \
    --num_epochs 4 \
    --num_minibatches 4 \
    
    # 로깅
    --use_wandb true \
    --wandb_entity your-entity \
    --wandb_project mappo-twoarm
```

## 알고리즘 세부사항

### MAPPO 업데이트

1. **Rollout Collection**: 각 에이전트가 환경과 상호작용하며 경험 수집
2. **GAE Computation**: Generalized Advantage Estimation으로 이점 계산
3. **PPO Update**: Clipped surrogate objective로 정책 업데이트
4. **Value Update**: MSE loss로 중앙집중식 critic 업데이트

### Loss Functions

```
L_policy = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
L_value = 0.5 * E[(V(s) - R)^2]
L_entropy = -c_ent * E[H(π)]
L_total = L_policy + c_value * L_value + L_entropy
```

### Centralized Value Network

```
Global State = [proprio_agent0, proprio_agent1] (with history)
Value = MLP(Global State)
```

## 모니터링

### TensorBoard

```bash
tensorboard --logdir runs/mappo
```

### Weights & Biases

훈련 시작 시 자동으로 W&B에 로깅됩니다.

## 체크포인트

체크포인트는 다음 위치에 저장됩니다:
- `runs/mappo/{run_id}/checkpoint_{step}/`
- `runs/mappo/{run_id}/checkpoint_{step}_best/` (최고 성능)

## 코드 구조

### MAPPOConfig

```python
@dataclass
class MAPPOConfig:
    # 모델 설정
    pretrained_checkpoint: str
    num_images_in_input: int = 4  # (front + wrist) * history
    history_length: int = 2
    num_actions_chunk: int = 2
    
    # MAPPO 하이퍼파라미터
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ...
```

### MultiAgentVLAPolicy

```python
class MultiAgentVLAPolicy(nn.Module):
    def get_actions(self, agent_inputs, agent_proprios, deterministic=False):
        """모든 에이전트의 액션 샘플링"""
        ...
    
    def evaluate_actions(self, agent_inputs, agent_actions, agent_proprios):
        """주어진 액션의 log probability 계산"""
        ...
```

### CentralizedValueNetwork

```python
class CentralizedValueNetwork(nn.Module):
    def forward(self, global_state):
        """글로벌 상태에서 value 예측"""
        ...
```

## 문제 해결

### CUDA Out of Memory

- `num_steps_per_rollout` 감소
- `num_minibatches` 증가
- `store_images=False` (기본값)

### 느린 훈련

- `LightweightCentralizedCritic` 사용 (기본값)
- 이미지 인코딩 없이 proprio만 사용

### 수렴 문제

- `entropy_coef` 증가 (탐색 장려)
- `clip_epsilon` 조정
- `learning_rate` 감소

## 참고 문헌

- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [OpenVLA-OFT](https://github.com/openvla/openvla-oft)
- [robosuite](https://robosuite.ai/)
