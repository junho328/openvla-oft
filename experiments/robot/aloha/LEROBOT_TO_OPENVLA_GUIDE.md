# LeRobot v2.0 ALOHA 데이터셋으로 OpenVLA-OFT 훈련하기

이 가이드는 LeRobot v2.0 형식(parquet + mp4 비디오)의 ALOHA 데이터셋을 OpenVLA-OFT로 훈련시키는 전체 과정을 설명합니다.

## 데이터 변환 파이프라인

```
LeRobot v2.0 (parquet + mp4) 
    ↓ [convert_lerobot_to_hdf5.py]
HDF5 (raw)
    ↓ [preprocess_split_aloha_data.py]
HDF5 (preprocessed, 256x256, train/val split)
    ↓ [rlds_dataset_builder]
RLDS (TensorFlow Datasets)
    ↓ [finetune.py]
Fine-tuned OpenVLA Model
```

## 전제 조건

- LeRobot v2.0 형식의 ALOHA 데이터셋
- OpenVLA-OFT 환경 설정 완료 (SETUP.md 참조)
- 충분한 GPU 메모리 (권장: 8x A100 80GB 또는 유사)

## 단계별 가이드

### 1단계: LeRobot → HDF5 변환

LeRobot v2.0 형식의 데이터를 HDF5 형식으로 변환합니다.

```bash
cd /home/work/aipr-jhna/openvla-oft

python experiments/robot/aloha/convert_lerobot_to_hdf5.py \
  --lerobot_path /path/to/your/lerobot/dataset/ \
  --output_path /path/to/output/hdf5_raw/your_task_name/ \
  --task_name "your task description"
```

**입력 구조 (LeRobot v2.0):**
```
/path/to/your/lerobot/dataset/
├── data/
│   ├── train-00000-of-00049.parquet
│   ├── train-00001-of-00049.parquet
│   └── ...
├── videos/
│   ├── observation.images.cam_high_episode_000000.mp4
│   ├── observation.images.cam_left_wrist_episode_000000.mp4
│   └── ...
└── meta/
    └── info.json
```

**출력 구조 (HDF5):**
```
/path/to/output/hdf5_raw/your_task_name/
├── episode_0.hdf5
├── episode_1.hdf5
└── ...
```

### 2단계: HDF5 전처리 (이미지 리사이즈 + train/val 분할)

이미지를 256x256으로 리사이즈하고 train/val 세트로 분할합니다.

```bash
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /path/to/output/hdf5_raw/your_task_name/ \
  --out_base_dir /path/to/preprocessed/ \
  --percent_val 0.05 \
  --img_resize_size 256
```

**출력 구조:**
```
/path/to/preprocessed/your_task_name/
├── train/
│   ├── episode_0.hdf5
│   └── ...
└── val/
    ├── episode_0.hdf5
    └── ...
```

### 3단계: RLDS 형식으로 변환

#### 3.1 rlds_dataset_builder 설치

```bash
git clone https://github.com/moojink/rlds_dataset_builder.git
cd rlds_dataset_builder
pip install -e .
```

#### 3.2 데이터셋 빌더 생성

`experiments/robot/aloha/rlds_builder_template/your_aloha_dataset_builder.py` 파일을 참고하여 빌더를 생성합니다.

1. 파일을 `rlds_dataset_builder/` 디렉토리에 복사
2. 다음 값들을 수정:
   - `DATASET_NAME`: 데이터셋 이름 (예: `my_aloha_task_49_demos`)
   - `DATA_PATHS`: 전처리된 HDF5 경로
   - `LANGUAGE_INSTRUCTION`: 태스크 설명

#### 3.3 RLDS 빌드 실행

```bash
cd rlds_dataset_builder
tfds build --data_dir /path/to/rlds/output/
```

### 4단계: OpenVLA 데이터셋 등록

RLDS 데이터셋을 OpenVLA에 등록합니다.

#### 4.1 configs.py 수정

`prismatic/vla/datasets/rlds/oxe/configs.py`에 추가:

```python
# 파일 끝부분의 OXE_DATASET_CONFIGS 딕셔너리에 추가
"my_aloha_task_49_demos": {
    "image_obs_keys": {
        "primary": "image",
        "secondary": None,
        "left_wrist": "left_wrist_image",
        "right_wrist": "right_wrist_image"
    },
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.JOINT_BIMANUAL,
    "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
},
```

#### 4.2 transforms.py 수정

`prismatic/vla/datasets/rlds/oxe/transforms.py`에 추가:

```python
# OXE_STANDARDIZATION_TRANSFORMS 딕셔너리에 추가
"my_aloha_task_49_demos": aloha_dataset_transform,
```

#### 4.3 mixtures.py 수정

`prismatic/vla/datasets/rlds/oxe/mixtures.py`에 추가:

```python
# OXE_NAMED_MIXTURES 딕셔너리에 추가
"my_aloha_task_49_demos": [
    ("my_aloha_task_49_demos", 1.0),
],
```

### 5단계: Fine-tuning 실행

#### 5.1 Action Chunk Size 설정

`prismatic/vla/constants.py`에서 ALOHA action chunk size를 설정합니다:

```python
# ALOHA_CONSTANTS에서 NUM_ACTIONS_CHUNK 확인
# 50 Hz 데이터의 경우 50으로 설정 (1초 분량)
# 25 Hz 데이터의 경우 25로 설정
```

#### 5.2 Fine-tuning 실행

```bash
# GPU 개수에 맞게 --nproc-per-node 값 수정
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds/output/ \
  --dataset_name my_aloha_task_49_demos \
  --run_root_dir /path/to/checkpoints/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note my_aloha_task_training
```

## 훈련 팁

1. **Learning Rate Decay**: 기본 learning rate `5e-4`로 시작하여 loss가 천천히 감소하면 `5e-5`로 decay
2. **목표 Loss**: Training L1 loss가 0.01 이하로 떨어지고 plateau될 때까지 훈련
3. **데이터셋 크기에 따른 조정**:
   - 작은 데이터셋 (< 50 demos): 더 빨리 decay, 더 짧은 훈련
   - 큰 데이터셋 (> 300 demos): 더 늦게 decay, 더 긴 훈련
4. **FiLM 사용**: 언어 grounding이 필요한 경우 `--use_film True`, 단일 태스크면 `False`도 가능

## 문제 해결

### 비디오 로딩 오류
- OpenCV가 AV1 코덱을 지원하는지 확인: `pip install opencv-python-headless`
- FFmpeg 설치 확인

### 메모리 부족
- `--batch_size` 줄이기
- `--gradient_accumulation_steps` 늘리기

### 데이터셋을 찾을 수 없음
- `--data_root_dir` 경로 확인
- RLDS 데이터셋이 올바르게 빌드되었는지 확인

## 참고 자료

- [OpenVLA-OFT ALOHA.md](ALOHA.md)
- [rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder)
- [LeRobot](https://github.com/huggingface/lerobot)
