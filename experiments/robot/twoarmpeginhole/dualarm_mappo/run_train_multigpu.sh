#!/bin/bash
# Multi-GPU Dual-Arm MAPPO Training Script for TwoArmPegInHole Environment
#
# This script launches distributed MAPPO training using PyTorch DDP (DistributedDataParallel).
# Each GPU runs its own environment instance with different random seeds for diverse experience collection.
#
# Dual-Arm MAPPO uses a bimanual VLA model (ALOHA-style) with:
# - 14-dim action space (7 per arm)
# - 14-dim proprioceptive state (7 per arm)
# - Agent-specific image and proprio padding
# - Shared action head, separate value heads
#
# Usage:
#   ./run_train_multigpu.sh [NUM_GPUS] [pretrained_checkpoint_path] [resume_checkpoint_path]
#
# Examples:
#   # Use all available GPUs
#   ./run_train_multigpu.sh
#   
#   # Use 2 GPUs
#   ./run_train_multigpu.sh 2
#   
#   # Use 2 GPUs with specific checkpoint
#   ./run_train_multigpu.sh 2 /path/to/pretrained_checkpoint
#
#   # Resume training from a saved checkpoint
#   ./run_train_multigpu.sh 2 /path/to/pretrained_checkpoint /path/to/resume_checkpoint
#
# Environment Variables:
#   CUDA_VISIBLE_DEVICES - Specify which GPUs to use (e.g., "0,1")
#   RUN_ROOT_DIR         - Output directory for checkpoints and logs

set -e

# Parse arguments
NUM_GPUS="${1:-auto}"
PRETRAINED_CHECKPOINT="${2:-/home/work/aipr-jhna/huggingface_hub/openvla-7b+aloha_combined_task+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--my_aloha_task_training--4000_chkpt}"
RESUME_CHECKPOINT="${3:-}"  # Optional: path to checkpoint to resume from

# Determine number of GPUs
if [ "$NUM_GPUS" = "auto" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

# Validate GPU count
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: At least 1 GPU is required"
    exit 1
fi

# Default configuration
WANDB_ENTITY="acpo"
WANDB_PROJECT="dualarm-mappo"

# Output directory
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/work/aipr-jhna/output/twoarmpeginhole/dualarm_mappo}"

# Training hyperparameters (adjusted for multi-GPU)
# Note: Effective batch size = BATCH_SIZE * NUM_GPUS
TOTAL_TIMESTEPS=10000
NUM_STEPS_PER_ROLLOUT=16
BATCH_SIZE=4
# Conservative learning rates for VLA-based multi-agent RL
# Actor (policy): Lower lr for stable policy updates
# Critic (value): Slightly higher lr for faster value function convergence
ACTOR_LR=5e-5
CRITIC_LR=1e-4

# Dual-Arm MAPPO specific parameters
NUM_ACTIONS_CHUNK=25  # Bimanual model uses 25 action chunks (actually it was trained for 50 action chunks since 50 fps)

# Environment settings
REWARD_SHAPING=true
MAX_EPISODE_STEPS=500

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../.."

# Setup environment
export OMP_NUM_THREADS=4

# NCCL settings for optimal performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand (enable if available on your cluster)
export NCCL_P2P_DISABLE=0  # Enable peer-to-peer (set to 1 if issues occur)

# For debugging CUDA issues
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# Timestamp for run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Multi-GPU Dual-Arm MAPPO Training (DDP)"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output directory: $RUN_ROOT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout (per GPU): $NUM_STEPS_PER_ROLLOUT"
echo "Effective rollout steps: $((NUM_STEPS_PER_ROLLOUT * NUM_GPUS))"
echo "Action chunk size: $NUM_ACTIONS_CHUNK"
echo "=============================================="
echo ""

# Create log directory
LOG_DIR="${RUN_ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train_multigpu_${TIMESTAMP}.log"

# Common training arguments
TRAIN_ARGS=(
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --run_root_dir "$RUN_ROOT_DIR"
    --total_timesteps $TOTAL_TIMESTEPS
    --num_steps_per_rollout $NUM_STEPS_PER_ROLLOUT
    --num_minibatches $BATCH_SIZE
    --num_epochs 1
    --actor_lr $ACTOR_LR
    --critic_lr $CRITIC_LR
    --gamma 0.99
    --gae_lambda 0.95
    --clip_epsilon 0.2
    --entropy_coef 0.01
    --value_loss_coef 0.5
    --reward_shaping $REWARD_SHAPING
    --reaching_weight 1.0
    --perpendicular_weight 1.0
    --parallel_weight 1.0
    --alignment_weight 1.0
    --max_episode_steps $MAX_EPISODE_STEPS
    --use_wandb true
    --wandb_entity "$WANDB_ENTITY"
    --wandb_project "$WANDB_PROJECT"
    --use_proprio true
    --use_l1_regression true
    --use_film true
    # Video saving options
    --save_eval_videos true
    --num_eval_videos 2
    # Evaluation and checkpoint frequency
    --eval_freq 100
    --save_freq 100
    --num_eval_episodes 5
    --history_length 2
    --num_actions_chunk $NUM_ACTIONS_CHUNK
    --seed 42
    --run_id_note "dualarm_mappo_multigpu_${NUM_GPUS}gpus"
    # Freeze/train options
    --freeze_vla_backbone true
    --train_action_head true
    --train_value_head true
    --train_proprio_projector true
    # Dual-arm specific dimensions
    --model_action_dim 14
    --agent_action_dim 7
    --model_proprio_dim 14
    --agent_proprio_dim 7
)

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Starting distributed training with $NUM_GPUS GPUs..."
echo "Log file: $LOG_FILE"
echo ""

# Run with torchrun for distributed training
# --standalone: Single node training
# --nnodes=1: Number of nodes
# --nproc_per_node: Number of processes (GPUs) per node
# nohup torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:29500 \
#     -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \
#     "${TRAIN_ARGS[@]}" > "$LOG_FILE" 2>&1 &

# For foreground execution (debugging), uncomment below:
export WANDB_MODE=offline

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \
    "${TRAIN_ARGS[@]}"

PID=$!
echo "Training started with PID: $PID"
echo "Monitor progress: tail -f $LOG_FILE"
echo ""
echo "To stop training: kill $PID"
