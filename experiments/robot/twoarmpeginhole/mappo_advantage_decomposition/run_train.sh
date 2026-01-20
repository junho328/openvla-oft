#!/bin/bash
# MAPPO Training Script for TwoArmPegInHole Environment
#
# This script launches MAPPO training for multi-agent VLA on the 
# TwoArmPegInHole robosuite environment.
#
# Supports both Single-GPU and Multi-GPU training modes.
#
# Usage:
#   Single GPU:  ./run_train.sh [pretrained_checkpoint_path]
#   Multi-GPU:   ./run_train.sh [pretrained_checkpoint_path] --multi-gpu [NUM_GPUS]
#
# Examples:
#   # Single GPU training
#   ./run_train.sh /path/to/openvla-7b-oft-checkpoint
#   
#   # Multi-GPU training with all available GPUs
#   ./run_train.sh /path/to/openvla-7b-oft-checkpoint --multi-gpu
#   
#   # Multi-GPU training with specific number of GPUs
#   ./run_train.sh /path/to/openvla-7b-oft-checkpoint --multi-gpu 4

set -e

# Parse arguments
PRETRAINED_CHECKPOINT="${1:-/home/work/aipr-jhna/huggingface_hub/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
MULTI_GPU=false
NUM_GPUS=""

# Check for multi-gpu flag
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu)
            MULTI_GPU=true
            shift
            # Check if next argument is a number (num GPUs)
            if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
                NUM_GPUS=$1
                shift
            fi
            ;;
        *)
            shift
            ;;
    esac
done

# Default configuration
WANDB_ENTITY="acpo"
WANDB_PROJECT="mappo-twoarm"

# Output directory
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/work/aipr-jhna/output/twoarmpeginhole/mappo}"

# Training hyperparameters
TOTAL_TIMESTEPS=1000
NUM_STEPS_PER_ROLLOUT=16
BATCH_SIZE=4
LEARNING_RATE=3e-4
ACTOR_LR=1e-4
CRITIC_LR=5e-4

# Environment settings
REWARD_SHAPING=true
MAX_EPISODE_STEPS=300

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../.."

# Ensure wandb is in online mode
export WANDB_MODE=online

# Common training arguments
TRAIN_ARGS=(
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --run_root_dir "$RUN_ROOT_DIR"
    --total_timesteps $TOTAL_TIMESTEPS
    --num_steps_per_rollout $NUM_STEPS_PER_ROLLOUT
    --num_minibatches $BATCH_SIZE
    --learning_rate $LEARNING_RATE
    --actor_lr $ACTOR_LR
    --critic_lr $CRITIC_LR
    --reward_shaping $REWARD_SHAPING
    --max_episode_steps $MAX_EPISODE_STEPS
    --use_wandb true
    --wandb_entity "$WANDB_ENTITY"
    --wandb_project "$WANDB_PROJECT"
    --use_proprio true
    --use_l1_regression true
    --save_eval_videos true
    --num_eval_videos 3
    --eval_freq 100
    --save_freq 100
    --history_length 2
    --num_actions_chunk 2
    --seed 42
)

echo "=============================================="
echo "MAPPO Training for Multi-Agent VLA"
echo "=============================================="
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output directory: $RUN_ROOT_DIR"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout: $NUM_STEPS_PER_ROLLOUT"

if [ "$MULTI_GPU" = true ]; then
    # Multi-GPU training with torchrun
    
    # Determine number of GPUs
    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    fi
    
    echo "Mode: Multi-GPU (DDP)"
    echo "Number of GPUs: $NUM_GPUS"
    echo "=============================================="
    
    # Set environment variables for better distributed training
    export OMP_NUM_THREADS=4
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
    
    # Run with torchrun for distributed training
    nohup torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        -m experiments.robot.twoarmpeginhole.mappo.train_mappo \
        "${TRAIN_ARGS[@]}" > train_multigpu.log 2>&1 &
    
    echo "Multi-GPU training started in background. Check train_multigpu.log for progress."
    
else
    # Single-GPU training
    echo "Mode: Single-GPU"
    echo "=============================================="
    
    nohup python -m experiments.robot.twoarmpeginhole.mappo.train_mappo \
        "${TRAIN_ARGS[@]}" > train.log 2>&1 &
    
    echo "Single-GPU training started in background. Check train.log for progress."
fi

echo "Training process started!"
