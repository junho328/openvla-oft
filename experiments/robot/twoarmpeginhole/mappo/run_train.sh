#!/bin/bash
# MAPPO Training Script for TwoArmPegInHole Environment
#
# This script launches MAPPO training for multi-agent VLA on the 
# TwoArmPegInHole robosuite environment.
#
# Usage:
#   ./run_train.sh [pretrained_checkpoint_path]
#
# Example:
#   ./run_train.sh /path/to/openvla-7b-oft-checkpoint

set -e

# Default configuration
# Use fine-tuned checkpoint (has trained proprio_projector and action_head)
# Options: moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}
# Or base VLA: openvla/openvla-7b (will initialize heads from scratch)
PRETRAINED_CHECKPOINT="${1:-/home/work/aipr-jhna/huggingface_hub/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
WANDB_ENTITY="acpo"
WANDB_PROJECT="mappo-twoarm"

# Output directory (change this to save checkpoints to a different location)
# Examples:
#   RUN_ROOT_DIR="runs/mappo"                    # Relative path (default)
#   RUN_ROOT_DIR="/data/mappo_outputs"           # Absolute path
#   RUN_ROOT_DIR="/mnt/nfs/experiments/mappo"    # NFS mount
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/work/aipr-jhna/output/twoarmpeginhole/mappo}"

# Training hyperparameters
TOTAL_TIMESTEPS=1000000
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

echo "=============================================="
echo "MAPPO Training for Multi-Agent VLA"
echo "=============================================="
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output directory: $RUN_ROOT_DIR"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout: $NUM_STEPS_PER_ROLLOUT"
echo "=============================================="

# Run training
python -m experiments.robot.twoarmpeginhole.mappo.train_mappo \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --total_timesteps $TOTAL_TIMESTEPS \
    --num_steps_per_rollout $NUM_STEPS_PER_ROLLOUT \
    --num_minibatches $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --actor_lr $ACTOR_LR \
    --critic_lr $CRITIC_LR \
    --reward_shaping $REWARD_SHAPING \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --use_wandb true \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --use_proprio true \
    --use_l1_regression true \
    --save_eval_videos true \
    --num_eval_videos 3 \
    --eval_freq 100 \
    --save_freq 100 \
    --history_length 2 \
    --num_actions_chunk 2 \
    --seed 42

echo "Training completed!"
