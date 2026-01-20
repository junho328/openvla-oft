#!/bin/bash
# Single GPU ACPPO Training Script for TwoArmPegInHole Environment
#
# This script launches single-GPU ACPPO training.
#
# ACPPO extends MAPPO with:
# - Agent chaining: Agent 1 estimates Agent 0's action distribution as additional input
# - Microstep-based GAE for proper credit assignment
# - Per-agent value functions
#
# Usage:
#   ./run_train.sh [pretrained_checkpoint_path] [resume_checkpoint_path]
#
# Examples:
#   # Use default checkpoint
#   ./run_train.sh
#   
#   # Use specific checkpoint
#   ./run_train.sh /path/to/pretrained_checkpoint
#
#   # Resume training from a saved checkpoint
#   ./run_train.sh /path/to/pretrained_checkpoint /path/to/resume_checkpoint
#
# Environment Variables:
#   CUDA_VISIBLE_DEVICES - Specify which GPU to use (e.g., "0")
#   RUN_ROOT_DIR         - Output directory for checkpoints and logs

set -e

# Parse arguments
PRETRAINED_CHECKPOINT="${1:-/home/work/aipr-jhna/huggingface_hub/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
RESUME_CHECKPOINT="${2:-}"  # Optional: path to checkpoint to resume from

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Default configuration
WANDB_ENTITY="acpo"
WANDB_PROJECT="acppo-twoarm"

# Output directory
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/work/aipr-jhna/output/twoarmpeginhole/acppo}"

# Training hyperparameters
TOTAL_TIMESTEPS=1000000
NUM_STEPS_PER_ROLLOUT=256
BATCH_SIZE=4
# Conservative learning rates for VLA-based multi-agent RL
# Actor (policy): Lower lr for stable policy updates
# Critic (value): Slightly higher lr for faster value function convergence
ACTOR_LR=5e-5
CRITIC_LR=1e-4

# ACPPO specific parameters
GAMMA_PRIME=0.99
LAMBDA_PRIME=0.95
NUM_ACTIONS_CHUNK=2  # ACPPO uses chunk size 4 (vs MAPPO's 2)

# Environment settings
REWARD_SHAPING=true
MAX_EPISODE_STEPS=300

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../.."

# Setup environment
export WANDB_MODE=online
export OMP_NUM_THREADS=4

# Timestamp for run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Single GPU ACPPO Training"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output directory: $RUN_ROOT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout: $NUM_STEPS_PER_ROLLOUT"
echo "Action chunk size: $NUM_ACTIONS_CHUNK"
echo "ACPPO gamma_prime: $GAMMA_PRIME"
echo "ACPPO lambda_prime: $LAMBDA_PRIME"
echo "=============================================="
echo ""

# Create log directory
LOG_DIR="${RUN_ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train_single_${TIMESTAMP}.log"

# Common training arguments
TRAIN_ARGS=(
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --run_root_dir "$RUN_ROOT_DIR"
    --total_timesteps $TOTAL_TIMESTEPS
    --num_steps_per_rollout $NUM_STEPS_PER_ROLLOUT
    --num_minibatches $BATCH_SIZE
    --actor_lr $ACTOR_LR
    --critic_lr $CRITIC_LR
    --gamma_prime $GAMMA_PRIME
    --lambda_prime $LAMBDA_PRIME
    --reward_shaping $REWARD_SHAPING
    --reaching_weight 0.4
    --perpendicular_weight 1.2
    --parallel_weight 0.6
    --alignment_weight 1.2
    --max_episode_steps $MAX_EPISODE_STEPS
    --use_wandb true
    --wandb_entity "$WANDB_ENTITY"
    --wandb_project "$WANDB_PROJECT"
    --use_proprio true
    --use_l1_regression true
    --save_eval_videos true
    --num_eval_videos 3
    --eval_freq 5000
    --save_freq 10000
    --history_length 2
    --num_actions_chunk $NUM_ACTIONS_CHUNK
    --seed 42
    --run_id_note "single_gpu"
    # ACPPO specific
    --use_action_dist_input true
    --detach_action_dist_grad true
    --use_per_agent_value true
)

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Starting single-GPU training..."
echo "Log file: $LOG_FILE"
echo ""

# Run training
nohup python -m experiments.robot.twoarmpeginhole.acppo.train_acppo \
    "${TRAIN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
# python -m experiments.robot.twoarmpeginhole.acppo.train_acppo \
#     "${TRAIN_ARGS[@]}"

PID=$!
echo "Training started with PID: $PID"
echo "Monitor progress: tail -f $LOG_FILE"
echo ""
echo "To stop training: kill $PID"
