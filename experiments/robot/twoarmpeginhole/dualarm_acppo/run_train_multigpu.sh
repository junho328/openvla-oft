#!/bin/bash
# Multi-GPU Dual-Arm ACPPO Training Script for TwoArmPegInHole Environment
#
# This script launches distributed ACPPO training using PyTorch DDP (DistributedDataParallel).
# Each GPU runs its own environment instance with different random seeds for diverse experience collection.
#
# Dual-Arm ACPPO extends single-arm ACPPO with:
# - Bimanual VLA model (14-dim action, 7 per arm)
# - Agent chaining: Agent 1 estimates Agent 0's action distribution as additional input
# - Microstep-based GAE for proper credit assignment
# - Per-agent value functions
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
WANDB_PROJECT="dualarm_acppo"

# Output directory
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/work/aipr-jhna/output/twoarmpeginhole/dualarm_acppo}"

# Training hyperparameters (adjusted for multi-GPU)
# Note: Effective batch size = BATCH_SIZE * NUM_GPUS
TOTAL_TIMESTEPS=10000
NUM_STEPS_PER_ROLLOUT=4
BATCH_SIZE=4
# Conservative learning rates for VLA-based multi-agent RL
# Actor (policy): Lower lr for stable policy updates
# Critic (value): Slightly higher lr for faster value function convergence
ACTOR_LR=5e-5
CRITIC_LR=1e-4

# Dual-Arm ACPPO specific parameters
GAMMA=0.99
GAMMA_PRIME=0.99
GAE_LAMBDA=0.95
LAMBDA_PRIME=0.95
NUM_ACTIONS_CHUNK=10

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
echo "Multi-GPU Dual-Arm ACPPO Training (DDP)"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output directory: $RUN_ROOT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout (per GPU): $NUM_STEPS_PER_ROLLOUT"
echo "Effective rollout steps: $((NUM_STEPS_PER_ROLLOUT * NUM_GPUS))"
echo "Action chunk size: $NUM_ACTIONS_CHUNK"
echo "ACPPO gamma: $GAMMA"
echo "ACPPO gamma_prime: $GAMMA_PRIME"
echo "ACPPO lambda_prime: $LAMBDA_PRIME"
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
    --actor_lr $ACTOR_LR
    --critic_lr $CRITIC_LR
    --gamma $GAMMA
    --gamma_prime $GAMMA_PRIME
    --gae_lambda $GAE_LAMBDA
    --lambda_prime $LAMBDA_PRIME
    --clip_epsilon 0.2
    --entropy_coef 0.01
    --value_loss_coef 0.5
    --num_epochs 1
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
    --run_id_note "dualarm_acppo"
    # Dual-Arm ACPPO specific
    --use_action_dist_input true
    --detach_action_dist_grad true
    --use_per_agent_value true
    --gae_mode "microstep"
    # Freeze/train options
    --freeze_vla_backbone true
    --train_action_head true
    --train_proprio_projector true
    --train_value_head true
    --train_action_dist_projector true
)

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Starting distributed training with $NUM_GPUS GPUs..."
echo "Log file: $LOG_FILE"
echo ""

export WANDB_MODE=offline

# Run with torchrun for distributed training
# --standalone: Single node training
# --nnodes=1: Number of nodes
# --nproc_per_node: Number of processes (GPUs) per node
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    -m experiments.robot.twoarmpeginhole.dualarm_acppo.train_acppo \
    "${TRAIN_ARGS[@]}" > "$LOG_FILE" 2>&1 &

# For foreground execution (debugging), uncomment below:
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:29501 \
#     -m experiments.robot.twoarmpeginhole.dualarm_acppo.train_acppo \
#     "${TRAIN_ARGS[@]}"

PID=$!
echo "Training started with PID: $PID"
echo "Monitor progress: tail -f $LOG_FILE"
echo ""
echo "To stop training: kill $PID"
