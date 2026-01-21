#!/bin/bash
# Single GPU Training Script for Dual-Arm MAPPO
#
# This trains a bimanual VLA model (ALOHA-style) with MAPPO on TwoArmPegInHole.
# The VLA outputs 14-dim actions (7 per arm), which are split per agent.

set -e

# Default parameters
CHECKPOINT=${CHECKPOINT:-"/path/to/your/aloha/bimanual/checkpoint"}
RUN_NOTE=${RUN_NOTE:-"dualarm_mappo_run"}
SEED=${SEED:-42}

# Training parameters
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-256}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-4}
NUM_EPOCHS=${NUM_EPOCHS:-4}
ACTOR_LR=${ACTOR_LR:-5e-5}
CRITIC_LR=${CRITIC_LR:-1e-4}

# Environment parameters
REWARD_SHAPING=${REWARD_SHAPING:-true}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}

# W&B settings
USE_WANDB=${USE_WANDB:-true}
WANDB_ENTITY=${WANDB_ENTITY:-"acpo"}
WANDB_PROJECT=${WANDB_PROJECT:-"dualarm-mappo-twoarm"}

# Print configuration
echo "=============================================="
echo "Dual-Arm MAPPO Training (Single GPU)"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Run note: $RUN_NOTE"
echo "Seed: $SEED"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per rollout: $NUM_STEPS_PER_ROLLOUT"
echo "Reward shaping: $REWARD_SHAPING"
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/../../../.."

# Run training
python -m experiments.robot.twoarmpeginhole.dualarm_mappo.train_mappo \
    --pretrained_checkpoint "$CHECKPOINT" \
    --run_id_note "$RUN_NOTE" \
    --seed "$SEED" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --num_steps_per_rollout "$NUM_STEPS_PER_ROLLOUT" \
    --num_minibatches "$NUM_MINIBATCHES" \
    --num_epochs "$NUM_EPOCHS" \
    --actor_lr "$ACTOR_LR" \
    --critic_lr "$CRITIC_LR" \
    --reward_shaping "$REWARD_SHAPING" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --use_wandb "$USE_WANDB" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --freeze_vla_backbone true \
    --train_action_head true \
    --train_value_head true \
    --train_proprio_projector true \
    --use_proprio true \
    --use_film true \
    --use_l1_regression true \
    --num_actions_chunk 2 \
    --history_length 2 \
    --model_action_dim 14 \
    --agent_action_dim 7 \
    --model_proprio_dim 14 \
    --agent_proprio_dim 7

echo "Training completed!"
