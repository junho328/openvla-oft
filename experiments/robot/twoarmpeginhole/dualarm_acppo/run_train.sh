#!/bin/bash
# Run Dual-Arm ACPPO training on single GPU

set -e

# Activate environment
source /home/work/aipr-jhna/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Change to project root
cd /home/work/aipr-jhna/openvla-oft

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training configuration
CHECKPOINT_PATH="/path/to/your/aloha/bimanual/checkpoint"  # Update this!
RUN_NOTE="dualarm_acppo_single_gpu"

# Run training
python -m experiments.robot.twoarmpeginhole.dualarm_acppo.train_acppo \
    --pretrained_checkpoint "$CHECKPOINT_PATH" \
    --run_id_note "$RUN_NOTE" \
    --num_steps_per_rollout 128 \
    --num_actions_chunk 25 \
    --actor_lr 5e-5 \
    --critic_lr 1e-4 \
    --gamma 0.99 \
    --gamma_prime 0.99 \
    --gae_lambda 0.95 \
    --lambda_prime 0.95 \
    --clip_epsilon 0.2 \
    --entropy_coef 0.01 \
    --value_loss_coef 0.5 \
    --num_epochs 4 \
    --num_minibatches 4 \
    --total_timesteps 1000000 \
    --max_episode_steps 300 \
    --reward_shaping True \
    --normalize_rewards True \
    --normalize_advantages True \
    --use_action_dist_input True \
    --detach_action_dist_grad True \
    --gae_mode "shared_reward" \
    --freeze_vla_backbone True \
    --train_action_head True \
    --train_proprio_projector True \
    --train_value_head True \
    --train_action_dist_projector True \
    --use_wandb True \
    --wandb_project "dualarm-acppo-twoarm" \
    --eval_freq 5000 \
    --save_freq 10000 \
    --num_eval_episodes 10 \
    --seed 42

echo "Training completed!"
