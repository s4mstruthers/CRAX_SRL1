#!/bin/bash
# =============================================================================
# PPO vs PPO-Lag, lock-step vs desynced robots (experiment 01, DONE 30 Sep 2026)
#
# 6 runs on SafeGoal Point Level 1, 30M steps each (~114 policy updates,
# ~14.6 episodes per robot), 2 seeds per configuration:
#
#   config 0: ppo      (lock-step)  unconstrained baseline, cost logged only
#   config 1: ppo_lag  (lock-step)  same setting as the 28 Sep test, longer
#   config 2: ppo_lag  (desync)     robots start at random episode steps
#
# What each comparison tests:
#   0 vs 1 : how much safer is PPO-Lag during training? does its tail (p99/CVaR)
#            stay above budget while the mean is satisfied?
#   1 vs 2 : is the per-update metric biased by lock-step episodes? With desync
#            the periodic bump every ~7.6 updates should disappear.
#   1 alone: does the lambda oscillation continue over ~114 updates?
#
# Expected cost: ~7 min per run at 85,900 steps/s -> ~15 SBU/run, ~90 SBU total.
# Hard cap from --time: 6 x 30 min x 128 SBU/h = 384 SBU (only if every run hangs).
#
# Submit:  sbatch scripts/snellius/01_ppo_vs_lag_lockstep_vs_desync.sh
# Check:   squeue -u $USER      Cancel: scancel <jobid>
# =============================================================================
#SBATCH --job-name=crax_ppo_vs_lag_desync
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# --- Software stack (validated interactively on gcn3, 2026-10-03) -------------
# jax-cuda12-plugin 0.6.0 is compiled against cuDNN 9.8 -> need cuDNN >= 9.8
# (2024 stack only ships 9.5, which XLA rejects). 2025 stack has 9.10.1.4.
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load 2025
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0
source ~/venvs/crax/bin/activate
cd ~/CRAX_SRL1
mkdir -p logs

# --- Map array index -> (configuration, seed) ---------------------------------
ALGS=(ppo ppo_lag ppo_lag)
DESYNC=(0 0 1)
IDX=${SLURM_ARRAY_TASK_ID}
CFG=$(( IDX / 2 ))     # 0,0,1,1,2,2
SEED=$(( IDX % 2 ))    # 0,1,0,1,0,1
ALG=${ALGS[$CFG]}
export CRAX_DESYNC_EPISODES=${DESYNC[$CFG]}
SYNC_LABEL=$([ "$CRAX_DESYNC_EPISODES" = "1" ] && echo desync || echo lockstep)

NUM_TIMESTEPS="${NUM_TIMESTEPS:-3e7}"
echo "Task ${IDX}: alg=${ALG} ${SYNC_LABEL} seed=${SEED} steps=${NUM_TIMESTEPS}"
python -c "import jax; print(jax.devices())"

# --- Train ----------------------------------------------------------------------
# --training_metrics_steps 262144: exactly one logged row per policy update
#   (2048 envs x 8 steps x 16 rollouts), so safety/ metrics are per update.
# --num_evals 20: eval curve for the train-time vs eval-time comparison.
# The wandb group encodes the configuration so runs are easy to filter.
python -m training.train_env \
  --env_name safe_goal_point --alg "${ALG}" --difficulty 1 \
  --num_timesteps "${NUM_TIMESTEPS}" \
  --num_envs 2048 \
  --training_metrics_steps 262144 \
  --num_evals 20 \
  --safety_bound 25 \
  --seeds "${SEED}" \
  --store_model false --skip_rollout --skip_video --quiet \
  --wandb_project crax-srl \
  --wandb_group "safety_${ALG}_${SYNC_LABEL}"
