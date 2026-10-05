#!/bin/bash
# =============================================================================
# Fixed baselines and tail-driven PPO-Lag (experiment 09)
#
# PID as in Stooke et al., P3O with a faster / fixed kappa, Saute without budget discount,
# PPO-Lag with a tighter target, and PPO-Lag driven by the violation rate or CVaR.
#
# Runs: 28   Time limit per run: 00:40:00   Estimated cost: ~400-450 SBU (estimate from ~16 SBU per 35M-step run)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/09_fixed_baselines.sh list
# Submit from the repo root:  sbatch scripts/snellius/09_fixed_baselines.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_fixed_baselines
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --array=0-27
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# Repo root: the folder sbatch was run from, or this script's repo when run with bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "${REPO}/scripts/snellius/lib/run_safety_task.sh" ]; then
  echo "Cannot find scripts/snellius/lib/run_safety_task.sh under ${REPO}. Submit from the repo root."
  exit 1
fi
source "${REPO}/scripts/snellius/lib/run_safety_task.sh"

# Keep the final policies, so they can be re-evaluated later without retraining.
export STORE_MODEL=true

define_fixed_baselines
run_tasks "$@"
