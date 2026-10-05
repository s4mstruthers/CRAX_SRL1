#!/bin/bash
# =============================================================================
# Core baselines with the new metrics (experiment 10)
#
# PPO, PPO-Lag, CRPO, FOCOPS x 5 seeds: as experiment 03, plus the final greedy/stochastic
# evaluation (1000 episodes each), D_norm+ and the CDF.
#
# Runs: 20   Time limit per run: 00:40:00   Estimated cost: ~300 SBU (estimate)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/10_core_rerun.sh list
# Submit from the repo root:  sbatch scripts/snellius/10_core_rerun.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_core_rerun
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --array=0-19
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

define_core_rerun
run_tasks "$@"
