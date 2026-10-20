#!/bin/bash
# =============================================================================
# Baselines for the penalty-floor experiment (experiment 16)
#
# The CRAX-default arms that experiment 15 is compared with, rerun with the current code so that
# every arm keeps its 1000 raw final-evaluation episode costs (needed for the cost-distribution
# figure fig37). See define_penalty_floor_baselines in lib/run_safety_task.sh.
#   Goal Point L1 (T = 1000, 30M):  PPO-Lag default, FOCOPS default, P3O kappa fixed at 20
#   Button L1     (T = 2000, 50M):  PPO-Lag default, FOCOPS default
#
# Runs: 15   Time limit per run: 00:40:00   Estimated cost: ~300 SBU (runs of exp. 15 took 7-14 min)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/16_penalty_floor_baselines.sh list
# Submit from the repo root:  sbatch scripts/snellius/16_penalty_floor_baselines.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_floor_baselines
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --array=0-14
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

define_penalty_floor_baselines
run_tasks "$@"
