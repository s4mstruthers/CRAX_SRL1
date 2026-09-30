#!/bin/bash
# =============================================================================
# Run all experiments in one batch
#
# ONE BATCH: experiments 03-08 in a single array job (113 runs).
#   03 compare safe algorithms (35)   04 parallel-robots sweep (24)   05 long runs (6)
#   06 harder levels (24)             07 Ant velocity (12)            08 lambda-rate sweep (12)
# The time limit is the longest any run needs; you are billed only for actual run time.
# Check the first finished runs (~10-15 min in) with: grep -l Traceback logs/safety_all_experiments_*
#
# Runs: 113   Time limit per run: 01:30:00   Estimated cost: ~2,100-2,700 SBU
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/run_all_experiments.sh list
# Submit from the repo root:  sbatch scripts/snellius/run_all_experiments.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_all_experiments
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --array=0-112
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# Repo root: the folder sbatch was run from, or this script's repo when run with bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "${REPO}/scripts/snellius/lib/run_safety_task.sh" ]; then
  echo "Cannot find scripts/snellius/lib/run_safety_task.sh under ${REPO}. Submit from the repo root."
  exit 1
fi
source "${REPO}/scripts/snellius/lib/run_safety_task.sh"

define_compare_algos
define_parallel_envs
define_long_runs
define_harder_levels
define_ant_velocity
define_lambda_lr
run_tasks "$@"
