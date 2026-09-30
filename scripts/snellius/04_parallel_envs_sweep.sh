#!/bin/bash
# =============================================================================
# Parallel envs sweep (experiment 04)
#
# RQ3: does the number of parallel robots change training-time safety? 512-8192 robots.
#
# Runs: 24   Time limit per run: 01:30:00   Estimated cost: ~350-700 SBU
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/04_parallel_envs_sweep.sh list
# Submit from the repo root:  sbatch scripts/snellius/04_parallel_envs_sweep.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_parallel_envs
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# Repo root: the folder sbatch was run from, or this script's repo when run with bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "${REPO}/scripts/snellius/lib/run_safety_task.sh" ]; then
  echo "Cannot find scripts/snellius/lib/run_safety_task.sh under ${REPO}. Submit from the repo root."
  exit 1
fi
source "${REPO}/scripts/snellius/lib/run_safety_task.sh"

define_parallel_envs
run_tasks "$@"
