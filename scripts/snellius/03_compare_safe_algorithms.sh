#!/bin/bash
# =============================================================================
# Compare safe algorithms (experiment 03)
#
# RQ1 + RQ2: how unsafe is training for each safe RL algorithm? 7 algorithms x 5 seeds.
#
# Runs: 35   Time limit per run: 00:30:00   Estimated cost: ~560 SBU
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/03_compare_safe_algorithms.sh list
# Submit from the repo root:  sbatch scripts/snellius/03_compare_safe_algorithms.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_compare_algos
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --array=0-34
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
run_tasks "$@"
