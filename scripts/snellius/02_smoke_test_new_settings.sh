#!/bin/bash
# =============================================================================
# Smoke test new settings (experiment 02)
#
# Quick check (3M steps each) that every new algorithm and setting runs and logs the episode-level
# metrics; measures each setting's speed. Optional when using run_all_experiments.sh.
#
# Runs: 13   Time limit per run: 00:20:00   Estimated cost: ~60-90 SBU
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/02_smoke_test_new_settings.sh list
# Submit from the repo root:  sbatch scripts/snellius/02_smoke_test_new_settings.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_smoke_test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --array=0-12
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# Repo root: the folder sbatch was run from, or this script's repo when run with bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "${REPO}/scripts/snellius/lib/run_safety_task.sh" ]; then
  echo "Cannot find scripts/snellius/lib/run_safety_task.sh under ${REPO}. Submit from the repo root."
  exit 1
fi
source "${REPO}/scripts/snellius/lib/run_safety_task.sh"

define_smoke_test
run_tasks "$@"
