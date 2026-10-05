#!/bin/bash
# =============================================================================
# Smoke test of the new code (experiment 12)
#
# Every new option once for 3M steps, with a 200-episode final evaluation. Run this first:
# each log should end with two "[ppo/train] final eval" lines (stochastic and greedy).
#
# Runs: 10   Time limit per run: 00:15:00   Estimated cost: ~25 SBU (estimate)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/12_smoke_test_new_code.sh list
# Submit from the repo root:  sbatch scripts/snellius/12_smoke_test_new_code.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_smoke_new
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --array=0-9
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
export STORE_MODEL=false
export FINAL_EVAL_EPISODES=200

define_smoke_new
run_tasks "$@"
