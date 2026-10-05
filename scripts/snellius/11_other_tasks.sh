#!/bin/bash
# =============================================================================
# Other tasks in the setting of Spoor et al. (experiment 11)
#
# Circle, Push, Button (Level 1, T = 2000, d = 25) x PPO, PPO-Lag, FOCOPS, PPO-PID (absolute)
# x 3 seeds, 50M steps; plus PPO-Lag on Goal Point at T = 2000 as a bridge.
#
# Runs: 39   Time limit per run: 01:15:00   Estimated cost: ~900 SBU (estimate; Push/T=2000 runs are slower)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/11_other_tasks.sh list
# Submit from the repo root:  sbatch scripts/snellius/11_other_tasks.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_other_tasks
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:15:00
#SBATCH --array=0-38
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

define_other_tasks
run_tasks "$@"
