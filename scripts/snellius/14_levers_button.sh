#!/bin/bash
# =============================================================================
# Levers on Button (experiment 14, direction brief of 11 Oct)
#
# The two interventions that reduced per-episode violations on Goal Point L1, replicated on
# Button (Level 1, T = 2000, d = 25, same setting as experiment 11), 3 seeds, 50M steps:
#   * PPO-Lag trained to 12.5, measured against 25 (--safety_bound 12.5 --metric_safety_bound 25)
#   * P3O with a fixed penalty kappa = 20
# Compare with experiment 11's Button runs (PPO, PPO-Lag, FOCOPS, PPO-PID) in fig32 and fig34.
#
# Runs: 6   Time limit per run: 01:15:00   Estimated cost: ~140 SBU (estimate from experiment 11)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/14_levers_button.sh list
# Submit from the repo root:  sbatch scripts/snellius/14_levers_button.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_levers_button
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:15:00
#SBATCH --array=0-5
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

define_levers_button
run_tasks "$@"
