#!/bin/bash
# =============================================================================
# CPO (experiment 13)
#
# CPO (Achiam et al., 2017) on Goal Point L1 x 5 seeds (as experiments 03/10) and on
# Circle, Push, Button x 3 seeds (as experiment 11). Run after CPO passes the smoke test (12).
#
# Runs: 14   Time limit per run: 01:15:00   Estimated cost: ~300 SBU (estimate; CPO timing unmeasured) (estimate)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/13_cpo.sh list
# Submit from the repo root:  sbatch scripts/snellius/13_cpo.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_cpo
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:15:00
#SBATCH --array=0-13
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

define_cpo
run_tasks "$@"
