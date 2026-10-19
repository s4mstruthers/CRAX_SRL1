#!/bin/bash
# =============================================================================
# Penalty floor across algorithms (experiment 15, supervisor feedback of 12 Oct)
#
# Can one hyperparameter change, with the budget left at d = 25, cut per-episode violations
# for several algorithms at once? The cost-penalty weight (PPO-Lag lambda, FOCOPS nu, P3O kappa)
# keeps adapting but is floored at ~3x the value it settles at in a default run
# (--penalty_floor; 0 = CRAX default). See define_penalty_floor in lib/run_safety_task.sh.
#   Goal Point L1 (T = 1000, 30M steps):  PPO-Lag floor 2, FOCOPS floor 2, P3O default
#   Button L1     (T = 2000, 50M steps):  PPO-Lag floor 8, FOCOPS floor 1.5, P3O default
# Compared with: default PPO-Lag / FOCOPS (experiments 10 and 11) and P3O with kappa fixed at 20
# (experiments 09 and 14). Every run ends with the 1000 + 1000 episode final-policy evaluation.
#
# Runs: 18   Time limit per run: 01:15:00   Estimated cost: ~350 SBU (9 x ~16 SBU Goal, 9 x ~23 SBU Button,
# from experiments 09/10 and 14)
# (A100 = 128 SBU per hour of actual run time; the time limit only caps a hung run.)
#
# Preview (starts nothing):   bash scripts/snellius/15_penalty_floor.sh list
# Submit from the repo root:  sbatch scripts/snellius/15_penalty_floor.sh
# Monitor: squeue -u $USER     Cancel everything: scancel <jobid>
# Download results:           python safety_analysis/download_wandb.py
# =============================================================================
#SBATCH --job-name=safety_penalty_floor
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:15:00
#SBATCH --array=0-17
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

define_penalty_floor
run_tasks "$@"
