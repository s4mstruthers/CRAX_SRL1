#!/bin/bash
# =============================================================================
# Shared runner for the training-time safety experiments (sourced, not submitted).
#
# Each experiment's runs are defined once here, in a define_<experiment> function.
# A job script sources this file, calls one or more define_ functions, and ends with
# `run_tasks "$@"`:
#   - `bash <script> list`  prints the run table and checks the #SBATCH --array range;
#   - under sbatch, each array task trains the run at its index.
#
# Every run uses desynced robots (CRAX_DESYNC_EPISODES=1) and logs once per policy
# update, including the episode-level metrics (safety_ep/*).
# =============================================================================

TASKS=()

# add_task EXPERIMENT NUM_EVALS ALG ENV LEVEL NUM_ENVS SEED STEPS [EXTRA_ARGS]
add_task() { TASKS+=("$1|$2|$3|$4|$5|$6|$7|$8|${9:-}"); }

# ---------------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------------

# 02 Smoke test: every new algorithm / setting once for 3M steps (checks code, measures speed).
define_smoke_test() {
  local a n
  for a in ppo_lag ppo_pid crpo focops p3o ppo_saute; do add_task smoke_test 3 $a safe_goal_point 1 2048 0 3e6; done
  for n in 512 1024 4096 8192; do add_task smoke_test 3 ppo_lag safe_goal_point 1 $n 0 3e6; done
  add_task smoke_test 3 ppo_lag safe_goal_point 2 2048 0 3e6
  add_task smoke_test 3 ppo_lag safe_goal_point 3 2048 0 3e6
  add_task smoke_test 3 ppo_lag safe_velocity_ant 1 2048 0 3e6
}

# 03 Compare safe algorithms (RQ1, RQ2): 7 algorithms x 5 seeds, Goal Point L1, 35M steps.
define_compare_algos() {
  local a s
  for a in ppo ppo_lag ppo_pid crpo focops p3o ppo_saute; do
    for s in 0 1 2 3 4; do add_task compare_algos 20 $a safe_goal_point 1 2048 $s 3e7; done
  done
}

# 04 Parallel-robots sweep (RQ3): PPO and PPO-Lag x {512, 1024, 4096, 8192} robots x 3 seeds.
define_parallel_envs() {
  local a n s
  for a in ppo ppo_lag; do for n in 512 1024 4096 8192; do
    for s in 0 1 2; do add_task parallel_envs 20 $a safe_goal_point 1 $n $s 3e7; done
  done; done
}

# 05 Long runs: PPO-Lag and PPO-PID x 3 seeds, 100M steps (does the oscillation settle?).
define_long_runs() {
  local a s
  for a in ppo_lag ppo_pid; do for s in 0 1 2; do add_task long_runs 20 $a safe_goal_point 1 2048 $s 1e8; done; done
}

# 06 Harder levels: PPO, PPO-Lag, PPO-PID, CRPO x Goal Point Level {2, 3} x 3 seeds.
define_harder_levels() {
  local a l s
  for a in ppo ppo_lag ppo_pid crpo; do for l in 2 3; do
    for s in 0 1 2; do add_task harder_levels 20 $a safe_goal_point $l 2048 $s 3e7; done
  done; done
}

# 07 Second environment: Ant velocity limit; PPO, PPO-Lag, PPO-PID, CRPO x 3 seeds, 50M steps.
define_ant_velocity() {
  local a s
  for a in ppo ppo_lag ppo_pid crpo; do for s in 0 1 2; do add_task ant_velocity 20 $a safe_velocity_ant 1 2048 $s 5e7; done; done
}

# 08 Lambda learning-rate sweep: PPO-Lag with rate {1, 3, 30, 100} (default 10) x 3 seeds.
define_lambda_lr() {
  local r s
  for r in 1 3 30 100; do
    for s in 0 1 2; do add_task lambda_lr 20 ppo_lag safe_goal_point 1 2048 $s 3e7 "--lagrangian_coef_rate $r"; done
  done
}

# ---------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------
run_tasks() {
  local mode="${1:-run}" n=${#TASKS[@]}

  # ---- list mode: show what would run, and check the header's array range ----
  if [ "$mode" = "list" ]; then
    echo "${n} runs:"
    local i exp ev a e l ne s st x
    for i in "${!TASKS[@]}"; do
      IFS='|' read -r exp ev a e l ne s st x <<< "${TASKS[$i]}"
      printf "  %3d  %-14s %-9s %-18s L%s  envs=%-5s seed=%s  steps=%-4s evals=%-2s %s\n" \
        "$i" "$exp" "$a" "$e" "$l" "$ne" "$s" "$st" "$ev" "$x"
    done
    local header_array
    header_array=$(grep -m1 '^#SBATCH --array=' "$0" | sed 's/^#SBATCH --array=//')
    if [ "${header_array%%%*}" = "0-$((n - 1))" ]; then
      echo "Header --array=${header_array} matches the ${n} runs."
    else
      echo "WARNING: header --array=${header_array} but there are ${n} runs (expected 0-$((n - 1)))."
    fi
    echo "Submit from the repo root with:  sbatch ${0#./}"
    return 0
  fi

  # ---- job mode ----
  local idx=${SLURM_ARRAY_TASK_ID:?"Submit with sbatch, or preview with: bash $0 list"}
  if [ "$idx" -ge "$n" ]; then
    echo "Array index ${idx} is outside this job (0-$((n - 1))). Nothing run."
    return 0
  fi
  local experiment num_evals alg env level num_envs seed steps extra
  IFS='|' read -r experiment num_evals alg env level num_envs seed steps extra <<< "${TASKS[$idx]}"

  module purge
  module load 2024
  module load Python/3.12.3-GCCcore-13.3.0
  source ~/venvs/crax/bin/activate
  cd "${REPO}"
  mkdir -p logs

  export CRAX_DESYNC_EPISODES=1
  local tag=""
  [[ "$extra" == *lagrangian_coef_rate* ]] && tag="_lr${extra##* }"
  local group="safety_${experiment}_${alg}_${env#safe_}_L${level}_n${num_envs}${tag}"
  echo "Run ${idx}/${n} (${experiment}): alg=${alg} env=${env} level=${level} envs=${num_envs} seed=${seed} steps=${steps} ${extra} -> wandb group ${group}"
  python -c "import jax; print(jax.devices())"

  # One policy update is always 262,144 env steps (batch 1024 x 32 minibatches x unroll 8),
  # whatever num_envs is, so --training_metrics_steps 262144 logs exactly once per update.
  # shellcheck disable=SC2086
  python -m training.train_env \
    --env_name "${env}" --alg "${alg}" --difficulty "${level}" \
    --num_timesteps "${steps}" \
    --num_envs "${num_envs}" \
    --training_metrics_steps 262144 \
    --num_evals "${num_evals}" \
    --safety_bound 25 \
    --seeds "${seed}" \
    --store_model false --skip_rollout --skip_video --quiet \
    --wandb_project crax-srl \
    --wandb_group "${group}" \
    ${extra}
}
