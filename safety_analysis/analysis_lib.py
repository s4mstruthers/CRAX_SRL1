"""Shared loading, cleaning and metrics for the training-time safety experiments.

Everything the figures (make_figures.py) and the cited numbers (key_numbers.py) use is
defined here once, so the two can never disagree. Only the Python standard library and
numpy are needed.

Data flow
---------
download_wandb.py writes one CSV per run to safety_analysis/wandb_exports/<group>/<run>.csv.
Each CSV row is one wandb history row. `load_runs()` turns every CSV into a `Run`:

  * update rows   one row per policy update (the rows that carry training/safety/*),
                  with chunk-averaged rows removed (see `averaged_rows`);
  * eval rows     rows that carry eval/* metrics (evaluation of the current policy on
                  fresh episodes), kept separately because eval values are genuine even
                  in rows whose training values were chunk-averaged.

Terminology used in the reports
-------------------------------
  budget d          allowed cost per episode (25 in every experiment).
  slice metrics     training/safety/*: each robot's cost over the steps it ran in one
                    update (128 steps at 2048 robots), scaled up to a full episode. The
                    MEAN is accurate; tails (p90/p99/max) are inflated by the short window.
  episode metrics   training/safety_ep/*: the true total cost of every episode that
                    finished during the update. Used for all tail / share-over-budget
                    statements. Missing until the first full episodes finish (~2M steps
                    with desynced robots, because the shortened first episodes are excluded).
  over budget       the update's mean (slice) cost is above d.
  clear crossing    the mean moves from below 0.9 d to above 1.1 d or back (a +-10% band,
                    so wobbles around d are not counted).
"""
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(HERE, "wandb_exports")
BUDGET = 25.0
UPDATE_STEPS = 262_144          # env steps per policy update in every experiment

# Penalty / multiplier signal logged by each constrained algorithm.
PENALTY_KEY = {
    "ppo_lag": "training/lambda_lagr",
    "ppo_pid": "training/lambda_lagr",
    "focops": "training/nu",
    "p3o": "training/kappa",
    "crpo": "training/regime",      # 1 = this update optimised reward, 0 = optimised cost (see crpo/train.py)
}

ALG_LABEL = {"ppo": "PPO (no constraint)", "ppo_lag": "PPO-Lag", "ppo_pid": "PPO-PID", "crpo": "CRPO",
             "focops": "FOCOPS", "p3o": "P3O", "ppo_saute": "PPO-Saute"}


def num(value) -> Optional[float]:
    """Parse a CSV cell to float; empty, non-numeric and non-finite cells become None."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_group(group: str) -> Dict:
    """Split a wandb group name into its experimental settings.

    New groups:  safety_<experiment>_<alg>_<env>_L<level>_n<num_envs>[_lr<rate> | _<variant>]
    where <variant> spells out any other extra arguments, e.g.
    'pid_lambda_mode-absolute-pid_ki-10-pid_kp-10-pid_kd-0' (see run_safety_task.sh).
    The six runs of the first study (30 Sep, before the experiment scripts) are named
    safety_<alg>_<lockstep|desync> and are labelled experiment 'study01'.
    """
    m = re.match(r"safety_(compare_algos|parallel_envs|long_runs|harder_levels|ant_velocity|lambda_lr|smoke_test|"
                 r"fixed_baselines|core_rerun|other_tasks|smoke_new|cpo)_"
                 r"(ppo_lag|ppo_pid|ppo_saute|ppo|crpo|cpo|focops|p3o)_(.+?)_L(\d)_n(\d+)(?:_lr(\d+)|_(.+))?$", group)
    if m:
        exp, alg, env, level, n, lr, variant = m.groups()
        return dict(experiment=exp, alg=alg, env="safe_" + env, level=int(level), num_envs=int(n),
                    lr=float(lr) if lr else 10.0, sync="desync", variant=variant or "")
    m = re.match(r"safety_(ppo_lag|ppo)_(lockstep|desync)$", group)
    if m:
        alg, sync = m.groups()
        return dict(experiment="study01", alg=alg, env="safe_goal_point", level=1, num_envs=2048,
                    lr=10.0, sync=sync, variant="")
    raise ValueError(f"Unrecognised group name: {group}")


def is_chunk_average(row: Dict, num_envs: int) -> bool:
    """True if a row's training values are certainly an average over several updates.

    Before the fix 'Stop progress_fn overwriting per-update metrics' (commit 720b86c), the
    trainer also logged the average of each epoch's updates at the epoch's last step,
    overwriting that update. A genuine update always has a whole number of robots with cost
    and a whole number of finished episodes; an averaged row usually does not. 'Usually':
    an average of 7 whole numbers is itself whole about 1 time in 7, so this test alone
    misses some averaged rows. `averaged_rows` closes that gap.
    """
    frac = num(row.get("training/safety/frac_envs_with_cost"))
    count = num(row.get("training/safety_ep/count"))
    robots_whole = frac is None or abs(frac * num_envs - round(frac * num_envs)) < 1e-3
    count_whole = count is None or abs(count - round(count)) < 1e-6
    return not (robots_whole and count_whole)


def averaged_rows(update_rows: List[Dict], num_envs: int) -> List[bool]:
    """Flag every chunk-averaged row of one run (True = drop from the per-update series).

    Two facts are combined:
      1. a row failing the whole-number test (`is_chunk_average`) is certainly averaged;
      2. the old code logged the averaged values at the same step as the epoch's
         evaluation, so in a run with ANY failing row, every row that also carries
         evaluation values (after step 0) is averaged too, even if its numbers happen to
         be whole. (Checked on all 119 runs: every failing row except PPO-Saute's single
         final row is an evaluation row.)
    Runs trained with the fixed code have no failing rows, so their evaluation rows are
    kept as ordinary updates.
    """
    certain = [is_chunk_average(r, num_envs) for r in update_rows]
    old_code = any(certain)
    return [c or (old_code and num(r.get("eval/episode_cost")) is not None and (num(r.get("_step")) or 0) > 0)
            for c, r in zip(certain, update_rows)]


@dataclass
class Run:
    """One training run: settings plus cleaned per-update and evaluation series."""
    group: str
    name: str
    seed: int
    settings: Dict
    steps: np.ndarray                        # env steps (millions) of each genuine update row
    cols: Dict[str, np.ndarray] = field(default_factory=dict)       # per-update series (NaN = missing)
    eval_steps: np.ndarray = None
    eval_cols: Dict[str, np.ndarray] = field(default_factory=dict)
    n_averaged_dropped: int = 0
    # Steps (millions) of every update row, kept or dropped, in which at least one episode
    # finished (episodic/cost logged). Episodic values are never averaged, so these are
    # exact; the lock-step check uses them to find the episode boundaries.
    episode_end_steps: np.ndarray = None
    # wandb run summary (<run>.summary.json from download_wandb.py), which holds the final-policy
    # evaluation: final/final_eval/<stochastic|greedy>/<metric> and the raw .../episode_costs.
    summary: Dict = field(default_factory=dict)

    def final_eval(self, mode: str, metric: str):
        """Final-policy metric, e.g. final_eval('greedy', 'violation_rate'); None if not logged."""
        return self.summary.get(f"final/final_eval/{mode}/{metric}")

    def get(self, key: str) -> np.ndarray:
        """Per-update series for `key` (all NaN if the run never logged it)."""
        return self.cols.get(key, np.full(len(self.steps), np.nan))

    def __getattr__(self, item):
        if item in ("experiment", "alg", "env", "level", "num_envs", "lr", "sync", "variant"):
            return self.settings.get(item, "")
        raise AttributeError(item)


def load_runs(exports: str = EXPORTS) -> List[Run]:
    """Load every exported run, dropping chunk-averaged rows from the per-update series."""
    runs = []
    for path in sorted(glob.glob(os.path.join(exports, "*", "*.csv"))):
        group = os.path.basename(os.path.dirname(path))
        try:
            settings = parse_group(group)
        except ValueError:
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        name = os.path.splitext(os.path.basename(path))[0]
        seed = int(re.search(r"_seed(\d+)_", name).group(1))
        upd = [r for r in rows if num(r.get("training/safety/env_cost_mean")) is not None]
        drop = averaged_rows(upd, settings["num_envs"])
        keep = [r for r, d in zip(upd, drop) if not d]
        ends = np.array([num(r["_step"]) / 1e6 for r in upd if num(r.get("episodic/cost")) is not None])
        keys = set().union(*(r.keys() for r in keep)) if keep else set()
        cols = {k: np.array([np.nan if num(r.get(k)) is None else num(r.get(k)) for r in keep])
                for k in keys if k.startswith(("training/", "episodic/"))}
        ev = [r for r in rows if num(r.get("eval/episode_cost")) is not None]
        ekeys = set().union(*(r.keys() for r in ev)) if ev else set()
        eval_cols = {k: np.array([np.nan if num(r.get(k)) is None else num(r.get(k)) for r in ev])
                     for k in ekeys if k.startswith("eval/")}
        summary_path = os.path.splitext(path)[0] + ".summary.json"
        summary = {}
        if os.path.exists(summary_path):
            with open(summary_path) as fh:
                summary = json.load(fh)
        runs.append(Run(group=group, name=name, seed=seed, settings=settings, summary=summary,
                        steps=np.array([num(r["_step"]) / 1e6 for r in keep]), cols=cols,
                        eval_steps=np.array([num(r["_step"]) / 1e6 for r in ev]), eval_cols=eval_cols,
                        n_averaged_dropped=len(upd) - len(keep), episode_end_steps=ends))
    return runs


def select(runs: List[Run], **criteria) -> List[Run]:
    """Runs whose settings match every keyword, e.g. select(runs, experiment='lambda_lr', lr=30)."""
    return [r for r in runs if all(r.settings.get(k) == v for k, v in criteria.items())]


def reward_series(run: Run) -> np.ndarray:
    """Episode return of finished training episodes (episodic/sum_reward exists for every env)."""
    return run.get("episodic/sum_reward")


# ---------------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------------
def run_metrics(run: Run, max_step_m: Optional[float] = None, budget: float = BUDGET,
                final_fraction: float = 0.25) -> Dict[str, float]:
    """Summary metrics for one run, optionally only up to `max_step_m` million steps.

    'final phase' = the last `final_fraction` of the (possibly truncated) run's updates.
    """
    sel = np.ones(len(run.steps), bool) if max_step_m is None else run.steps <= max_step_m + 1e-9
    steps = run.steps[sel]
    mean = run.get("training/safety/env_cost_mean")[sel]
    n = len(mean)
    final = np.arange(n) >= int(round(n * (1 - final_fraction)))

    # Clear crossings of the budget with a +-10% hysteresis band.
    state, crossings = None, 0
    for v in mean:
        new = "over" if v > 1.1 * budget else ("under" if v < 0.9 * budget else state)
        if state is not None and new != state:
            crossings += 1
        state = new

    excess = np.maximum(0.0, mean - budget)
    under = np.where(mean <= budget)[0]

    # Episode-level: share of all finished training episodes whose cost exceeded d / 2d,
    # weighting each update by how many episodes finished in it.
    count = run.get("training/safety_ep/count")[sel]
    f_over = run.get("training/safety_ep/frac_over_budget")[sel]
    f_over2 = run.get("training/safety_ep/frac_over_2x_budget")[sel]
    ok = np.isfinite(count) & np.isfinite(f_over) & (count > 0)

    def share(mask, frac):
        m = ok & mask
        return float(np.sum(frac[m] * count[m]) / np.sum(count[m]) * 100) if m.any() else np.nan

    def avg(key, mask):
        v = run.get(key)[sel][mask]
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else np.nan

    reward = reward_series(run)[sel]
    reward_last = reward[np.isfinite(reward)][-10:]
    ev_sel = np.ones(len(run.eval_steps), bool) if max_step_m is None else run.eval_steps <= max_step_m + 1e-9
    ev_cost = run.eval_cols.get("eval/episode_cost", np.array([]))[ev_sel] if len(run.eval_steps) else np.array([])
    ev_rew = run.eval_cols.get("eval/episode_reward", np.array([]))[ev_sel] if len(run.eval_steps) else np.array([])
    pen = run.get(PENALTY_KEY.get(run.alg, "none"))[sel]
    sps = run.get("training/sps")[sel]

    return {
        "updates": n,
        "last_step_m": float(steps[-1]) if n else np.nan,
        "avg_cost": float(np.mean(mean)),                              # mean over updates of the update mean
        "avg_excess": float(np.mean(excess)),                          # mean over updates of max(0, mean - d)
        "pct_over": float(np.mean(mean > budget) * 100),
        "pct_over_10pct": float(np.mean(mean > 1.1 * budget) * 100),
        "clear_crossings": crossings,
        "first_under_m": float(steps[under[0]]) if len(under) else np.nan,
        "excess_share_first_2m": float(np.sum(excess[steps <= 2.1]) / np.sum(excess) * 100) if excess.sum() > 0 else np.nan,
        "final_avg_cost": float(np.mean(mean[final])),
        "final_pct_over": float(np.mean(mean[final] > budget) * 100),
        "ep_share_over": share(np.ones(n, bool), f_over),               # all training episodes (after the first ~2M steps)
        "ep_share_over_2x": share(np.ones(n, bool), f_over2),
        "ep_final_share_over": share(final, f_over),
        "ep_final_share_over_2x": share(final, f_over2),
        "ep_final_mean": avg("training/safety_ep/cost_mean", final),
        "ep_final_p50": avg("training/safety_ep/cost_p50", final),
        "ep_final_p90": avg("training/safety_ep/cost_p90", final),
        "ep_final_p99": avg("training/safety_ep/cost_p99", final),
        "slice_final_p99": avg("training/safety/env_cost_p99", final),
        "slice_final_mean": avg("training/safety/env_cost_mean", final),
        "frac_robots_final": avg("training/safety/frac_envs_with_cost", final) * 100,
        "unsafe_steps_final": avg("training/safety/frac_unsafe_steps", final) * 100,   # % of env steps with cost > 0
        "ep_count_median": float(np.nanmedian(count)) if np.isfinite(count).any() else np.nan,
        "reward_final": float(np.mean(reward_last)) if len(reward_last) else np.nan,
        "eval_cost_final": float(ev_cost[-1]) if len(ev_cost) else np.nan,
        "eval_reward_final": float(ev_rew[-1]) if len(ev_rew) else np.nan,
        "penalty_zero_pct": float(np.mean(pen[np.isfinite(pen)] == 0) * 100) if np.isfinite(pen).any() else np.nan,
        "sps_median": float(np.nanmedian(sps)) if np.isfinite(sps).any() else np.nan,
    }


def group_metrics(runs: List[Run], **kw) -> Dict[str, Dict[str, float]]:
    """Mean, min, max and n over runs for every metric in run_metrics."""
    per_run = [run_metrics(r, **kw) for r in runs]
    out = {}
    for key in per_run[0]:
        vals = np.array([m[key] for m in per_run], float)
        vals = vals[np.isfinite(vals)]
        out[key] = {"mean": float(vals.mean()) if len(vals) else np.nan,
                    "min": float(vals.min()) if len(vals) else np.nan,
                    "max": float(vals.max()) if len(vals) else np.nan,
                    "n": int(len(vals)), "values": [float(v) for v in vals]}
    return out


# ---------------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------------
def on_grid(run: Run, key: str, grid: np.ndarray, series: Optional[np.ndarray] = None) -> np.ndarray:
    """Linearly interpolate a run's per-update series onto `grid` (million steps).

    Dropped (chunk-averaged) rows leave small gaps; interpolation bridges them for
    plotting only. Points outside the run's range are NaN.
    """
    y = run.get(key) if series is None else series
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, run.steps[ok], y[ok])
    out[(grid < run.steps[ok][0]) | (grid > run.steps[ok][-1])] = np.nan
    return out


def seed_curves(runs: List[Run], key: str, max_step_m: Optional[float] = None, series_fn=None):
    """Common grid plus a (seeds x grid) array of a metric, for mean/range plots."""
    last = min(r.steps[-1] for r in runs)
    if max_step_m is not None:
        last = min(last, max_step_m)
    grid = np.arange(UPDATE_STEPS / 1e6, last + 1e-9, UPDATE_STEPS / 1e6)
    arr = np.array([on_grid(r, key, grid, None if series_fn is None else series_fn(r)) for r in runs])
    return grid, arr


def rolling(values: np.ndarray, window: int, fn=np.nanstd) -> np.ndarray:
    """Trailing-window statistic (NaN until the window is full)."""
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = fn(values[i - window + 1:i + 1])
    return out


# ---------------------------------------------------------------------------------
# Lock-step measurement check (study01 runs)
# ---------------------------------------------------------------------------------
def lockstep_position_effect(runs: List[Run], half_window: int = 4, after_m: float = 5.0):
    """Cost of each update relative to its local trend, by position within the episode.

    In the lock-step runs every robot finishes its episode in the same update; those
    updates are the ones where episodic/cost was logged (taken from all rows, including
    dropped averaged rows, whose episodic values are genuine). Each later update is labelled
    1st, 2nd, ... after that boundary. The trend is the mean of the surrounding
    2*half_window+1 genuine updates. Desynced runs use the same boundaries (the update
    schedule is identical) as a control where no pattern is expected.

    Returns {group: {position: (mean_pct, standard_error_pct, n)}} pooled over seeds.
    """
    ref = sorted(select(runs, experiment="study01", alg="ppo_lag", sync="lockstep"), key=lambda r: r.seed)[0]
    boundary_steps = ref.episode_end_steps
    out = {}
    for group in ("safety_ppo_lockstep", "safety_ppo_lag_lockstep", "safety_ppo_lag_desync"):
        residuals = {}
        for run in [r for r in runs if r.group == group]:
            mean = run.get("training/safety/env_cost_mean")
            for i, s in enumerate(run.steps):
                if s <= after_m or i < half_window or i + half_window >= len(mean):
                    continue
                prev = boundary_steps[boundary_steps < s - 1e-9]
                if not len(prev):
                    continue
                pos = int(round((s - prev[-1]) / (UPDATE_STEPS / 1e6)))
                trend = np.mean(mean[i - half_window:i + half_window + 1])
                residuals.setdefault(pos, []).append((mean[i] / trend - 1) * 100)
        out[group] = {p: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
                      for p, v in sorted(residuals.items()) if 1 <= p <= 8 and len(v) > 2}
    return out
