"""Compute every number quoted in the training-time safety reports.

Usage (repo folder, crax env active; needs numpy only):
    python safety_analysis/key_numbers.py              # print all tables + write figures/key_numbers.json
    python safety_analysis/key_numbers.py compare      # print only sections whose name contains 'compare'

All metrics come from analysis_lib.run_metrics, the same function the figures use, so a
number in the reports can be traced to one line of output here. Values are the mean over
seeds; [min-max] over seeds is printed next to them where it matters.

Windows used (the same as in the figures):
  * algorithm comparison   updates up to 30.2M steps (PPO-Saute trained for 30M steps);
  * every other section    the full run (35M steps; long runs 100M; Ant 55M).
  * 'final quarter'        the last 25% of the updates in that window.
"""
import json
import math
import os
import sys

import numpy as np

import analysis_lib as A

CMP_MAX = 30.2          # algorithm comparison window (million steps), as in make_figures.py
FULL = 34.95            # 35M-step runs: every update (used where compare_algos is reused as a baseline)
OUT_JSON = os.path.join(A.HERE, "figures", "key_numbers.json")

RESULTS = {}            # everything printed is also stored here and written to JSON


# ---------------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------------
def fmt(x, digits=1):
    """Format a number for the tables ('-' for missing)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def show_groups(section, labelled_runs, metrics, window=None, digits=None):
    """Print one row per group (mean over seeds, with the seed range) and store it.

    labelled_runs: list of (label, runs); metrics: list of run_metrics keys.
    """
    digits = digits or {}
    print(f"\n=== {section} ===")
    header = f"{'group':<26}{'n':>3}  " + "  ".join(f"{m:>22}" for m in metrics)
    print(header)
    store = {}
    for label, runs in labelled_runs:
        if not runs:
            continue
        g = A.group_metrics(runs, max_step_m=window)
        cells = []
        for m in metrics:
            d = digits.get(m, 1)
            v = g[m]
            cells.append(f"{fmt(v['mean'], d):>8} [{fmt(v['min'], d)}-{fmt(v['max'], d)}]".rjust(22))
        print(f"{label:<26}{len(runs):>3}  " + "  ".join(cells))
        store[label] = {m: g[m] for m in g}
        store[label]["seeds"] = sorted(r.seed for r in runs)
    RESULTS[section] = store
    return store


def note(section, key, value, text):
    """Print and store a single derived number."""
    RESULTS.setdefault(section, {})[key] = value
    print(f"  {text}: {value if not isinstance(value, float) else fmt(value, 2)}")


# ---------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------
def inventory(runs):
    """What was loaded: runs per experiment, updates kept, averaged rows dropped."""
    print("\n=== inventory ===")
    rows = {}
    for exp in sorted({r.experiment for r in runs}):
        rs = [r for r in runs if r.experiment == exp]
        upd = [len(r.steps) for r in rs]
        dropped = [r.n_averaged_dropped for r in rs]
        last = [r.steps[-1] for r in rs]
        evals = [len(r.eval_steps) for r in rs]
        rows[exp] = dict(runs=len(rs), updates_kept=[min(upd), max(upd)], averaged_rows_dropped=[min(dropped), max(dropped)],
                         last_step_m=[round(min(last), 2), round(max(last), 2)], evals=[min(evals), max(evals)])
        print(f"  {exp:<15} runs={len(rs):>3}  updates kept {min(upd)}-{max(upd)}  averaged rows dropped "
              f"{min(dropped)}-{max(dropped)}  last step {min(last):.1f}-{max(last):.1f}M  evals {min(evals)}-{max(evals)}")
    rows["total_runs"] = len(runs)
    rows["new_runs"] = len([r for r in runs if r.experiment != "study01"])
    print(f"  total runs {rows['total_runs']} (new batch {rows['new_runs']}, first study {rows['total_runs'] - rows['new_runs']})")
    saute = A.select(runs, experiment="compare_algos", alg="ppo_saute")
    rows["saute_updates"] = sorted({len(r.steps) + r.n_averaged_dropped for r in saute})
    print(f"  PPO-Saute update rows (incl. dropped averages): {rows['saute_updates']}, evaluations: "
          f"{sorted({len(r.eval_steps) for r in saute})}")
    RESULTS["inventory"] = rows


CORE = ["avg_cost", "avg_excess", "pct_over", "pct_over_10pct", "clear_crossings", "first_under_m",
        "final_avg_cost", "final_pct_over", "reward_final"]
EPISODE = ["ep_share_over", "ep_share_over_2x", "ep_final_share_over", "ep_final_share_over_2x",
           "ep_final_mean", "ep_final_p50", "ep_final_p90", "ep_final_p99"]
EXTRA = ["slice_final_p99", "eval_cost_final", "eval_reward_final", "excess_share_first_2m",
         "frac_robots_final", "unsafe_steps_final", "ep_count_median", "sps_median"]
ALGS = ["ppo", "ppo_lag", "ppo_pid", "crpo", "focops", "p3o", "ppo_saute"]


def compare_algos(runs):
    """Experiment 03: seven algorithms, 5 seeds, Goal Point L1, window 0-30.2M steps."""
    groups = [(a, A.select(runs, experiment="compare_algos", alg=a)) for a in ALGS]
    show_groups("compare_algos (<=30.2M) core", groups, CORE, window=CMP_MAX)
    show_groups("compare_algos (<=30.2M) episodes", groups, EPISODE, window=CMP_MAX)
    show_groups("compare_algos (<=30.2M) extra", groups, EXTRA, window=CMP_MAX, digits={"sps_median": 0})

    # PPO vs PPO-Lag: 'how much safer' depends on whether total cost or cost above budget is compared.
    ppo = A.group_metrics(A.select(runs, experiment="compare_algos", alg="ppo"), max_step_m=CMP_MAX)
    lag = A.group_metrics(A.select(runs, experiment="compare_algos", alg="ppo_lag"), max_step_m=CMP_MAX)
    print()
    note("ratios", "ppo_over_lag_total_cost", ppo["avg_cost"]["mean"] / lag["avg_cost"]["mean"],
         "PPO / PPO-Lag average training cost")
    note("ratios", "ppo_over_lag_excess", ppo["avg_excess"]["mean"] / lag["avg_excess"]["mean"],
         "PPO / PPO-Lag average cost ABOVE budget")
    note("ratios", "lag_reward_cost_pct", (1 - lag["reward_final"]["mean"] / ppo["reward_final"]["mean"]) * 100,
         "PPO-Lag reward below PPO (%)")
    note("ratios", "lag_p99_over_budget", lag["ep_final_p99"]["mean"] / A.BUDGET,
         "PPO-Lag episode p99 (final quarter) / budget")


def mechanisms(runs):
    """Numbers that explain each algorithm's behaviour (from its logged penalty signal)."""
    print("\n=== mechanisms (compare_algos, <=30.2M) ===")
    sec = "mechanisms"

    def seed_mean_curve(alg, key):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        grid, arr = A.seed_curves(rs, key, max_step_m=CMP_MAX)
        return grid, np.nanmean(arr, axis=0)

    # PPO-Lag / PPO-PID: size of lambda and how often it sits at zero
    for alg in ("ppo_lag", "ppo_pid"):
        grid, lam = seed_mean_curve(alg, "training/lambda_lagr")
        note(sec, f"{alg}_lambda_peak", float(np.nanmax(lam)), f"{alg}: peak of seed-mean lambda")
        note(sec, f"{alg}_lambda_peak_step_m", float(grid[np.nanargmax(lam)]), f"{alg}: step of that peak (M)")
        g = A.group_metrics(A.select(runs, experiment="compare_algos", alg=alg), max_step_m=CMP_MAX)
        note(sec, f"{alg}_lambda_zero_pct", g["penalty_zero_pct"]["mean"], f"{alg}: % of updates with lambda exactly 0")

    # CRPO: share of updates spent optimising cost (regime 0)
    rs = A.select(runs, experiment="compare_algos", alg="crpo")
    shares = []
    for r in rs:
        reg = r.get("training/regime")[r.steps <= CMP_MAX + 1e-9]
        reg = reg[np.isfinite(reg)]
        shares.append(float(np.mean(reg == 0) * 100))
    note(sec, "crpo_cost_regime_pct", float(np.mean(shares)), "CRPO: % of updates optimising cost (regime 0)")

    # FOCOPS: nu rises slowly (nu <- nu + 1.0 * per-step violation)
    grid, nu = seed_mean_curve("focops", "training/nu")
    note(sec, "focops_nu_peak", float(np.nanmax(nu)), "FOCOPS: peak of seed-mean nu")
    note(sec, "focops_nu_peak_step_m", float(grid[np.nanargmax(nu)]), "FOCOPS: step of that peak (M)")

    # P3O: kappa <- min(1.1 * kappa, 50) while violated, starting at 0.01
    grid, kappa = seed_mean_curve("p3o", "training/kappa")
    at_cap = np.where(kappa >= 49.9)[0]
    note(sec, "p3o_kappa_cap_step_m", float(grid[at_cap[0]]) if len(at_cap) else float("nan"),
         "P3O: first step where seed-mean kappa reaches the cap of 50 (M)")
    n_needed = math.ceil(math.log(50 / 0.01) / math.log(1.1))
    note(sec, "p3o_updates_to_cap_theory", n_needed, "P3O: updates needed in theory, 0.01 * 1.1^n >= 50")
    note(sec, "p3o_steps_to_cap_theory_m", n_needed * A.UPDATE_STEPS / 1e6, "P3O: that many updates in M steps")
    below1 = np.where(kappa >= 1.0)[0]
    note(sec, "p3o_kappa_reaches_1_step_m", float(grid[below1[0]]) if len(below1) else float("nan"),
         "P3O: first step where seed-mean kappa reaches 1 (M)")

    # PPO-Saute: how often its own (discounted) budget ran out, vs its true episode cost
    viol_first, viol_last, cost_last = [], [], []
    for r in A.select(runs, experiment="compare_algos", alg="ppo_saute"):
        v = r.get("episodic/saute_violated")
        c = r.get("episodic/cost")
        v, c = v[np.isfinite(v)], c[np.isfinite(c)]
        viol_first.append(float(np.mean(v[:5])))
        viol_last.append(float(np.mean(v[-5:])))
        cost_last.append(float(np.mean(c[-5:])))
    note(sec, "saute_violated_steps_first5", float(np.mean(viol_first)), "PPO-Saute: steps per episode with its budget exhausted, first 5 logs")
    note(sec, "saute_violated_steps_last5", float(np.mean(viol_last)), "PPO-Saute: same, last 5 logs (seed mean)")
    note(sec, "saute_violated_steps_last5_range", [min(viol_last), max(viol_last)], "PPO-Saute: last-5 range over seeds")
    note(sec, "saute_episode_cost_last5", float(np.mean(cost_last)), "PPO-Saute: true episode cost, last 5 logs (seed mean)")


def curves(runs):
    """Turning points of the seed-mean curves in figures 3 and 4 (compare_algos, <=30.2M).

    PPO-Lag and PPO-PID swing around the budget in two waves; the windows below bracket
    the first dip, first peak, second dip and second peak of their seed-mean cost curve.
    The reward speed is the first step at which the seed-mean training reward reaches 30
    (about 83% of PPO's final reward of 36).
    """
    print("\n=== curve landmarks (seed means, compare_algos, <=30.2M) ===")
    sec = "curves"
    cost_key = "training/safety/env_cost_mean"
    for alg in ("ppo_lag", "ppo_pid"):
        grid, arr = A.seed_curves(A.select(runs, experiment="compare_algos", alg=alg), cost_key, max_step_m=CMP_MAX)
        mean = np.nanmean(arr, axis=0)
        for name, lo, hi, pick in (("dip1", 2, 6, np.nanargmin), ("peak1", 6, 13, np.nanargmax),
                                   ("dip2", 13, 20, np.nanargmin), ("peak2", 20, CMP_MAX, np.nanargmax)):
            idx = np.where((grid >= lo) & (grid <= hi))[0]
            i = idx[pick(mean[idx])]
            note(sec, f"{alg}_{name}", [float(mean[i]), float(grid[i])], f"{alg}: {name} in {lo}-{hi}M (cost, step M)")
    for alg in ("ppo", "p3o", "ppo_saute"):
        grid, arr = A.seed_curves(A.select(runs, experiment="compare_algos", alg=alg), cost_key, max_step_m=CMP_MAX)
        mean = np.nanmean(arr, axis=0)
        early = (grid >= 1) & (grid <= 5)
        note(sec, f"{alg}_early_max", float(np.nanmax(mean[early])), f"{alg}: highest seed-mean cost in 1-5M")
        note(sec, f"{alg}_last", float(mean[np.isfinite(mean)][-1]), f"{alg}: seed-mean cost at the last common update")
    for alg in ALGS:
        grid, arr = A.seed_curves(A.select(runs, experiment="compare_algos", alg=alg), None, max_step_m=CMP_MAX,
                                  series_fn=A.reward_series)
        mean = np.nanmean(arr, axis=0)
        hit = np.where(mean >= 30)[0]
        note(sec, f"{alg}_reward30_step_m", float(grid[hit[0]]) if len(hit) else float("nan"),
             f"{alg}: first step where the seed-mean reward reaches 30 (M)")


def study01(runs):
    """The first six runs (before the experiment scripts): lock-step vs desynced."""
    groups = [(g, [r for r in runs if r.group == g]) for g in
              ("safety_ppo_lockstep", "safety_ppo_lag_lockstep", "safety_ppo_lag_desync")]
    show_groups("study01 (first six runs, full length)", groups,
                ["avg_cost", "avg_excess", "pct_over", "pct_over_10pct", "clear_crossings", "first_under_m", "reward_final"])
    print("\n=== lock-step check (cost vs local trend, % +- standard error, n) ===")
    res = A.lockstep_position_effect(runs)
    out = {}
    for g, by_pos in res.items():
        print(f"  {g:<26}" + "  ".join(f"{p}:{m:+.1f}±{se:.1f}(n={n})" for p, (m, se, n) in by_pos.items()))
        out[g] = {str(p): {"mean_pct": m, "se_pct": se, "n": n} for p, (m, se, n) in by_pos.items()}
    RESULTS["lockstep"] = out


def long_runs(runs):
    """100M-step runs: does the oscillation settle?"""
    groups = [(a, A.select(runs, experiment="long_runs", alg=a)) for a in ("ppo_lag", "ppo_pid")]
    show_groups("long_runs (100M) core", groups, CORE)
    show_groups("long_runs (100M) episodes", groups, EPISODE)
    print("\n=== long_runs: swing size (std of the mean cost over updates, per seed, then averaged) ===")
    out = {}
    for alg, rs in groups:
        for lo, hi in ((5, 25), (25, 50), (50, 75), (75, 105)):
            stds = [float(np.std(r.get("training/safety/env_cost_mean")[(r.steps > lo) & (r.steps <= hi)])) for r in rs]
            out[f"{alg}_{lo}_{hi}M"] = float(np.mean(stds))
            print(f"  {alg:<8} {lo:>3}-{hi:<3}M  std {np.mean(stds):5.2f}  (seeds {', '.join(f'{s:.2f}' for s in stds)})")
    RESULTS["long_runs_swing_std"] = out

    # When do the clear crossings happen, and how often is the mean >10% over budget early on?
    print("\n=== long_runs: first 30M steps vs the whole run ===")
    early_cross, all_cross = 0, 0
    for alg, rs in groups:
        early = [A.run_metrics(r, max_step_m=CMP_MAX) for r in rs]
        whole = [A.run_metrics(r) for r in rs]
        early_cross += sum(m["clear_crossings"] for m in early)
        all_cross += sum(m["clear_crossings"] for m in whole)
        note("long_runs_early", f"{alg}_pct_over_10pct_first30m", float(np.mean([m["pct_over_10pct"] for m in early])),
             f"{alg}: % of updates >10% over budget in the first 30.2M steps")
    note("long_runs_early", "crossings_first30m", early_cross, "clear crossings in the first 30.2M steps (both algorithms, all seeds)")
    note("long_runs_early", "crossings_total", all_cross, "clear crossings over the whole run (both algorithms, all seeds)")


def lambda_lr(runs):
    """Lambda learning-rate sweep (lr 10 = default = compare_algos, 5 seeds; others 3 seeds)."""
    groups = []
    for lr in (1, 3, 10, 30, 100):
        rs = A.select(runs, experiment="compare_algos", alg="ppo_lag") if lr == 10 else A.select(runs, experiment="lambda_lr", lr=float(lr))
        groups.append((f"lr{lr}", rs))
    show_groups("lambda_lr (full 35M) core", groups, CORE, window=FULL)
    show_groups("lambda_lr (full 35M) episodes", groups, EPISODE + ["eval_cost_final"], window=FULL)


def parallel_envs(runs):
    """Number of parallel robots (2048 = compare_algos seeds 0-2)."""
    groups = []
    for alg in ("ppo", "ppo_lag"):
        for n in (512, 1024, 2048, 4096, 8192):
            rs = ([r for r in A.select(runs, experiment="compare_algos", alg=alg) if r.seed <= 2] if n == 2048
                  else A.select(runs, experiment="parallel_envs", alg=alg, num_envs=n))
            groups.append((f"{alg}_n{n}", rs))
    show_groups("parallel_envs (full 35M) core", groups, CORE, window=FULL)
    show_groups("parallel_envs (full 35M) episodes/measurement", groups,
                ["ep_share_over", "ep_final_p99", "slice_final_p99", "sps_median", "eval_cost_final"], window=FULL,
                digits={"sps_median": 0})
    print("\n=== parallel_envs: seeds that did not learn (reward in the last 10 updates < 10) ===")
    out = {}
    for label, rs in groups:
        ms = [A.run_metrics(r, max_step_m=FULL) for r in sorted(rs, key=lambda r: r.seed)]
        rew = [m["reward_final"] for m in ms]
        cost = [m["final_avg_cost"] for m in ms]
        failed = sum(v < 10 for v in rew)
        out[label] = {"failed": failed, "n": len(rs), "rewards": [round(v, 1) for v in rew],
                      "final_quarter_cost": [round(v, 1) for v in cost]}
        print(f"  {label:<14} {failed}/{len(rs)} failed   rewards {', '.join(f'{v:.1f}' for v in rew)}"
              f"   final-quarter mean cost {', '.join(f'{v:.1f}' for v in cost)}  (seed order)")
    RESULTS["parallel_envs_failed"] = out


def harder_levels(runs):
    """Goal Point Levels 1-3 (Level 1 = compare_algos, full 35M, 5 seeds)."""
    groups = []
    for alg in ("ppo", "ppo_lag", "ppo_pid", "crpo"):
        for level in (1, 2, 3):
            rs = (A.select(runs, experiment="compare_algos", alg=alg) if level == 1
                  else A.select(runs, experiment="harder_levels", alg=alg, level=level))
            groups.append((f"{alg}_L{level}", rs))
    show_groups("harder_levels (full 35M) core", groups, CORE, window=FULL)
    show_groups("harder_levels (full 35M) episodes", groups, EPISODE, window=FULL)


def ant_velocity(runs):
    """Ant velocity (budget 25 per 2000-step episode; 55M steps)."""
    groups = [(a, A.select(runs, experiment="ant_velocity", alg=a)) for a in ("ppo", "ppo_lag", "ppo_pid", "crpo")]
    show_groups("ant_velocity core", groups, CORE)
    show_groups("ant_velocity episodes", groups, EPISODE + ["eval_cost_final", "eval_reward_final"])
    print("\n=== ant_velocity per seed ===")
    out = {}
    for alg, rs in groups:
        for r in sorted(rs, key=lambda r: r.seed):
            m = A.run_metrics(r)
            out[f"{alg}_seed{r.seed}"] = {k: m[k] for k in ("final_avg_cost", "ep_final_share_over", "reward_final", "eval_cost_final")}
            print(f"  {alg:<8} seed {r.seed}: final-quarter mean cost {m['final_avg_cost']:6.1f}  episodes over budget "
                  f"{fmt(m['ep_final_share_over'])}%  reward {m['reward_final']:6.1f}  final eval cost {fmt(m['eval_cost_final'])}")
    RESULTS["ant_per_seed"] = out

    # Figure 18 cuts the constrained panels at 100: how many single updates lie above that?
    vals = np.concatenate([r.get("training/safety/env_cost_mean") for a in ("ppo_lag", "ppo_pid", "crpo")
                           for r in A.select(runs, experiment="ant_velocity", alg=a)])
    vals = vals[np.isfinite(vals)]
    note("ant_spikes", "updates_over_100", int(np.sum(vals > 100)), "PPO-Lag/PID/CRPO updates with mean cost above 100 (all seeds)")
    note("ant_spikes", "updates_total", int(len(vals)), "out of this many updates")
    note("ant_spikes", "max_update_cost", float(vals.max()), "largest single-update mean cost")

    # Largest spike of every PPO-Lag / PPO-PID seed, and the mean cost from 2M steps after it
    # to the end of the run (does the policy become over-cautious after a large spike?).
    per_seed = {}
    print("  largest single-update cost per PPO-Lag / PPO-PID seed:")
    for alg in ("ppo_lag", "ppo_pid"):
        for r in sorted(A.select(runs, experiment="ant_velocity", alg=alg), key=lambda r: r.seed):
            c, lam = r.get("training/safety/env_cost_mean"), r.get("training/lambda_lagr")
            k = int(np.nanargmax(c))
            later = c[r.steps > r.steps[k] + 2.0]
            per_seed[f"{alg}_seed{r.seed}"] = {"max_cost": float(c[k]), "step_m": float(r.steps[k]),
                                               "lambda_before": float(lam[k - 1]), "lambda_max_after": float(np.nanmax(lam[k:k + 4])),
                                               "mean_cost_after": float(np.nanmean(later)) if len(later) else float("nan")}
            d = per_seed[f"{alg}_seed{r.seed}"]
            print(f"    {alg:<8} seed {r.seed}: {d['max_cost']:6.0f} at {d['step_m']:.1f}M, lambda {d['lambda_before']:.2f} -> "
                  f"{d['lambda_max_after']:.2f}, mean cost from 2M later to the end {d['mean_cost_after']:.1f}")
    RESULTS["ant_spike_per_seed"] = per_seed

    # The largest spike, in a PPO-Lag run: how lambda and the reward respond to it.
    lag_runs = A.select(runs, experiment="ant_velocity", alg="ppo_lag")
    run = max(lag_runs, key=lambda r: np.nanmax(r.get("training/safety/env_cost_mean")))
    cost, lam, rew = run.get("training/safety/env_cost_mean"), run.get("training/lambda_lagr"), A.reward_series(run)
    i = int(np.nanargmax(cost))
    after = slice(i + 1, i + 21)                       # the next 20 updates (~5M steps)
    sec = "ant_largest_spike"
    note(sec, "seed", run.seed, "PPO-Lag seed with the largest single-update cost")
    note(sec, "step_m", float(run.steps[i]), "step of that update (M)")
    note(sec, "cost", float(cost[i]), "its mean cost per 2000-step episode")
    note(sec, "lambda_before", float(lam[i - 1]), "lambda at the update before")
    note(sec, "lambda_max_after", float(np.nanmax(lam[i:i + 4])), "highest lambda in the next few updates")
    note(sec, "reward_before", float(np.nanmean(rew[i - 10:i])), "mean training reward over the 10 updates before")
    note(sec, "reward_next_5m", float(np.nanmean(rew[after])), "mean training reward over the next 20 updates")
    note(sec, "lambda_end", float(lam[np.isfinite(lam)][-1]), "lambda at the last update")
    # Fall of lambda per update once the cost is ~0. Removed averaged rows leave gaps of two
    # updates, so each change is divided by the number of updates between the two rows.
    upd = run.steps / (A.UPDATE_STEPS / 1e6)
    j = np.arange(i + 4, len(lam) - 1)
    ok = (cost[j + 1] < 1) & np.isfinite(lam[j]) & np.isfinite(lam[j + 1])
    rate = (lam[j] - lam[j + 1])[ok] / (upd[j + 1] - upd[j])[ok]
    note(sec, "lambda_drop_per_update_when_cost_below_1", float(np.mean(rate)),
         "mean fall of lambda per update while cost < 1 (update rule: 10 x 25/2000 = 0.125 at zero cost)")


def measurement(runs):
    """Slice-based (per-update window) vs episode-based metrics, Goal Point runs."""
    print("\n=== measurement: slice vs episode metrics (final quarter, Goal Point, new batch) ===")
    ratios_mean, ratios_p99 = [], {}
    for r in runs:
        if r.experiment == "study01" or r.env != "safe_goal_point":
            continue
        m = A.run_metrics(r)
        if not (np.isfinite(m["ep_final_mean"]) and np.isfinite(m["ep_final_p99"])):
            continue
        ratios_mean.append(m["slice_final_mean"] / m["ep_final_mean"])
        ratios_p99.setdefault(r.num_envs, []).append(m["slice_final_p99"] / m["ep_final_p99"])
    q = np.percentile(ratios_mean, [5, 50, 95])
    note("measurement", "mean_ratio_p5_p50_p95", [float(v) for v in q], "slice mean / episode mean, 5th/50th/95th percentile over runs")
    for n in sorted(ratios_p99):
        v = ratios_p99[n]
        note("measurement", f"p99_ratio_n{n}", float(np.median(v)), f"slice p99 / episode p99, {n} robots (median of {len(v)} runs)")


def cross_cutting(runs):
    """Statements that pool experiments (constrained runs that learned normally)."""
    print("\n=== cross-cutting ===")
    sec = "cross_cutting"
    safe = [r for r in runs if r.alg in ("ppo_lag", "ppo_pid", "crpo", "focops") and r.experiment != "study01"
            and not (r.experiment == "parallel_envs" and r.num_envs >= 4096)]
    ms = [A.run_metrics(r) for r in safe]
    ok = [m for m in ms if np.isfinite(m["eval_cost_final"])]
    within = [m for m in ok if m["eval_cost_final"] <= A.BUDGET]
    note(sec, "runs_considered", len(ok), "constrained runs with a final evaluation (fig 20)")
    note(sec, "final_eval_within_budget", len(within), "of those, final evaluation cost <= 25")
    note(sec, "pct_over_range_when_final_safe", [min(m["pct_over"] for m in within), max(m["pct_over"] for m in within)],
         "% of training updates over budget in those runs (min, max)")
    note(sec, "pct_over_mean_when_final_safe", float(np.mean([m["pct_over"] for m in within])), "mean of that")

    # When the mean episode cost sits at the budget, how many episodes exceed it?
    goal = [A.run_metrics(r) for r in runs if r.env == "safe_goal_point" and r.experiment != "study01"
            and not (r.experiment == "parallel_envs" and r.num_envs >= 4096)]
    near = [m["ep_final_share_over"] for m in goal if np.isfinite(m["ep_final_mean"]) and 23.75 <= m["ep_final_mean"] <= 26.25]
    note(sec, "share_over_when_mean_at_budget_n", len(near), "Goal Point runs whose final-quarter mean episode cost is within 5% of 25")
    note(sec, "share_over_when_mean_at_budget", [float(np.mean(near)), float(min(near)), float(max(near))],
         "their % of episodes over budget (mean, min, max)")
    for label, levels in (("L1", (1,)), ("L23", (2, 3))):
        vals = [A.run_metrics(r)["ep_final_share_over"] for r in runs
                if r.env == "safe_goal_point" and r.experiment != "study01" and r.level in levels
                and not (r.experiment == "parallel_envs" and r.num_envs >= 4096)
                and np.isfinite(A.run_metrics(r)["ep_final_mean"]) and 23.75 <= A.run_metrics(r)["ep_final_mean"] <= 26.25]
        note(sec, f"share_over_when_mean_at_budget_{label}", [float(np.mean(vals)), float(min(vals)), float(max(vals)), len(vals)],
             f"  {label}: % of episodes over budget (mean, min, max, n runs), fig 21 annotation")
    med = [m["ep_final_p50"] / m["ep_final_mean"] for m in goal if np.isfinite(m["ep_final_p50"]) and m["ep_final_mean"] > 0]
    note(sec, "median_over_mean_ratio", float(np.median(med)), "episode median / episode mean (skew of the cost distribution)")

    # Which runs keep fewer than 10% of their episodes over budget (final quarter)?
    low = []
    for r in runs:
        if r.env != "safe_goal_point" or r.experiment == "study01" or (r.experiment == "parallel_envs" and r.num_envs >= 4096):
            continue
        m = A.run_metrics(r)
        if np.isfinite(m["ep_final_share_over"]) and m["ep_final_share_over"] < 10:
            low.append({"group": r.group, "seed": r.seed, "ep_final_mean": m["ep_final_mean"],
                        "ep_final_share_over": m["ep_final_share_over"]})
    low.sort(key=lambda d: d["ep_final_mean"])
    RESULTS[sec]["runs_under_10pct_over"] = low
    print(f"  Goal Point runs with <10% of final-quarter episodes over budget: {len(low)}")
    for d in low:
        print(f"    {d['group']} seed {d['seed']}: mean episode cost {d['ep_final_mean']:.1f}, "
              f"{d['ep_final_share_over']:.1f}% over budget")


SECTIONS = [("inventory", inventory), ("compare", compare_algos), ("mechanisms", mechanisms), ("curves", curves),
            ("study01", study01),
            ("long", long_runs), ("lambda", lambda_lr), ("parallel", parallel_envs), ("levels", harder_levels),
            ("ant", ant_velocity), ("measurement", measurement), ("cross", cross_cutting)]


def _jsonable(x):
    """Convert numpy types / NaN so json.dump works (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    # The numbers of the first-batch report: first-batch runs only, so later experiments
    # (core_rerun, fixed_baselines, cpo, other_tasks) never change them.
    runs = A.batch1(A.load_runs())
    print(f"Loaded {len(runs)} runs from {A.EXPORTS}")
    np.seterr(all="ignore")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)       # NaN means / empty slices are expected
    for name, fn in SECTIONS:
        if only in name:
            fn(runs)
    if only:
        # a filtered run only computes some sections; do not overwrite the full JSON with them
        print(f"\n(Filtered to sections containing '{only}': {OUT_JSON} was not rewritten.)")
        return
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(_jsonable(RESULTS), fh, indent=1)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
