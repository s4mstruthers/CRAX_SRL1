"""Summarise training-time safety for every exported run.

Reads the CSVs written by download_wandb.py (safety_analysis/wandb_exports/<group>/<run>.csv),
computes one row of summary metrics per run, writes them to safety_analysis/run_summary.csv,
and prints the mean and range per wandb group.

Usage (repo folder, crax env active; standard library only):
    python safety_analysis/summarise_runs.py
    python safety_analysis/summarise_runs.py --budget 25 --after 5

Metric definitions (d = budget per 1000-step episode):
  Per-update slice metrics (training/safety/*): each robot's cost over the update's slice
  (128 steps at 2048 envs), scaled to an episode. Accurate for the MEAN; the tail
  (p90/p99/max) is inflated by the short window.
  Episode-level metrics (training/safety_ep/*, runs from 30 Sep 2026 onwards): the true
  cost of episodes that finished during the update. Use these for tail statements.
"""
import argparse
import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(HERE, "wandb_exports")


def num(v):
    """Parse a CSV cell to float; empty or non-numeric cells become None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def summarise(path, budget, after_m):
    """Compute the summary metrics for one run's CSV."""
    rows = list(csv.DictReader(open(path)))
    upd = [r for r in rows if num(r.get("training/safety/env_cost_mean")) is not None]
    if not upd:
        return None
    col = lambda key: [num(r.get(key)) for r in upd]
    steps = [num(r["_step"]) / 1e6 for r in upd]
    mean = col("training/safety/env_cost_mean")
    lam = col("training/lambda_lagr")
    late = [i for i, s in enumerate(steps) if s > after_m]

    # Crossings of the budget with a +-10% hysteresis band, so wobbles around d do not count.
    state, clear_crossings = None, 0
    for v in mean:
        new = "over" if v > 1.1 * budget else ("under" if v < 0.9 * budget else state)
        if state is not None and new != state:
            clear_crossings += 1
        state = new

    excess = [max(0.0, v - budget) for v in mean]
    ep_frac = col("training/safety_ep/frac_over_budget")
    ep_frac2 = col("training/safety_ep/frac_over_2x_budget")
    ep_p90, ep_p99 = col("training/safety_ep/cost_p90"), col("training/safety_ep/cost_p99")
    ep_mean, ep_count = col("training/safety_ep/cost_mean"), col("training/safety_ep/count")
    rew = [num(r.get("episodic/reward")) for r in upd[-10:]]
    evc = [num(r.get("eval/episode_cost")) for r in rows if num(r.get("eval/episode_cost")) is not None]
    evr = [num(r.get("eval/episode_reward")) for r in rows if num(r.get("eval/episode_reward")) is not None]

    return {
        "group": os.path.basename(os.path.dirname(path)),
        "run": os.path.splitext(os.path.basename(path))[0],
        "updates": len(mean),
        "total_cost": sum(mean),
        "total_excess": sum(excess),
        "excess_share_first8": sum(excess[:8]) / sum(excess) if sum(excess) else None,
        "pct_updates_over": 100 * sum(v > budget for v in mean) / len(mean),
        "pct_updates_over_10pct": 100 * sum(v > 1.1 * budget for v in mean) / len(mean),
        "clear_crossings": clear_crossings,
        "lambda_zero_updates": sum(1 for v in lam if v == 0) if lam[0] is not None else None,
        "slice_p99_median": st.median(col("training/safety/env_cost_p99")),
        "slice_frac_envs_late": mean_or_none([col("training/safety/frac_envs_with_cost")[i] for i in late]),
        # Episode-level metrics (only present for runs with the safety_ep/* logging).
        "ep_updates_with_data": sum(1 for v in ep_count if v),
        "ep_mean_late": mean_or_none([ep_mean[i] for i in late]),
        "ep_p90_late": mean_or_none([ep_p90[i] for i in late]),
        "ep_p99_late": mean_or_none([ep_p99[i] for i in late]),
        "ep_pct_over_budget_late": None if mean_or_none([ep_frac[i] for i in late]) is None
        else 100 * mean_or_none([ep_frac[i] for i in late]),
        "ep_pct_over_2x_budget_late": None if mean_or_none([ep_frac2[i] for i in late]) is None
        else 100 * mean_or_none([ep_frac2[i] for i in late]),
        "reward_last10": mean_or_none(rew),
        "eval_cost_final": evc[-1] if evc else None,
        "eval_reward_final": evr[-1] if evr else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=float, default=25.0, help="cost budget d per episode")
    parser.add_argument("--after", type=float, default=5.0,
                        help="'late' metrics use updates after this many million steps")
    args = parser.parse_args()

    results = [r for p in sorted(glob.glob(os.path.join(EXPORTS, "*", "*.csv")))
               if (r := summarise(p, args.budget, args.after))]
    if not results:
        print(f"No exported runs found in {EXPORTS}. Run download_wandb.py first.")
        return

    out = os.path.join(HERE, "run_summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} runs to {out}\n")

    # Per-group mean and range for the headline metrics.
    keys = ["total_cost", "total_excess", "pct_updates_over", "pct_updates_over_10pct", "clear_crossings",
            "slice_p99_median", "ep_p99_late", "ep_pct_over_budget_late", "ep_pct_over_2x_budget_late",
            "reward_last10", "eval_cost_final"]
    groups = sorted({r["group"] for r in results})
    for g in groups:
        rs = [r for r in results if r["group"] == g]
        print(f"{g}  ({len(rs)} run{'s' if len(rs) > 1 else ''})")
        for k in keys:
            vals = [r[k] for r in rs if r[k] is not None]
            if vals:
                print(f"    {k:28s} mean {st.mean(vals):9.2f}   range {min(vals):9.2f} - {max(vals):9.2f}")
        print()


if __name__ == "__main__":
    main()
