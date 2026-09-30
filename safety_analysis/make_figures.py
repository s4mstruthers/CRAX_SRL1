"""Produce every figure in the training-time safety reports.

Usage (repo folder, crax env active; needs numpy + matplotlib):
    python safety_analysis/make_figures.py            # all 21 figures -> safety_analysis/figures/*.png
    python safety_analysis/make_figures.py fig07      # only figures whose name starts with fig07
    python safety_analysis/make_figures.py --pdf      # also write vector PDFs (for the paper)

Every figure is built from analysis_lib (loading, cleaning, metrics), the same code
key_numbers.py uses for the numbers quoted in the reports.

Conventions used in every plot
------------------------------
  * dashed black line = budget d = 25 per episode;
  * thin lines = individual seeds, thick line = mean over seeds;
  * dots in dot plots = individual seeds, short horizontal bar = mean over seeds;
  * algorithm-comparison plots (figures 3-9) stop at 30.2M steps, so that PPO-Saute
    (which trained for 30M steps) is compared over the same window as the others;
  * figure numbers (figNN_*.png) match the figure numbers in the in-depth report.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np

import analysis_lib as A

# Seed means over grid points where no seed has a value yet (e.g. before the first full
# episodes finish) are NaN by design; silence numpy's warning about those empty slices.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

OUT = os.path.join(A.HERE, "figures")
D = A.BUDGET
CMP_MAX = 30.2                    # algorithm comparison window (million steps)

# ---- palette (validated categorical order; baseline PPO in neutral grey) ----
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ALGS = ["ppo", "ppo_lag", "ppo_pid", "crpo", "focops", "p3o", "ppo_saute"]
ALG_COLOR = {"ppo": "#8a8983", "ppo_lag": "#2a78d6", "ppo_pid": "#eb6834", "crpo": "#1baf7a",
             "focops": "#eda100", "p3o": "#e87ba4", "ppo_saute": "#008300"}
ORDINAL5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]      # light -> dark blue
LIGHT_BLUE = "#86b6ef"            # PPO-Lag with desynced robots (same algorithm, lighter step)
FAILED_SHADE = "#d9d8d2"          # background for settings where learning failed
LABEL = A.ALG_LABEL

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def budget_line(ax, band=False, label=True, x=None, corner=False):
    """Dashed budget line, optionally with the +-10% band used for 'clear crossings'.

    With band=True or corner=True the label becomes a small key in the top-right corner
    instead of a label on the line, for plots where the curves sit on the line itself.
    """
    if band:
        ax.axhspan(0.9 * D, 1.1 * D, color="#8a8983", alpha=0.13, lw=0)
    ax.axhline(D, color=INK, lw=1.0, ls="--", zorder=1)
    if label and (band or corner):
        key = "dashed: budget (25)" + ("\ngrey band: ±10%" if band else "")
        ax.text(0.98, 0.96, key, transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK2, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    elif label:
        xmax = ax.get_xlim()[1] if x is None else x
        ax.text(xmax, D * 1.04, "budget", ha="right", va="bottom", fontsize=8, color=INK)


def seeds_and_mean(ax, runs, key, color, max_step=None, series_fn=None, lw=2.0, seed_alpha=0.35):
    """Thin line per seed plus a thick mean line."""
    grid, arr = A.seed_curves(runs, key, max_step_m=max_step, series_fn=series_fn)
    for row in arr:
        ax.plot(grid, row, color=color, lw=0.8, alpha=seed_alpha)
    mean = np.nanmean(arr, axis=0)
    ax.plot(grid, mean, color=color, lw=lw)
    return grid, mean


def dotplot(ax, cats, values, colors, fmt="{:.0f}", ylim=None, logy=False, labels=None, never_y=None):
    """Seed dots plus a mean bar per category; the mean is written above the dots.

    never_y: if given, missing values (NaN, e.g. 'never got under budget') are drawn as
    open circles at this height and the category gets no mean unless every seed has a value.
    """
    rng = np.random.default_rng(0)
    span = (ax.get_ylim()[1] if ylim is None else ylim[1])
    for i, (c, vals) in enumerate(zip(cats, values)):
        vals = np.array(vals, float)
        ok = np.isfinite(vals)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        if never_y is not None and (~ok).any():
            ax.scatter(i + jitter[~ok], np.full((~ok).sum(), never_y), s=22, facecolor="white",
                       edgecolor=colors[i], linewidth=1.1, zorder=3)
        if not ok.any():
            continue
        ax.scatter(i + jitter[ok], vals[ok], s=22, color=colors[i], edgecolor="white", linewidth=0.6, zorder=3)
        top = max(vals[ok].max(), vals[ok].mean())
        if never_y is not None and (~ok).any():
            ax.annotate(f"{ok.sum()}/{len(vals)}", (i, top), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
            continue
        m = vals[ok].mean()
        ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=1.6, zorder=4)
        # label 5 points above the highest dot (works on linear, log and symlog axes)
        ax.annotate(fmt.format(m), (i, top), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.8, color=INK)
    if never_y is not None:
        ax.axhline(never_y - span * 0.04, color=AXIS, lw=0.7)
        ticks = [t for t in ax.get_yticks() if 0 <= t < never_y - span * 0.06]
        ax.set_yticks(ticks + [never_y])
        ax.set_yticklabels([f"{t:g}" for t in ticks] + ["never"])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels or cats, rotation=30 if len(cats) > 5 else 0, ha="right" if len(cats) > 5 else "center",
                       rotation_mode="anchor", fontsize=8)
    ax.grid(axis="x", visible=False)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


def trend_dots(ax, xs, per_x_values, color, label=None, logx=False):
    """Seed dots at each x plus a line through the means (for ordered sweeps)."""
    means = []
    for x, vals in zip(xs, per_x_values):
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals):
            ax.scatter(np.full(len(vals), x), vals, s=18, color=color, alpha=0.55, edgecolor="white", linewidth=0.5, zorder=3)
            means.append(vals.mean())
        else:
            means.append(np.nan)
    ax.plot(xs, means, color=color, lw=2, marker="o", ms=4.5, label=label, zorder=4)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.minorticks_off()
    return means


def end_labels(ax, x, ys, names, min_gap=0.075, log=False, **text_kw):
    """Write `names` at height `ys` (line ends), nudged apart so they never overlap.

    min_gap is the smallest allowed vertical gap as a fraction of the axis height
    (measured in log10 units when log=True).
    """
    lo, hi = ax.get_ylim()
    tf = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    gap = min_gap * (tf(hi) - tf(lo))
    order = np.argsort(ys)
    placed = []
    for i in order:
        y = tf(ys[i])
        if placed and y < placed[-1] + gap:
            y = placed[-1] + gap
        placed.append(y)
    for i, y in zip(order, placed):
        ax.text(x, inv(y), names[i], va="center", **text_kw)


def plain_log_ticks(ax, ticks, both=False):
    """Plain-number tick labels (25, 50, 100 ...) on a log axis instead of 10^x."""
    axes_to_set = [ax.yaxis, ax.xaxis] if both else [ax.yaxis]
    for axis in axes_to_set:
        axis.set_ticks(ticks)
        axis.set_ticklabels([f"{t:g}" for t in ticks])
        axis.set_minor_locator(matplotlib.ticker.NullLocator())


def metric_values(runs, key, **kw):
    """One run_metrics value per run (e.g. one value per seed)."""
    return [A.run_metrics(r, **kw)[key] for r in runs]


def shade_failed_learning(ax, label=True, y_text=0.97, x_text=np.sqrt(4096 * 8192)):
    """Grey background over 4096 and 8192 robots (log x-axis), where learning failed.

    With the default hyperparameters, 1-2 of 3 seeds at 4096 robots and all 3 seeds at
    8192 robots never learned the task (PPO too, so it is not a safety-method effect).
    Values there mix safety with that failure and should not be read as a safety trend.
    """
    x0 = np.sqrt(2048 * 4096)                          # halfway between 2048 and 4096 on a log axis
    ax.axvspan(x0, 8192 * 1.6, color=FAILED_SHADE, alpha=0.45, lw=0, zorder=0)
    ax.set_xlim(512 / 1.35, 8192 * 1.35)
    if label:
        # x in data units, y as a fraction of the axis height (works on linear and log axes)
        tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(x_text, y_text, "learning\nfailed in\nsome seeds", transform=tr,
                ha="center", va="top", fontsize=7.5, color=INK2, zorder=5)


SAVE_PDF = False                  # set by the --pdf command-line flag


def save(fig, name):
    """Write the figure to safety_analysis/figures/<name>.png (and .pdf with --pdf), then close it."""
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"))
    if SAVE_PDF:
        fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------------------------
# Run selection shared by several figures
# ---------------------------------------------------------------------------------

LRS = [1, 3, 10, 30, 100]


def lr_runs(runs, lr):
    """PPO-Lag runs with a given lambda learning rate (10 = the default, from compare_algos)."""
    if lr == 10:
        return A.select(runs, experiment="compare_algos", alg="ppo_lag")
    return A.select(runs, experiment="lambda_lr", lr=float(lr))


NENVS = [512, 1024, 2048, 4096, 8192]


def env_runs(runs, alg, n):
    """Runs of the parallel-robots sweep (2048 robots comes from compare_algos, seeds 0-2)."""
    if n == 2048:
        return [r for r in A.select(runs, experiment="compare_algos", alg=alg) if r.seed <= 2]
    return A.select(runs, experiment="parallel_envs", alg=alg, num_envs=n)


LEVEL_ALGS = ["ppo_lag", "ppo_pid", "crpo"]


def level_runs(runs, alg, level):
    """Runs of one algorithm at one Goal Point level (Level 1 comes from compare_algos, 5 seeds)."""
    if level == 1:
        return A.select(runs, experiment="compare_algos", alg=alg)
    return A.select(runs, experiment="harder_levels", alg=alg, level=level)


ANT_ALGS = ["ppo", "ppo_lag", "ppo_pid", "crpo"]


def safe_runs_for_crosscut(runs):
    """Constrained runs that learned normally (excludes PPO, Saute, P3O and >=4096 robots)."""
    return [r for r in runs if r.alg in ("ppo_lag", "ppo_pid", "crpo", "focops") and r.experiment != "study01"
            and not (r.experiment == "parallel_envs" and r.num_envs >= 4096)]


SETTING_MARKER = {"Goal Point, Level 1": "o", "Goal Point, Levels 2-3": "^", "Ant velocity": "s"}


def setting_of(run):
    """Environment setting of a run, used for marker shapes in the cross-cutting plots."""
    if run.env == "safe_velocity_ant":
        return "Ant velocity"
    return "Goal Point, Level 1" if run.level == 1 else "Goal Point, Levels 2-3"


# ---------------------------------------------------------------------------------
# Report section 2: can the measurements be trusted?
# ---------------------------------------------------------------------------------

def fig01_measurement_mean_vs_tail(runs):
    """Slice-based vs episode-based metrics: the mean agrees, the tail does not."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    usable = [r for r in runs if r.experiment != "study01" and np.isfinite(A.run_metrics(r)["ep_final_mean"])]
    ms = [(r, A.run_metrics(r)) for r in usable]
    goal = [(r, m) for r, m in ms if r.env == "safe_goal_point"]
    # (a) means
    ax = axes[0]
    x = [m["ep_final_mean"] for r, m in goal]
    y = [m["slice_final_mean"] for r, m in goal]
    ax.scatter(x, y, s=16, color="#2a78d6", alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.plot([1, 300], [1, 300], color=INK, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5, 250); ax.set_ylim(5, 250)
    plain_log_ticks(ax, [5, 10, 25, 50, 100, 200], both=True)
    ax.set_xlabel("True mean episode cost (final quarter)")
    ax.set_ylabel("Slice-based mean (final quarter)")
    ax.set_title("Mean: slice estimate matches episodes", loc="left", color=INK)
    # (b) p99
    ax = axes[1]
    for r, m in goal:
        col = ORDINAL5[NENVS.index(r.num_envs)]
        ax.scatter(m["ep_final_p99"], m["slice_final_p99"], s=16, color=col, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.plot([1, 3000], [1, 3000], color=INK, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(20, 2000); ax.set_ylim(20, 2000)
    plain_log_ticks(ax, [25, 50, 100, 200, 500, 1000], both=True)
    ax.set_xlabel("True p99 of episode cost (final quarter)")
    ax.set_ylabel("Slice-based p99 (final quarter)")
    ax.set_title("Tail: slice p99 overstates it", loc="left", color=INK)
    ax.text(0.03, 0.95, "colour = number of robots\n(light 512 → dark 8192)\ndotted line = equal", transform=ax.transAxes,
            fontsize=8, color=INK2, va="top")
    # (c) p99 vs robots for PPO-Lag
    ax = axes[2]
    for key, col, lab in [("slice_final_p99", "#86b6ef", "slice-based p99"), ("ep_final_p99", "#1c5cab", "true episode p99")]:
        trend_dots(ax, NENVS, [metric_values(env_runs(runs, "ppo_lag", n), key, max_step_m=34.95) for n in NENVS], col, label=lab, logx=True)
    ax.set_yscale("log")
    ax.set_ylim(30, 2000)
    ax.set_yticks([50, 100, 200, 500, 1000])
    ax.set_yticklabels(["50", "100", "200", "500", "1000"])
    ax.set_xticklabels([f"{n}\n({A.UPDATE_STEPS // n} steps)" for n in NENVS], fontsize=8)
    ax.set_xlabel("Parallel robots (steps each robot runs per update)")
    ax.set_title("PPO-Lag: worst 1% by method", loc="left", color=INK)
    shade_failed_learning(ax, y_text=0.97, x_text=np.sqrt(2896 * 8192))    # left of the 8192 dots
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig01_measurement_mean_vs_tail")


def fig02_lockstep(runs):
    """Lock-step vs desynced robots (first study, 2 seeds per condition).

    Left: desyncing does not change how PPO-Lag learns (mean cost per update).
    Right: with lock-step robots, the first update after every episode boundary reads
    lower than its local trend; with desynced robots it does not.
    """
    res = A.lockstep_position_effect(runs)
    fig, (a, ax) = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.2})
    # (a) learning curves: lock-step (dark) vs desynced (light), thin = seeds
    for sync, col in (("lockstep", ALG_COLOR["ppo_lag"]), ("desync", LIGHT_BLUE)):
        seeds_and_mean(a, A.select(runs, experiment="study01", alg="ppo_lag", sync=sync),
                       "training/safety/env_cost_mean", col, lw=1.8)
    a.set_ylim(0, 60)
    a.set_xlim(0, 35.5)
    budget_line(a, corner=True)
    a.set_xlabel("Training steps (millions)")
    a.set_ylabel("Mean cost per episode")
    a.set_title("PPO-Lag learns the same either way", loc="left", color=INK)
    a.legend(handles=[plt.Line2D([], [], color=ALG_COLOR["ppo_lag"], lw=2, label="lock-step robots"),
                      plt.Line2D([], [], color=LIGHT_BLUE, lw=2, label="desynced robots")],
             loc="upper right", bbox_to_anchor=(1.0, 0.84), fontsize=8)
    ax.set_title("Lock-step leaves a dip at each episode start", loc="left", color=INK)
    # (b) position effect
    cols = {"safety_ppo_lockstep": ALG_COLOR["ppo"], "safety_ppo_lag_lockstep": ALG_COLOR["ppo_lag"], "safety_ppo_lag_desync": LIGHT_BLUE}
    labs = {"safety_ppo_lockstep": "PPO, lock-step", "safety_ppo_lag_lockstep": "PPO-Lag, lock-step", "safety_ppo_lag_desync": "PPO-Lag, desynced"}
    w = 0.26
    for j, g in enumerate(cols):
        pos = sorted(res[g])
        ax.bar([p + (j - 1) * w for p in pos], [res[g][p][0] for p in pos], w, color=cols[g], label=labs[g],
               yerr=[res[g][p][1] for p in pos], error_kw=dict(ecolor=INK2, elinewidth=0.9, capsize=2))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])
    ax.set_xlabel("Update number after an episode boundary (lock-step schedule)")
    ax.set_ylabel("Cost vs local trend (%)")
    ax.set_ylim(-10, 5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.99, 0.97, "error bars: ±1 standard error (all seeds pooled)", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.8, color=INK2)
    save(fig, "fig02_lockstep")


# ---------------------------------------------------------------------------------
# Report section 3: seven algorithms (experiment 'compare_algos', <= 30.2M steps)
# ---------------------------------------------------------------------------------

def fig03_algos_cost_curves(runs):
    """Mean cost per update for the 7 algorithms (small multiples, 5 seeds each)."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, alg in zip(axes.flat, ALGS):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], max_step=CMP_MAX)
        ax.set_title(LABEL[alg], loc="left", color=INK)
        ax.set_ylim(0, 130)
        ax.set_xlim(0, 30.5)
        budget_line(ax, label=(alg == "ppo"))
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.62, "Mean cost per episode of the\ncurrent policy, at every update.\n\nThin lines: 5 seeds\nThick line: mean of seeds\n"
            "Dashed line: budget (25)", transform=ax.transAxes, fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean cost per episode")
    save(fig, "fig03_algos_cost_curves")


def fig04_algos_reward_curves(runs):
    """Episode reward over training for the 7 algorithms, with unconstrained PPO as reference.

    Reward = return of the training episodes that finished in each update
    (episodic/sum_reward). The dashed grey line in every panel is PPO's seed mean, so the
    reward each constrained method gives up can be read directly.
    """
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    ppo_grid, ppo_arr = A.seed_curves(A.select(runs, experiment="compare_algos", alg="ppo"), None,
                                      max_step_m=CMP_MAX, series_fn=A.reward_series)
    ppo_mean = np.nanmean(ppo_arr, axis=0)
    for ax, alg in zip(axes.flat, ALGS):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        if alg != "ppo":
            ax.plot(ppo_grid, ppo_mean, color=ALG_COLOR["ppo"], lw=1.2, ls="--", zorder=2)
        seeds_and_mean(ax, rs, None, ALG_COLOR[alg], max_step=CMP_MAX, series_fn=A.reward_series)
        ax.set_title(LABEL[alg], loc="left", color=INK)
        ax.set_ylim(-4, 40)          # room below 0 for PPO-Saute's penalised reward early on
        ax.set_xlim(0, 30.5)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.55, "Episode reward (return) of the\ntraining episodes finishing in\neach update.\n\nThin lines: 5 seeds\n"
            "Thick line: mean of seeds\nDashed grey: PPO (no constraint),\nfor reference\n\n"
            "PPO-Saute's logged reward includes\nits -1 penalty for steps taken after\nits own budget ran out, so it starts\n"
            "slightly below 0.", transform=ax.transAxes, fontsize=8.8, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward")
    save(fig, "fig04_algos_reward_curves")


def fig05_algos_metrics(runs):
    """Dot plots of the key training-time safety metrics per algorithm."""
    panels = [
        ("avg_cost", "Average training cost per episode", "{:.0f}", (0, 125), False),
        ("pct_over", "% of updates over budget", "{:.0f}%", (0, 115), False),
        ("ep_share_over", "% of finished training episodes over budget", "{:.0f}%", (0, 115), False),
        ("ep_final_p99", "Worst 1% of episodes (p99), final quarter", "{:.0f}", (0, 185), False),
        ("first_under_m", "First update under budget (M steps)", "{:.1f}", (0, 14.2), False),
        ("reward_final", "Episode reward, last 10 updates", "{:.1f}", (0, 45), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), gridspec_kw={"hspace": 0.48})
    short = ["PPO", "Lag", "PID", "CRPO", "FOCOPS", "P3O", "Saute"]
    for ax, (key, title, fmt, ylim, logy) in zip(axes.flat, panels):
        vals = [metric_values(A.select(runs, experiment="compare_algos", alg=a), key, max_step_m=CMP_MAX) for a in ALGS]
        ax.set_ylim(*ylim)
        dotplot(ax, ALGS, vals, [ALG_COLOR[a] for a in ALGS], fmt=fmt, ylim=ylim, logy=logy, labels=short,
                never_y=13.2 if key == "first_under_m" else None)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        if key in ("avg_cost", "ep_final_p99"):
            budget_line(ax, label=False)
    save(fig, "fig05_algos_metrics")


def fig06_algos_tradeoff(runs):
    """Reward vs training cost: small dots = seeds, large ringed dot = mean of the 5 seeds.

    Labels sit at fixed positions (data coordinates) with a thin leader line to the mean,
    because PPO-Lag, PPO-PID, CRPO and FOCOPS overlap near the budget.
    """
    label_at = {"ppo": (84, 38.9), "ppo_lag": (4, 36.2), "ppo_pid": (4, 38.4), "crpo": (36, 27.0),
                "focops": (38, 37.6), "p3o": (62, 37.6), "ppo_saute": (80, 29.0)}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axvspan(0, D, color="#1baf7a", alpha=0.06, lw=0)
    for alg in ALGS:
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        ms = [A.run_metrics(r, max_step_m=CMP_MAX) for r in rs]
        x = np.array([m["avg_cost"] for m in ms])
        y = np.array([m["reward_final"] for m in ms])
        ax.scatter(x, y, s=18, color=ALG_COLOR[alg], alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter([x.mean()], [y.mean()], s=70, color=ALG_COLOR[alg], edgecolor=INK, linewidth=0.9, zorder=4)
        ax.annotate(LABEL[alg], (x.mean(), y.mean()), xytext=label_at[alg], textcoords="data", fontsize=8.5,
                    color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=5))
    ax.axvline(D, color=INK, lw=1, ls="--")
    ax.text(D + 1, 39.3, "budget (25)", fontsize=8, color=INK, ha="left")
    ax.text(1.5, 23.0, "average within budget", fontsize=8, color="#0f7a53", ha="left")
    ax.set_xlabel("Average training cost per episode (updates up to 30M steps)")
    ax.set_ylabel("Episode reward at the end (last 10 updates)")
    ax.set_xlim(0, 110)
    ax.set_ylim(22, 40)
    save(fig, "fig06_algos_tradeoff")


def fig07_algos_penalties(runs):
    """Cost (top) and each algorithm's penalty signal (bottom), seed means + seeds."""
    algs = ["ppo_lag", "ppo_pid", "crpo", "focops", "p3o"]
    pen_title = {"ppo_lag": "λ (Lagrange multiplier)", "ppo_pid": "λ (PID output)", "crpo": "% of updates optimising cost",
                 "focops": "ν (FOCOPS multiplier)", "p3o": "κ (penalty; log scale, cap 50)"}
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.0), sharex=True)
    for j, alg in enumerate(algs):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        ax = axes[0, j]
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], max_step=CMP_MAX, lw=1.8)
        ax.set_ylim(0, 110)
        budget_line(ax, label=(alg == "p3o"))       # the only panel where the label does not sit on a curve
        ax.set_title(LABEL[alg], loc="left", color=INK)
        ax = axes[1, j]
        if alg == "crpo":
            # regime 1 = reward step, 0 = cost step; show a trailing 8-update share of cost steps
            fn = lambda r: 100 * (1 - A.rolling(r.get("training/regime"), 8, np.nanmean))
            seeds_and_mean(ax, rs, None, ALG_COLOR[alg], max_step=CMP_MAX, series_fn=fn, lw=1.8)
            ax.set_ylim(0, 100)
        else:
            seeds_and_mean(ax, rs, A.PENALTY_KEY[alg], ALG_COLOR[alg], max_step=CMP_MAX, lw=1.8)
        if alg == "p3o":
            # kappa grows geometrically (x1.1 per violating update from 0.01), so a log axis
            # shows it as a straight line and makes 'kappa = 1 only after ~13M steps' visible
            ax.set_yscale("log")
            ax.set_ylim(0.006, 100)
            plain_log_ticks(ax, [0.01, 0.1, 1, 10, 50])
            ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
        ax.set_title(pen_title[alg], loc="left", color=INK2, fontsize=8.8, fontweight="normal")
        ax.set_xlabel("Training steps (M)")
    axes[0, 0].set_ylabel("Mean cost per episode")
    axes[1, 0].set_ylabel("Penalty signal")
    save(fig, "fig07_algos_penalties")


def fig08_algos_episode_tail(runs):
    """Episode-level cost distribution over training: mean, p90, p99 (log scale)."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, alg in zip(axes.flat, ALGS):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        ends, names = [], []
        for key, shade, name in [("training/safety_ep/cost_p99", 0.35, "p99"), ("training/safety_ep/cost_p90", 0.65, "p90"),
                                 ("training/safety_ep/cost_mean", 1.0, "mean")]:
            grid, arr = A.seed_curves(rs, key, max_step_m=CMP_MAX)
            curve = np.nanmean(arr, 0)               # NaN before the first full episodes finish
            ax.plot(grid, curve, color=ALG_COLOR[alg], lw=1.8, alpha=shade)
            if np.isfinite(curve).any():
                ends.append(curve[np.isfinite(curve)][-1])
                names.append(name)
        ax.set_yscale("log")
        ax.set_ylim(4, 400)
        end_labels(ax, 30.6, ends, names, log=True, fontsize=7.5, color=INK2)
        ax.set_yticks([5, 10, 25, 50, 100, 200])
        ax.set_yticklabels(["5", "10", "25", "50", "100", "200"])
        ax.set_xlim(0, 34)
        budget_line(ax, label=False)
        ax.set_title(LABEL[alg], loc="left", color=INK)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.6, "True cost of the episodes that\nfinished in each update\n(seed mean).\n\nDarkest: mean\nMid: p90\nLightest: p99 (worst 1%)\n\n"
            "Starts ~2M steps in: shortened\nfirst episodes are excluded.", transform=ax.transAxes, fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode cost (log scale)")
    save(fig, "fig08_algos_episode_tail")


def fig09_algos_share_over(runs):
    """Share of finished episodes over budget, per update."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.0), sharex=True, sharey=True)
    fn = lambda r: 100 * r.get("training/safety_ep/frac_over_budget")
    for ax, alg in zip(axes.flat, ALGS):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        seeds_and_mean(ax, rs, None, ALG_COLOR[alg], max_step=CMP_MAX, series_fn=fn, lw=1.8)
        ax.axhline(50, color=AXIS, lw=0.8)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 30.5)
        ax.set_title(LABEL[alg], loc="left", color=INK)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.6, "Share of the episodes finishing\nin each update whose total\ncost exceeded the budget (25).\n\n"
            "Thin: seeds, thick: mean.\nGrey horizontal line: 50%.",
            transform=ax.transAxes, fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    for ax in axes[:, 0]:
        ax.set_ylabel("% of episodes over budget")
    save(fig, "fig09_algos_share_over")


# ---------------------------------------------------------------------------------
# Report section 4: 100M-step runs (experiment 'long_runs')
# ---------------------------------------------------------------------------------

def fig10_long_cost_curves(runs):
    """100M-step runs: mean cost with the +-10% band."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, alg in zip(axes, ["ppo_lag", "ppo_pid"]):
        rs = A.select(runs, experiment="long_runs", alg=alg)
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], lw=1.6)
        ax.set_ylim(0, 60)
        ax.set_xlim(0, 106)
        budget_line(ax, band=True, label=(alg == "ppo_lag"))
        ax.set_title(LABEL[alg] + " (3 seeds, 100M steps)", loc="left", color=INK)
        ax.set_xlabel("Training steps (millions)")
    axes[0].set_ylabel("Mean cost per episode")
    save(fig, "fig10_long_cost_curves")


def fig11_long_damping(runs):
    """Oscillation amplitude and share of episodes over budget over 100M steps."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    win = 20                                   # 20 updates ~= 5.2M steps
    for alg in ["ppo_lag", "ppo_pid"]:
        rs = A.select(runs, experiment="long_runs", alg=alg)
        seeds_and_mean(axes[0], rs, None, ALG_COLOR[alg], series_fn=lambda r: A.rolling(r.get("training/safety/env_cost_mean"), win), lw=1.8)
        seeds_and_mean(axes[1], rs, None, ALG_COLOR[alg],
                       series_fn=lambda r: 100 * A.rolling(r.get("training/safety_ep/frac_over_budget"), win, np.nanmean), lw=1.8)
    axes[0].set_title("Size of the swings: std of mean cost over ~5M steps", loc="left", color=INK)
    axes[0].set_ylabel("Std of mean cost")
    axes[0].set_ylim(0, 12)
    axes[1].set_title("% of episodes over budget (~5M-step window)", loc="left", color=INK)
    axes[1].set_ylabel("% of episodes over budget")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50, color=AXIS, lw=0.8)
    for ax in axes:
        ax.set_xlim(0, 106)
        ax.set_xlabel("Training steps (millions)")
    axes[0].text(60, 9.5, "PPO-Lag", color=ALG_COLOR["ppo_lag"], fontsize=9, fontweight="bold")
    axes[0].text(60, 8.3, "PPO-PID", color=ALG_COLOR["ppo_pid"], fontsize=9, fontweight="bold")
    save(fig, "fig11_long_damping")


# ---------------------------------------------------------------------------------
# Report section 5: lambda learning rate (experiment 'lambda_lr'; rate 10 = compare_algos)
# ---------------------------------------------------------------------------------

def fig12_lr_curves(runs):
    """Mean cost per update for each lambda learning rate."""
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3), sharey=True)
    for ax, lr, col in zip(axes, LRS, ORDINAL5):
        seeds_and_mean(ax, lr_runs(runs, lr), "training/safety/env_cost_mean", col, max_step=35, lw=1.8)
        ax.set_ylim(0, 60)
        ax.set_xlim(0, 35.5)
        budget_line(ax, band=True, label=(lr == 1))
        ax.set_title(f"λ-rate {lr}" + (" (default)" if lr == 10 else ""), loc="left", color=INK)
        ax.set_xlabel("Training steps (M)")
    axes[0].set_ylabel("Mean cost per episode")
    save(fig, "fig12_lr_curves")


def fig13_lr_dose_response(runs):
    """Metrics against the lambda learning rate (log scale)."""
    panels = [("avg_excess", "Average cost above budget", (0, 12)),
              ("pct_over_10pct", "% of updates >10% over budget", (0, 55)),
              ("first_under_m", "First update under budget (M steps)", (0, 14)),
              ("clear_crossings", "Clear crossings of the budget (±10%)", (0, 9)),
              ("ep_share_over", "% of training episodes over budget", (0, 55)),
              ("reward_final", "Episode reward, last 10 updates", (20, 40))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        # all runs of this sweep are 35M steps long, so the full run is used (max_step_m=34.95)
        trend_dots(ax, LRS, [metric_values(lr_runs(runs, lr), key, max_step_m=34.95) for lr in LRS], "#1c5cab", logx=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("λ learning rate (log scale; default 10)")
    save(fig, "fig13_lr_dose_response")


# ---------------------------------------------------------------------------------
# Report section 6: number of parallel robots (experiment 'parallel_envs'; 2048 = compare_algos seeds 0-2)
# ---------------------------------------------------------------------------------

def fig14_envs_curves(runs):
    """Parallel robots: PPO-Lag cost (row 1) and reward (row 2), and PPO reward (row 3).

    Every seed is shown (thin) with the seed mean (thick). Row 3 shows that unconstrained
    PPO fails to learn at 8192 robots as well, so the failure is not caused by the
    safety method.
    """
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.6), sharex=True)
    for j, (n, col) in enumerate(zip(NENVS, ORDINAL5)):
        rs = env_runs(runs, "ppo_lag", n)
        seeds_and_mean(axes[0, j], rs, "training/safety/env_cost_mean", col, max_step=35, seed_alpha=0.6, lw=1.4)
        axes[0, j].set_ylim(0, 220)
        budget_line(axes[0, j], label=(j == 0), corner=True)
        axes[0, j].set_title(f"{n} robots\n({A.UPDATE_STEPS // n} steps each per update)", loc="left", color=INK, fontsize=9)
        seeds_and_mean(axes[1, j], rs, None, col, max_step=35, series_fn=A.reward_series, seed_alpha=0.6, lw=1.4)
        axes[1, j].set_ylim(0, 40)
        seeds_and_mean(axes[2, j], env_runs(runs, "ppo", n), None, ALG_COLOR["ppo"], max_step=35,
                       series_fn=A.reward_series, seed_alpha=0.6, lw=1.4)
        axes[2, j].set_ylim(0, 40)
        axes[2, j].set_xlabel("Training steps (M)")
    axes[0, 0].set_ylabel("PPO-Lag:\nmean cost per episode")
    axes[1, 0].set_ylabel("PPO-Lag:\nepisode reward")
    axes[2, 0].set_ylabel("PPO (no constraint):\nepisode reward")
    save(fig, "fig14_envs_curves")


def fig15_envs_metrics(runs):
    """Safety, reward and throughput against the number of parallel robots."""
    panels = [("ep_share_over", "% of training episodes over budget", (0, 105)),
              ("reward_final", "Episode reward, last 10 updates", (0, 42)),
              ("avg_cost", "Average training cost per episode", (0, 160)),
              ("sps_median", "Training throughput (thousand steps/s)", (0, 265))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (key, title, ylim) in zip(axes, panels):
        for alg in ["ppo", "ppo_lag"]:
            vals = [metric_values(env_runs(runs, alg, n), key, max_step_m=34.95) for n in NENVS]
            if key == "sps_median":
                vals = [[v / 1000 for v in vs] for vs in vals]
            trend_dots(ax, NENVS, vals, ALG_COLOR[alg], label=LABEL[alg], logx=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("Parallel robots (log scale)")
        # throughput is unaffected by the learning failure, so that panel is not shaded
        if key != "sps_median":
            shade_failed_learning(ax, label=(key == "reward_final"), y_text=0.97)
        ax.set_xlim(512 / 1.35, 8192 * 1.35)
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, "fig15_envs_metrics")


# ---------------------------------------------------------------------------------
# Report section 7: harder levels (experiment 'harder_levels'; Level 1 = compare_algos)
# ---------------------------------------------------------------------------------

def fig16_levels_curves(runs):
    """Mean cost per update by level (rows = algorithm, columns = level)."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    for i, alg in enumerate(LEVEL_ALGS):
        for j, level in enumerate([1, 2, 3]):
            ax = axes[i, j]
            seeds_and_mean(ax, level_runs(runs, alg, level), "training/safety/env_cost_mean", ALG_COLOR[alg], max_step=35, lw=1.6)
            ax.set_ylim(0, 60)
            ax.set_xlim(0, 35.5)
            budget_line(ax, band=True, label=(i == 0 and j == 0))
            ax.set_title(f"{LABEL[alg]} · Level {level}", loc="left", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Training steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean cost per episode")
    save(fig, "fig16_levels_curves")


def fig17_levels_metrics(runs):
    """Metrics against difficulty level for PPO and the three constrained algorithms."""
    panels = [("avg_cost", "Average training cost per episode", (10, 200), [10, 25, 50, 100, 200]),
              ("ep_share_over", "% of training episodes over budget", (0, 105), None),
              ("ep_share_over_2x", "% of episodes over 2× budget", (0, 105), None),
              ("ep_final_p99", "Episode p99 (worst 1%), final quarter", (20, 1000), [25, 50, 100, 200, 500, 1000]),
              ("reward_final", "Episode reward, last 10 updates", (0, 42), None),
              ("pct_over_10pct", "% of updates >10% over budget", (0, 105), None)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), gridspec_kw={"hspace": 0.35, "wspace": 0.22})
    for ax, (key, title, ylim, log_ticks) in zip(axes.flat, panels):
        for alg in ["ppo"] + LEVEL_ALGS:
            trend_dots(ax, [1, 2, 3], [metric_values(level_runs(runs, alg, lv), key) for lv in [1, 2, 3]], ALG_COLOR[alg], label=LABEL[alg])
        if log_ticks:
            ax.set_yscale("log")
            plain_log_ticks(ax, log_ticks)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Level 1", "Level 2", "Level 3"])
        ax.set_xlim(0.75, 3.25)                     # margin so the edge labels are not clipped
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        if key in ("avg_cost", "ep_final_p99"):
            budget_line(ax, label=False)
    axes[0, 1].legend(loc="center right", fontsize=8)
    save(fig, "fig17_levels_metrics")


# ---------------------------------------------------------------------------------
# Report section 8: Ant velocity (experiment 'ant_velocity')
# ---------------------------------------------------------------------------------

def fig18_ant_curves(runs):
    """Ant velocity: mean cost per 2000-step episode (PPO on its own scale)."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    spikes = []                                      # constrained updates above the y-limit of 100
    for ax, alg in zip(axes, ANT_ALGS):
        rs = A.select(runs, experiment="ant_velocity", alg=alg)
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], seed_alpha=0.6, lw=1.3)
        ax.set_ylim(0, 2100 if alg == "ppo" else 100)   # short spikes above 100 are cut off (counted below)
        ax.set_xlim(0, 56)
        budget_line(ax, label=(alg == "ppo_lag"), corner=True)
        ax.set_title("PPO (own scale)" if alg == "ppo" else LABEL[alg], loc="left", color=INK)
        ax.set_xlabel("Training steps (M)")
        if alg != "ppo":
            spikes += [v for r in rs for v in r.get("training/safety/env_cost_mean") if np.isfinite(v)]
    spikes = np.array(spikes)
    fig.text(0.99, -0.03, f"PPO-Lag, PPO-PID and CRPO panels: {np.sum(spikes > 100)} of {len(spikes)} updates "
             f"({100 * np.mean(spikes > 100):.1f}%) lie above 100 and are cut off; the largest is {spikes.max():.0f}.",
             ha="right", va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("Mean cost per 2000-step episode")
    save(fig, "fig18_ant_curves")


def fig19_ant_metrics(runs):
    """Ant velocity: key metrics per algorithm, every seed shown."""
    panels = [("ep_final_share_over", "% of episodes over budget, final quarter", "{:.0f}%", (0, 115)),
              ("clear_crossings", "Clear crossings of the budget (±10%)", "{:.0f}", (0, 70)),
              ("reward_final", "Episode reward, last 10 updates", "{:.0f}", (0, 190)),
              ("eval_cost_final", "Cost of the final policy in evaluation", "{:.0f}", (0, 5000))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0), gridspec_kw={"hspace": 0.32, "wspace": 0.2})
    for ax, (key, title, fmt, ylim) in zip(axes.flat, panels):
        vals = [metric_values(A.select(runs, experiment="ant_velocity", alg=a), key) for a in ANT_ALGS]
        if key == "eval_cost_final":
            # symlog: linear from 0 to 10, logarithmic above, so a cost of exactly 0 can be shown
            ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.set_ylim(*ylim)
        dotplot(ax, ANT_ALGS, vals, [ALG_COLOR[a] for a in ANT_ALGS], fmt=fmt, ylim=ylim,
                labels=["PPO", "Lag", "PID", "CRPO"])
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        if key == "eval_cost_final":
            ax.set_yticks([0, 5, 10, 25, 100, 1000])
            ax.set_yticklabels(["0", "5", "10", "25", "100", "1000"])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            budget_line(ax, label=False)
    save(fig, "fig19_ant_metrics")


# ---------------------------------------------------------------------------------
# Report section 9: findings across experiments
# ---------------------------------------------------------------------------------

def fig20_final_vs_training(runs):
    """Final evaluation cost vs time spent over budget during training (colour = algorithm, shape = setting)."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axvspan(-1.5, D, color="#1baf7a", alpha=0.06, lw=0)
    safe = safe_runs_for_crosscut(runs)
    for alg in ["ppo_lag", "ppo_pid", "crpo", "focops"]:
        for setting, marker in SETTING_MARKER.items():
            ms = [A.run_metrics(r) for r in safe if r.alg == alg and setting_of(r) == setting]
            if ms:
                ax.scatter([m["eval_cost_final"] for m in ms], [m["pct_over"] for m in ms], s=30, marker=marker,
                           color=ALG_COLOR[alg], edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(D, color=INK, lw=1, ls="--")
    ax.text(D + 0.8, 2, "budget", fontsize=8, color=INK)
    ax.text(1, 69, "final policy within budget\n(left of the dashed line), yet\nmany training updates were over it",
            fontsize=8.5, color=INK2, va="top")
    # two-part legend: colour = algorithm, shape = setting
    alg_handles = [plt.Line2D([], [], ls="", marker="o", color=ALG_COLOR[a], label=LABEL[a])
                   for a in ["ppo_lag", "ppo_pid", "crpo", "focops"]]
    set_handles = [plt.Line2D([], [], ls="", marker=m, color=INK2, label=k) for k, m in SETTING_MARKER.items()]
    leg = ax.legend(handles=alg_handles, loc="upper right", fontsize=8, title="Algorithm", title_fontsize=8)
    ax.add_artist(leg)
    ax.legend(handles=set_handles, loc="lower right", fontsize=8, title="Setting", title_fontsize=8)
    ax.set_xlim(-1.5, 60)                     # a little room so a final cost of exactly 0 is visible
    ax.set_ylim(0, 70)
    ax.set_xlabel("Cost of the final policy in evaluation (last evaluation)")
    ax.set_ylabel("% of training updates over budget")
    save(fig, "fig20_final_vs_training")


def near_budget_share(runs, levels):
    """% of episodes over budget in Goal Point runs whose mean episode cost is within 5% of d.

    Uses the final quarter of each run (same selection as key_numbers.cross_cutting).
    Returns (mean, min, max, number of runs).
    """
    vals = []
    for r in runs:
        if r.env != "safe_goal_point" or r.experiment == "study01" or r.level not in levels:
            continue
        if r.experiment == "parallel_envs" and r.num_envs >= 4096:
            continue
        m = A.run_metrics(r)
        if np.isfinite(m["ep_final_mean"]) and 0.95 * D <= m["ep_final_mean"] <= 1.05 * D:
            vals.append(m["ep_final_share_over"])
    return float(np.mean(vals)), float(min(vals)), float(max(vals)), len(vals)


def fig21_mean_vs_share(runs):
    """Mean episode cost vs share of episodes over budget (final quarter), all Goal Point runs.

    Colour = algorithm, shape = level. Runs at 4096/8192 robots (learning failed) and the
    first study are left out. The annotation numbers are computed here from the data.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for alg in ALGS:
        rs = [r for r in runs if r.alg == alg and r.env == "safe_goal_point" and r.experiment != "study01"
              and not (r.experiment == "parallel_envs" and r.num_envs >= 4096)]
        for levels, marker in (((1,), "o"), ((2, 3), "^")):
            ms = [A.run_metrics(r) for r in rs if r.level in levels]
            if ms:
                ax.scatter([m["ep_final_mean"] for m in ms], [m["ep_final_share_over"] for m in ms], s=26, marker=marker,
                           color=ALG_COLOR[alg], edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(D, color=INK, lw=1, ls="--")
    ax.axhline(50, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(8, 200)
    plain_log_ticks(ax, [10, 25, 50, 100, 200], both=True)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([str(t) for t in range(0, 101, 20)])
    ax.set_ylim(0, 105)
    ax.text(D * 0.97, 101, "budget", fontsize=8, color=INK, ha="right", va="top")
    l1, l23 = near_budget_share(runs, (1,)), near_budget_share(runs, (2, 3))
    ax.annotate(f"mean cost within 5% of the budget:\nLevel 1: {l1[1]:.0f}-{l1[2]:.0f}% of episodes over it "
                f"({l1[3]} runs)\nLevels 2-3: {l23[1]:.0f}-{l23[2]:.0f}% ({l23[3]} runs)",
                (D, 45), xytext=(8.6, 76), fontsize=8.3, color=INK2, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlabel("Mean episode cost, final quarter (log scale)")
    ax.set_ylabel("% of episodes over budget, final quarter")
    alg_handles = [plt.Line2D([], [], ls="", marker="o", color=ALG_COLOR[a], label=LABEL[a]) for a in ALGS]
    lvl_handles = [plt.Line2D([], [], ls="", marker="o", color=INK2, label="Level 1"),
                   plt.Line2D([], [], ls="", marker="^", color=INK2, label="Levels 2-3")]
    leg = ax.legend(handles=alg_handles, loc="lower right", fontsize=7.6, title="Algorithm", title_fontsize=7.6)
    ax.add_artist(leg)
    ax.legend(handles=lvl_handles, loc="lower right", bbox_to_anchor=(0.73, 0.0), fontsize=7.6, title="Shape",
              title_fontsize=7.6)
    save(fig, "fig21_mean_vs_share")


FIGURES = [fig01_measurement_mean_vs_tail, fig02_lockstep, fig03_algos_cost_curves,
           fig04_algos_reward_curves, fig05_algos_metrics, fig06_algos_tradeoff,
           fig07_algos_penalties, fig08_algos_episode_tail, fig09_algos_share_over,
           fig10_long_cost_curves, fig11_long_damping, fig12_lr_curves,
           fig13_lr_dose_response, fig14_envs_curves, fig15_envs_metrics,
           fig16_levels_curves, fig17_levels_metrics, fig18_ant_curves,
           fig19_ant_metrics, fig20_final_vs_training, fig21_mean_vs_share]


def remove_old_figures():
    """Delete figNN_*.png/.pdf files that no current figure writes (left by older versions).

    Figure numbers shifted when figures were added, so an old fig05_... would otherwise sit
    next to the new fig05_... and could be mistaken for it.
    """
    if not os.path.isdir(OUT):
        return
    current = {f.__name__ for f in FIGURES}
    for fname in sorted(os.listdir(OUT)):
        stem, ext = os.path.splitext(fname)
        if ext in (".png", ".pdf") and stem[:3] == "fig" and stem[3:5].isdigit() and stem not in current:
            try:
                os.remove(os.path.join(OUT, fname))
                print("  removed old figure", fname)
            except OSError as err:
                print(f"  could not remove old figure {fname}: {err}")


def main():
    global SAVE_PDF
    args = [a for a in sys.argv[1:] if a != "--pdf"]
    SAVE_PDF = "--pdf" in sys.argv[1:]
    only = args[0] if args else ""
    runs = A.load_runs()
    print(f"Loaded {len(runs)} runs from {A.EXPORTS}")
    if not only:
        remove_old_figures()
    made = 0
    for f in FIGURES:
        if f.__name__.startswith(only):
            f(runs)
            made += 1
    if made == 0:
        print(f"No figure name starts with '{only}'. Names: " + ", ".join(f.__name__ for f in FIGURES))


if __name__ == "__main__":
    main()
