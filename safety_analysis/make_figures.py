"""Produce every figure in the training-time safety reports.

Usage (repo folder, crax env active; needs numpy + matplotlib):
    python safety_analysis/make_figures.py            # all 33 figures -> safety_analysis/figures/*.png
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
import matplotlib.patches
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
        ax.text(x, inv(y), names[i], va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6), **text_kw)


def legend_above(fig, handles, top=0.9, ncol=None, fontsize=9.5):
    """One shared legend in a row above all panels, so it never covers any data.

    top is where the panels end (figure fraction); the legend sits just above the panel
    titles (22 points higher, whatever the figure height).
    """
    fig.subplots_adjust(top=top)
    gap = 22 / (fig.get_figheight() * 72)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top + gap), ncol=ncol or len(handles),
               fontsize=fontsize, handlelength=2.2, columnspacing=2.0, frameon=False)


def line_key(color, label, lw=2.0, ls="-", marker=None):
    """Legend entry for a line (optionally with a marker)."""
    return plt.Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, ms=5, label=label)


def seed_key(color=INK2):
    """Legend entry for the small dots that mark individual seeds."""
    return plt.Line2D([], [], ls="", marker="o", ms=4.5, color=color, alpha=0.55, label="one seed")


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
    ax.plot([1, 300], [1, 300], color=INK, lw=0.9, ls=":", label="equal (slice = episode)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5, 250); ax.set_ylim(5, 250)
    ax.legend(loc="upper left", fontsize=7.8)
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
    robot_keys = [plt.Line2D([], [], ls="", marker="o", ms=5.5, color=ORDINAL5[i], label=f"{n} robots")
                  for i, n in enumerate(NENVS)]
    ax.legend(handles=robot_keys + [line_key(INK, "equal (slice = episode)", lw=0.9, ls=":")],
              loc="upper left", fontsize=7.8)
    # (c) p99 vs robots for PPO-Lag
    ax = axes[2]
    for key, col, lab in [("slice_final_p99", "#86b6ef", "slice-based p99"), ("ep_final_p99", "#1c5cab", "true episode p99")]:
        trend_dots(ax, NENVS, [metric_values(env_runs(runs, "ppo_lag", n), key, max_step_m=34.95) for n in NENVS], col, label=lab, logx=True)
    ax.set_yscale("log")
    ax.set_ylim(30, 2000)
    ax.set_yticks([50, 100, 200, 500, 1000])
    ax.set_yticklabels(["50", "100", "200", "500", "1000"])
    ax.set_xticklabels([f"{n}\n({A.UPDATE_STEPS // n})" for n in NENVS], fontsize=8)
    ax.set_xlabel("Parallel robots (steps per robot per update)")
    ax.set_ylabel("p99 of episode cost (final quarter)")
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
    budget_line(a, label=False)
    a.set_xlabel("Training steps (millions)")
    a.set_ylabel("Mean cost per episode")
    a.set_title("PPO-Lag learns the same either way", loc="left", color=INK)
    a.legend(handles=[line_key(ALG_COLOR["ppo_lag"], "lock-step robots (mean of 2 seeds)"),
                      line_key(LIGHT_BLUE, "desynced robots (mean of 2 seeds)"),
                      line_key(INK, "budget (25)", lw=1.0, ls="--")], loc="upper right", fontsize=8)
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
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45})   # room for the x-label under the top-right panel
    for ax, alg in zip(axes.flat, ALGS):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], max_step=CMP_MAX)
        ax.set_title(LABEL[alg], loc="left", color=INK)
        ax.set_ylim(0, 140)
        ax.set_xlim(0, 30.5)
        budget_line(ax, label=(alg == "ppo"))
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.62, "Mean cost per episode of the\ncurrent policy, at every update.\n\nThin lines: 5 seeds\nThick line: mean of seeds\n"
            "Dashed line: budget (25)", transform=ax.transAxes, fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    axes[0, 3].set_xlabel("Training steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean cost per episode")
    save(fig, "fig03_algos_cost_curves")


def fig04_algos_reward_curves(runs):
    """Episode reward over training for the 7 algorithms, with unconstrained PPO as reference.

    Reward = return of the training episodes that finished in each update
    (episodic/sum_reward). The dashed grey line in every panel is PPO's seed mean, so the
    reward each constrained method gives up can be read directly.
    """
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45})   # room for the x-label under the top-right panel
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
    axes[0, 3].set_xlabel("Training steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward")
    save(fig, "fig04_algos_reward_curves")


def fig05_algos_metrics(runs):
    """Dot plots of the key training-time safety metrics per algorithm."""
    panels = [
        ("avg_cost", "Average training cost per episode", "{:.0f}", (0, 125), False),
        ("pct_over", "% of updates over budget", "{:.0f}%", (0, 115), False),
        ("ep_share_over", "% of training episodes over budget", "{:.0f}%", (0, 115), False),
        ("ep_final_p99", "Worst 1% of episodes (p99), final quarter", "{:.0f}", (0, 185), False),
        ("first_under_m", "First update under budget (M steps)", "{:.1f}", (0, 14.2), False),
        ("reward_final", "Episode reward, last 10 updates", "{:.1f}", (0, 45), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), gridspec_kw={"hspace": 0.48, "wspace": 0.28})
    short = ["PPO", "Lag", "PID", "CRPO", "FOCOPS", "P3O", "Saute"]
    for ax, (key, title, fmt, ylim, logy) in zip(axes.flat, panels):
        vals = [metric_values(A.select(runs, experiment="compare_algos", alg=a), key, max_step_m=CMP_MAX) for a in ALGS]
        ax.set_ylim(*ylim)
        dotplot(ax, ALGS, vals, [ALG_COLOR[a] for a in ALGS], fmt=fmt, ylim=ylim, logy=logy, labels=short,
                never_y=13.2 if key == "first_under_m" else None)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        if key in ("avg_cost", "ep_final_p99"):
            budget_line(ax)                          # dashed line, labelled "budget" at the right end
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
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4.5, color=INK2, alpha=0.55, label="one seed"),
                       plt.Line2D([], [], ls="", marker="o", ms=8.5, markerfacecolor=INK2, markeredgecolor=INK,
                                  label="mean of the 5 seeds")], loc="lower right", fontsize=8)
    save(fig, "fig06_algos_tradeoff")


def fig07_algos_penalties(runs):
    """Cost (top) and each algorithm's penalty signal (bottom), seed means + seeds."""
    algs = ["ppo_lag", "ppo_pid", "crpo", "focops", "p3o"]
    pen_title = {"ppo_lag": "λ (Lagrange multiplier)", "ppo_pid": "λ (PID output)", "crpo": "% of updates optimising cost",
                 "focops": "ν (FOCOPS multiplier)", "p3o": "κ (penalty; log scale, cap 50)"}
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.0), sharex=True, gridspec_kw={"wspace": 0.3})
    for j, alg in enumerate(algs):
        rs = A.select(runs, experiment="compare_algos", alg=alg)
        ax = axes[0, j]
        seeds_and_mean(ax, rs, "training/safety/env_cost_mean", ALG_COLOR[alg], max_step=CMP_MAX, lw=1.8)
        ax.set_ylim(0, 125)
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
    legend_above(fig, [line_key(INK2, "mean of the 5 seeds"), line_key(INK2, "single seeds", lw=0.8),
                       line_key(INK, "budget (25)", lw=1.0, ls="--")], top=0.88)
    save(fig, "fig07_algos_penalties")


def fig08_algos_episode_tail(runs):
    """Episode-level cost distribution over training: mean, p90, p99 (log scale)."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45})   # room for the x-label under the top-right panel
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
            "Dashed line: budget (25).\n\nStarts ~2M steps in: shortened\nfirst episodes are excluded.", transform=ax.transAxes,
            fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions)")
    axes[0, 3].tick_params(labelbottom=True)          # the panel below it is the text box
    axes[0, 3].set_xlabel("Training steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode cost (log scale)")
    save(fig, "fig08_algos_episode_tail")


def fig09_algos_share_over(runs):
    """Share of finished episodes over budget, per update."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45})   # room for the x-label under the top-right panel
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
    axes[0, 3].set_xlabel("Training steps (millions)")
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
    axes[0].set_title("Size of the swings (~5M-step window)", loc="left", color=INK)
    axes[0].set_ylabel("Std of the mean cost per episode")
    axes[0].set_ylim(0, 12)
    axes[1].set_title("% of episodes over budget (~5M-step window)", loc="left", color=INK)
    axes[1].set_ylabel("% of episodes over budget")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50, color=AXIS, lw=0.8)
    for ax in axes:
        ax.set_xlim(0, 106)
        ax.set_xlabel("Training steps (millions)")
    legend_above(fig, [line_key(ALG_COLOR["ppo_lag"], "PPO-Lag (mean of 3 seeds)"),
                       line_key(ALG_COLOR["ppo_pid"], "PPO-PID (mean of 3 seeds)"),
                       line_key(INK2, "single seeds", lw=0.8), line_key(AXIS, "50% (right panel)", lw=1.0)], top=0.86)
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
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), gridspec_kw={"hspace": 0.5, "wspace": 0.25})
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        # all runs of this sweep are 35M steps long, so the full run is used (max_step_m=34.95)
        trend_dots(ax, LRS, [metric_values(lr_runs(runs, lr), key, max_step_m=34.95) for lr in LRS], "#1c5cab", logx=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
    for ax in axes.flat:
        ax.set_xlabel("λ learning rate (log scale; default 10)")
    legend_above(fig, [line_key("#1c5cab", "PPO-Lag, mean over seeds", marker="o"), seed_key("#1c5cab")], top=0.9)
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
    panels = [("ep_share_over", "% of episodes over budget", (0, 105)),
              ("reward_final", "Reward (last 10 updates)", (0, 42)),
              ("avg_cost", "Average cost per episode", (0, 160)),
              ("sps_median", "Throughput (1000 steps/s)", (0, 265))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), gridspec_kw={"wspace": 0.28})
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
            shade_failed_learning(ax, label=False)
        ax.set_xlim(512 / 1.35, 8192 * 1.35)
    shade_key = matplotlib.patches.Patch(facecolor=FAILED_SHADE, alpha=0.45, label="learning failed in some seeds")
    legend_above(fig, [line_key(ALG_COLOR["ppo"], LABEL["ppo"], marker="o"), line_key(ALG_COLOR["ppo_lag"], LABEL["ppo_lag"], marker="o"),
                       seed_key(), shade_key], top=0.84)
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
    # One shared legend in a row above the panels, so it never covers any data.
    handles, _ = axes[0, 0].get_legend_handles_labels()
    legend_above(fig, handles + [seed_key(), line_key(INK, "budget (25)", lw=1.0, ls="--")], top=0.9)
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
    # both legends sit outside the plot area, to the right, so they never cover a run
    leg = ax.legend(handles=alg_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5,
                    title="Colour: algorithm", title_fontsize=8.5, alignment="left")
    ax.add_artist(leg)
    leg.set_clip_on(False)        # add_artist clips to the axes; unclipped, the saved image keeps the whole legend
    ax.legend(handles=set_handles, loc="upper left", bbox_to_anchor=(1.02, 0.62), fontsize=8.5,
              title="Shape: setting", title_fontsize=8.5, alignment="left")
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
    # both legends sit outside the plot area, to the right, so they never cover a run
    leg = ax.legend(handles=alg_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5,
                    title="Colour: algorithm", title_fontsize=8.5, alignment="left")
    ax.add_artist(leg)
    leg.set_clip_on(False)        # add_artist clips to the axes; unclipped, the saved image keeps the whole legend
    ax.legend(handles=lvl_handles, loc="upper left", bbox_to_anchor=(1.02, 0.45), fontsize=8.5,
              title="Shape: level", title_fontsize=8.5, alignment="left")
    save(fig, "fig21_mean_vs_share")


# =================================================================================
# Batch 2 (6 Oct 2026): final-policy evaluation, fixed baselines, CPO and other tasks
# (experiments core_rerun, fixed_baselines, cpo and other_tasks)
# =================================================================================
CPO_COLOR = "#4a3aa7"             # slot 7 (violet) of the validated categorical palette
ALG_COLOR["cpo"] = CPO_COLOR
NOISE_KEY = "open: with exploration noise (stochastic)"
GREEDY_KEY = "filled: greedy (deterministic)"


class Method:
    """One method on Goal Point Level 1: label, short axis label, colour and run selection."""

    def __init__(self, key, label, short, color, **sel):
        self.key, self.label, self.short, self.color, self.sel = key, label, short, color, sel

    def runs(self, runs):
        return A.select(runs, **self.sel)


# Colour = algorithm family (hue); variants of one method are lighter / darker steps of
# its hue, and every category is named on its axis or in a legend.
GOAL_METHODS = [
    Method("ppo", "PPO (no constraint)", "PPO", ALG_COLOR["ppo"], experiment="core_rerun", alg="ppo"),
    Method("ppo_lag", "PPO-Lag", "Lag", ALG_COLOR["ppo_lag"], experiment="core_rerun", alg="ppo_lag"),
    Method("lag_tight", "PPO-Lag, target 12.5", "Lag d'=12.5", "#0d366b", experiment="fixed_baselines",
           alg="ppo_lag", variant="safety_bound-12.5-metric_safety_bound-25"),
    Method("lag_tail_v", "PPO-Lag, tail signal V", "Lag tail V", "#86b6ef", experiment="fixed_baselines",
           alg="ppo_lag", variant="lagrangian_signal-violation_rate-chance_delta-0.05"),
    Method("lag_tail_cvar", "PPO-Lag, tail signal CVaR", "Lag tail CVaR", "#5598e7", experiment="fixed_baselines",
           alg="ppo_lag", variant="lagrangian_signal-cvar"),
    Method("pid10", "PPO-PID (Stooke), Kp 10", "PID Kp10", ALG_COLOR["ppo_pid"], experiment="fixed_baselines",
           alg="ppo_pid", variant="pid_lambda_mode-absolute-pid_ki-10-pid_kp-10-pid_kd-0"),
    Method("pid50", "PPO-PID (Stooke), Kp 50", "PID Kp50", "#f5a27f", experiment="fixed_baselines",
           alg="ppo_pid", variant="pid_lambda_mode-absolute-pid_ki-10-pid_kp-50-pid_kd-0"),
    Method("crpo", "CRPO", "CRPO", ALG_COLOR["crpo"], experiment="core_rerun", alg="crpo"),
    Method("focops", "FOCOPS", "FOCOPS", ALG_COLOR["focops"], experiment="core_rerun", alg="focops"),
    Method("cpo", "CPO", "CPO", CPO_COLOR, experiment="cpo", alg="cpo", env="safe_goal_point"),
    Method("p3o_fast", "P3O, fast κ", "P3O fast κ", ALG_COLOR["p3o"], experiment="fixed_baselines", alg="p3o",
           variant="initial_kappa-1-kappa_decrease_factor-1.0"),
    Method("p3o_fixed", "P3O, fixed κ = 20", "P3O κ=20", "#c2477a", experiment="fixed_baselines", alg="p3o",
           variant="initial_kappa-20-kappa_increase_factor-1.0-kappa_decrease_factor-1.0"),
    Method("saute", "PPO-Saute, budget discount 1", "Saute γb=1", ALG_COLOR["ppo_saute"], experiment="fixed_baselines",
           alg="ppo_saute", variant="saute-gamma-budget-1.0"),
]
GM = {m.key: m for m in GOAL_METHODS}
CORE_KEYS = ["ppo", "ppo_lag", "pid10", "crpo", "focops", "cpo"]
VARIANT_KEYS = ["ppo_lag", "lag_tight", "lag_tail_v", "lag_tail_cvar", "p3o_fast", "p3o_fixed", "saute"]

TASKS = [("safe_circle_point", "Circle"), ("safe_push_point", "Push"), ("safe_button_point", "Button")]
TASK_ALGS = ["ppo", "ppo_lag", "ppo_pid", "focops", "cpo"]
TASK_SHORT = {"ppo": "PPO", "ppo_lag": "Lag", "ppo_pid": "PID", "focops": "FOCOPS", "cpo": "CPO"}
TASK_LABEL = {**LABEL, "ppo_pid": "PPO-PID (Stooke), Kp 10"}


def task_runs(runs, alg, env):
    """Runs on the other tasks (episode length 2000, budget 25); CPO comes from experiment 'cpo'."""
    exp = "cpo" if alg == "cpo" else "other_tasks"
    return [r for r in runs if r.experiment == exp and r.alg == alg and r.env == env]


def final_values(runs, mode, metric, scale=1.0):
    """One final-evaluation value per run (NaN if not logged)."""
    vals = []
    for r in runs:
        v = r.final_eval(mode, metric)
        vals.append(np.nan if v is None else float(v) * scale)
    return vals


def paired_dotplot(ax, labels, stoch, greedy, colors, fmt="{:.0f}", ylim=None, logy=False, write_mean=True,
                   floor=None):
    """Per category: stochastic seeds as open circles (left), greedy seeds filled (right).

    Short bars mark the seed means; the greedy mean is written above the category.
    floor (for log axes): values below it, e.g. exact zeros, are drawn at the floor; the
    means are always computed from the raw values.
    """
    rng = np.random.default_rng(1)
    show = (lambda v: np.maximum(v, floor)) if floor is not None else (lambda v: v)
    for i, (s_vals, g_vals, color) in enumerate(zip(stoch, greedy, colors)):
        for vals, dx, filled in ((s_vals, -0.17, False), (g_vals, 0.17, True)):
            vals = np.array(vals, float)
            ok = np.isfinite(vals)
            if not ok.any():
                continue
            x = i + dx + rng.uniform(-0.05, 0.05, ok.sum())
            ax.scatter(x, show(vals[ok]), s=20, facecolor=color if filled else "white", edgecolor=color,
                       linewidth=1.0, zorder=3)
            m = show(vals[ok].mean())
            ax.plot([i + dx - 0.12, i + dx + 0.12], [m, m], color=INK, lw=1.3, zorder=4)
        g = np.array(g_vals, float)
        s = np.array(s_vals, float)
        if write_mean and np.isfinite(g).any():
            top = show(np.nanmax(np.concatenate([g[np.isfinite(g)], s[np.isfinite(s)]])))
            ax.annotate(fmt.format(np.nanmean(g)), (i, top), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.3, color=INK)
    ax.set_xticks(range(len(labels)))
    rot = len(labels) > 5
    ax.set_xticklabels(labels, rotation=35 if rot else 0, ha="right" if rot else "center",
                       rotation_mode="anchor", fontsize=8)
    ax.grid(axis="x", visible=False)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


def noise_keys():
    """Legend entries for the open (stochastic) / filled (greedy) markers."""
    return [plt.Line2D([], [], ls="", marker="o", ms=5.5, markerfacecolor="white", markeredgecolor=INK2, label=NOISE_KEY),
            plt.Line2D([], [], ls="", marker="o", ms=5.5, color=INK2, label=GREEDY_KEY),
            plt.Line2D([], [], color=INK, lw=1.3, label="bar: mean of seeds")]


def mean_curve(ax, runs, key, color, label=None, ls="-", lw=1.8, series_fn=None, max_step=None, seeds=False):
    """Seed-mean curve (optionally with thin seed lines); returns the last finite value."""
    if not runs:
        return np.nan
    grid, arr = A.seed_curves(runs, key, max_step_m=max_step, series_fn=series_fn)
    if seeds:
        for row in arr:
            ax.plot(grid, row, color=color, lw=0.7, alpha=0.3, ls=ls)
    mean = np.nanmean(arr, axis=0)
    ax.plot(grid, mean, color=color, lw=lw, ls=ls, label=label)
    fin = mean[np.isfinite(mean)]
    return fin[-1] if len(fin) else np.nan


def fig22_final_eval_goal(runs):
    """Final policy on Goal Point L1: violation rate, cost, D_norm+, reward; stochastic vs greedy.

    1000 fresh episodes per run and mode (final_eval). Open circles: with exploration noise
    (the policy as trained); filled: greedy (the mean action, as usually deployed).
    """
    ms = GOAL_METHODS
    panels = [("violation_rate", 100, "Episodes over budget, V (%)", "{:.0f}%", (-3, 112), False),
              ("cost_mean", 1, "Mean episode cost", "{:.0f}", (0, 118), False),
              ("d_norm_plus", 1, "Mean overshoot of violating episodes, D_norm+\n(× budget, log scale; bottom edge = no violations)",
               "{:.2f}", (0.02, 40), True),
              ("reward_mean", 1, "Mean episode reward", "{:.0f}", (-15, 45), False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), gridspec_kw={"hspace": 0.62, "wspace": 0.16})
    for ax, (metric, scale, title, fmt, ylim, logy) in zip(axes.flat, panels):
        stoch = [final_values(m.runs(runs), "stochastic", metric, scale) for m in ms]
        greedy = [final_values(m.runs(runs), "greedy", metric, scale) for m in ms]
        # D_norm+ is 0 when no episode violates: drawn at the bottom edge of the log axis
        paired_dotplot(ax, [m.short for m in ms], stoch, greedy, [m.color for m in ms], fmt=fmt, ylim=ylim, logy=logy,
                       floor=ylim[0] * 1.15 if logy else None)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        if metric == "cost_mean":
            budget_line(ax)
        if metric == "violation_rate":
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        if logy:
            plain_log_ticks(ax, [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25])
    legend_above(fig, noise_keys(), top=0.91)
    fig.text(0.5, 0.005, "Goal Point Level 1, final policy after ~35M steps (Saute 30M), 1000 evaluation episodes per run "
             "and mode; labels: greedy mean.", ha="center", fontsize=8, color=INK2)
    save(fig, "fig22_final_eval_goal")


def _cdf_panel(ax, runs, methods, which):
    """Seed-mean CDF of D_norm (training-time pooled, or final stochastic / greedy)."""
    xs = np.array(A.CDF_THRESHOLDS)
    for m in methods:
        rs = m.runs(runs)
        if which == "training":
            arr = np.array([A.training_cdf(r, max_step_m=CMP_MAX) for r in rs])
        else:
            arr = np.array([A.final_cdf(r, which) for r in rs])
        if not len(arr) or not np.isfinite(arr).any():
            continue
        ax.plot(xs, np.nanmean(arr, 0), color=m.color, lw=1.8, marker="o", ms=3.5, drawstyle="steps-post", label=m.label)
    ax.axvline(0, color=INK, lw=1, ls="--", zorder=1)
    ax.set_xlim(-1.1, 4.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1, 2, 4])
    ax.set_xticklabels(["-1\n(no cost)", "-0.5", "0\n(budget)", "0.5", "1", "2", "4"], fontsize=7.8)


def fig23_cdfs(runs):
    """Empirical CDF of D_norm = (episode cost - d)/d, as in Spoor et al. (2026, Fig. 2).

    Left: all training episodes (pooled over updates, up to 30.2M steps). Middle and right:
    the final policy with and without exploration noise. The height at 0 is the share of
    episodes within budget (1 - V); the further right a curve reaches 1, the larger the
    overshoots. Values are logged at fixed thresholds (dots), drawn as steps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.08})
    cols = [("training", "During training (all episodes)"), ("stochastic", "Final policy, with exploration noise"),
            ("greedy", "Final policy, greedy")]
    for row, keys, name in ((0, CORE_KEYS, "Methods"), (1, VARIANT_KEYS, "Variants")):
        for j, (which, title) in enumerate(cols):
            ax = axes[row, j]
            _cdf_panel(ax, runs, [GM[k] for k in keys], which)
            ax.set_title(f"{name}: {title}", loc="left", color=INK, fontsize=9.3)
            ax.set_xlabel("D_norm = (episode cost − budget) / budget")
        axes[row, 0].set_ylabel("Share of episodes ≤ D_norm")
        axes[row, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.text(0.5, 0.005, "Goal Point Level 1. Height at 0 = share of episodes within budget. Seed means; dots = logged thresholds.",
             ha="center", fontsize=8, color=INK2)
    save(fig, "fig23_cdfs")


def fig24_training_vs_final(runs):
    """Training-time vs final-policy safety, per run: share of training episodes over budget
    vs share of final greedy episodes over budget. Points on the diagonal: training predicts
    the deployed policy; below it: the final policy is safer than training suggested."""
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1, zorder=1)
    ax.text(97, 92, "same share during training\nand for the final policy", fontsize=7.8, color=INK2, ha="right", va="top")
    handles = []
    for m in GOAL_METHODS:
        rs = m.runs(runs)
        x = [A.run_metrics(r, max_step_m=CMP_MAX)["ep_share_over"] for r in rs]
        y = final_values(rs, "greedy", "violation_rate", 100)
        ax.scatter(x, y, s=34, marker="o", color=m.color, edgecolor="white", linewidth=0.6, zorder=3)
        handles.append(plt.Line2D([], [], ls="", marker="o", color=m.color, label=m.label))
    task_marker = {"safe_circle_point": "s", "safe_push_point": "^", "safe_button_point": "D"}
    for env, name in TASKS:
        for alg in TASK_ALGS:
            rs = task_runs(runs, alg, env)
            x = [A.run_metrics(r)["ep_share_over"] for r in rs]
            y = final_values(rs, "greedy", "violation_rate", 100)
            ax.scatter(x, y, s=30, marker=task_marker[env], color=ALG_COLOR[alg], edgecolor=INK, linewidth=0.4, zorder=3)
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                    title="Goal Point L1 (circles)", title_fontsize=8.3, alignment="left")
    ax.add_artist(leg)
    leg.set_clip_on(False)
    shape_handles = [plt.Line2D([], [], ls="", marker=task_marker[e], color=INK2, markeredgecolor=INK, label=n)
                     for e, n in TASKS]
    ax.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(1.02, 0.18), fontsize=8,
              title="Other tasks (colour = algorithm)", title_fontsize=8.3, alignment="left")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("% of training episodes over budget (whole run)")
    ax.set_ylabel("% of final greedy episodes over budget (V)")
    save(fig, "fig24_training_vs_final")


TIER_COLORS = ["#f6f6f3", "#dbe9f8", "#b3d0f1", "#7eafe6", "#3f87d9"]   # tier 0 -> 4, one hue light -> dark


def fig25_safety_tiers(runs):
    """Safety tiers of Spoor et al. (2026, Table 1), during training and for the final greedy policy.

    Tier 1: average cost within budget (D_norm <= 0); 2: also V <= 0.5; 3: also V <= 0.1 and
    D_norm+ <= 0.1; 4: no violations. Metrics are seed means (one task, so the mean replaces
    the paper's IQM across tasks). Training = average over updates (Spoor Eq. 6).
    """
    ms = GOAL_METHODS
    rows = []
    for m in ms:
        vals = [A.spoor_metrics(r, max_step_m=CMP_MAX) for r in m.runs(runs)]
        mean = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
        rows.append((m, mean, A.safety_tier(mean["train_d_norm"], mean["train_v"], mean["train_d_norm_plus"]),
                     A.safety_tier(mean["final_d_norm"], mean["final_v"], mean["final_d_norm_plus"])))
    cols = ["D_norm", "V", "D_norm+", "Tier", "D_norm", "V", "D_norm+", "Tier"]
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 1.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(len(rows) + 1.3, -0.2)
    ax.axis("off")
    x_cols = [3.2 + 0.9 * i + (0.4 if i >= 4 else 0) for i in range(8)]
    ax.text(np.mean(x_cols[:4]), -0.05, "During training", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(np.mean(x_cols[4:]), -0.05, "Final policy, greedy", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    for x, c in zip(x_cols, cols):
        ax.text(x, 0.55, c, ha="center", fontsize=8.5, color=INK2)
    for i, (m, mean, t_train, t_final) in enumerate(rows):
        y = i + 1.25
        ax.text(0.1, y, m.label, va="center", fontsize=8.5, color=INK)
        ax.scatter([2.75], [y], s=40, color=m.color, zorder=3)
        cells = [mean["train_d_norm"], mean["train_v"], mean["train_d_norm_plus"], t_train,
                 mean["final_d_norm"], mean["final_v"], mean["final_d_norm_plus"], t_final]
        for j, (x, v) in enumerate(zip(x_cols, cells)):
            if j in (3, 7):
                ax.add_patch(matplotlib.patches.FancyBboxPatch((x - 0.3, y - 0.32), 0.6, 0.64, boxstyle="round,pad=0.02",
                                                               facecolor=TIER_COLORS[v], edgecolor="none"))
                ax.text(x, y, str(v), ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if v >= 3 else INK)
            else:
                ax.text(x, y, f"{v:+.2f}" if j in (0, 4) else f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.plot([x_cols[3] + 0.6, x_cols[3] + 0.6], [0.3, len(rows) + 0.9], color=AXIS, lw=0.8)
    ax.text(0.1, len(rows) + 1.15, "Tier 0 unsafe (mean over budget) · 1 mean within budget · 2 + V ≤ 0.5 · "
            "3 + V ≤ 0.1 and D_norm+ ≤ 0.1 · 4 no violations.  Goal Point L1, seed means.", fontsize=7.8, color=INK2)
    save(fig, "fig25_safety_tiers")


def _fixed_groups(runs):
    """(title, [(label, colour, runs, linestyle), ...]) for the fixed-baseline comparisons.

    Grey dashed = the CRAX default of that method from the first batch (experiment 03).
    """
    old = lambda alg: A.select(runs, experiment="compare_algos", alg=alg)
    lag = GM["ppo_lag"].runs(runs)
    return [
        ("PID: λ set to the PID output", [("PPO-PID, CRAX default", INK2, old("ppo_pid"), "--"),
                                          ("PPO-Lag", ALG_COLOR["ppo_lag"], lag, "-"),
                                          ("PID (Stooke), Kp 10", GM["pid10"].color, GM["pid10"].runs(runs), "-"),
                                          ("PID (Stooke), Kp 50", GM["pid50"].color, GM["pid50"].runs(runs), "-")]),
        ("P3O: faster / fixed κ", [("P3O, CRAX default", INK2, old("p3o"), "--"),
                                   ("P3O, fast κ (from 1, no decay)", GM["p3o_fast"].color, GM["p3o_fast"].runs(runs), "-"),
                                   ("P3O, fixed κ = 20", GM["p3o_fixed"].color, GM["p3o_fixed"].runs(runs), "-")]),
        ("Saute: no budget discount", [("PPO-Saute, CRAX default", INK2, old("ppo_saute"), "--"),
                                       ("PPO-Saute, budget discount 1", GM["saute"].color, GM["saute"].runs(runs), "-")]),
        ("PPO-Lag: tighter target", [("PPO-Lag, target 25", ALG_COLOR["ppo_lag"], lag, "-"),
                                     ("PPO-Lag, target 12.5", GM["lag_tight"].color, GM["lag_tight"].runs(runs), "-")]),
        ("PPO-Lag: tail-driven λ", [("PPO-Lag, mean signal", ALG_COLOR["ppo_lag"], lag, "-"),
                                    ("tail signal V (δ = 0.05)", GM["lag_tail_v"].color, GM["lag_tail_v"].runs(runs), "-"),
                                    ("tail signal CVaR95", GM["lag_tail_cvar"].color, GM["lag_tail_cvar"].runs(runs), "-")]),
    ]


def fig26_fixed_baselines(runs):
    """Fixed baselines vs CRAX defaults on Goal Point L1: cost (top) and reward (bottom) per update."""
    groups = _fixed_groups(runs)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharex=True, gridspec_kw={"wspace": 0.22, "hspace": 0.18})
    for j, (title, entries) in enumerate(groups):
        for label, color, rs, ls in entries:
            mean_curve(axes[0, j], rs, "training/safety/env_cost_mean", color, label=label, ls=ls, max_step=CMP_MAX)
            mean_curve(axes[1, j], rs, None, color, ls=ls, series_fn=A.reward_series, max_step=CMP_MAX)
        axes[0, j].set_title(title, loc="left", color=INK, fontsize=9.5)
        axes[0, j].set_ylim(0, 140)
        axes[1, j].set_ylim(-15, 42)
        axes[1, j].axhline(0, color=AXIS, lw=0.8, zorder=1)
        budget_line(axes[0, j], label=False)
        handles, _ = axes[0, j].get_legend_handles_labels()
        axes[0, j].legend(handles=handles + [line_key(INK, "budget (25)", lw=1.0, ls="--")], loc="upper right",
                          fontsize=7.3, handlelength=1.8)
        axes[1, j].set_xlabel("Training steps (millions)")
        axes[0, j].set_xlim(0, 30.5)
    axes[0, 0].set_ylabel("Mean cost per episode")
    axes[1, 0].set_ylabel("Episode reward")
    fig.text(0.5, -0.01, "Goal Point Level 1, seed means (3-5 seeds). Grey dashed: the method's CRAX default from the first "
             "batch (30 Sep).", ha="center", fontsize=8, color=INK2)
    save(fig, "fig26_fixed_baselines")


def fig27_fixed_penalties(runs):
    """Multiplier / penalty of the fixed baselines (seed means): λ for PID and the PPO-Lag
    variants, κ for P3O (log scale)."""
    old = lambda alg: A.select(runs, experiment="compare_algos", alg=alg)
    lag = GM["ppo_lag"].runs(runs)
    panels = [
        ("λ: PID vs PPO-Lag", "training/lambda_lagr", False,
         [("PPO-PID, CRAX default", INK2, old("ppo_pid"), "--"), ("PPO-Lag", ALG_COLOR["ppo_lag"], lag, "-"),
          ("PID (Stooke), Kp 10", GM["pid10"].color, GM["pid10"].runs(runs), "-"),
          ("PID (Stooke), Kp 50", GM["pid50"].color, GM["pid50"].runs(runs), "-")]),
        ("κ: P3O (log scale)", "training/kappa", True,
         [("P3O, CRAX default", INK2, old("p3o"), "--"), ("fast κ", GM["p3o_fast"].color, GM["p3o_fast"].runs(runs), "-"),
          ("fixed κ = 20", GM["p3o_fixed"].color, GM["p3o_fixed"].runs(runs), "-")]),
        ("λ: PPO-Lag variants (log scale; λ = 0 drawn at 0.001)", "training/lambda_lagr", True,
         [("mean signal", ALG_COLOR["ppo_lag"], lag, "-"), ("target 12.5", GM["lag_tight"].color, GM["lag_tight"].runs(runs), "-"),
          ("tail V", GM["lag_tail_v"].color, GM["lag_tail_v"].runs(runs), "-"),
          ("tail CVaR", GM["lag_tail_cvar"].color, GM["lag_tail_cvar"].runs(runs), "-")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.25})
    for ax, (title, key, logy, entries) in zip(axes, panels):
        for label, color, rs, ls in entries:
            if logy:   # multipliers can be exactly 0; floor them so the log axis can show them
                fn = lambda r, k=key: np.maximum(r.get(k), 1e-3)
                mean_curve(ax, rs, None, color, label=label, ls=ls, series_fn=fn, max_step=CMP_MAX)
            else:
                mean_curve(ax, rs, key, color, label=label, ls=ls, max_step=CMP_MAX)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("Training steps (millions)")
        ax.set_xlim(0, 30.5)
        ax.legend(fontsize=7.6, loc="best")
    axes[0].set_ylabel("Multiplier / penalty")
    save(fig, "fig27_fixed_penalties")


CASE_NAMES = ["0 infeasible recovery", "1 feasible recovery", "2 constrained step", "3 TRPO step (region safe)",
              "4 TRPO step (no cost gradient)"]
CASE_COLORS = ["#b8431c", "#f2a17a", "#a9c9ef", "#2a78d6", "#0d366b"]   # unsafe (warm) -> safe (cool)


def fig28_cpo_diagnostics(runs):
    """CPO on Goal Point L1: cost and reward vs PPO-Lag and FOCOPS, and CPO's own diagnostics."""
    cpo = GM["cpo"].runs(runs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    ax = axes[0, 0]
    for k in ("ppo_lag", "focops", "cpo"):
        mean_curve(ax, GM[k].runs(runs), "training/safety/env_cost_mean", GM[k].color, label=GM[k].label,
                   max_step=CMP_MAX, seeds=(k == "cpo"))
    budget_line(ax, label=False)
    ax.set_ylim(0, 120)
    ax.set_title("Mean cost per episode (thin: CPO seeds)", loc="left", color=INK, fontsize=9.5)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for k in ("ppo_lag", "focops", "cpo"):
        mean_curve(ax, GM[k].runs(runs), None, GM[k].color, label=GM[k].label, series_fn=A.reward_series, max_step=CMP_MAX)
    ax.set_title("Episode reward", loc="left", color=INK, fontsize=9.5)
    ax.set_ylim(0, 40)
    # optimisation cases: share of updates in each case, trailing 8 updates, pooled over seeds
    ax = axes[0, 2]
    grid, arr = A.seed_curves(cpo, "training/cpo/optim_case", max_step_m=CMP_MAX)
    shares = []
    for c in range(5):
        hit = np.where(np.isfinite(arr), (np.round(arr) == c).astype(float), np.nan)
        pooled = np.nanmean(hit, axis=0)
        shares.append(A.rolling(pooled, 8, np.nanmean) * 100)
    shares = np.nan_to_num(np.array(shares))
    ax.stackplot(grid, shares, colors=CASE_COLORS, labels=CASE_NAMES, edgecolor="white", linewidth=0.3)
    ax.set_ylim(0, 100)
    ax.set_title("Share of updates per optimisation case (%)", loc="left", color=INK, fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    ax = axes[1, 0]
    fn_acc = lambda r: 100 * A.rolling(r.get("training/cpo/step_accepted"), 8, np.nanmean)
    fn_frac = lambda r: 100 * A.rolling(r.get("training/cpo/step_fraction"), 8, np.nanmean)
    mean_curve(ax, cpo, None, CPO_COLOR, label="line search accepted a step", series_fn=fn_acc, max_step=CMP_MAX)
    mean_curve(ax, cpo, None, CPO_COLOR, label="mean accepted step size (% of full)", ls="--", series_fn=fn_frac, max_step=CMP_MAX)
    ax.set_ylim(0, 105)
    ax.set_title("Line search (trailing 8 updates, %)", loc="left", color=INK, fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 1]
    mean_curve(ax, cpo, "training/cpo/kl", CPO_COLOR, max_step=CMP_MAX, seeds=True)
    ax.axhline(0.01, color=INK, lw=1, ls="--")
    ax.text(0.5, 0.0103, "trust region δ = 0.01", fontsize=7.8, color=INK, va="bottom")
    ax.set_ylim(0, 0.013)
    ax.set_title("KL between successive policies", loc="left", color=INK, fontsize=9.5)
    ax = axes[1, 2]
    fn_c = lambda r: 1000.0 * r.get("training/cpo/c")       # per-step c -> per 1000-step episode
    mean_curve(ax, cpo, None, CPO_COLOR, series_fn=fn_c, max_step=CMP_MAX, seeds=True)
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.set_ylim(-30, 100)
    ax.set_title("Constraint value c = J_C − d (per episode)", loc="left", color=INK, fontsize=9.5)
    for a in axes[1]:
        a.set_xlabel("Training steps (millions)")
    for a in axes.flat:
        a.set_xlim(0, 30.5)
    save(fig, "fig28_cpo_diagnostics")


def fig29_tasks_curves(runs):
    """Other tasks (episode length 2000, budget 25): cost and reward per update, seed means.

    Last column: PPO-Lag on Goal Point with 1000- vs 2000-step episodes (same budget 25),
    the bridge to the setting of Spoor et al. (2026).
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), gridspec_kw={"wspace": 0.22, "hspace": 0.2})
    for j, (env, name) in enumerate(TASKS):
        for alg in TASK_ALGS:
            rs = task_runs(runs, alg, env)
            mean_curve(axes[0, j], rs, "training/safety/env_cost_mean", ALG_COLOR[alg], label=TASK_LABEL[alg])
            mean_curve(axes[1, j], rs, None, ALG_COLOR[alg], series_fn=A.reward_series)
        axes[0, j].set_title(f"{name} (Level 1, T = 2000)", loc="left", color=INK, fontsize=9.5)
    lag1000 = GM["ppo_lag"].runs(runs)
    lag2000 = [r for r in runs if r.experiment == "other_tasks" and r.alg == "ppo_lag" and r.env == "safe_goal_point"]
    for rs, ls, label in ((lag1000, "-", "PPO-Lag, T = 1000"), (lag2000, "--", "PPO-Lag, T = 2000")):
        mean_curve(axes[0, 3], rs, "training/safety/env_cost_mean", ALG_COLOR["ppo_lag"], label=label, ls=ls)
        mean_curve(axes[1, 3], rs, None, ALG_COLOR["ppo_lag"], ls=ls, series_fn=A.reward_series)
    axes[0, 3].set_title("Goal Point: episode length", loc="left", color=INK, fontsize=9.5)
    axes[0, 3].legend(fontsize=7.8, loc="upper right")
    for j in range(4):
        axes[0, j].set_ylim(0, 140)
        budget_line(axes[0, j], label=False)
        axes[1, j].set_xlabel("Training steps (millions)")
        axes[0, j].set_xlim(0, 56)
        axes[1, j].set_xlim(0, 56)
    axes[0, 0].set_ylabel("Mean cost per episode")
    axes[1, 0].set_ylabel("Episode reward")
    legend_above(fig, [line_key(ALG_COLOR[a], TASK_LABEL[a]) for a in TASK_ALGS]
                 + [line_key(INK, "budget (25)", lw=1.0, ls="--")], top=0.9, fontsize=8.8)
    save(fig, "fig29_tasks_curves")


def fig30_tasks_final(runs):
    """Final policy on the other tasks: V, D_norm+ and reward, stochastic vs greedy (3 seeds)."""
    panels = [("violation_rate", 100, "Episodes over budget, V (%)", "{:.0f}%", (-3, 112), False),
              ("d_norm_plus", 1, "D_norm+ (× budget, log)", "{:.2f}", (0.02, 90), True),
              ("reward_mean", 1, "Mean episode reward", "{:.0f}", None, False)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), gridspec_kw={"hspace": 0.45, "wspace": 0.2})
    for j, (env, name) in enumerate(TASKS):
        for i, (metric, scale, title, fmt, ylim, logy) in enumerate(panels):
            ax = axes[i, j]
            stoch = [final_values(task_runs(runs, a, env), "stochastic", metric, scale) for a in TASK_ALGS]
            greedy = [final_values(task_runs(runs, a, env), "greedy", metric, scale) for a in TASK_ALGS]
            paired_dotplot(ax, [TASK_SHORT[a] for a in TASK_ALGS], stoch, greedy, [ALG_COLOR[a] for a in TASK_ALGS],
                           fmt=fmt, ylim=ylim, logy=logy, floor=ylim[0] * 1.15 if logy else None)
            if logy:
                plain_log_ticks(ax, [0.05, 0.25, 1, 5, 25])
            ax.set_title(f"{name}: {title}", loc="left", color=INK, fontsize=9.2)
    legend_above(fig, noise_keys(), top=0.93)
    fig.text(0.5, 0.005, "Level 1, episode length 2000, budget 25; ~55M steps; 1000 evaluation episodes per run and mode. "
             "D_norm+ = 0 (no violation) is drawn at the bottom edge.", ha="center", fontsize=8, color=INK2)
    save(fig, "fig30_tasks_final")


def fig31_tails_goal(runs):
    """Worst cases on Goal Point L1: worst 1% of training episodes (final quarter), and the
    final policy's CVaR95 (mean of the worst 5% of episodes) and worst single episode."""
    ms = GOAL_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.18})
    ax = axes[0]
    vals = [[A.run_metrics(r, max_step_m=CMP_MAX)["ep_final_p99"] for r in m.runs(runs)] for m in ms]
    ax.set_yscale("log")
    dotplot(ax, [m.key for m in ms], vals, [m.color for m in ms], fmt="{:.0f}", ylim=(3, 600), labels=[m.short for m in ms])
    ax.set_yscale("log")
    plain_log_ticks(ax, [5, 10, 25, 50, 100, 250, 500])
    budget_line(ax)
    ax.set_title("Training: worst 1% of episodes (p99), final quarter", loc="left", color=INK, fontsize=9.3)
    for ax, metric, title in ((axes[1], "cost_cvar95", "Final policy: CVaR95 (mean of worst 5%)"),
                              (axes[2], "cost_max", "Final policy: worst episode of 1000")):
        stoch = [final_values(m.runs(runs), "stochastic", metric) for m in ms]
        greedy = [final_values(m.runs(runs), "greedy", metric) for m in ms]
        # zero-cost tails are drawn at the bottom edge of the log axis
        paired_dotplot(ax, [m.short for m in ms], stoch, greedy, [m.color for m in ms], fmt="{:.0f}", ylim=(0.5, 4000),
                       logy=True, floor=0.6)
        plain_log_ticks(ax, [1, 5, 25, 100, 500, 2000])
        budget_line(ax, label=False)
        ax.set_title(title, loc="left", color=INK, fontsize=9.3)
    legend_above(fig, noise_keys(), top=0.86)
    fig.text(0.5, -0.06, "Log scales. Dashed: budget 25. Values of 0 are drawn at the bottom edge.", ha="center",
             fontsize=8, color=INK2)
    save(fig, "fig31_tails_goal")


def fig32_tradeoff_final(runs):
    """Reward vs safety of the final greedy policy: small dots = seeds, ringed dot = mean."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1, 1], "wspace": 0.22})

    def panel(ax, groups):
        for label, color, rs in groups:
            x = np.array(final_values(rs, "greedy", "violation_rate", 100))
            y = np.array(final_values(rs, "greedy", "reward_mean"))
            if not len(x):
                continue
            ax.scatter(x, y, s=16, color=color, alpha=0.55, edgecolor="none", zorder=3)
            ax.scatter([np.nanmean(x)], [np.nanmean(y)], s=70, color=color, edgecolor=INK, linewidth=0.9, zorder=4, label=label)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("Final greedy episodes over budget, V (%)")

    panel(axes[0], [(m.label, m.color, m.runs(runs)) for m in GOAL_METHODS])
    axes[0].set_title("Goal Point L1 (T = 1000)", loc="left", color=INK, fontsize=9.5)
    axes[0].set_ylabel("Final greedy episode reward")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=3, fontsize=7.8, title="Goal Point L1",
                   title_fontsize=8.3, alignment="left")
    for ax, (env, name) in zip(axes[1:], TASKS):
        panel(ax, [(TASK_LABEL[a], ALG_COLOR[a], task_runs(runs, a, env)) for a in TASK_ALGS])
        ax.set_title(f"{name} (T = 2000)", loc="left", color=INK, fontsize=9.5)
    axes[2].legend(loc="upper left", bbox_to_anchor=(-0.6, -0.2), ncol=5, fontsize=7.8, title="Circle, Push, Button",
                   title_fontsize=8.3, alignment="left")
    save(fig, "fig32_tradeoff_final")


def fig33_reproducibility(runs):
    """Same settings, run twice: first batch (experiment 03, 30 Sep) vs rerun (experiment 10, 6 Oct).

    Seed means with the seed range shaded. The training code differs only in logging, so the
    curves should agree within seed noise.
    """
    algs = ["ppo", "ppo_lag", "crpo", "focops"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, alg in zip(axes, algs):
        for exp, color, ls, label in (("compare_algos", INK2, "--", "first batch (30 Sep)"),
                                      ("core_rerun", ALG_COLOR[alg], "-", "rerun (6 Oct)")):
            rs = A.select(runs, experiment=exp, alg=alg)
            if not rs:
                continue
            grid, arr = A.seed_curves(rs, "training/safety/env_cost_mean", max_step_m=CMP_MAX)
            ax.fill_between(grid, np.nanmin(arr, 0), np.nanmax(arr, 0), color=color, alpha=0.12, lw=0)
            ax.plot(grid, np.nanmean(arr, 0), color=color, ls=ls, lw=1.8, label=label)
        ax.set_title(LABEL[alg], loc="left", color=INK)
        ax.set_ylim(0, 140)
        ax.set_xlim(0, 30.5)
        budget_line(ax, label=False)
        ax.set_xlabel("Training steps (millions)")
        ax.legend(fontsize=7.8, loc="upper right")
    axes[0].set_ylabel("Mean cost per episode")
    save(fig, "fig33_reproducibility")


FIGURES = [fig01_measurement_mean_vs_tail, fig02_lockstep, fig03_algos_cost_curves,
           fig04_algos_reward_curves, fig05_algos_metrics, fig06_algos_tradeoff,
           fig07_algos_penalties, fig08_algos_episode_tail, fig09_algos_share_over,
           fig10_long_cost_curves, fig11_long_damping, fig12_lr_curves,
           fig13_lr_dose_response, fig14_envs_curves, fig15_envs_metrics,
           fig16_levels_curves, fig17_levels_metrics, fig18_ant_curves,
           fig19_ant_metrics, fig20_final_vs_training, fig21_mean_vs_share,
           # batch 2 (6 Oct): final-policy evaluation, fixed baselines, CPO, other tasks
           fig22_final_eval_goal, fig23_cdfs, fig24_training_vs_final, fig25_safety_tiers,
           fig26_fixed_baselines, fig27_fixed_penalties, fig28_cpo_diagnostics, fig29_tasks_curves,
           fig30_tasks_final, fig31_tails_goal, fig32_tradeoff_final, fig33_reproducibility]


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
    first_report = set(FIGURES[:21])            # fig01-fig21: first-batch report, first-batch runs only
    for f in FIGURES:
        if f.__name__.startswith(only):
            f(A.batch1(runs) if f in first_report else runs)
            made += 1
    if made == 0:
        print(f"No figure name starts with '{only}'. Names: " + ", ".join(f.__name__ for f in FIGURES))


if __name__ == "__main__":
    main()
