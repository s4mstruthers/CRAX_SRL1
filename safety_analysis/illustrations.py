"""Illustrations for talks and the paper's introduction (no run data).

Usage (repo folder; needs numpy + matplotlib):
    python safety_analysis/illustrations.py      # -> safety_analysis/figures/illustration/intro_same_mean_different_risk.(png|pdf)

intro_same_mean_different_risk (optional introduction figure): three per-episode cost distributions (policies A, B, C) with the
SAME mean, equal to the budget d = 25, so all three satisfy the CMDP constraint E[C] <= d exactly. Each stands for a
FINAL policy (after training), run on fresh episodes, not for episodes seen during training.
A: tightly bunched around the mean, so half the episodes are over budget but only just. B: a wide
symmetric spread (as on easy tasks): still half over, now by more. C: right-skewed (as on harder
tasks): fewer episodes over budget, but the worst ones far over. The shapes are chosen to resemble our Goal Point Level 1 and Level 3 numbers (worst 1% about
2x and 6x the budget); they are illustrations, not fitted to data. Every number printed on the
figure is computed from the distribution, not typed in.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "illustration")
D = 25.0                                   # budget per episode, as in every experiment

# Same palette and style as make_figures.py
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
WITHIN = "#d9d8d2"                         # neutral fill: episodes within budget
EASY, HARD = "#2a78d6", "#eb6834"          # blue / orange: told apart by lightness too

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def lognormal_pdf(x, mu, sigma):
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.exp(-0.5 * ((np.log(x[pos]) - mu) / sigma) ** 2) / (x[pos] * sigma * math.sqrt(2 * math.pi))
    return out


def summarise(x, pdf):
    """Mean, V = P(C > d), D+ = E[(C - d)/d | C > d] and the 99th percentile, by numerical integration."""
    dx = x[1] - x[0]
    p = pdf * dx
    p = p / p.sum()                                      # the grid is wide enough that this only fixes rounding
    over = x > D
    v = p[over].sum()
    dplus = ((x[over] - D) / D * p[over]).sum() / v
    p99 = x[np.searchsorted(np.cumsum(p), 0.99)]
    return float((x * p).sum()), float(v), float(dplus), float(p99)


def intro_same_mean_different_risk():
    """Single-column figure for the introduction: three policies stacked, sharing the x-axis.

    Only two short data labels per row (V and the worst-1% cost); what the shading, the dashed
    line and the policies mean is explained in the paper's caption, not in the figure.
    """
    # A wide grid (normal tails to -60, lognormal tail to 3000) so the integrals are exact to rounding.
    x = np.linspace(-60, 3000, 300000)
    skew_sigma = 1.0                                     # lognormal with mean exactly d
    skew_mu = math.log(D) - skew_sigma ** 2 / 2
    cases = [
        ("A", "#1baf7a", normal_pdf(x, D, 3.0)),
        ("B", EASY, normal_pdf(x, D, 10.0)),
        ("C", HARD, lognormal_pdf(x, skew_mu, skew_sigma)),
    ]
    with plt.rc_context({"font.size": 7.5, "xtick.labelsize": 7, "axes.labelsize": 7.5, "savefig.pad_inches": 0.02}):
        fig, axes = plt.subplots(3, 1, figsize=(3.9, 2.4), sharex=True, gridspec_kw={"hspace": 0.12})
        for ax, (name, color, pdf) in zip(axes, cases):
            mean, v, dplus, p99 = summarise(x, pdf)
            top = pdf.max()
            # Draw only the visible part of the curve on a coarser grid (~2000 points): the full
            # 300k-point grid is needed for exact integrals, but would make a 7 MB PDF.
            vis = np.flatnonzero((x >= 0) & (x <= 185))[::50]
            xs, ps = x[vis], pdf[vis]
            ax.fill_between(xs, ps, where=xs <= D, color=WITHIN, lw=0, interpolate=True)
            ax.fill_between(xs, ps, where=xs >= D, color=color, alpha=0.85, lw=0, interpolate=True)
            ax.plot(xs, ps, color=INK2, lw=0.8)
            ax.axvline(D, color=INK, lw=0.8, ls="--")
            # Worst 1%: a short tick on the axis at the 99th percentile.
            ax.plot([p99, p99], [0, top * 0.45], color=INK, lw=1.2)
            label = f"p99 = {p99 / D:.1f}d"
            if p99 > 110:                                # long tail: label on the left of the tick
                ax.text(p99 - 3, top * 0.45, label, color=INK, va="top", ha="right", fontsize=6.5)
            else:
                ax.text(p99 + 3, top * 0.45, label, color=INK, va="top", fontsize=6.5)
            ax.text(178, top * 0.95, f"V = {100 * v:.0f}%", color=INK, ha="right", va="top", fontsize=6.5)
            ax.set_ylabel(name, rotation=0, ha="right", va="center", color=INK, fontsize=8, labelpad=4)
            ax.set_xlim(0, 180)
            ax.set_ylim(0, top * 1.1)
            ax.set_yticks([])
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_axisbelow(True)                       # gridlines behind the shaded areas
            print(f"  Policy {name}: mean {mean:.2f}, V {100 * v:.1f}%, D+ {dplus:.2f}, p99 {p99:.0f}")
        axes[-1].set_xticks([0, 25, 50, 100, 150])
        axes[-1].set_xlabel("Episode cost")
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, "intro_same_mean_different_risk.png"))
        fig.savefig(os.path.join(OUT, "intro_same_mean_different_risk.pdf"))
    plt.close(fig)
    print("  wrote illustration/intro_same_mean_different_risk (.png and .pdf)")


if __name__ == "__main__":
    intro_same_mean_different_risk()
