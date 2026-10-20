"""Illustrations for talks and the paper's introduction (no run data).

Usage (repo folder; needs numpy + matplotlib):
    python safety_analysis/illustrations.py      # -> safety_analysis/figures/illustration/

ill01_same_mean_different_risk: three per-episode cost distributions (policies A, B, C) with the
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


def ill01_same_mean_different_risk():
    # A wide grid (normal tails to -60, lognormal tail to 3000) so the integrals are exact to rounding.
    x = np.linspace(-60, 3000, 300000)
    skew_sigma = 1.0                                     # lognormal with mean exactly d
    skew_mu = math.log(D) - skew_sigma ** 2 / 2
    cases = [
        ("Policy A: tightly bunched", "#1baf7a", normal_pdf(x, D, 3.0)),
        ("Policy B: wide, symmetric", EASY, normal_pdf(x, D, 10.0)),
        ("Policy C: long right tail", HARD, lognormal_pdf(x, skew_mu, skew_sigma)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (title, color, pdf) in zip(axes, cases):
        mean, v, dplus, p99 = summarise(x, pdf)
        ax.fill_between(x, pdf, where=x <= D, color=WITHIN, lw=0)
        ax.fill_between(x, pdf, where=x >= D, color=color, alpha=0.85, lw=0)
        ax.plot(x, pdf, color=INK2, lw=1.2)
        top = pdf.max()
        ax.axvline(D, color=INK, lw=1.4, ls="--")
        ax.text(D + 3, top * 1.12, f"budget = mean = {mean:.0f}", color=INK, va="bottom")
        # Share over budget, written beside the coloured area.
        ax.text(70, top * 0.72, f"{100 * v:.0f}% of episodes\nover budget (V)", color=INK, va="center")
        # Worst 1%: a tick on the axis at the 99th percentile, labelled on whichever side has room.
        ax.plot([p99, p99], [0, top * 0.3], color=INK, lw=2.2)
        label = f"worst 1%:\n\u2265 {p99:.0f}  ({p99 / D:.1f}\u00d7 budget)"
        if p99 > 110:
            ax.text(p99 - 4, top * 0.34, label, color=INK, va="bottom", ha="right")
        else:
            ax.text(p99 + 4, top * 0.34, label, color=INK, va="bottom")
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, top * 1.35)
        ax.set_yticks([])
        ax.set_xlabel("Cost of one episode")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)                          # gridlines behind the shaded areas
        print(f"  {title}: mean {mean:.2f}, V {100 * v:.1f}%, D+ {dplus:.2f}, p99 {p99:.0f}")
    axes[0].set_ylabel("Share of episodes\n(final policy, after training)")
    fig.tight_layout(w_pad=2.5)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "ill01_same_mean_different_risk.png"))
    plt.close(fig)
    print("  wrote illustration/ill01_same_mean_different_risk")


if __name__ == "__main__":
    ill01_same_mean_different_risk()
