"""One boosted rule vs one unboosted rule, as added context and bias vary.

Three panels on a shared bias scale: the two firing-rate curves against K, and the tolerance
trade-off.  `B = 0` is drawn dashed and black: with the bias off the two rules are exchangeable --
same form, same distribution, and which draw gets boosted is a coin flip -- so the two curve panels
must agree along it and the trade-off point must sit where the two tolerances are equal.

The combined figure is also written against the attention multiplier `M = exp(B)`, where the
baseline is `M = 1`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "boost_tradeoff"
CMAP = "magma"

# What is being swept.  The same figures are drawn against the bias and against the multiplier,
# so the plotting is written against this description of the control variable.
BIAS_CTRL = {"col": "attn_bias", "label": "Attention bias $B$", "fmt": lambda v: f"B={v:g}",
             "base": 0.0}
# The same sweep read as the attention multiplier M = exp(B), the quantity a practitioner sets:
# a bias of B multiplies the instruction rows' attention weight by exp(B).  The colour scale is
# linear in M, so the low-bias curves bunch together and the high-bias ones spread out.
MULT_CTRL = {"col": "boost_mult", "label": "Attention multiplier $M = e^{B}$",
             "fmt": lambda v: f"M={v:.2f}", "base": 1.0}

# Figures are generated at exactly the width they are included at, so LaTeX never rescales them
# and the point sizes below are the point sizes on the page.  Pass --width-pt with the value of
# \the\linewidth from the target document.
DEFAULT_WIDTH_PT = 469.75
DPI = 400
PT_PER_INCH = 72.27
FS_SUP, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 9.5, 8.5, 8.0, 7.0, 6.5
# Weights are in points too, so they are set for the final size rather than for something that
# will be shrunk on the page.
LW_CURVE, LW_BASE, LW_FIT = 1.3, 1.7, 0.9
MS_POINT, MEW_POINT = 20, 0.5


def _figsize(width_pt, ratio):
    w = width_pt / PT_PER_INCH
    return w, w * ratio


def _colour(B, norm):
    return colormaps[CMAP](norm(B) * 0.9)


def _tolerance(K, y, level):
    """Largest K still firing at >= `level`, interpolated.

    Returns (value, status).  `"censored"` means the rate never fell below the level inside the
    swept K range, so only a lower bound is known.  `"unreachable"` means the rate is already
    below the level at K = 0: there is no K to report, and reporting 0 would put a point on the
    axis that reads as a measured frontier rather than a floor.  Only `"ok"` is plottable.
    """
    if y[0] < level:
        return 0.0, "unreachable"
    below = np.nonzero(y < level)[0]
    if below.size == 0:
        return float(K.max()), "censored"
    i = below[0]
    val = K[i - 1] + (y[i - 1] - level) * (K[i] - K[i - 1]) / (y[i - 1] - y[i])
    return float(val), "ok"


def _tradeoff_panel(ax, d, level, norm, annotate=True, verbose=True, biases=None,
                    ctrl=BIAS_CTRL):
    """Scatter the two tolerances against each other, one point per bias, with a linear fit.

    A bias is plotted only when both tolerances are measured.  Right-censored ones are dropped
    because only a lower bound is known, and unreachable ones because the rule never hits the
    threshold at any K -- plotting those at zero would stack a row of points along the axis that
    looks like a measured frontier.  Returns (slope, r, n) or None.
    """
    pts, dropped = [], {"censored": 0, "unreachable": 0}
    for B in (sorted(d[ctrl["col"]].unique()) if biases is None else biases):
        g = d[d[ctrl["col"]] == B].sort_values("num_competing")
        K = g.num_competing.values
        tb, sb = _tolerance(K, g.boosted_firing_rate.values, level)
        tu, su = _tolerance(K, g.unboosted_firing_rate.values, level)
        if sb == "ok" and su == "ok":
            pts.append((B, tb, tu))
        else:
            for st in (sb, su):
                if st != "ok":
                    dropped[st] += 1
    if len(pts) < 2:
        ax.set_axis_off()
        return None
    x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
    sl, ic = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, ic + sl * xs, "--", color="0.55", lw=LW_FIT, zorder=2)
    ax.scatter(x, y, c=[p[0] for p in pts], cmap=colormaps[CMAP], norm=norm,
               s=MS_POINT, edgecolor="white", linewidth=MEW_POINT, zorder=4)
    if annotate:
        # Both labels go horizontally inward, into the gap the frontier leaves beside its end
        # markers: the fitted line rises to the left of the last point and falls to the right of
        # the first, so neither label meets a marker or the line.  Placing them diagonally
        # instead would push the axis limits out to make room, stretching the range past the
        # data just to seat a piece of text.
        lo, hi = min(pts, key=lambda q: q[1]), max(pts, key=lambda q: q[1])
        for (B, tb, tu), (dx, dy, ha, va) in zip(
                [lo, hi], [(10, 0, "left", "center"), (-10, 0, "right", "center")]):
            ax.annotate(ctrl["fmt"](B), (tb, tu), textcoords="offset points", xytext=(dx, dy),
                        fontsize=FS_ANNOT, color="#374151", ha=ha, va=va)
    # Pad the data, then round the bottom of the y-axis down to a whole tick so it reads 30
    # rather than 30.5 and no blank strip sits under the lowest point.  Only that edge is snapped:
    # rounding every edge picks a coarse step and pushes the axis out to 0.
    xr, yr = x.max() - x.min(), max(y.max() - y.min(), 1e-9)
    ax.set_xlim(x.min() - 0.06 * xr, x.max() + 0.06 * xr)
    ylo, yhi = y.min() - 0.06 * yr, y.max() + 0.06 * yr
    step = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(ylo, yhi)
    step = step[1] - step[0]
    ax.set_ylim(np.floor(ylo / step) * step, yhi)
    if verbose:
        note = ", ".join(f"{v} {k}" for k, v in dropped.items() if v)
        print(f"  {level:.0%} threshold: slope={sl:.3f}, r={r:+.4f}, n={len(pts)}"
              + (f"  (dropped: {note})" if note else ""))
    return sl, r, len(pts)


def plot_combined(d: pd.DataFrame, out: Path, level: float, width_pt=DEFAULT_WIDTH_PT,
                  ctrl=BIAS_CTRL) -> None:
    biases = sorted(d[ctrl["col"]].unique())
    norm = Normalize(vmin=min(biases), vmax=max(biases))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=_figsize(width_pt, 0.365))

    for ax in axes:
        ax.tick_params(labelsize=FS_TICK, pad=1.5)
    for ax, (col, title) in zip(axes[:2], [("boosted_firing_rate", "Boosted rule fires"),
                                           ("unboosted_firing_rate", "Unboosted rule fires")]):
        for B in biases:
            g = d[d[ctrl["col"]] == B].sort_values("num_competing")
            base = abs(B - ctrl.get("base", 0.0)) < 1e-9
            ax.plot(g.num_competing, g[col], color="black" if base else _colour(B, norm),
                    ls="--" if base else "-", lw=LW_BASE if base else LW_CURVE, zorder=4 if base else 3)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel("Number of added competing rules", fontsize=FS_LABEL)
        ax.set_xlim(0, d.num_competing.max())
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Firing rate", fontsize=FS_LABEL)
    axes[1].set_yticklabels([])

    _tradeoff_panel(axes[2], d, level, norm, ctrl=ctrl)
    axes[2].set_title(f"{level:.0%} firing threshold trade-off", fontsize=FS_TITLE)
    axes[2].set_xlabel("Boosted-rule tolerance", fontsize=FS_LABEL)
    axes[2].set_ylabel("Unboosted-rule tolerance", fontsize=FS_LABEL)

    fig.subplots_adjust(left=0.068, right=0.905, bottom=0.20, top=0.90, wspace=0.20)
    cax = fig.add_axes((0.922, 0.22, 0.014, 0.66))
    fig.colorbar(ScalarMappable(norm=norm, cmap=colormaps[CMAP]), cax=cax,
                 label=ctrl["label"])
    cax.tick_params(labelsize=FS_TICK, pad=1.5)
    cax.yaxis.label.set_size(FS_LABEL)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig); plt.style.use("default")
    print(f"wrote {out.with_suffix('.png')}")


def plot_thresholds(d: pd.DataFrame, out: Path, levels, width_pt=DEFAULT_WIDTH_PT,
                    ctrl=BIAS_CTRL) -> None:
    """The same trade-off read at several firing thresholds.

    The threshold is a free parameter -- "tolerance" only means anything once you say how often
    the rule has to fire -- so the point is to show that the frontier's shape survives the choice.

    Each panel runs over every bias it can measure, so the panels end at different biases, and
    for different reasons.  A loose threshold leaves the boosted rule censored early (still above
    30% at K = 200 once B >= 0.9).  A strict one makes the unboosted rule unreachable: it fires
    below 90% at K = 0 from B = 1.2 on, and since firing only falls as K grows, no K recovers it.
    Neither is a gap in the sweep, so neither is worth trimming the other panels to hide.
    """
    biases = sorted(d[ctrl["col"]].unique())
    norm = Normalize(vmin=min(biases), vmax=max(biases))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(levels), figsize=_figsize(width_pt, 0.285))
    axes = np.atleast_1d(axes)
    print("  trade-off fits by threshold:")
    for ax, level in zip(axes, levels):
        _tradeoff_panel(ax, d, level, norm, ctrl=ctrl)
        ax.set_title(f"{level:.0%} firing threshold", fontsize=FS_TITLE)
        ax.set_xlabel("Boosted-rule tolerance", fontsize=FS_LABEL)
        # Each panel keeps its own scale -- the tolerances shrink as the threshold rises -- so the
        # tick labels cannot be shared.  Fewer, smaller ones is what buys the panels room to sit
        # closer together.
        ax.tick_params(labelsize=FS_TICK, pad=1.5)
    axes[0].set_ylabel("Unboosted-rule tolerance", fontsize=FS_LABEL)

    right = 0.905
    fig.subplots_adjust(left=0.065, right=right, bottom=0.21, top=0.89, wspace=0.28)
    cax = fig.add_axes((right + 0.017, 0.23, 0.014, 0.63))
    fig.colorbar(ScalarMappable(norm=norm, cmap=colormaps[CMAP]), cax=cax,
                 label=ctrl["label"])
    cax.tick_params(labelsize=FS_TICK, pad=1.5)
    cax.yaxis.label.set_size(FS_LABEL)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig); plt.style.use("default")
    print(f"wrote {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "learned_boost_tradeoff.csv")
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--width-pt", type=float, default=DEFAULT_WIDTH_PT,
                    help="target \\linewidth in points; the figure is drawn at exactly this width")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
                    help="firing thresholds for the multi-threshold trade-off figure")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    for col in ("num_init_facts", "applicable_fraction"):
        if d[col].nunique() > 1:
            keep = d[col].value_counts().idxmax()
            print(f"note: several {col} values; plotting {col}={keep}")
            d = d[d[col] == keep]
    d = d.assign(boost_mult=np.exp(d.attn_bias))
    stem = args.csv.stem.replace("learned_", "", 1)
    out_dir = args.csv.parent
    plot_combined(d, out_dir / f"{stem}_combined", args.level, args.width_pt)
    plot_combined(d, out_dir / f"{stem}_multiplier_combined", args.level, args.width_pt,
                  ctrl=MULT_CTRL)
    plot_thresholds(d, out_dir / f"{stem}_thresholds", args.thresholds, args.width_pt)


if __name__ == "__main__":
    main()
