"""Attention to applicable vs non-applicable rule rows, clean (unboosted) reasoner.

Two quantities, because "attention given" can mean either side of the softmax:
  * POST-softmax attention weight per rule row -- a probability in (0, 1), plotted on a log10
    x-axis because the groups sit orders of magnitude apart.
  * PRE-softmax attention logit per rule row -- the quantity the theory's lambda is stated in,
    taken as-is. Softmax is shift-invariant, so these are in principle only defined up to a
    per-example offset, but that offset is small here (std 0.35) and centring by it widens the
    applicable groups rather than tightening them; `logit_centred` is kept in the CSV.

Each is drawn as a smoothed Gaussian KDE at `<stem>.png` (y-axis a density). `--hist` instead
draws raw histograms at `<stem>_hist.png`, whose y-axis is a readable fraction of rows. Both pool all reasoning steps
into one panel and split the applicable rules into "not yet fired" and "already fired" -- that
split, not the step index, is what separates the peaks. The pre-softmax figure is additionally drawn
per reasoning step (`applicable_attention_per_step_logits`), with rules grouped by the step at
which they first become applicable.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/lambda_separation/applicable_attention_rows.csv"
OUT_DIR = ROOT / "results/lambda_separation"

# Slots 1-2 of the validated default palette. Both applicable groups share the blue hue, so
# applicability reads as one family; the already-fired group is separated by a dashed edge
# rather than by a third colour. Non-applicable takes the contrasting warm hue.
GROUPS = [
    ("applicable_new", "Applied at this step", "#2a78d6", "solid"),
    ("applicable_fired", "Applied at an earlier step", "#2a78d6", (0, (2.2, 1.4))),
    ("non_applicable", "Not applicable", "#eb6834", "solid"),
]
# A distinct hue per step, from the validated palette: blue / aqua / violet clear every
# all-pairs gate together with the warm hue that "never applicable" keeps (which also stays
# dashed, so it does not rely on colour alone).
FIRST_APPLICABLE_GROUPS = [
    (1, "First applicable at step 1", "#2a78d6", "solid"),
    (2, "First applicable at step 2", "#1baf7a", "solid"),
    (3, "First applicable at step 3", "#4a3aa7", "solid"),
    (0, "Never applicable", "#eb6834", (0, (2.2, 1.4))),
]
INK = "#0b0b0b"
INK_SOFT = "#52514e"

# Groups can land almost on top of each other, so both fills and strokes are translucent.
POOLED_ALPHA = (0.14, 1.0)   # (fill, stroke)
PER_STEP_ALPHA = (0.20, 0.6)

METRICS = {
    "attn_weight": dict(
        slug="weights",
        log=True,
        xlabel="Attention weight (post-softmax probability, log scale)",
        title="Attention concentrates on applicable rules",
    ),
    "logit_raw": dict(
        slug="logits",
        log=False,
        xlabel="Pre-softmax attention logit",
        title="Attention logits separate by rule applicability",
    ),
}


def _style(ax, log: bool, clamp_y: bool = True) -> None:
    if log:
        ax.set_xscale("log")
    if clamp_y:
        ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=7.5, colors=INK_SOFT)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")


def _edges(vals: pd.Series, log: bool, n: int):
    """Bin edges (histogram) or an evaluation grid (KDE), in log10 space on a log axis."""
    v = vals[vals > 0] if log else vals
    lo, hi = (np.log10(v.min()), np.log10(v.max())) if log else (v.min(), v.max())
    pad = 0.06 * (hi - lo)
    space = np.logspace if log else np.linspace
    return space(lo - pad, hi + pad, n)


def _draw(ax, vals, support, color, label, log, kde, linestyle="solid", alpha=POOLED_ALPHA) -> None:
    if vals.empty:
        return
    fill_alpha, line_alpha = alpha
    if kde:
        # Estimated in log10 space on a log axis, where each group is actually unimodal.
        x = np.log10(vals[vals > 0]) if log else vals
        y = gaussian_kde(x)(np.log10(support) if log else support)
        ax.fill_between(support, y, color=color, alpha=fill_alpha, linewidth=0, zorder=2)
        ax.plot(support, y, color=color, linewidth=1.8, linestyle=linestyle, alpha=line_alpha,
                zorder=3, label=label or "_nolegend_")
    else:
        weights = np.full(len(vals), 1.0 / len(vals))
        ax.hist(vals, bins=support, weights=weights, color=color, alpha=fill_alpha, zorder=2)
        ax.hist(
            vals, bins=support, weights=weights, histtype="step", linestyle=linestyle,
            color=color, linewidth=1.6, alpha=line_alpha, zorder=3, label=label or "_nolegend_",
        )


def _ylabel(kde: bool) -> str:
    return "Density (within group)" if kde else "Fraction of rows in group"


def figure_pooled(df: pd.DataFrame, metric: str, spec: dict, kde: bool) -> None:
    support = _edges(df[metric], spec["log"], 400 if kde else 46)
    fig, ax = plt.subplots(figsize=(4.3, 2.9), layout="constrained")

    for key, label, color, linestyle in GROUPS:
        _draw(ax, df.loc[df["group"] == key, metric], support, color, label, spec["log"], kde, linestyle)

    _style(ax, spec["log"])
    ax.set_ylabel(_ylabel(kde), fontsize=8.5)
    ax.set_xlabel(spec["xlabel"], fontsize=8.5, color=INK, linespacing=1.4)
    ax.legend(
        fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK,
        handlelength=2.0, borderaxespad=0.4, labelspacing=0.4,
    )
    fig.suptitle(spec["title"], fontsize=9.5, color=INK)
    _save(fig, OUT_DIR / f"applicable_attention_{spec['slug']}{'' if kde else '_hist'}")


def figure_per_step(df: pd.DataFrame, metric: str, spec: dict, kde: bool) -> None:
    """One panel per reasoning step, with rules grouped by when they FIRST become applicable.

    That grouping is a property of the rule rather than of the step, so the same four curves
    appear in every panel and a group can be tracked across the proof. The legend is built from
    proxy handles so the full key shows even where a group is empty.
    """
    support = _edges(df[metric], spec["log"], 400 if kde else 46)
    steps = sorted(df["step"].unique())

    fig, axes = plt.subplots(
        1, len(steps), figsize=(7.4, 2.7), sharex=True, sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, step in zip(axes, steps):
        sub = df[df["step"] == step]
        for key, _label, color, linestyle in FIRST_APPLICABLE_GROUPS:
            _draw(ax, sub.loc[sub["first_applicable_step"] == key, metric], support, color,
                  None, spec["log"], kde, linestyle, alpha=PER_STEP_ALPHA)
        _style(ax, spec["log"], clamp_y=False)
        ax.set_title(f"Step {int(step)}", fontsize=9.5, color=INK)

    # Shared y-limit sized to the tallest panel, not just the first.
    axes[0].set_ylim(0, max(ax.dataLim.y1 for ax in axes) * 1.08)
    axes[0].set_ylabel(_ylabel(kde), fontsize=8.5)
    axes[len(axes) // 2].set_xlabel(spec["xlabel"].replace("\n", "  "), fontsize=8.5, color=INK)

    handles = [
        Line2D([], [], color=color, linewidth=1.8, linestyle=linestyle, label=label)
        for _key, label, color, linestyle in FIRST_APPLICABLE_GROUPS
    ]
    fig.legend(
        handles=handles, fontsize=8, frameon=False, ncol=4, loc="outside lower center",
        labelcolor=INK, handlelength=2.0,
    )
    fig.suptitle(spec["title"], fontsize=10, color=INK)
    _save(fig, OUT_DIR / f"applicable_attention_per_step_{spec['slug']}{'' if kde else '_hist'}")


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {stem}.png / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hist", action="store_true",
        help="Draw raw histograms (written as <stem>_hist) instead of the default KDE curves.",
    )
    args = parser.parse_args()

    if not CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV}. Run runners/run_applicable_attention.py first.")
    df = pd.read_csv(CSV)

    kde = not args.hist
    for metric, spec in METRICS.items():
        figure_pooled(df, metric, spec, kde=kde)
        if metric == "logit_raw":
            figure_per_step(df, metric, spec, kde=kde)


if __name__ == "__main__":
    main()
