"""defense_full_summary, rebuilt on the append-mode sweeps.

Same three panels as plot_full_defense_summary.py -- failure induction rate against suffix
repetitions for fact amnesia and rule suppression, across-x0 variance for state coercion --
but reading the append-mode CSVs, where the adversarial suffix is added to the rulebook
instead of carved out of it. That lifts the k <= 26 ceiling (the FIR sweeps run to k = 100)
and removes the confound where raising k also deleted legitimate rules.

Written as a separate script rather than a flag on the original so the published figure and
its in-progress edits are left alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps, colors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
FACT_CSV = ROOT / "results/learned_fact_amnesia_rule_attention_bias_append.csv"
SUPPRESS_CSV = ROOT / "results/learned_suppress_rule_rule_attention_bias_append.csv"
# The coercion variance runner has its own construction (delta dataset fixed at 4 attack
# tokens, independent of k), so it was never capped or confounded and needs no append variant.
COERCE_VAR_CSV = ROOT / "results/learned_coerce_state_variance_rule_attention_bias.csv"
OUT_DIR = ROOT / "results/subversion_lower_bounds"
OUT_STEM = OUT_DIR / "defense_full_summary_append"
MAX_BIAS = 2.5

# Paper sizing, matching plotting/plot_boost_tradeoff.py: drawn at exactly the width it is
# included at, so LaTeX never rescales and these point sizes are the page point sizes.
WIDTH_PT = 469.75
PT_PER_INCH = 72.27
DPI = 400
RATIO = 0.40  # three panels plus the bias legend below them
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 8.5, 8.0, 7.0, 7.0


def _figsize(ratio, width_pt=WIDTH_PT):
    w = width_pt / PT_PER_INCH
    return w, w * ratio


def _set_sparse_ticks(ax, k_vals):
    k_vals = sorted(int(k) for k in k_vals)
    k_min, k_max = k_vals[0], k_vals[-1]
    if k_max >= 60:
        # The append sweeps run to k=100; a tick every 20 keeps the axis readable.
        step = 20
        ticks = [k_min] + list(range(step, k_max + 1, step))
    elif k_max >= 25:
        ticks = [k_min, 5, 10, 15, 20, k_max]
    elif len(k_vals) > 6:
        step = max(1, round((k_max - k_min) / 4))
        ticks = [k_min] + list(range(((k_min + step - 1) // step) * step, k_max, step))
        if ticks[-1] != k_max:
            ticks.append(k_max)
    else:
        ticks = k_vals
    ax.set_xticks(sorted(set(ticks)))


def _plot_panel(ax, df, y_col, title, y_label, biases, cmap, norm, label_map):
    for bias in biases:
        sub = df[df["attn_bias"] == bias].sort_values("num_attack_tokens")
        if sub.empty:
            continue
        is_base = abs(float(bias)) < 1e-12
        color = "#222222" if is_base else cmap(0.95 - 0.75 * norm(bias))
        ax.plot(
            sub["num_attack_tokens"],
            sub[y_col],
            color=color,
            linewidth=1.7,
            linestyle=("--" if is_base else "-"),
            label=label_map[bias],
        )

    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel("Suffix repetitions", fontsize=FS_LABEL)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK, pad=1.5)
    _set_sparse_ticks(ax, df["num_attack_tokens"].unique())


def main() -> None:
    for p in [FACT_CSV, SUPPRESS_CSV, COERCE_VAR_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    frames = {}
    for key, path, needed in [
        ("fact", FACT_CSV, {"num_attack_tokens", "attn_bias", "adv_ns3_state_acc"}),
        ("suppress", SUPPRESS_CSV, {"num_attack_tokens", "attn_bias", "adv_ns3_state_acc"}),
        ("coerce", COERCE_VAR_CSV, {"num_attack_tokens", "attn_bias", "variance"}),
    ]:
        df = pd.read_csv(path)
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        frames[key] = df[df["attn_bias"] <= MAX_BIAS].copy()

    biases = sorted(set().union(*(set(d["attn_bias"].unique()) for d in frames.values())))
    norm = colors.Normalize(vmin=min(biases), vmax=max(biases))
    cmap = colormaps["magma"]
    label_map = {b: f"B={b:.1f}" for b in biases}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=_figsize(RATIO))

    _plot_panel(axes[0], frames["fact"], "adv_ns3_state_acc", "Fact Amnesia",
                "Failure Induction Rate", biases, cmap, norm, label_map)
    axes[0].set_ylim(-0.02, 1.02)

    _plot_panel(axes[1], frames["suppress"], "adv_ns3_state_acc", "Rule Suppression",
                "Failure Induction Rate", biases, cmap, norm, label_map)
    axes[1].set_ylim(-0.02, 1.02)

    _plot_panel(axes[2], frames["coerce"], "variance", "State Coercion",
                "Variance", biases, cmap, norm, label_map)
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)),
               fontsize=FS_LEGEND, frameon=True, title=r"Instruction-attention bias B",
               title_fontsize=FS_LEGEND, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_STEM.with_suffix('.png')} / .pdf")


if __name__ == "__main__":
    main()
