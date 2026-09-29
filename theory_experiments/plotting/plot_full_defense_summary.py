from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACT_CSV = ROOT / "results/learned_fact_amnesia_rule_attention_bias.csv"
SUPPRESS_CSV = ROOT / "results/learned_suppress_rule_rule_attention_bias.csv"
COERCE_VAR_CSV = ROOT / "results/learned_coerce_state_variance_rule_attention_bias.csv"
OUT_FIG_PNG = ROOT / "results/defense_full_summary.png"
OUT_FIG_PDF = ROOT / "results/defense_full_summary.pdf"
MAX_BIAS = 2.5


def _set_sparse_ticks(ax, k_vals):
    k_vals = sorted(int(k) for k in k_vals)
    k_min, k_max = k_vals[0], k_vals[-1]
    if k_max >= 25:
        ticks = [k_min, 5, 10, 15, 20, k_max]
        ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
    elif len(k_vals) > 6:
        step = max(1, round((k_max - k_min) / 4))
        ticks = [k_min]
        ticks.extend(list(range(((k_min + step - 1) // step) * step, k_max, step)))
        if ticks[-1] != k_max:
            ticks.append(k_max)
        ax.set_xticks(sorted(set(ticks)))
    else:
        ax.set_xticks(k_vals)


def _plot_panel(ax, df: pd.DataFrame, y_col: str, title: str, y_label: str, biases, cmap, norm, label_map):
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
            linewidth=(1.7),
            linestyle=("--" if is_base else "-"),
            label=label_map[bias],
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Suffix repetitions", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    _set_sparse_ticks(ax, df["num_attack_tokens"].unique())


def main() -> None:
    for p in [FACT_CSV, SUPPRESS_CSV, COERCE_VAR_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    fact_df = pd.read_csv(FACT_CSV)
    suppress_df = pd.read_csv(SUPPRESS_CSV)
    coerce_df = pd.read_csv(COERCE_VAR_CSV)

    fact_df = fact_df[fact_df["attn_bias"] <= MAX_BIAS].copy()
    suppress_df = suppress_df[suppress_df["attn_bias"] <= MAX_BIAS].copy()
    coerce_df = coerce_df[coerce_df["attn_bias"] <= MAX_BIAS].copy()

    req_fact_sup = {"num_attack_tokens", "attn_bias", "adv_ns3_state_acc", "raw_state_acc"}
    req_coerce = {"num_attack_tokens", "attn_bias", "variance"}
    if req_fact_sup - set(fact_df.columns):
        raise ValueError("Fact-amnesia CSV missing required columns.")
    if req_fact_sup - set(suppress_df.columns):
        raise ValueError("Suppress-rule CSV missing required columns.")
    if req_coerce - set(coerce_df.columns):
        raise ValueError("Coerce-state variance CSV missing required columns.")

    biases = sorted(
        set(fact_df["attn_bias"].unique())
        .union(set(suppress_df["attn_bias"].unique()))
        .union(set(coerce_df["attn_bias"].unique()))
    )
    norm = colors.Normalize(vmin=min(biases), vmax=max(biases))
    cmap = colormaps["magma"]

    label_map = {}
    for b in biases:
        if abs(float(b)) < 1e-12:
            label_map[b] = f"B={b:.1f}"
        else:
            label_map[b] = f"B={b:.1f}"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.7))

    _plot_panel(
        axes[0], fact_df, "adv_ns3_state_acc", "Fact Amnesia", "Failure Induction Rate", biases, cmap, norm, label_map
    )
    axes[0].set_ylim(-0.02, 1.02)

    _plot_panel(
        axes[1], suppress_df, "adv_ns3_state_acc", "Rule Suppression", "Failure Induction Rate", biases, cmap, norm, label_map
    )
    axes[1].set_ylim(-0.02, 1.02)

    _plot_panel(
        axes[2], coerce_df, "variance", "State Coercion", "Variance", biases, cmap, norm, label_map
    )
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(labels)),
        fontsize=8,
        frameon=True,
        title=r"Instruction-attention bias B",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.03),
    )

    # Reserve bottom space for legend to avoid overlap.
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    OUT_FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PNG, dpi=240)
    fig.savefig(OUT_FIG_PDF)
    plt.close(fig)
    print(f"Saved: {OUT_FIG_PNG}")
    print(f"Saved: {OUT_FIG_PDF}")


if __name__ == "__main__":
    main()
