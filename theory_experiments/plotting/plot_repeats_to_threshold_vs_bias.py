"""Suffix repetitions needed to induce failure, as a function of the rule-attention bias B.

Empirical support for the subversion lower-bound corollary: the competing magnitude needed
to induce a failure grows at least exponentially with B. We measure that magnitude as the
number of suffix repetitions k* at which an attack first reaches a fixed degree of success,
and plot k* against B, all three attacks on one axes.  A second figure reads the same k*
against the attention multiplier M = exp(B), where the cost is close to linear.

The sweeps are the append-mode ones: the suffix is appended and the rulebook kept whole, so
the attack can run past the rulebook cap instead of replacing rules.

Fact amnesia and rule suppression use the Failure Induction Rate (adv_ns3_state_acc).
State coercion has no usable FIR: its exact-match success is identically 0 at every (B, k)
measured, because coercion must hit a full 3-step target trajectory rather than drop a
single variable. Its analogue is the coercion-variance experiment, which measures
Var_{x0}[pred] -- the spread of the predicted state across initial states x0 for a fixed
attack target. Low variance means the suffix pins the state regardless of x0, so "repetitions
to halve the variance (relative to k=1 at the same B)" is the same kind of quantity.

Biases whose attack never reaches the threshold within the tested range are censored, and
shown at the k* the fitted exponential PREDICTS, as x markers (see FIT_ALL_POINTS for how
they enter the fit).

Output: results/subversion_lower_bounds/repeats_to_threshold_{merged,multiplier}_append
as png and pdf, plus the k* table as repeats_to_threshold_vs_bias_append.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACT_CSV = RESULTS / "learned_fact_amnesia_rule_attention_bias_append.csv"
SUPPRESS_CSV = RESULTS / "learned_suppress_rule_rule_attention_bias_append.csv"
COERCE_VAR_CSV = RESULTS / "learned_coerce_state_variance_rule_attention_bias.csv"
# Figures and derived tables for this analysis live together, away from the raw sweep CSVs.
OUT_DIR = RESULTS / "subversion_lower_bounds"
STEM = "_append"

MAX_BIAS = 2.5
THRESHOLD = 0.5
FIR_COL = "adv_ns3_state_acc"

# k* is read off the curve by interpolation on the crossing segment, so the estimate is not
# quantised to the integer grid (set False to use the integer ceiling).
INTERPOLATE = True

FIR_ATTACKS = [("Fact Amnesia", FACT_CSV), ("Rule Suppression", SUPPRESS_CSV)]

# Paper sizing, matching plotting/plot_boost_tradeoff.py: the figure is drawn at exactly the
# width it is included at, so LaTeX never rescales it and the point sizes below are the point
# sizes on the page. WIDTH_PT is \the\linewidth of the target document.
WIDTH_PT = 469.75
PT_PER_INCH = 72.27
DPI = 400
# A single panel does not span the full text width: it is drawn at WIDTH_FRAC_SINGLE of the
# linewidth and included at that fraction, so the point sizes below still land correctly.
WIDTH_FRAC_SINGLE = 0.5
RATIO_SINGLE = 0.70   # of the single-panel width, not of the linewidth
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 8.5, 8.0, 7.0, 6.5


def _figsize(ratio, width_pt=WIDTH_PT):
    w = width_pt / PT_PER_INCH
    return w, w * ratio


FIT = "#6b7280"
INK = "#374151"

# Merged single-panel view: one colour per failure mode, from the Okabe-Ito colourblind-safe
# qualitative palette (blue / vermillion / bluish green). Worst all-pairs separation is
# dE 11.0 deutan and 8.6 tritan -- the only candidate tested that clears the dE 8 floor under
# both. Each series also carries a marker shape, so identity never rests on colour alone.
MERGED_STYLE = {
    "Fact Amnesia": ("#0072B2", "o"),
    "Rule Suppression": ("#D55E00", "s"),
    "State Coercion": ("#009E73", "D"),
}
PREDICTED_MARKER = "x"
PREDICTED_LABEL = "Predicted by fit"
# Fit the exponential to every bias, censored ones included. A censored k* sits at the tested
# ceiling, which is a lower bound rather than a measurement, so including it pulls the fitted
# rate down; set False to fit only the biases that actually crossed the threshold.
FIT_ALL_POINTS = True

def load_fir(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = {"num_attack_tokens", "attn_bias", FIR_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
    return df[df["attn_bias"] <= MAX_BIAS].copy()


def load_coercion_variance() -> pd.DataFrame:
    if not COERCE_VAR_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {COERCE_VAR_CSV}")
    dv = pd.read_csv(COERCE_VAR_CSV)
    return dv[(dv["attn_bias"] <= MAX_BIAS) & dv["variance"].notna()].copy()


def repeats_to_fir(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Smallest k with FIR >= threshold, per bias. `censored` flags biases that never reach it."""
    k_max = int(df["num_attack_tokens"].max())
    rows = []
    for bias, sub in df.groupby("attn_bias"):
        sub = sub.sort_values("num_attack_tokens")
        k = sub["num_attack_tokens"].to_numpy(dtype=float)
        y = sub[FIR_COL].to_numpy(dtype=float)
        hits = np.flatnonzero(y >= threshold)
        if hits.size == 0:
            rows.append({"attn_bias": bias, "k_star": float(k_max), "k_int": k_max,
                         "censored": True, "best": y.max(), "k_tested": k_max})
            continue
        i = int(hits[0])
        k_star = k[i] if (i == 0 or not INTERPOLATE) else \
            k[i - 1] + (threshold - y[i - 1]) * (k[i] - k[i - 1]) / (y[i] - y[i - 1])
        rows.append({"attn_bias": bias, "k_star": float(k_star), "k_int": int(k[i]),
                     "censored": False, "best": y.max(), "k_tested": k_max})
    return pd.DataFrame(rows).sort_values("attn_bias").reset_index(drop=True)


def coercion_frac(severity: float) -> float:
    """Map a failure severity p to the remaining-variance fraction that represents it.

    Severity is defined so that larger means a harder attack target across all three panels:
    for FIR it is "FIR >= p", for coercion it is "variance reduced BY p", i.e. remaining
    variance <= (1 - p) of baseline. Without this flip a "75%" coercion curve would be an
    EASIER target than "25%", and the panel would read backwards against the other two.
    """
    return 1.0 - severity


def repeats_to_variance_drop(dv: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Smallest k whose across-x0 variance falls to `frac` of its own k=1 value at that bias.

    The k=1 variance itself grows with B, so normalising per bias is what makes the
    comparison across biases fair. Variance decays geometrically in k, so the crossing is
    interpolated in log-variance.
    """
    k_max = int(dv["num_attack_tokens"].max())
    rows = []
    for bias, sub in dv.groupby("attn_bias"):
        sub = sub.sort_values("num_attack_tokens")
        k = sub["num_attack_tokens"].to_numpy(dtype=float)
        v = sub["variance"].to_numpy(dtype=float)
        target = v[0] * frac
        hits = np.flatnonzero(v <= target)
        if hits.size == 0:
            rows.append({"attn_bias": bias, "k_star": float(k_max), "k_int": k_max,
                         "censored": True, "best": v.min() / v[0], "k_tested": k_max})
            continue
        i = int(hits[0])
        if i == 0 or not INTERPOLATE:
            k_star = k[i]
        else:
            lo, hi, tgt = np.log(v[i - 1]), np.log(v[i]), np.log(target)
            k_star = k[i - 1] + (tgt - lo) * (k[i] - k[i - 1]) / (hi - lo)
        rows.append({"attn_bias": bias, "k_star": float(k_star), "k_int": int(k[i]),
                     "censored": False, "best": v.min() / v[0], "k_tested": k_max})
    return pd.DataFrame(rows).sort_values("attn_bias").reset_index(drop=True)


def fit_exponential(thr: pd.DataFrame):
    """Least-squares fit of log k* against B (see FIT_ALL_POINTS for censored handling)."""
    obs = thr if FIT_ALL_POINTS else thr[~thr["censored"]]
    if len(obs) < 2:
        return None
    b = obs["attn_bias"].to_numpy(dtype=float)
    log_k = np.log(obs["k_star"].to_numpy(dtype=float))
    rate, intercept = np.polyfit(b, log_k, 1)
    ss_tot = ((log_k - log_k.mean()) ** 2).sum()
    ss_res = ((log_k - (intercept + rate * b)) ** 2).sum()
    return rate, intercept, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))



def predicted_censored(thr: pd.DataFrame, fit) -> pd.DataFrame:
    """Where the fitted exponential puts k* for the biases the sweep could not reach."""
    cen = thr[thr["censored"]]
    if fit is None or cen.empty:
        return pd.DataFrame({"attn_bias": [], "k_pred": []})
    rate, intercept, _ = fit
    b = cen["attn_bias"].to_numpy(dtype=float)
    return pd.DataFrame({"attn_bias": b, "k_pred": np.exp(intercept + rate * b)})


def _draw_series(ax, thr, color, label=None, linestyle="none", marker="o"):
    """Plot the measured k* per bias; censored biases are handled by _draw_predicted."""
    obs = thr[~thr["censored"]]
    ax.plot(obs["attn_bias"], obs["k_star"], color=color, marker=marker, markersize=5.2,
            linestyle=linestyle, linewidth=1.6, markeredgecolor="white",
            markeredgewidth=0.8, label=label, zorder=4, clip_on=False)


def _draw_predicted(ax, thr, fit, color) -> float:
    """Mark the fit's prediction for each censored bias; returns the highest value drawn."""
    pred = predicted_censored(thr, fit)
    if pred.empty:
        return 0.0
    ax.plot(pred["attn_bias"], pred["k_pred"], linestyle="none", marker=PREDICTED_MARKER,
            markersize=6.2, markeredgewidth=1.8, color=color, zorder=5, clip_on=False)
    return float(pred["k_pred"].max())


def _draw_fit(ax, thr, biases, color=FIT):
    """Dashed exponential fit over the measured biases, dotted where it extrapolates."""
    fit = fit_exponential(thr)
    if fit is None:
        return None
    rate, intercept, _ = fit
    b_last = thr.loc[~thr["censored"], "attn_bias"].max()
    fitted = np.linspace(min(biases), b_last, 60)
    ax.plot(fitted, np.exp(intercept + rate * fitted), color=color, linewidth=1.4,
            linestyle="-", zorder=2)
    if b_last < max(biases):
        # Beyond the last measured bias the line is extrapolation, not evidence.
        extra = np.linspace(b_last, max(biases), 30)
        ax.plot(extra, np.exp(intercept + rate * extra), color=color, linewidth=1.0,
                linestyle=":", alpha=0.75, zorder=2)
    return fit


def _style_axes(ax, biases, k_hi):
    ax.set_xlabel(r"Attention bias $B$", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK, pad=1.5)
    ax.set_xticks(biases)
    # B is a non-negative quantity: start the axis at 0 rather than padding into negative bias.
    ax.set_xlim(0, max(biases) + 0.15)
    ax.set_ylim(0, k_hi * 1.12)
    step = 5 if k_hi <= 40 else 10
    ax.set_yticks(np.arange(0, k_hi * 1.12, step))


def make_merged_figure(panels, out_stem: Path) -> None:
    """All three failure modes on one axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=_figsize(RATIO_SINGLE, WIDTH_PT * WIDTH_FRAC_SINGLE))
    biases = sorted({b for _, thr, _ in panels for b in thr["attn_bias"]})
    y_hi, handles = max(thr["k_star"].max() for _, thr, _ in panels), []

    for name, thr, _ in panels:
        color, marker = MERGED_STYLE[name]
        fit = fit_exponential(thr)
        _draw_fit(ax, thr, biases, color=color)
        _draw_series(ax, thr, color, marker=marker)
        y_hi = max(y_hi, _draw_predicted(ax, thr, fit, color))
        handles.append(Line2D([0], [0], color=color, marker=marker, markersize=5.2,
                              linestyle="-", linewidth=1.4, markeredgecolor="white",
                              markeredgewidth=0.8, label=name))
    if any(thr["censored"].any() for _, thr, _ in panels):
        handles.append(Line2D([0], [0], color=INK, linestyle="none",
                              marker=PREDICTED_MARKER, markersize=6.2, markeredgewidth=1.8,
                              label=PREDICTED_LABEL))
    ax.legend(handles=handles, fontsize=FS_ANNOT, frameon=True, loc="upper left", labelspacing=0.3,
              handlelength=1.9, borderpad=0.45)
    ax.set_title(f"Subversion cost at {THRESHOLD:.0%} failure", fontsize=FS_TITLE)
    ax.set_ylabel("Suffix repetitions", fontsize=FS_LABEL)
    _style_axes(ax, biases, y_hi)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf")


def fit_linear_in_multiplier(thr: pd.DataFrame):
    """Least-squares fit of k* against M = exp(B): slope, intercept, R^2.

    On this axis the relationship is affine rather than a power law -- an exponential in B
    re-expressed in M gives k* ~ M^alpha, but a line fits the measurements better (the
    exponential's alpha < 1 is standing in for a positive intercept).
    """
    if len(thr) < 2:
        return None
    m = np.exp(thr["attn_bias"].to_numpy(dtype=float))
    k = thr["k_star"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(m, k, 1)
    ss_tot = ((k - k.mean()) ** 2).sum()
    ss_res = ((k - (intercept + slope * m)) ** 2).sum()
    return slope, intercept, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def make_multiplier_figure(panels, out_stem: Path) -> None:
    """The same k*, plotted against the attention multiplier M = exp(B) rather than B.

    A bias of B multiplies the instruction rows' attention weight by exp(B), so M is the
    quantity a practitioner actually sets. Since k* ~ exp(alpha B), against M it is a power
    law k* ~ M^alpha, and the fitted alphas are near 1 -- so on this axis the cost of
    subverting the model is close to linear in the multiplier.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=_figsize(RATIO_SINGLE, WIDTH_PT * WIDTH_FRAC_SINGLE))
    biases = np.array(sorted({b for _, thr, _ in panels for b in thr["attn_bias"]}), dtype=float)
    handles = []

    grid_m = np.linspace(1.0, float(np.exp(biases.max())), 200)
    for name, thr, _ in panels:
        color, marker = MERGED_STYLE[name]
        fit = fit_linear_in_multiplier(thr)
        if fit is not None:
            slope, intercept, _ = fit
            ax.plot(grid_m, intercept + slope * grid_m, color=color, linewidth=1.4,
                    linestyle="-", zorder=2)
        ax.plot(np.exp(thr["attn_bias"].to_numpy(dtype=float)), thr["k_star"], color=color,
                marker=marker, markersize=5.2, linestyle="none", markeredgecolor="white",
                markeredgewidth=0.8, zorder=4, clip_on=False)
        handles.append(Line2D([0], [0], color=color, marker=marker, markersize=5.2,
                              linestyle="-", linewidth=1.4, markeredgecolor="white",
                              markeredgewidth=0.8, label=name))

    ax.legend(handles=handles, fontsize=FS_ANNOT, frameon=True, loc="upper left",
              labelspacing=0.3, handlelength=1.9, borderpad=0.45)
    ax.set_title(f"Subversion cost at {THRESHOLD:.0%} failure", fontsize=FS_TITLE)
    ax.set_xlabel(r"Attention multiplier $M = e^{B}$", fontsize=FS_LABEL)
    ax.set_ylabel("Suffix repetitions", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK, pad=1.5)
    m_hi = float(np.exp(biases.max()))
    ax.set_xlim(1.0, m_hi * 1.03)          # M = 1 is no boost; below it is not a defence
    ax.set_xticks([t for t in (1, 2, 4, 6, 8, 10, 12, 14) if t <= m_hi * 1.03])
    k_hi = max(thr["k_star"].max() for _, thr, _ in panels)
    ax.set_ylim(0, k_hi * 1.12)
    ax.set_yticks(np.arange(0, k_hi * 1.12, 10))

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf")


def main() -> None:
    frames = [(name, load_fir(csv_path)) for name, csv_path in FIR_ATTACKS]
    coerce_dv = load_coercion_variance()

    panels = [(name, repeats_to_fir(df, THRESHOLD), r"$50\%$ FIR") for name, df in frames]
    panels.append(("State Coercion",
                   repeats_to_variance_drop(coerce_dv, coercion_frac(THRESHOLD)),
                   r"$50\%$ variance drop"))

    table = pd.concat([thr.assign(attack=name) for name, thr, _ in panels], ignore_index=True)
    table = table[["attack", "attn_bias", "k_star", "k_int", "censored", "best"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"repeats_to_threshold_vs_bias{STEM}.csv", index=False)
    print(table.to_string(index=False))
    for name, thr, _ in panels:
        fit = fit_exponential(thr)
        if fit is not None:
            print(f"{name}: k* ~ exp({fit[0]:.3f} B), R^2 = {fit[2]:.4f}")
        lin = fit_linear_in_multiplier(thr)
        if lin is not None:
            print(f"{name}: k* = {lin[0]:.2f} M + {lin[1]:.2f}, R^2 = {lin[2]:.4f}  (M = e^B)")

    make_merged_figure(panels, OUT_DIR / f"repeats_to_threshold_merged{STEM}")
    make_multiplier_figure(panels, OUT_DIR / f"repeats_to_threshold_multiplier{STEM}")


if __name__ == "__main__":
    main()
