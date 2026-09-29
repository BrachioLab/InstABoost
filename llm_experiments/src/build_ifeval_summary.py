#!/usr/bin/env python
"""Build the IFEval summary markdown, LaTeX, and figure from results/ifeval/.

Reads each per-method results.json under
    results/ifeval/{model_slug}/{method}/results.json
and emits:
    results/ifeval_summary.md  — three table layouts (group, two-line, slash)
    results/ifeval_summary.tex — same three layouts in LaTeX (booktabs + makecell)
    results/ifeval_summary.png/.pdf — single-panel grouped bar chart

Numbers are formatted as percent with 1 decimal place, e.g. ``65.0 ± 4.6``,
where the value before ± is strict accuracy and the value after is the
half-width of the bootstrap 95% CI (both in percentage points).

Usage:
    python src/build_ifeval_summary.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "ifeval"
OUT_MD = REPO_ROOT / "results" / "ifeval_summary.md"
OUT_TEX = REPO_ROOT / "results" / "ifeval_summary.tex"
OUT_PNG = REPO_ROOT / "results" / "ifeval_summary.png"
OUT_PDF = REPO_ROOT / "results" / "ifeval_summary.pdf"

# Display name and order for models. Maps directory slug -> column label.
MODEL_DISPLAY = {
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "gemma-7b-it": "Gemma-7b-it",
    "Qwen3-14B": "Qwen3-14B",
    "gpt-oss-20b": "GPT-OSS-20B",
}
METHODS = ["prompt", "pasta", "spotlight", "instaboost"]

METHOD_COLORS = {
    "prompt": "#9aa0a6",
    "pasta": "#f4b400",
    "spotlight": "#4285f4",
    "instaboost": "#34a853",
}
LEVELS = ("sample", "instruction")
LEVEL_HATCH = {"sample": "", "instruction": "//"}
LEVEL_LABEL = {"sample": "sample-level", "instruction": "instruction-level"}


def half_width(ci: dict) -> float:
    return (ci["upper"] - ci["lower"]) / 2.0


def load_data() -> dict[str, dict[str, dict[str, tuple[float, float] | None]]]:
    """Returns DATA[display_name][level][method] = (acc_pct, hw_pct) or None."""
    data: dict = {}
    for slug, display in MODEL_DISPLAY.items():
        mdir = RESULTS_ROOT / slug
        if not mdir.is_dir():
            print(f"[warn] missing model dir: {mdir}")
            continue
        data[display] = {"sample": {}, "instruction": {}}
        for method in METHODS:
            path = mdir / method / "results.json"
            if not path.exists():
                data[display]["sample"][method] = None
                data[display]["instruction"][method] = None
                continue
            d = json.loads(path.read_text())
            data[display]["sample"][method] = (
                d["prompt_level_strict_acc"] * 100,
                half_width(d["prompt_level_ci"]) * 100,
            )
            data[display]["instruction"][method] = (
                d["instruction_level_strict_acc"] * 100,
                half_width(d["instruction_level_ci"]) * 100,
            )
    return data


def fmt(entry: tuple[float, float] | None, bold: bool = False) -> str:
    if entry is None:
        return "—"
    s = f"{entry[0]:.1f} ± {entry[1]:.1f}"
    return f"**{s}**" if bold else s


def best_methods(level_data: dict[str, tuple[float, float] | None]) -> set[str]:
    best_acc = -1.0
    best = set()
    for method, entry in level_data.items():
        if entry is None:
            continue
        if entry[0] > best_acc:
            best_acc = entry[0]
            best = {method}
        elif entry[0] == best_acc:
            best.add(method)
    return best


def write_markdown(data: dict) -> None:
    models = list(data.keys())

    lines: list[str] = []
    lines.append("# IFEval results")
    lines.append("")
    lines.append(
        "All values reported as **strict accuracy ± half-width of 95% CI**, "
        "in percentage points (e.g. `65.0 ± 4.6` means 65.0% accuracy with "
        "a ±4.6 pp half-width)."
    )
    lines.append("")
    lines.append("Bold = best method per (model, level).")
    lines.append("")

    # Layout 1: original 8-column grouped layout
    header = (
        "| Model | sample-prompt | sample-pasta | sample-spotlight | "
        "sample-instaboost | inst-prompt | inst-pasta | inst-spotlight | inst-instaboost |"
    )
    sep = "|---|" + ":---:|" * 8
    lines.append(header)
    lines.append(sep)
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            cells.append(fmt(sample[method], bold=method in sample_best))
        for method in METHODS:
            cells.append(fmt(inst[method], bold=method in inst_best))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Layout 2: two lines per cell (top: sample, bottom: instruction)
    lines.append("## Alternative layout 1 — two lines per cell (top: sample-level, bottom: instruction-level)")
    lines.append("")
    lines.append(
        "*Each cell shows sample-level strict accuracy on top, instruction-level "
        "on bottom. Bold = best per (model, level).*"
    )
    lines.append("")
    lines.append("| Model | prompt | pasta | spotlight | instaboost |")
    lines.append("|---|" + ":---:|" * 4)
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            top = fmt(sample[method], bold=method in sample_best)
            bot = fmt(inst[method], bold=method in inst_best)
            if sample[method] is None and inst[method] is None:
                cells.append("—")
            else:
                cells.append(f"{top}<br>{bot}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Layout 3: slash per cell
    lines.append("## Alternative layout 2 — sample / instruction in one cell")
    lines.append("")
    lines.append(
        "*Each cell shows `sample-level / instruction-level` strict accuracy. "
        "Bold = best per (model, level).*"
    )
    lines.append("")
    lines.append("| Model | prompt | pasta | spotlight | instaboost |")
    lines.append("|---|" + ":---:|" * 4)
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            top = fmt(sample[method], bold=method in sample_best)
            bot = fmt(inst[method], bold=method in inst_best)
            if sample[method] is None and inst[method] is None:
                cells.append("—")
            else:
                cells.append(f"{top} / {bot}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


def fmt_tex(entry: tuple[float, float] | None, bold: bool = False) -> str:
    """LaTeX cell value: math mode with \\pm; bold via \\mathbf for math-mode bold."""
    if entry is None:
        return "--"
    inner = f"{entry[0]:.1f} \\pm {entry[1]:.1f}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def write_latex(data: dict) -> None:
    models = list(data.keys())

    lines: list[str] = []
    lines.append("% IFEval results — auto-generated by src/build_ifeval_summary.py")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{makecell}")
    lines.append(
        "% Values: strict accuracy (%) ± half-width of 95% CI (percentage points)."
    )
    lines.append("% Bold = best method per (model, level).")
    lines.append("")

    # Layout 1: 8-column grouped layout (sample-* then inst-*)
    lines.append("% --- Layout 1: 8 grouped columns ---")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "c" * 8 + "}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{4}{c}{Sample-level (prompt-strict)} "
        "& \\multicolumn{4}{c}{Instruction-level (instruction-strict)} \\\\"
    )
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    lines.append(
        "Model & prompt & pasta & spotlight & instaboost "
        "& prompt & pasta & spotlight & instaboost \\\\"
    )
    lines.append("\\midrule")
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            cells.append(fmt_tex(sample[method], bold=method in sample_best))
        for method in METHODS:
            cells.append(fmt_tex(inst[method], bold=method in inst_best))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{IFEval strict accuracy by model and method (\\%, "
        "$\\pm$ half-width of 95\\% CI). Best per (model, level) in bold.}"
    )
    lines.append("\\label{tab:ifeval_grouped}")
    lines.append("\\end{table}")
    lines.append("")

    # Layout 2: two lines per cell (top: sample, bottom: instruction)
    lines.append("% --- Layout 2: two lines per cell ---")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * 4 + "}")
    lines.append("\\toprule")
    lines.append("Model & prompt & pasta & spotlight & instaboost \\\\")
    lines.append("\\midrule")
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            top = fmt_tex(sample[method], bold=method in sample_best)
            bot = fmt_tex(inst[method], bold=method in inst_best)
            if sample[method] is None and inst[method] is None:
                cells.append("--")
            else:
                cells.append(f"\\makecell{{{top} \\\\ {bot}}}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{IFEval strict accuracy. Each cell shows sample-level "
        "(top) and instruction-level (bottom) accuracy in \\%. Best per "
        "(model, level) in bold.}"
    )
    lines.append("\\label{tab:ifeval_twoline}")
    lines.append("\\end{table}")
    lines.append("")

    # Layout 3: slash per cell
    lines.append("% --- Layout 3: sample / instruction in one cell ---")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * 4 + "}")
    lines.append("\\toprule")
    lines.append("Model & prompt & pasta & spotlight & instaboost \\\\")
    lines.append("\\midrule")
    for m in models:
        sample = data[m]["sample"]
        inst = data[m]["instruction"]
        sample_best = best_methods(sample)
        inst_best = best_methods(inst)
        cells = [m]
        for method in METHODS:
            top = fmt_tex(sample[method], bold=method in sample_best)
            bot = fmt_tex(inst[method], bold=method in inst_best)
            if sample[method] is None and inst[method] is None:
                cells.append("--")
            else:
                cells.append(f"{top} / {bot}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{IFEval strict accuracy. Each cell: "
        "sample-level / instruction-level (\\%). Best per (model, level) in bold.}"
    )
    lines.append("\\label{tab:ifeval_slash}")
    lines.append("\\end{table}")
    lines.append("")

    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


def build_figure(data: dict) -> None:
    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(14, 6))

    n_models = len(models)
    n_methods = len(METHODS)
    n_levels = len(LEVELS)

    group_width = 0.86
    method_width = group_width / n_methods
    bar_width = method_width / n_levels
    x = np.arange(n_models)

    for mi, method in enumerate(METHODS):
        method_offset = (mi - (n_methods - 1) / 2) * method_width
        for li, level in enumerate(LEVELS):
            level_offset = (li - (n_levels - 1) / 2) * bar_width
            positions = x + method_offset + level_offset

            accs, errs, present = [], [], []
            for m in models:
                entry = data[m][level].get(method)
                if entry is None:
                    accs.append(0.0); errs.append(0.0); present.append(False)
                else:
                    accs.append(entry[0]); errs.append(entry[1]); present.append(True)
            accs_a = np.array(accs); errs_a = np.array(errs); mask = np.array(present)

            ax.bar(
                positions[mask],
                accs_a[mask],
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                hatch=LEVEL_HATCH[level],
                yerr=errs_a[mask],
                capsize=2,
                error_kw={"elinewidth": 0.7, "ecolor": "black"},
            )
            for pos, acc, err, ok in zip(positions, accs, errs, present):
                if ok:
                    ax.text(
                        pos, acc + err + 1.0, f"{acc:.1f}",
                        ha="center", va="bottom", fontsize=6,
                    )
                else:
                    ax.text(pos, 0.5, "—", ha="center", va="bottom", fontsize=7, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("IFEval strict accuracy (%)")
    ax.set_title("IFEval strict accuracy by model and method (95% CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    method_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=m)
        for m in METHODS
    ]
    level_handles = [
        mpatches.Patch(
            facecolor="white", edgecolor="black", hatch=LEVEL_HATCH[l],
            label=LEVEL_LABEL[l],
        )
        for l in LEVELS
    ]
    leg1 = ax.legend(
        handles=method_handles, title="method",
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=level_handles, title="level",
        loc="upper left", bbox_to_anchor=(1.01, 0.55), frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


def main():
    data = load_data()
    if not data:
        raise SystemExit(f"No model dirs found under {RESULTS_ROOT}")
    write_markdown(data)
    write_latex(data)
    build_figure(data)


if __name__ == "__main__":
    main()
