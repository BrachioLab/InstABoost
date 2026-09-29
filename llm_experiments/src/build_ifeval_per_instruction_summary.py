#!/usr/bin/env python
"""Per-instruction-type accuracy summary for IFEval.

Walks each results/ifeval/<model>/<method>/results_by_sample.json and emits
TWO views of per-instruction pass rates:

  Coarse (category): groups instruction_ids by the prefix before ":"
    e.g. `keywords:letter_frequency` and `keywords:forbidden_words` both
    count under `keywords`. ~9 categories.
      results/ifeval_per_category.md  — table per model
      results/ifeval_per_category.png/.pdf  — grouped bar chart (2x2 grid)

  Fine (instruction_id): full ids preserved (~30 unique).
      results/ifeval_per_instruction_id.md  — table per model
      results/ifeval_per_instruction_id.png/.pdf — 1x4 heatmap row

Numbers are mean of ``per_instruction_eval[i]["score"]`` over all instances
of that category / instruction_id.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "ifeval"
OUT_CAT_MD = REPO_ROOT / "results" / "ifeval_per_category.md"
OUT_CAT_PNG = REPO_ROOT / "results" / "ifeval_per_category.png"
OUT_CAT_PDF = REPO_ROOT / "results" / "ifeval_per_category.pdf"
OUT_ID_MD = REPO_ROOT / "results" / "ifeval_per_instruction_id.md"
OUT_ID_PNG = REPO_ROOT / "results" / "ifeval_per_instruction_id.png"
OUT_ID_PDF = REPO_ROOT / "results" / "ifeval_per_instruction_id.pdf"

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


def category_of(instruction_id: str) -> str:
    """`keywords:letter_frequency` -> `keywords`."""
    return instruction_id.split(":", 1)[0]


def load_grouped(model_slug: str, method: str, key_fn) -> dict[str, list[float]] | None:
    """Returns {key: [score, ...]} grouped by key_fn(instruction_id)."""
    path = RESULTS_ROOT / model_slug / method / "results_by_sample.json"
    if not path.exists():
        return None
    samples = json.loads(path.read_text())
    grouped: dict[str, list[float]] = defaultdict(list)
    for entry in samples.values():
        for instr in entry.get("per_instruction_eval", []):
            grouped[key_fn(instr["instruction_id"])].append(float(instr["score"]))
    return grouped


def collect(key_fn) -> tuple[dict, list[str]]:
    """Returns (data, sorted_keys).

    data[model_display][method][key] = (mean_pct, n_count) or None.
    """
    data: dict = {}
    all_keys: set[str] = set()
    for slug, display in MODEL_DISPLAY.items():
        data[display] = {}
        for method in METHODS:
            grouped = load_grouped(slug, method, key_fn)
            if grouped is None:
                data[display][method] = None
                continue
            data[display][method] = {
                k: (float(np.mean(scores)) * 100, len(scores))
                for k, scores in grouped.items()
            }
            all_keys.update(grouped.keys())
    return data, sorted(all_keys)


def fmt_cell(entry):
    if entry is None:
        return "—"
    pct, n = entry
    return f"{pct:.1f} (n={n})"


def best_method(per_method: dict[str, dict | None], key: str) -> set[str]:
    best = -1.0
    winners = set()
    for method in METHODS:
        cell = per_method[method]
        if cell is None or key not in cell:
            continue
        pct, _ = cell[key]
        if pct > best:
            best = pct
            winners = {method}
        elif pct == best:
            winners.add(method)
    return winners


def write_markdown(data: dict, keys: list[str], out_path: Path,
                   title: str, key_label: str, blurb: str) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(blurb)
    lines.append("")
    for model in data:
        per_method = data[model]
        lines.append(f"## {model}")
        lines.append("")
        header = f"| {key_label} | " + " | ".join(METHODS) + " |"
        sep = "|---|" + ":---:|" * len(METHODS)
        lines.append(header)
        lines.append(sep)
        for k in keys:
            winners = best_method(per_method, k)
            row = [k]
            for method in METHODS:
                cell = per_method[method]
                if cell is None or k not in cell:
                    row.append("—")
                    continue
                rendered = fmt_cell(cell[k])
                if method in winners:
                    rendered = f"**{rendered}**"
                row.append(rendered)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def build_category_figure(data: dict, categories: list[str]) -> None:
    n_cats = len(categories)
    n_methods = len(METHODS)
    bar_width = 0.8 / n_methods
    x = np.arange(n_cats)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    for ax, (model, per_method) in zip(axes.flat, data.items()):
        for mi, method in enumerate(METHODS):
            cell = per_method[method]
            offset = (mi - (n_methods - 1) / 2) * bar_width
            if cell is None:
                continue
            heights = []
            mask = []
            for cat in categories:
                if cat in cell:
                    heights.append(cell[cat][0])
                    mask.append(True)
                else:
                    heights.append(0)
                    mask.append(False)
            heights_a = np.array(heights)
            mask_a = np.array(mask)
            positions = x + offset
            ax.bar(
                positions[mask_a],
                heights_a[mask_a],
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.4,
                label=method,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(model)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Per-instruction pass rate (%)")
    axes[1, 0].set_ylabel("Per-instruction pass rate (%)")

    handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=m)
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        ncol=len(METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("IFEval per-instruction-category pass rate", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(OUT_CAT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_CAT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_CAT_PNG}")
    print(f"Wrote {OUT_CAT_PDF}")


def build_instruction_id_figure(data: dict, instruction_ids: list[str]) -> None:
    """One heatmap per model. Rows = instruction_ids, cols = methods, cell = pass rate."""
    n_ids = len(instruction_ids)
    n_methods = len(METHODS)
    fig, axes = plt.subplots(
        1, len(data), figsize=(4 * len(data), max(6, 0.32 * n_ids)), sharey=True
    )
    if len(data) == 1:
        axes = [axes]

    for ax, (model, per_method) in zip(axes, data.items()):
        grid = np.full((n_ids, n_methods), np.nan)
        for ri, key in enumerate(instruction_ids):
            for ci, method in enumerate(METHODS):
                cell = per_method[method]
                if cell is None or key not in cell:
                    continue
                grid[ri, ci] = cell[key][0]
        im = ax.imshow(grid, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        ax.set_title(model, fontsize=10)
        # Annotate
        for ri in range(n_ids):
            for ci in range(n_methods):
                v = grid[ri, ci]
                if np.isnan(v):
                    ax.text(ci, ri, "—", ha="center", va="center", color="gray", fontsize=6)
                else:
                    color = "white" if v < 35 or v > 80 else "black"
                    ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                            color=color, fontsize=6)

    axes[0].set_yticks(range(n_ids))
    axes[0].set_yticklabels(instruction_ids, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Pass rate (%)")
    fig.suptitle("IFEval per-instruction-id pass rate", fontsize=13)
    fig.savefig(OUT_ID_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_ID_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_ID_PNG}")
    print(f"Wrote {OUT_ID_PDF}")


def main() -> None:
    cat_data, categories = collect(category_of)
    id_data, instruction_ids = collect(lambda x: x)
    if not categories:
        raise SystemExit(f"No per_instruction_eval entries found under {RESULTS_ROOT}")

    write_markdown(
        cat_data, categories, OUT_CAT_MD,
        title="IFEval per-instruction-category pass rate",
        key_label="category",
        blurb=(
            "Each cell shows the mean pass rate (\\%) on individual instruction "
            "instances of a given category, with `n` = total instructions of that "
            "category in the test set. Bold = best method per (model, category).\n\n"
            "Categories come from splitting `instruction_id` on `:` — e.g. "
            "`keywords:letter_frequency` and `keywords:forbidden_words` both count "
            "under `keywords`."
        ),
    )
    write_markdown(
        id_data, instruction_ids, OUT_ID_MD,
        title="IFEval per-instruction-id pass rate (full granularity)",
        key_label="instruction_id",
        blurb=(
            "Each cell shows the mean pass rate (\\%) on individual instruction "
            "instances with this exact `instruction_id`, with `n` = total "
            "instructions of that id in the test set. Bold = best method per "
            "(model, instruction_id)."
        ),
    )
    build_category_figure(cat_data, categories)
    build_instruction_id_figure(id_data, instruction_ids)


if __name__ == "__main__":
    main()
