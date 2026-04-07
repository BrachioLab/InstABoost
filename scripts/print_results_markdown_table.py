#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PREFERRED_METHOD_ORDER = [
    "baseline",
    "prompt",
    "prompt-attention",
    "refusal",
    "random",
    "mean",
    "pca",
    "repe",
    "linear",
]

PREFERRED_DATASET_ORDER = [
    "emotions:anger",
    "emotions:disgust",
    "emotions:fear",
    "emotions:joy",
    "emotions:sadness",
    "emotions:surprise",
    "ai-risk-mcq:power",
    "ai-risk-mcq:wealth",
    "ai-risk-qa:power",
    "ai-risk-qa:wealth",
    "triviaqa",
    "truthfulqa",
    "safety",
    "jbb",
    "toxicity",
]


def normalize_model_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def dataset_label_from_path(path: Path) -> str | None:
    parts = path.parts
    if "smoke" in parts:
        return None

    try:
        idx = parts.index("results")
    except ValueError:
        return None

    rel = parts[idx + 1 :]
    if len(rel) < 3:
        return None

    if rel[0] == "emotions" and len(rel) >= 3:
        return f"emotions:{rel[1]}"
    if rel[0] in {"ai-risk-mcq", "ai-risk-qa"} and len(rel) >= 3:
        return f"{rel[0]}:{rel[1]}"
    if rel[0] in {"triviaqa", "truthfulqa", "safety", "jbb", "toxicity"}:
        return rel[0]
    return None


def find_result_files(results_root: Path, model_short: str) -> list[Path]:
    return sorted(
        path
        for path in results_root.glob(f"**/{model_short}/all_methods_results.json")
        if "smoke" not in path.parts
    )


def format_metric(method_results: dict, metric_key: str, ci_key: str) -> str:
    if metric_key not in method_results or method_results[metric_key] is None:
        return "-"

    value = float(method_results[metric_key])
    ci = method_results.get(ci_key)
    if not ci:
        return f"{value:.3f}"

    lower = float(ci["lower"])
    upper = float(ci["upper"])
    upper_delta = upper - value
    lower_delta = value - lower
    error = (upper_delta + lower_delta) / 2.0
    return f"{value:.3f} +- {error:.3f}"


def ordered_methods(methods: set[str]) -> list[str]:
    preferred = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    extras = sorted(method for method in methods if method not in PREFERRED_METHOD_ORDER)
    return preferred + extras


def ordered_datasets(datasets: set[str]) -> list[str]:
    preferred = [dataset for dataset in PREFERRED_DATASET_ORDER if dataset in datasets]
    extras = sorted(dataset for dataset in datasets if dataset not in PREFERRED_DATASET_ORDER)
    return preferred + extras


def render_table(title: str, methods: list[str], datasets: list[str], results_by_dataset: dict, metric_key: str, ci_key: str) -> str:
    lines = [f"## {title}", ""]
    header = ["Dataset"] + methods
    rows = []
    for dataset in datasets:
        row = [dataset]
        for method in methods:
            method_results = results_by_dataset.get(dataset, {}).get(method)
            row.append(format_metric(method_results, metric_key, ci_key) if method_results else "-")
        rows.append(row)

    widths = [
        max(len(str(cell)) for cell in [header[col_idx]] + [row[col_idx] for row in rows])
        for col_idx in range(len(header))
    ]

    def format_row(row: list[str]) -> str:
        padded = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(padded) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"

    lines.append(format_row(header))
    lines.append(divider)
    for row in rows:
        lines.append(format_row(row))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print markdown tables for current experiment results.")
    parser.add_argument("model", help="Model name or short name, e.g. openai/gpt-oss-20b or gpt-oss-20b")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    args = parser.parse_args()

    model_short = normalize_model_name(args.model)
    results_root = Path(args.results_root)
    files = find_result_files(results_root, model_short)

    if not files:
        raise SystemExit(f"No all_methods_results.json files found for model {model_short!r} under {results_root}")

    results_by_dataset: dict[str, dict] = {}
    all_methods: set[str] = set()

    for path in files:
        dataset_label = dataset_label_from_path(path)
        if dataset_label is None:
            continue
        with path.open() as f:
            payload = json.load(f)
        methods = payload.get("methods", {})
        if not isinstance(methods, dict):
            continue
        results_by_dataset[dataset_label] = methods
        all_methods.update(methods.keys())

    datasets = ordered_datasets(set(results_by_dataset.keys()) | set(PREFERRED_DATASET_ORDER))
    methods = ordered_methods(all_methods)

    if not datasets:
        raise SystemExit(f"No non-smoke dataset results found for model {model_short!r}")

    print(f"# Results for `{model_short}`")
    print()
    print(render_table("Steering Success Score", methods, datasets, results_by_dataset, "score", "score_ci"))
    print(render_table("Fluency", methods, datasets, results_by_dataset, "fluency", "fluency_ci"))


if __name__ == "__main__":
    main()
