#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PREFERRED_METHOD_ORDER = [
    "baseline",
    "prompt",
    "prompt-attention",
    "spotlight",
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
    return f"{value:.3f} ± {error:.3f}"


def format_delta_percentage_points(
    results_by_dataset: dict[str, dict], dataset: str, improved_method: str, reference_method: str, metric_key: str
) -> str:
    improved = results_by_dataset.get(dataset, {}).get(improved_method)
    reference = results_by_dataset.get(dataset, {}).get(reference_method)
    if not improved or not reference:
        return "-"

    improved_value = improved.get(metric_key)
    reference_value = reference.get(metric_key)
    if improved_value is None or reference_value is None:
        return "-"

    delta_pp = (float(improved_value) - float(reference_value)) * 100.0
    return f"{delta_pp:+.0f}%"


def ordered_methods(methods: set[str]) -> list[str]:
    preferred = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    extras = sorted(method for method in methods if method not in PREFERRED_METHOD_ORDER)
    return preferred + extras


def ordered_datasets(datasets: set[str]) -> list[str]:
    preferred = [dataset for dataset in PREFERRED_DATASET_ORDER if dataset in datasets]
    extras = sorted(dataset for dataset in datasets if dataset not in PREFERRED_DATASET_ORDER)
    return preferred + extras


def average_improvement_percentage_points(
    results_by_dataset: dict[str, dict], improved_method: str, reference_method: str
) -> tuple[float | None, int]:
    deltas: list[float] = []
    for methods in results_by_dataset.values():
        improved = methods.get(improved_method)
        reference = methods.get(reference_method)
        if not improved or not reference:
            continue
        improved_score = improved.get("score")
        reference_score = reference.get("score")
        if improved_score is None or reference_score is None:
            continue
        deltas.append((float(improved_score) - float(reference_score)) * 100.0)

    if not deltas:
        return None, 0
    return sum(deltas) / len(deltas), len(deltas)


def render_steering_success_summary(
    results_by_dataset: dict[str, dict], instaboost_method: str, baseline_method: str, prompt_method: str
) -> str:
    lines = ["## Steering Success Summary", ""]

    avg_vs_baseline, count_vs_baseline = average_improvement_percentage_points(
        results_by_dataset, instaboost_method, baseline_method
    )
    avg_vs_prompt, count_vs_prompt = average_improvement_percentage_points(
        results_by_dataset, instaboost_method, prompt_method
    )

    lines.append(f"- InstABoost method key: `{instaboost_method}`")
    if avg_vs_baseline is None:
        lines.append(f"- Avg increase vs `{baseline_method}`: unavailable (no overlapping scores)")
    else:
        lines.append(
            f"- Avg increase vs `{baseline_method}`: {avg_vs_baseline:+.2f} percentage points "
            f"(across {count_vs_baseline} datasets)"
        )

    if avg_vs_prompt is None:
        lines.append(f"- Avg increase vs `{prompt_method}`: unavailable (no overlapping scores)")
    else:
        lines.append(
            f"- Avg increase vs `{prompt_method}`: {avg_vs_prompt:+.2f} percentage points "
            f"(across {count_vs_prompt} datasets)"
        )

    lines.append("")
    return "\n".join(lines)


def render_table(
    title: str,
    methods: list[str],
    datasets: list[str],
    results_by_dataset: dict,
    metric_key: str,
    ci_key: str,
    include_deltas: bool = False,
    instaboost_method: str = "prompt-attention",
    baseline_method: str = "baseline",
    prompt_method: str = "prompt",
) -> str:
    lines = [f"## {title}", ""]
    header = ["Dataset"] + methods
    if include_deltas:
        header.extend(
            [
                f"delta({baseline_method})",
                f"delta({prompt_method})",
            ]
        )
    rows = []
    for dataset in datasets:
        row = [dataset]
        for method in methods:
            method_results = results_by_dataset.get(dataset, {}).get(method)
            row.append(format_metric(method_results, metric_key, ci_key) if method_results else "-")
        if include_deltas:
            row.append(
                format_delta_percentage_points(
                    results_by_dataset, dataset, instaboost_method, baseline_method, metric_key
                )
            )
            row.append(
                format_delta_percentage_points(
                    results_by_dataset, dataset, instaboost_method, prompt_method, metric_key
                )
            )
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
    parser.add_argument(
        "--instaboost-method",
        default="prompt-attention",
        help="Method key to treat as InstABoost for summary statistics",
    )
    parser.add_argument("--baseline-method", default="baseline", help="Baseline method key for summary statistics")
    parser.add_argument("--prompt-method", default="prompt", help="Prompting method key for summary statistics")
    parser.add_argument(
        "--include_deltas",
        action="store_true",
        help="Include per-dataset delta columns (InstABoost-baseline and InstABoost-prompt) in Steering Success table",
    )
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
    print(
        render_table(
            "Steering Success Score",
            methods,
            datasets,
            results_by_dataset,
            "score",
            "score_ci",
            include_deltas=args.include_deltas,
            instaboost_method=args.instaboost_method,
            baseline_method=args.baseline_method,
            prompt_method=args.prompt_method,
        )
    )
    print(
        render_steering_success_summary(
            results_by_dataset,
            instaboost_method=args.instaboost_method,
            baseline_method=args.baseline_method,
            prompt_method=args.prompt_method,
        )
    )
    print(render_table("Fluency", methods, datasets, results_by_dataset, "fluency", "fluency_ci"))


if __name__ == "__main__":
    main()
