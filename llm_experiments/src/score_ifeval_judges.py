"""Post-hoc fluency + relevance scoring for cached IFEval samples.

Walks ``results/ifeval/<model>/<method>/results_by_sample.json`` files; for each
sample, calls prompts.judge_fluency(generation) and prompts.judge_relevance(
base_question, generation). Adds "fluency" and "relevance" integer scores
(0-2) to each sample dict in place. Resumable — skips samples that already
have both scores.

After processing each (model, method), writes aggregate means to
``results/ifeval_judge_scores.json`` so you don't have to re-walk the
per-sample files later.

Requires GOOGLE_API_KEY (the judges call Gemini 2.0 Flash).

Usage:
    python src/score_ifeval_judges.py                              # all models / methods
    python src/score_ifeval_judges.py --models Qwen3-14B           # subset
    python src/score_ifeval_judges.py --methods prompt pasta       # subset
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from prompts import judge_fluency, judge_relevance


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "ifeval"
SUMMARY_PATH = REPO_ROOT / "results" / "ifeval_judge_scores.json"

DEFAULT_MODELS = ["Meta-Llama-3-8B-Instruct", "gemma-7b-it", "Qwen3-14B", "gpt-oss-20b"]
DEFAULT_METHODS = ["prompt", "pasta", "spotlight", "instaboost"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"Model dirs to process (default: {DEFAULT_MODELS}).")
    ap.add_argument("--methods", nargs="+", default=None,
                    help=f"Method dirs to process (default: {DEFAULT_METHODS}).")
    ap.add_argument("--save_every", type=int, default=10,
                    help="Persist per-sample JSON every N newly-judged samples (default 10).")
    return ap.parse_args()


def aggregate(samples: dict) -> dict:
    fluency = [s["fluency"] for s in samples.values() if "fluency" in s]
    relevance = [s["relevance"] for s in samples.values() if "relevance" in s]
    return {
        "n_total": len(samples),
        "n_fluency": len(fluency),
        "n_relevance": len(relevance),
        "fluency_mean": float(np.mean(fluency)) if fluency else None,
        "relevance_mean": float(np.mean(relevance)) if relevance else None,
        "fluency_std": float(np.std(fluency, ddof=1)) if len(fluency) > 1 else None,
        "relevance_std": float(np.std(relevance, ddof=1)) if len(relevance) > 1 else None,
    }


def score_one_method(model: str, method: str, save_every: int) -> dict | None:
    base = RESULTS_ROOT / model / method
    sample_path = base / "results_by_sample.json"
    if not sample_path.exists():
        print(f"  [{model}/{method}] no results_by_sample.json — skip")
        return None

    samples = json.loads(sample_path.read_text())
    keys = list(samples.keys())
    n_total = len(keys)
    needs_work = sum(
        1 for k in keys if "fluency" not in samples[k] or "relevance" not in samples[k]
    )
    print(f"  [{model}/{method}] {needs_work}/{n_total} samples need judging")
    if needs_work == 0:
        return aggregate(samples)

    n_processed = 0
    for k in keys:
        entry = samples[k]
        question = entry.get("base_question", "") or entry.get("input", "")
        generation = entry.get("generation", "")

        if "fluency" not in entry:
            entry["fluency"] = judge_fluency(generation)
        if "relevance" not in entry:
            entry["relevance"] = judge_relevance(question, generation)

        n_processed += 1
        if n_processed % save_every == 0:
            sample_path.write_text(json.dumps(samples, indent=2))
            print(f"    [{model}/{method}] {n_processed}/{needs_work} judged...")

    sample_path.write_text(json.dumps(samples, indent=2))
    return aggregate(samples)


def main():
    args = parse_args()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY env var is required (Gemini 2.0 Flash).")

    models = args.models or DEFAULT_MODELS
    methods = args.methods or DEFAULT_METHODS

    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())

    for model in models:
        summary.setdefault(model, {})
        for method in methods:
            print(f"\n>> {model} / {method}")
            agg = score_one_method(model, method, args.save_every)
            if agg is None:
                continue
            summary[model][method] = agg
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
            if agg["fluency_mean"] is not None and agg["relevance_mean"] is not None:
                print(
                    f"  fluency_mean={agg['fluency_mean']:.3f}  "
                    f"relevance_mean={agg['relevance_mean']:.3f}  "
                    f"(n={agg['n_total']})"
                )

    print(f"\nWrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
