from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy import stats

from ifeval_data_utils import clean_kwargs, get_instructions_registry


def bootstrap_confidence_interval(
    data: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    data_array = np.array(data)
    bootstrap_result = stats.bootstrap(
        (data_array,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        method="percentile",
    )
    return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high


def evaluate_sample(
    generation: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> dict[str, Any]:
    instructions_registry = get_instructions_registry()
    per_instruction: list[dict[str, Any]] = []
    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        checker_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if checker_cls is None:
            per_instruction.append(
                {
                    "instruction_id": instruction_id,
                    "passed": False,
                    "score": 0,
                    "note": "unknown instruction_id",
                }
            )
            continue

        checker = checker_cls(instruction_id)
        cleaned_kwargs = clean_kwargs(kwargs)
        try:
            checker.build_description(**cleaned_kwargs)
        except TypeError:
            checker.build_description()

        passed = bool(checker.check_following(generation))
        per_instruction.append(
            {
                "instruction_id": instruction_id,
                "passed": passed,
                "score": 1 if passed else 0,
            }
        )

    all_passed = all(item["passed"] for item in per_instruction) if per_instruction else False
    instruction_level_score = (
        sum(item["score"] for item in per_instruction) / len(per_instruction)
        if per_instruction
        else 0.0
    )

    return {
        "per_instruction": per_instruction,
        "sample_score": 1 if all_passed else 0,
        "instruction_level_score": instruction_level_score,
    }


def evaluate_generations(
    generations: list[str],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(generations) != len(samples):
        raise ValueError(
            f"Generation/sample length mismatch: got {len(generations)} generations "
            f"for {len(samples)} samples."
        )
    return [
        evaluate_sample(generation, sample["instruction_id_list"], sample["kwargs_list"])
        for generation, sample in zip(generations, samples)
    ]


def compute_aggregate_metrics(
    eval_results: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    sample_scores = [result["sample_score"] for result in eval_results]
    instruction_scores = [
        instruction_result["score"]
        for result in eval_results
        for instruction_result in result["per_instruction"]
    ]

    prompt_acc = float(np.mean(sample_scores)) if sample_scores else 0.0
    instruction_acc = float(np.mean(instruction_scores)) if instruction_scores else 0.0

    prompt_ci = (
        bootstrap_confidence_interval(sample_scores, n_bootstrap, confidence_level)
        if len(sample_scores) >= 2
        else (0.0, 0.0)
    )
    instruction_ci = (
        bootstrap_confidence_interval(instruction_scores, n_bootstrap, confidence_level)
        if len(instruction_scores) >= 2
        else (0.0, 0.0)
    )

    return {
        "prompt_level_strict_acc": prompt_acc,
        "prompt_level_ci": {
            "lower": prompt_ci[0],
            "upper": prompt_ci[1],
            "confidence_level": confidence_level,
        },
        "instruction_level_strict_acc": instruction_acc,
        "instruction_level_ci": {
            "lower": instruction_ci[0],
            "upper": instruction_ci[1],
            "confidence_level": confidence_level,
        },
        "n_samples": len(sample_scores),
        "n_instructions": len(instruction_scores),
    }


def save_method_outputs(
    method_output_dir: str,
    method_name: str,
    samples: list[dict[str, Any]],
    input_texts: list[str],
    generations: list[str],
    eval_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    timing: dict[str, Any],
    best_param: Any | None = None,
    per_sample_timing: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    os.makedirs(method_output_dir, exist_ok=True)

    by_sample: dict[str, dict[str, Any]] = {}
    for idx, (sample, input_text, generation, eval_result) in enumerate(
        zip(samples, input_texts, generations, eval_results)
    ):
        entry: dict[str, Any] = {
            "key": sample["key"],
            "instruction_id_list": sample["instruction_id_list"],
            "instruction_texts": sample["instruction_texts"],
            "instruction_block": sample["instruction_block"],
            "base_question": sample["base_question"],
            "input": input_text,
            "generation": generation,
            "per_instruction_eval": eval_result["per_instruction"],
            "sample_score": eval_result["sample_score"],
            "instruction_level_score": eval_result["instruction_level_score"],
        }
        if per_sample_timing is not None:
            entry.update(per_sample_timing[idx])
        by_sample[str(idx)] = entry

    with open(os.path.join(method_output_dir, "results_by_sample.json"), "w") as handle:
        json.dump(by_sample, handle, indent=2)

    method_payload = {
        "method": method_name,
        "best_param": best_param,
        **aggregate,
        "timing": timing,
    }
    with open(os.path.join(method_output_dir, "results.json"), "w") as handle:
        json.dump(method_payload, handle, indent=2)

    return method_payload
