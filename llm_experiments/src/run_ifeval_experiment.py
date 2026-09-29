from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import random
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from env_utils import load_repo_env
from gpt_oss_loading import load_gpt_oss_model
from ifeval_data_utils import get_instruction_prompt_spans, load_ifeval_samples
from ifeval_eval_utils import compute_aggregate_metrics, evaluate_generations, save_method_outputs
from model_interaction_utils import get_generations, tokenize_instructions


INSTABOOST_MULTIPLIERS = list(range(3, 20, 2))
SPOTLIGHT_TARGET_ATTENTIONS = [0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_METHODS = ["prompt", "instaboost", "spotlight", "pasta"]


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _cached_method_summary(method_dir: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    results_path = os.path.join(method_dir, "results.json")
    if not os.path.exists(results_path):
        return None

    payload = _load_json(results_path)
    summary = {key: value for key, value in payload.items() if key not in ("method", "best_param")}
    timing = payload.get("timing", {})
    return summary, timing


def _ensure_nltk_resources() -> None:
    try:
        import nltk
    except ImportError:
        return

    required_resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    missing_packages: list[str] = []
    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing_packages.append(package_name)

    for package_name in missing_packages:
        print(f"Downloading missing NLTK resource: {package_name}")
        try:
            nltk.download(package_name, quiet=True)
        except Exception as error:
            print(f"Warning: failed to download NLTK resource {package_name}: {error}")

    still_missing: list[str] = []
    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            still_missing.append(package_name)

    if still_missing:
        raise LookupError(
            "Missing required NLTK resources after download attempt: "
            f"{', '.join(still_missing)}. "
            "Please run: python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\""
        )


def _render_chat_prompt(
    tokenizer,
    user_content: str,
    reasoning_effort: str | None,
    enable_thinking: bool | None,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return user_content

    messages = [{"role": "user", "content": user_content}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizers do not support reasoning_effort/enable_thinking kwargs.
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _timing_payload(tokenizer, generations: list[str], started_at: float) -> dict[str, Any]:
    total_time = max(time.perf_counter() - started_at, 1e-8)
    total_output_tokens = sum(
        len(tokenizer.encode(generation, add_special_tokens=False))
        for generation in generations
    )
    return {
        "total_time_sec": round(total_time, 2),
        "total_output_tokens": total_output_tokens,
        "tokens_per_sec": round(total_output_tokens / total_time, 4),
    }


def _print_method_results(method_name: str, aggregate: dict[str, Any], timing: dict[str, Any]) -> None:
    prompt_ci = aggregate["prompt_level_ci"]
    instruction_ci = aggregate["instruction_level_ci"]
    print(
        f"  {method_name} prompt strict acc: {aggregate['prompt_level_strict_acc']:.3f} "
        f"(95% CI [{prompt_ci['lower']:.3f}, {prompt_ci['upper']:.3f}])"
    )
    print(
        f"  {method_name} instruction strict acc: {aggregate['instruction_level_strict_acc']:.3f} "
        f"(95% CI [{instruction_ci['lower']:.3f}, {instruction_ci['upper']:.3f}])"
    )
    print(
        f"  {method_name} tokens/sec: {timing['tokens_per_sec']:.2f} | "
        f"total time: {timing['total_time_sec']:.1f}s | "
        f"output tokens: {timing['total_output_tokens']}"
    )


def _load_hooked_model(args) -> HookedTransformer:
    trust_remote_code = (
        args.trust_remote_code
        or args.model_path.startswith("Qwen/Qwen3-")
        or args.model_path.startswith("openai/gpt-oss-")
    )
    if trust_remote_code:
        print(f"Loading {args.model_path} with trust_remote_code=True")

    if args.model_path == "openai/gpt-oss-20b":
        model = load_gpt_oss_model(device=args.device, n_layers=args.model_n_layers or 24)
    else:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_path,
            device=args.device,
            dtype=torch.bfloat16,
            default_padding_side="left",
            trust_remote_code=trust_remote_code,
        )

    if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model


def _hooked_generations(
    model: HookedTransformer,
    tokenize_instructions_fn,
    input_texts: list[str],
    max_tokens_generated: int,
    batch_size: int,
    prompt_spans: list[tuple[int, int, int]] | None = None,
    attention_pattern_multiplier: float | None = None,
    spotlight_target_attention: float | None = None,
    instaboost_hook_mode: str = "auto",
    instaboost_kv_cache: str = "auto",
) -> tuple[list[str], dict[str, Any]]:
    # IFEval instaboost uses the legacy slice + F.normalize hook AND disables
    # the KV cache, so end-to-end behaviour bit-matches the original
    # ifeval_instr_last reference implementation on non-sink models.
    #
    # Sink-based architectures (currently GPT-OSS) must keep the mass-preserving
    # ``prompt_attention_ablation_hook``. The legacy hook re-normalises each row
    # to sum=1.0 via F.normalize, which would erase the attention mass the sink
    # absorbed before pattern was sliced — see transformer_lens_compat.py:96-107.
    #
    # The ``instaboost_hook_mode`` and ``instaboost_kv_cache`` args allow
    # overriding both decisions, e.g. for ablation runs.
    is_instaboost = attention_pattern_multiplier is not None
    has_attention_sinks = (
        getattr(model.cfg, "original_architecture", "") == "GptOssForCausalLM"
    )
    if instaboost_hook_mode == "auto":
        use_legacy_hook = is_instaboost and not has_attention_sinks
    elif instaboost_hook_mode == "legacy":
        use_legacy_hook = is_instaboost
    elif instaboost_hook_mode == "mass_preserving":
        use_legacy_hook = False
    else:
        raise ValueError(f"Unknown instaboost_hook_mode: {instaboost_hook_mode!r}")

    if instaboost_kv_cache == "auto":
        disable_kv = is_instaboost
    elif instaboost_kv_cache == "off":
        disable_kv = is_instaboost
    elif instaboost_kv_cache == "on":
        disable_kv = False
    else:
        raise ValueError(f"Unknown instaboost_kv_cache: {instaboost_kv_cache!r}")

    started_at = time.perf_counter()
    generations = get_generations(
        model,
        input_texts,
        tokenize_instructions_fn,
        fwd_hooks=[],
        max_tokens_generated=max_tokens_generated,
        batch_size=batch_size,
        prompt_spans=prompt_spans,
        attention_pattern_multiplier=attention_pattern_multiplier,
        spotlight_target_attention=spotlight_target_attention,
        instaboost_legacy_renorm=use_legacy_hook,
        disable_kv_cache=disable_kv,
    )
    timing = _timing_payload(model.tokenizer, generations, started_at)
    return generations, timing


def _run_pasta_method(
    args,
    samples: list[dict[str, Any]],
    pasta_head_config: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    if not str(args.device).startswith("cuda"):
        raise ValueError("PASTA currently requires a CUDA device in this script.")

    from pastalib.pasta import PASTA

    trust_remote_code = (
        args.trust_remote_code
        or args.model_path.startswith("Qwen/Qwen3-")
        or args.model_path.startswith("openai/gpt-oss-")
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Dispatch: vanilla pastalib doesn't recognize Qwen3ForCausalLM, so fall
    # back to the patched subclass defined in autopasta_head_selection_ifeval.
    PastaCls = PASTA
    if hf_model.__class__.__name__ == "Qwen3ForCausalLM":
        from autopasta_head_selection_ifeval import Qwen3PASTA
        PastaCls = Qwen3PASTA

    pasta = PastaCls(
        model=hf_model,
        tokenizer=tokenizer,
        head_config=pasta_head_config,
        alpha=args.pasta_alpha,
        scale_position="exclude",
    )
    pasta_device = next(
        (parameter.device for parameter in hf_model.parameters() if parameter.device.type != "meta"),
        torch.device("cuda"),
    )

    input_texts = [sample["full_input"] for sample in samples]
    rendered_inputs = [
        _render_chat_prompt(
            tokenizer=tokenizer,
            user_content=sample["full_input"],
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        for sample in samples
    ]
    emphasized_substrings = [sample["instruction_block"] for sample in samples]

    generations: list[str] = []
    started_at = time.perf_counter()
    for index in range(0, len(samples), args.batch_size):
        batch_texts = rendered_inputs[index : index + args.batch_size]
        batch_substrings = emphasized_substrings[index : index + args.batch_size]
        inputs, offset_mapping = pasta.inputs_from_batch(batch_texts)
        inputs = inputs.to(pasta_device)

        with pasta.apply_steering(
            model=hf_model,
            strings=batch_texts,
            substrings=batch_substrings,
            model_input=inputs,
            offsets_mapping=offset_mapping,
        ) as steered_model:
            outputs = steered_model.generate(**inputs, max_new_tokens=args.max_tokens_generated)

        outputs = outputs[:, inputs.input_ids.shape[1] :]
        generations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    timing = _timing_payload(tokenizer, generations, started_at)

    del hf_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return input_texts, generations, timing


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run IFEval benchmark (instruction-last format) using current InstABoost method implementations."
    )
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="results/ifeval")
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens_generated", "--max_tokens", dest="max_tokens_generated", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", nargs="+", choices=["prompt", "instaboost", "spotlight", "pasta"], default=None)
    parser.add_argument("--skip_prompt", action="store_true")
    parser.add_argument("--skip_instaboost", action="store_true")
    parser.add_argument("--skip_spotlight", action="store_true")
    parser.add_argument("--skip_pasta", action="store_true")
    parser.add_argument("--pasta_head_config", type=str, default=None)
    parser.add_argument("--pasta_alpha", type=float, default=0.01)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Optional reasoning effort passed through tokenizer chat templates for models that support it.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking mode in tokenizer chat templates for models that support it.",
    )
    thinking_group.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode in tokenizer chat templates for models that support it.",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument(
        "--instaboost_hook_mode",
        choices=["auto", "legacy", "mass_preserving"],
        default="auto",
        help=(
            "Override which instaboost hook to use. "
            "'auto' (default): legacy slice + F.normalize hook on non-sink models, "
            "mass-preserving hook on sink models (GPT-OSS). "
            "'legacy': force the slice + F.normalize hook regardless of model. "
            "'mass_preserving': force the mass-preserving hook regardless of model "
            "(matches the GPT-OSS instaboost path; useful for comparing on non-sink models)."
        ),
    )
    parser.add_argument(
        "--instaboost_kv_cache",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Override KV cache for instaboost generation. "
            "'auto' (default): off (matches the verified Llama-3 fix path and the GPT-OSS path). "
            "'on': enable KV cache for instaboost (matches the original pre-fix code path)."
        ),
    )
    parser.add_argument(
        "--instaboost_multipliers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the instaboost val-grid multipliers (space-separated ints). "
            f"Default: {INSTABOOST_MULTIPLIERS}. Useful for narrowing the grid on "
            "expensive models where the val landscape is already known to peak at "
            "low multipliers."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--model_n_layers",
        type=int,
        default=None,
        help="Optional layer count override used only when loading openai/gpt-oss-20b.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    load_repo_env()
    args = _parse_args()
    _ensure_nltk_resources()

    selected_methods = set(args.methods or DEFAULT_METHODS)
    if args.skip_prompt:
        selected_methods.discard("prompt")
    if args.skip_instaboost:
        selected_methods.discard("instaboost")
    if args.skip_spotlight:
        selected_methods.discard("spotlight")
    if args.skip_pasta:
        selected_methods.discard("pasta")

    if "pasta" in selected_methods and args.pasta_head_config is None:
        raise ValueError("PASTA selected but --pasta_head_config was not provided.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_short = args.model_path.split("/")[-1]
    out_base = os.path.join(args.output_dir, model_short)
    os.makedirs(out_base, exist_ok=True)

    print(f"Starting IFEval run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_path}")
    print(f"Methods: {sorted(selected_methods)}")

    val_samples, test_samples = load_ifeval_samples(
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(f"Loaded samples -> val: {len(val_samples)} | test: {len(test_samples)}")

    all_method_results: dict[str, dict[str, Any]] = {}
    method_times: dict[str, Any] = {}

    model: HookedTransformer | None = None
    tokenize_instructions_fn = None
    val_spans = None
    test_spans = None

    if selected_methods.intersection({"prompt", "instaboost", "spotlight"}):
        model = _load_hooked_model(args)
        tokenize_instructions_fn = functools.partial(
            tokenize_instructions,
            tokenizer=model.tokenizer,
            model_name=args.model_path,
            apply_chat_template=True,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        print("Loaded TransformerLens model for prompt/instaboost/spotlight methods.")

    if selected_methods.intersection({"instaboost", "spotlight"}):
        assert tokenize_instructions_fn is not None
        print("Computing per-sample instruction spans...")
        val_spans = get_instruction_prompt_spans(
            val_samples,
            tokenizer=model.tokenizer,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        test_spans = get_instruction_prompt_spans(
            test_samples,
            tokenizer=model.tokenizer,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )

    if "prompt" in selected_methods:
        method_name = "prompt"
        method_dir = os.path.join(out_base, method_name)
        cached = _cached_method_summary(method_dir)
        if cached is not None:
            print("prompt: found cached results, skipping.")
            summary, timing = cached
            all_method_results[method_name] = summary
            method_times[method_name] = timing.get("total_time_sec")
        else:
            assert model is not None and tokenize_instructions_fn is not None
            print("Running prompt...")
            input_texts = [sample["full_input"] for sample in test_samples]
            generations, timing = _hooked_generations(
                model=model,
                tokenize_instructions_fn=tokenize_instructions_fn,
                input_texts=input_texts,
                max_tokens_generated=args.max_tokens_generated,
                batch_size=args.batch_size,
                instaboost_hook_mode=args.instaboost_hook_mode,
                instaboost_kv_cache=args.instaboost_kv_cache,
            )
            eval_results = evaluate_generations(generations, test_samples)
            aggregate = compute_aggregate_metrics(
                eval_results,
                n_bootstrap=args.n_bootstrap,
                confidence_level=args.confidence_level,
            )
            payload = save_method_outputs(
                method_output_dir=method_dir,
                method_name=method_name,
                samples=test_samples,
                input_texts=input_texts,
                generations=generations,
                eval_results=eval_results,
                aggregate=aggregate,
                timing=timing,
            )
            all_method_results[method_name] = {
                key: value for key, value in payload.items() if key not in ("method", "best_param")
            }
            method_times[method_name] = timing["total_time_sec"]
            _print_method_results(method_name, aggregate, timing)

    if "instaboost" in selected_methods:
        method_name = "instaboost"
        method_dir = os.path.join(out_base, method_name)
        cached = _cached_method_summary(method_dir)
        if cached is not None:
            print("instaboost: found cached results, skipping.")
            summary, timing = cached
            all_method_results[method_name] = summary
            method_times[method_name] = timing.get("total_time_sec")
        else:
            assert model is not None and tokenize_instructions_fn is not None
            assert val_spans is not None and test_spans is not None
            print("Running instaboost grid search on validation split...")
            os.makedirs(method_dir, exist_ok=True)
            val_progress_path = os.path.join(method_dir, "val_search_progress.json")
            best_params_path = os.path.join(method_dir, "best_params.json")

            if os.path.exists(best_params_path):
                best_payload = _load_json(best_params_path)
                best_multiplier = best_payload["best_multiplier"]
                val_scores = best_payload.get("val_scores", {})
                print(f"instaboost: using cached best_multiplier={best_multiplier}")
            else:
                val_scores: dict[str, float] = {}
                if os.path.exists(val_progress_path):
                    val_scores.update({key: float(value) for key, value in _load_json(val_progress_path).items()})

                best_multiplier = None
                best_score = -1.0
                multipliers_to_search = (
                    args.instaboost_multipliers
                    if args.instaboost_multipliers is not None
                    else INSTABOOST_MULTIPLIERS
                )
                for multiplier in multipliers_to_search:
                    key = str(multiplier)
                    if key in val_scores:
                        score = val_scores[key]
                    else:
                        val_generations, _ = _hooked_generations(
                            model=model,
                            tokenize_instructions_fn=tokenize_instructions_fn,
                            input_texts=[sample["full_input"] for sample in val_samples],
                            max_tokens_generated=args.max_tokens_generated,
                            batch_size=args.batch_size,
                            prompt_spans=val_spans,
                            attention_pattern_multiplier=multiplier,
                            instaboost_hook_mode=args.instaboost_hook_mode,
                            instaboost_kv_cache=args.instaboost_kv_cache,
                        )
                        val_eval_results = evaluate_generations(val_generations, val_samples)
                        val_aggregate = compute_aggregate_metrics(
                            val_eval_results,
                            n_bootstrap=args.n_bootstrap,
                            confidence_level=args.confidence_level,
                        )
                        score = float(val_aggregate["prompt_level_strict_acc"])
                        val_scores[key] = score
                        _save_json(val_progress_path, val_scores)

                    print(f"  multiplier={multiplier} | val prompt strict acc={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_multiplier = multiplier

                assert best_multiplier is not None
                _save_json(
                    best_params_path,
                    {
                        "best_multiplier": best_multiplier,
                        "val_scores": val_scores,
                    },
                )
                print(f"instaboost: best_multiplier={best_multiplier} (val score={best_score:.3f})")

            input_texts = [sample["full_input"] for sample in test_samples]
            generations, timing = _hooked_generations(
                model=model,
                tokenize_instructions_fn=tokenize_instructions_fn,
                input_texts=input_texts,
                max_tokens_generated=args.max_tokens_generated,
                batch_size=args.batch_size,
                prompt_spans=test_spans,
                attention_pattern_multiplier=best_multiplier,
                instaboost_hook_mode=args.instaboost_hook_mode,
                instaboost_kv_cache=args.instaboost_kv_cache,
            )
            eval_results = evaluate_generations(generations, test_samples)
            aggregate = compute_aggregate_metrics(
                eval_results,
                n_bootstrap=args.n_bootstrap,
                confidence_level=args.confidence_level,
            )
            payload = save_method_outputs(
                method_output_dir=method_dir,
                method_name=method_name,
                samples=test_samples,
                input_texts=input_texts,
                generations=generations,
                eval_results=eval_results,
                aggregate=aggregate,
                timing=timing,
                best_param=best_multiplier,
            )
            all_method_results[method_name] = {
                key: value for key, value in payload.items() if key not in ("method", "best_param")
            }
            method_times[method_name] = timing["total_time_sec"]
            _print_method_results(method_name, aggregate, timing)

    if "spotlight" in selected_methods:
        method_name = "spotlight"
        method_dir = os.path.join(out_base, method_name)
        cached = _cached_method_summary(method_dir)
        if cached is not None:
            print("spotlight: found cached results, skipping.")
            summary, timing = cached
            all_method_results[method_name] = summary
            method_times[method_name] = timing.get("total_time_sec")
        else:
            assert model is not None and tokenize_instructions_fn is not None
            assert val_spans is not None and test_spans is not None
            print("Running spotlight grid search on validation split...")
            os.makedirs(method_dir, exist_ok=True)
            val_progress_path = os.path.join(method_dir, "val_search_progress.json")
            best_params_path = os.path.join(method_dir, "best_params.json")

            if os.path.exists(best_params_path):
                best_payload = _load_json(best_params_path)
                best_target_attention = best_payload["best_target_attention"]
                val_scores = best_payload.get("val_scores", {})
                print(f"spotlight: using cached best_target_attention={best_target_attention}")
            else:
                val_scores: dict[str, float] = {}
                if os.path.exists(val_progress_path):
                    val_scores.update({key: float(value) for key, value in _load_json(val_progress_path).items()})

                best_target_attention = None
                best_score = -1.0
                for target_attention in SPOTLIGHT_TARGET_ATTENTIONS:
                    key = str(target_attention)
                    if key in val_scores:
                        score = val_scores[key]
                    else:
                        val_generations, _ = _hooked_generations(
                            model=model,
                            tokenize_instructions_fn=tokenize_instructions_fn,
                            input_texts=[sample["full_input"] for sample in val_samples],
                            max_tokens_generated=args.max_tokens_generated,
                            batch_size=args.batch_size,
                            prompt_spans=val_spans,
                            spotlight_target_attention=target_attention,
                        )
                        val_eval_results = evaluate_generations(val_generations, val_samples)
                        val_aggregate = compute_aggregate_metrics(
                            val_eval_results,
                            n_bootstrap=args.n_bootstrap,
                            confidence_level=args.confidence_level,
                        )
                        score = float(val_aggregate["prompt_level_strict_acc"])
                        val_scores[key] = score
                        _save_json(val_progress_path, val_scores)

                    print(f"  target_attention={target_attention} | val prompt strict acc={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_target_attention = target_attention

                assert best_target_attention is not None
                _save_json(
                    best_params_path,
                    {
                        "best_target_attention": best_target_attention,
                        "val_scores": val_scores,
                    },
                )
                print(
                    f"spotlight: best_target_attention={best_target_attention} "
                    f"(val score={best_score:.3f})"
                )

            input_texts = [sample["full_input"] for sample in test_samples]
            generations, timing = _hooked_generations(
                model=model,
                tokenize_instructions_fn=tokenize_instructions_fn,
                input_texts=input_texts,
                max_tokens_generated=args.max_tokens_generated,
                batch_size=args.batch_size,
                prompt_spans=test_spans,
                spotlight_target_attention=best_target_attention,
            )
            eval_results = evaluate_generations(generations, test_samples)
            aggregate = compute_aggregate_metrics(
                eval_results,
                n_bootstrap=args.n_bootstrap,
                confidence_level=args.confidence_level,
            )
            payload = save_method_outputs(
                method_output_dir=method_dir,
                method_name=method_name,
                samples=test_samples,
                input_texts=input_texts,
                generations=generations,
                eval_results=eval_results,
                aggregate=aggregate,
                timing=timing,
                best_param=best_target_attention,
            )
            all_method_results[method_name] = {
                key: value for key, value in payload.items() if key not in ("method", "best_param")
            }
            method_times[method_name] = timing["total_time_sec"]
            _print_method_results(method_name, aggregate, timing)

    if "pasta" in selected_methods:
        method_name = "pasta"
        method_dir = os.path.join(out_base, method_name)
        cached = _cached_method_summary(method_dir)
        if cached is not None:
            print("pasta: found cached results, skipping.")
            summary, timing = cached
            all_method_results[method_name] = summary
            method_times[method_name] = timing.get("total_time_sec")
        else:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()

            print("Running pasta...")
            pasta_head_config = _load_json(args.pasta_head_config)
            input_texts, generations, timing = _run_pasta_method(
                args=args,
                samples=test_samples,
                pasta_head_config=pasta_head_config,
            )
            eval_results = evaluate_generations(generations, test_samples)
            aggregate = compute_aggregate_metrics(
                eval_results,
                n_bootstrap=args.n_bootstrap,
                confidence_level=args.confidence_level,
            )
            payload = save_method_outputs(
                method_output_dir=method_dir,
                method_name=method_name,
                samples=test_samples,
                input_texts=input_texts,
                generations=generations,
                eval_results=eval_results,
                aggregate=aggregate,
                timing=timing,
                best_param={"alpha": args.pasta_alpha, "head_config": args.pasta_head_config},
            )
            all_method_results[method_name] = {
                key: value for key, value in payload.items() if key not in ("method", "best_param")
            }
            method_times[method_name] = timing["total_time_sec"]
            _print_method_results(method_name, aggregate, timing)

    summary_payload = {
        "model": args.model_path,
        "dataset": "ifeval_instruction_last",
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "methods": all_method_results,
        "method_times": method_times,
    }
    _save_json(os.path.join(out_base, "all_methods_results.json"), summary_payload)

    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saved results to: {out_base}")


if __name__ == "__main__":
    main()
