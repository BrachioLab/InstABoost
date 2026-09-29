"""AutoPASTA coarse-to-fine head selection on standard data_utils tasks.

Two phases (Zhang et al. 2024):
  Phase 1 (layer level): for each layer L, instantiate PASTA with
    head_config=[L] (steers ALL heads in layer L), score on the task's
    val split.
  Phase 2 (head level): take the top-l layers from Phase 1, then for each
    layer in that set sweep individual heads (head_config={L: [h]}).
  Selection: rank Phase 2 entries, take top-k. Emit head_config in
    pastalib's expected shape.

For IFEval, use src/autopasta_head_selection_ifeval.py instead — IFEval is
not in DATASET_CHOICES and uses per-sample instruction_block highlights.

Supports tasks listed in data_utils.DATASET_CHOICES:
  triviaqa, truthfulqa, safety, jbb, toxicity, ai-risk-mcq, ai-risk-qa,
  emotions-qa.

Outputs (under {output_dir}/{model_short}/{task_label}/autopasta/):
  layer_selection.json     — Phase 1 per-layer scores
  head_selection.json      — Phase 2 per-(layer, head) scores in top-l layers
  head_config_top{k}.json  — final {layer: [head, ...]} ready for pastalib

Each phase is independently resumable. Use --phase {1, 2, select, both}.

Dispatch by model class:
  - Qwen3ForCausalLM: Qwen3PASTA (the patched subclass, inlined below)
  - Llama / Gemma / Mistral / Phi3: vanilla pastalib.pasta.PASTA
Both rely on attn_implementation="eager" so the additive attention_mask
bias actually reaches the attention modules.
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pastalib.pasta import PASTA
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import DATASET_CHOICES, DATASET_CONFIGS, get_dataset
from evaluation_utils import evaluate_generations
from model_interaction_utils import tokenize_instructions
from prompts import EVAL_STRATEGY


# Datasets where evaluation_utils.evaluate_generations -> EVAL_STRATEGY is
# broken: the safety/jbb branch (evaluation_utils.py:180) calls
# ``EVAL_STRATEGY[eval_type](steer_dataset[i], generations[i])`` while
# prompts.py's safety_lmjudge_lazy expects ``(response, args_dict)`` and
# reads args["question"]. Rather than touch the shared modules, autopasta
# computes the score directly for these datasets.
SAFETY_LMJUDGE_DATASETS = {"safety", "jbb"}


def _safety_score(generations, val_samples, eval_type):
    """Score safety/jbb val generations by calling EVAL_STRATEGY[eval_type]
    with the (response, {"question": <prompt>}) convention prompts.py expects.
    Returns a dict matching evaluate_generations' shape (only "score" is used)."""
    scores = []
    for sample, gen in zip(val_samples, generations):
        q = sample if isinstance(sample, str) else _to_user_str(sample)
        # safety/jbb judges are "1.0 = unsafe / refused-to-refuse" → higher
        # score means the steering succeeded at making the model comply with
        # the harmful prompt. Same convention as the original safety pasta.
        result = EVAL_STRATEGY[eval_type](gen, {"question": q})
        scores.append(float(result))
    return {"score": sum(scores) / len(scores) if scores else 0.0}


# --------------------------------------------------------------------------
# Qwen3PASTA — pastalib subclass that adds Qwen3 support.
# Identical to the inline class in autopasta_head_selection_ifeval.py;
# duplicated so this script is self-contained.
# --------------------------------------------------------------------------


class Qwen3PASTA(PASTA):
    def setup_model(self, model):
        if model.__class__.__name__ == "Qwen3ForCausalLM":
            self.model_name = "qwen3"
            self.num_attn_head = model.config.num_attention_heads
            return
        super().setup_model(model)

    def edit_multisection_attention(
        self,
        module,
        input_args,
        input_kwargs,
        head_idx,
        token_ranges,
        input_len,
    ):
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs["attention_mask"].clone()
            arg_idx = None
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {module}")

        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()

        for token_range in token_ranges:
            for bi, (ti, tj) in enumerate(token_range.tolist()):
                if self.scale_position == "include":
                    attention_mask[bi, head_idx, :, ti:tj] += scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[bi, head_idx, :, :ti] += scale_constant
                    attention_mask[bi, head_idx, :, tj:input_len] += scale_constant
                elif self.scale_position == "generation":
                    attention_mask[bi, head_idx, :, :input_len] += scale_constant
                else:
                    raise ValueError(f"Unexpected {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= scale_constant

        if self.model_name in ["llama", "mistral", "qwen3"]:
            attention_mask.old_size = attention_mask.size
            attention_mask.size = lambda: (bsz, 1, tgt_len, src_len)

        if arg_idx is None:
            input_kwargs["attention_mask"] = attention_mask
            return input_args, input_kwargs
        return (
            (*input_args[:arg_idx], attention_mask, *input_args[arg_idx + 1:]),
            input_kwargs,
        )


Qwen3PASTA.ATTN_MODULE_NAME["qwen3"] = "model.layers.{}.self_attn"
Qwen3PASTA.ATTENTION_MASK_ARGIDX["qwen3"] = 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", default="results_pasta/autopasta")
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Required for ai-risk-{mcq,qa}, emotions-qa, and the *qa tasks. "
             "For ai-risk: must end with _<behavior> (e.g. ai-risk-mcq_wealth-seeking-inclination). "
             "For emotions: must end with _<emotion> (e.g. emotions_train_anger).",
    )
    ap.add_argument(
        "--task_label",
        default=None,
        help="Subdirectory label under {output_dir}/{model_short}/. "
             "Defaults to dataset name; pass an explicit value when running multiple "
             "behaviors of the same dataset (e.g. ai-risk-mcq_wealth).",
    )
    ap.add_argument("--answer_file_name", default="new_answer.txt")
    ap.add_argument("--baseline_generations_dir", default=None)
    ap.add_argument("--steer_to_pt", action="store_true")
    ap.add_argument("--cache_harmless", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--top_l", type=int, default=6)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument(
        "--enable_thinking",
        action="store_true",
        help="For Qwen3 only — apply chat template with thinking enabled.",
    )
    ap.add_argument("--use_fluency", action="store_true")
    ap.add_argument("--use_relevance", action="store_true")
    ap.add_argument(
        "--phase",
        choices=["both", "1", "2", "select"],
        default="both",
    )
    return ap.parse_args()


def get_pasta_class(model):
    if model.__class__.__name__ == "Qwen3ForCausalLM":
        return Qwen3PASTA
    return PASTA


def _to_user_str(sample):
    """Tasks like ai-risk-mcq return chat-message lists from PROMPTS; flatten."""
    if isinstance(sample, str):
        return sample
    user_parts = [m["content"] for m in sample if m.get("role") == "user"]
    return "\n\n".join(user_parts)


def format_texts(tokenizer, val_samples, steer_prompt, apply_chat_template, enable_thinking, model_class):
    raw_texts = [steer_prompt + _to_user_str(s) for s in val_samples]
    if not apply_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return raw_texts
    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_class == "Qwen3ForCausalLM" and not enable_thinking:
        template_kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], **template_kwargs
        )
        for t in raw_texts
    ]


def run_eval(
    pasta_obj,
    model,
    tokenizer,
    formatted_texts,
    steer_prompt,
    val_samples,
    val_questions,
    val_answers,
    val_alt_answers,
    eval_args,
    eval_type,
    dataset,
    batch_size,
    max_tokens,
    device,
    use_fluency,
    use_relevance,
):
    generations: list[str] = []
    for i in range(0, len(val_samples), batch_size):
        batch_texts = formatted_texts[i:i + batch_size]
        emphasized = [steer_prompt] * len(batch_texts)
        inputs, offset_mapping = pasta_obj.inputs_from_batch(batch_texts)
        inputs = inputs.to(device)
        with pasta_obj.apply_steering(
            model=model,
            strings=batch_texts,
            substrings=emphasized,
            model_input=inputs,
            offsets_mapping=offset_mapping,
        ) as steered:
            out = steered.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        out = out[:, inputs.input_ids.shape[1]:]
        generations.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    if dataset in SAFETY_LMJUDGE_DATASETS:
        return _safety_score(generations, val_samples, eval_type)
    return evaluate_generations(
        generations=generations,
        dataset=dataset,
        eval_type=eval_type,
        eval_args=eval_args,
        questions=val_questions,
        answers=val_answers,
        alt_answers=val_alt_answers,
        is_validation=True,
        use_fluency=use_fluency,
        use_relevance=use_relevance,
        steer_dataset=val_samples,
    )


def phase1_layer_sweep(
    args, model, tokenizer, val_samples, formatted_texts, steer_prompt,
    val_questions, val_answers, val_alt_answers, eval_args, eval_type,
    layer_path: Path, total_layers: int, max_tokens: int,
):
    PastaCls = get_pasta_class(model)
    results: dict = {}
    if layer_path.exists():
        results = json.loads(layer_path.read_text())
        print(f"Resuming Phase 1 — {len(results)} layers already scored")

    for layer in range(total_layers):
        if str(layer) in results:
            continue
        t0 = time.time()
        pasta_obj = PastaCls(
            model=model, tokenizer=tokenizer,
            head_config=[layer], alpha=args.alpha, scale_position="exclude",
        )
        out = run_eval(
            pasta_obj, model, tokenizer, formatted_texts, steer_prompt,
            val_samples, val_questions, val_answers, val_alt_answers,
            eval_args, eval_type, args.dataset,
            args.batch_size, max_tokens, args.device,
            args.use_fluency, args.use_relevance,
        )
        dt = time.time() - t0
        score = out["score"]
        print(f"  layer={layer:>3}  score={score:.3f}  ({dt:.1f}s)")
        results[str(layer)] = {"score": score, "time": dt}
        if args.use_fluency and "fluency" in out:
            results[str(layer)]["fluency"] = out["fluency"]
        layer_path.write_text(json.dumps(results, indent=2))
        del pasta_obj
        gc.collect()
        torch.cuda.empty_cache()

    return results


def phase2_head_sweep(
    args, model, tokenizer, val_samples, formatted_texts, steer_prompt,
    val_questions, val_answers, val_alt_answers, eval_args, eval_type,
    head_path: Path, top_layers: list[int], n_heads: int, max_tokens: int,
):
    PastaCls = get_pasta_class(model)
    results: dict = {}
    if head_path.exists():
        results = json.loads(head_path.read_text())
        n_done = sum(len(v) for v in results.values())
        print(f"Resuming Phase 2 — {n_done} (layer, head) entries already scored")

    for layer in top_layers:
        for head in range(n_heads):
            if str(layer) in results and str(head) in results[str(layer)]:
                continue
            t0 = time.time()
            pasta_obj = PastaCls(
                model=model, tokenizer=tokenizer,
                head_config={f"{layer}": [head]},
                alpha=args.alpha, scale_position="exclude",
            )
            out = run_eval(
                pasta_obj, model, tokenizer, formatted_texts, steer_prompt,
                val_samples, val_questions, val_answers, val_alt_answers,
                eval_args, eval_type, args.dataset,
                args.batch_size, max_tokens, args.device,
                args.use_fluency, args.use_relevance,
            )
            dt = time.time() - t0
            score = out["score"]
            print(f"  layer={layer:>3} head={head:>3}  score={score:.3f}  ({dt:.1f}s)")
            entry = {"score": score, "time": dt}
            if args.use_fluency and "fluency" in out:
                entry["fluency"] = out["fluency"]
            results.setdefault(str(layer), {})[str(head)] = entry
            head_path.write_text(json.dumps(results, indent=2))
            del pasta_obj
            gc.collect()
            torch.cuda.empty_cache()

    return results


def select_head_config(layer_results, head_results, top_l, top_k):
    layers_sorted = sorted(layer_results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top_layers = [int(L) for L, _ in layers_sorted[:top_l]]

    candidates = []
    for layer in top_layers:
        if str(layer) in head_results:
            for head, entry in head_results[str(layer)].items():
                candidates.append((layer, int(head), entry["score"]))
    candidates.sort(key=lambda t: t[2], reverse=True)

    if top_k is None:
        selected = candidates
    else:
        selected = []
        for i, (l, h, s) in enumerate(candidates):
            if i < top_k or (selected and s == selected[-1][2]):
                selected.append((l, h, s))
            else:
                break

    head_config: dict[str, list[int]] = {}
    for l, h, _ in selected:
        head_config.setdefault(str(l), []).append(h)
    return head_config, top_layers, selected


def main():
    args = parse_args()
    print(f"Starting AutoPASTA at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DATASET_CONFIGS[args.dataset]
    max_tokens = cfg["max_tokens_generated"]
    n_inst_train = cfg["n_inst_train"]
    n_inst_test = cfg["n_inst_test"]
    n_inst_val = cfg["n_inst_val"]
    eval_type = cfg["eval_type"]
    apply_chat_template = cfg.get("apply_chat_template", True)

    MODEL_PATH = args.model_path
    model_short = MODEL_PATH.split("/")[-1]
    task_label = args.task_label or args.dataset
    out_dir = Path(args.output_dir) / model_short / task_label / "autopasta"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_path = out_dir / "layer_selection.json"
    head_path = out_dir / "head_selection.json"
    config_filename = (
        f"head_config_top{args.top_k}.json" if args.top_k is not None else "head_config_all.json"
    )
    config_path = out_dir / config_filename

    print(f"Loading {MODEL_PATH} with attn_implementation=eager")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.tokenizer = tokenizer

    total_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Model: {model.__class__.__name__} | layers={total_layers} | heads={n_heads}")
    print(f"Output dir: {out_dir}")

    layer_results = json.loads(layer_path.read_text()) if layer_path.exists() else {}
    head_results = json.loads(head_path.read_text()) if head_path.exists() else {}

    val_samples = formatted_texts = steer_prompt = None
    val_questions = val_answers = val_alt_answers = eval_args = None
    if args.phase != "select":
        baseline_generations_dir = (
            args.baseline_generations_dir
            or os.path.join(args.output_dir, model_short, task_label)
        )
        tokenize_instructions_fn = functools.partial(
            tokenize_instructions,
            tokenizer=tokenizer,
            model_name=MODEL_PATH,
            apply_chat_template=apply_chat_template,
        )

        print(f"Loading dataset: {args.dataset} (dataset_dir={args.dataset_dir})")
        (
            harmful_inst_train, harmless_inst_train,
            steer_dataset, steer_val_dataset,
            steer_questions, steer_answers, steer_alt_answers,
            val_questions, val_answers, val_alt_answers,
            eval_args, steer_prompt,
        ) = get_dataset(
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
            answer_file_name=args.answer_file_name,
            n_inst_train=n_inst_train,
            n_inst_test=n_inst_test,
            n_inst_val=n_inst_val,
            steer_to_pt=args.steer_to_pt,
            tokenizer=tokenizer,
            model=model,
            baseline_generations_dir=baseline_generations_dir,
            cache_harmless=args.cache_harmless,
            tokenize_instructions_fn=tokenize_instructions_fn,
            max_tokens_generated=max_tokens,
            batch_size=args.batch_size,
        )
        val_samples = steer_val_dataset
        formatted_texts = format_texts(
            tokenizer, val_samples, steer_prompt, apply_chat_template,
            args.enable_thinking, model.__class__.__name__,
        )
        print(f"  steer_prompt: {steer_prompt[:80]!r}...")
        print(f"  val_samples: {len(val_samples)}")

    if args.phase in ("both", "1"):
        print(f"\n=== Phase 1: layer-level sweep (L={total_layers}) ===")
        layer_results = phase1_layer_sweep(
            args, model, tokenizer, val_samples, formatted_texts, steer_prompt,
            val_questions, val_answers, val_alt_answers, eval_args, eval_type,
            layer_path, total_layers, max_tokens,
        )

    if args.phase in ("both", "2"):
        if not layer_results:
            raise SystemExit(f"No Phase 1 results at {layer_path}. Run --phase 1 first.")
        sorted_layers = sorted(layer_results.items(), key=lambda kv: kv[1]["score"], reverse=True)
        top_layers = [int(L) for L, _ in sorted_layers[:args.top_l]]
        print(f"\n=== Phase 2: head-level sweep in top-{args.top_l} layers ===")
        for L, info in sorted_layers[:args.top_l]:
            print(f"  layer {L}: score={info['score']:.3f}")
        head_results = phase2_head_sweep(
            args, model, tokenizer, val_samples, formatted_texts, steer_prompt,
            val_questions, val_answers, val_alt_answers, eval_args, eval_type,
            head_path, sorted(top_layers), n_heads, max_tokens,
        )

    if not layer_results or not head_results:
        if args.phase == "select":
            raise SystemExit("Need both layer_selection.json and head_selection.json to produce head_config.")
        print("Skipping selection — Phase 2 produced no head results")
        return

    head_config, top_layers, selected = select_head_config(
        layer_results, head_results, args.top_l, args.top_k,
    )
    config_path.write_text(json.dumps(head_config, indent=2))
    print(f"\nWrote head config to {config_path}")
    print(f"  Top-{args.top_l} layers: {sorted(top_layers)}")
    print(f"  Selected heads: {sum(len(h) for h in head_config.values())}")
    print("  Top 5 (layer, head, score):")
    for l, h, s in selected[:5]:
        print(f"    layer={l:>3} head={h:>3} score={s:.3f}")


if __name__ == "__main__":
    main()
