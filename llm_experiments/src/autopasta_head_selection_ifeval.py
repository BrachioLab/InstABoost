"""AutoPASTA coarse-to-fine head selection on IFEval.

Two phases (Zhang et al. 2024 — AutoPASTA):

  Phase 1 (layer level): for each layer L in 0..N-1, instantiate PASTA with
    head_config=[L] (steers ALL heads in layer L), score on IFEval val.
  Phase 2 (head level): take the top-l layers from Phase 1, then for each
    layer in that set sweep individual heads (head_config={L: [h]}).
  Selection: rank all (layer, head) entries from Phase 2 by score, take
    top-k (or all). Emit head_config in pastalib's expected shape.

Total cost: L + l*H evaluations vs the full PASTA L*H. With l=6:
    Llama (32x32):   1024 -> 224  (4.6x)
    Gemma (28x16):    448 -> 124  (3.6x)
    Qwen3 (40x40):   1600 -> 280  (5.7x)

Dispatch by model class:
  - Qwen3ForCausalLM: Qwen3PASTA (the patched subclass)
  - Llama/Gemma/Mistral/Phi3: vanilla pastalib.pasta.PASTA
  Both rely on attn_implementation="eager" so the additive attention_mask
  bias actually reaches the attention modules.

Outputs (under {output_dir}/{model_short}/autopasta/):
  layer_selection.json     — Phase 1 per-layer scores
  head_selection.json      — Phase 2 per-(layer, head) scores in top-l layers
  head_config_top{k}.json  — final {layer: [head, ...]} ready for pastalib

Each phase is independently resumable. Use --phase {1, 2, select, both} to
control which steps run.

Example launches:
  # Full pipeline on Llama:
  python src/autopasta_head_selection_ifeval.py --model_path meta-llama/Meta-Llama-3-8B-Instruct
  # Just produce head_config from existing JSONs:
  python src/autopasta_head_selection_ifeval.py --model_path Qwen/Qwen3-14B --phase select --top_k 50
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pastalib.pasta import PASTA
from transformers import AutoModelForCausalLM, AutoTokenizer

from ifeval_data_utils import load_ifeval_samples
from ifeval_eval_utils import compute_aggregate_metrics, evaluate_generations


def _ensure_nltk_resources() -> None:
    """IFEval instruction checkers tokenize with nltk's punkt; download on demand."""
    try:
        import nltk
    except ImportError:
        return

    required = [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]
    missing = [pkg for path, pkg in required if not _has_nltk(nltk, path)]
    for pkg in missing:
        print(f"Downloading missing NLTK resource: {pkg}")
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"Warning: failed to download {pkg}: {e}")
    still_missing = [pkg for path, pkg in required if not _has_nltk(nltk, path)]
    if still_missing:
        raise LookupError(
            f"Missing NLTK resources: {', '.join(still_missing)}. Run: "
            "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\""
        )


def _has_nltk(nltk, path: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False
# --------------------------------------------------------------------------
# Qwen3PASTA — pastalib subclass that adds Qwen3 support.
# Upstream pastalib only knows Llama/Mistral/Gemma/Phi3/GPT-J. For Qwen3 we
# need: setup_model recognition, ATTN_MODULE_NAME / ATTENTION_MASK_ARGIDX
# entries, and the size-lying override extended to "qwen3" so HF code that
# calls attention_mask.size() doesn't break after pastalib expands the mask
# per query head. Requires loading the model with attn_implementation="eager"
# since pastalib's bias only flows through the eager attention forward path.
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
    ap.add_argument("--output_dir", default="results_pasta/ifeval_autopasta")
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--max_tokens_generated", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--top_l",
        type=int,
        default=6,
        help="Top layers to keep after Phase 1. Paper finds 6 sufficient for L<=32.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top heads to keep across top-l layers (default: keep all).",
    )
    ap.add_argument(
        "--enable_thinking",
        action="store_true",
        help="For Qwen3 only — apply chat template with thinking enabled (default: disabled).",
    )
    ap.add_argument(
        "--score_metric",
        choices=["instruction_level", "prompt_level"],
        default="instruction_level",
    )
    ap.add_argument(
        "--phase",
        choices=["both", "1", "2", "select"],
        default="both",
        help="'1' = layer sweep only; '2' = head sweep only (assumes Phase 1 done); "
             "'select' = read existing JSONs and emit head_config; 'both' = full pipeline.",
    )
    return ap.parse_args()


def get_pasta_class(model) -> type[PASTA]:
    if model.__class__.__name__ == "Qwen3ForCausalLM":
        return Qwen3PASTA
    return PASTA


def format_chat(tokenizer, val_samples, enable_thinking: bool, model_class: str) -> list[str]:
    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_class == "Qwen3ForCausalLM" and not enable_thinking:
        template_kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["full_input"]}], **template_kwargs
        )
        for s in val_samples
    ]


def run_eval(
    pasta_obj,
    model,
    tokenizer,
    formatted_texts: list[str],
    instruction_blocks: list[str],
    val_samples: list[dict],
    batch_size: int,
    max_tokens: int,
    device: str,
) -> dict:
    generations: list[str] = []
    for i in range(0, len(val_samples), batch_size):
        batch_texts = formatted_texts[i:i + batch_size]
        batch_blocks = instruction_blocks[i:i + batch_size]
        inputs, offset_mapping = pasta_obj.inputs_from_batch(batch_texts)
        inputs = inputs.to(device)
        with pasta_obj.apply_steering(
            model=model,
            strings=batch_texts,
            substrings=[batch_blocks],
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

    eval_results = evaluate_generations(generations, val_samples)
    return compute_aggregate_metrics(eval_results, n_bootstrap=200)


def _result_entry(agg: dict, dt: float) -> dict:
    return {
        "prompt_level_strict_acc": agg["prompt_level_strict_acc"],
        "instruction_level_strict_acc": agg["instruction_level_strict_acc"],
        "n_samples": agg["n_samples"],
        "n_instructions": agg["n_instructions"],
        "time": dt,
    }


def phase1_layer_sweep(
    args, model, tokenizer, val_samples, formatted_texts, instruction_blocks,
    layer_path: Path, score_key: str, total_layers: int,
) -> dict[str, dict]:
    PastaCls = get_pasta_class(model)
    results: dict[str, dict] = {}
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
        agg = run_eval(
            pasta_obj, model, tokenizer, formatted_texts, instruction_blocks,
            val_samples, args.batch_size, args.max_tokens_generated, args.device,
        )
        dt = time.time() - t0
        score = agg[score_key]
        print(f"  layer={layer:>3}  {args.score_metric}={score:.3f}  ({dt:.1f}s)")
        results[str(layer)] = {"score": score, **_result_entry(agg, dt)}
        layer_path.write_text(json.dumps(results, indent=2))
        del pasta_obj
        gc.collect()
        torch.cuda.empty_cache()

    return results


def phase2_head_sweep(
    args, model, tokenizer, val_samples, formatted_texts, instruction_blocks,
    head_path: Path, score_key: str, top_layers: list[int], n_heads: int,
) -> dict[str, dict[str, dict]]:
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
            agg = run_eval(
                pasta_obj, model, tokenizer, formatted_texts, instruction_blocks,
                val_samples, args.batch_size, args.max_tokens_generated, args.device,
            )
            dt = time.time() - t0
            score = agg[score_key]
            print(f"  layer={layer:>3} head={head:>3}  {args.score_metric}={score:.3f}  ({dt:.1f}s)")
            results.setdefault(str(layer), {})[str(head)] = {"score": score, **_result_entry(agg, dt)}
            head_path.write_text(json.dumps(results, indent=2))
            del pasta_obj
            gc.collect()
            torch.cuda.empty_cache()

    return results


def select_head_config(
    layer_results: dict, head_results: dict, top_l: int, top_k: int | None,
) -> tuple[dict[str, list[int]], list[int], list[tuple[int, int, float]]]:
    layers_sorted = sorted(
        layer_results.items(), key=lambda kv: kv[1]["score"], reverse=True
    )
    top_layers = [int(L) for L, _ in layers_sorted[:top_l]]

    candidates: list[tuple[int, int, float]] = []
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

    _ensure_nltk_resources()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    MODEL_PATH = args.model_path
    model_short = MODEL_PATH.split("/")[-1]
    out_dir = Path(args.output_dir) / model_short / "autopasta"
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

    score_key = (
        "instruction_level_strict_acc"
        if args.score_metric == "instruction_level"
        else "prompt_level_strict_acc"
    )

    layer_results = json.loads(layer_path.read_text()) if layer_path.exists() else {}
    head_results = json.loads(head_path.read_text()) if head_path.exists() else {}

    val_samples = formatted_texts = instruction_blocks = None
    if args.phase != "select":
        print(f"Loading IFEval val samples (n_val={args.n_val})")
        val_samples, _ = load_ifeval_samples(n_val=args.n_val, n_test=0, seed=args.seed)
        formatted_texts = format_chat(
            tokenizer, val_samples, args.enable_thinking, model.__class__.__name__
        )
        instruction_blocks = [s["instruction_block"] for s in val_samples]

    if args.phase in ("both", "1"):
        print(f"\n=== Phase 1: layer-level sweep (L={total_layers}) ===")
        layer_results = phase1_layer_sweep(
            args, model, tokenizer, val_samples, formatted_texts, instruction_blocks,
            layer_path, score_key, total_layers,
        )

    if args.phase in ("both", "2"):
        if not layer_results:
            raise SystemExit(
                f"No Phase 1 results at {layer_path}. Run with --phase 1 first."
            )
        sorted_layers = sorted(
            layer_results.items(), key=lambda kv: kv[1]["score"], reverse=True
        )
        top_layers = [int(L) for L, _ in sorted_layers[:args.top_l]]
        print(f"\n=== Phase 2: head-level sweep in top-{args.top_l} layers ===")
        for L, info in sorted_layers[:args.top_l]:
            print(f"  layer {L}: score={info['score']:.3f}")
        head_results = phase2_head_sweep(
            args, model, tokenizer, val_samples, formatted_texts, instruction_blocks,
            head_path, score_key, sorted(top_layers), n_heads,
        )

    # Selection
    if not layer_results or not head_results:
        if args.phase == "select":
            raise SystemExit(
                "Need both layer_selection.json and head_selection.json to produce head_config."
            )
        print("Skipping selection — Phase 2 produced no head results")
        return

    head_config, top_layers, selected = select_head_config(
        layer_results, head_results, args.top_l, args.top_k
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
