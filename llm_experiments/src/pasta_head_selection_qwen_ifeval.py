"""PASTA per-(layer, head) scoring sweep on IFEval, for Qwen3.

Reuses ``Qwen3PASTA`` from pasta_head_selection_qwen.py (the patched subclass
that knows about Qwen3ForCausalLM). Diverges from that script in two places:

  1. Data — uses load_ifeval_samples(...) instead of data_utils.get_dataset.
     IFEval is not registered in DATASET_CONFIGS, and its samples carry a
     dedicated per-sample "instruction_block" substring (the trailing
     "Your response should follow these constraints: ..." block) which is
     exactly what PASTA should highlight. So substrings are per-sample,
     not a single shared steer_prompt.

  2. Eval — uses ifeval_eval_utils.evaluate_generations + compute_aggregate
     and scores each head by ``instruction_level_strict_acc`` (denser signal
     than ``prompt_level_strict_acc`` since each sample has multiple
     instructions, so it ranks heads more reliably given a small val set).

Output JSON shape matches pasta_head_selection_qwen.py's:
    {output_dir}/{model_short}/pasta/layer_head_selection_results.json
or layer_head_selection_results.shard_{a}_{b}.json when sharded. Drop-in
compatible with src/pasta_head_selection.py's aggregator if you point its
TASK_DIRS at the resulting tree.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ifeval_data_utils import load_ifeval_samples
from ifeval_eval_utils import compute_aggregate_metrics, evaluate_generations
from pasta_head_selection_qwen import Qwen3PASTA


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output_dir", type=str, default="results_pasta/ifeval")
    parser.add_argument(
        "--n_val",
        type=int,
        default=50,
        help="Number of IFEval val samples for per-head scoring (default 50, matching other PASTA tasks).",
    )
    parser.add_argument(
        "--max_tokens_generated",
        type=int,
        default=512,
        help="Generation cap per val sample. Lower = faster sweep; relative head ranking is preserved even if some length constraints are truncated.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Apply Qwen3 chat template with thinking enabled (default: disabled, matching the IFEval runner).",
    )
    parser.add_argument(
        "--score_metric",
        choices=["instruction_level", "prompt_level"],
        default="instruction_level",
        help="Which IFEval metric to use as the per-head score.",
    )
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--head_start", type=int, default=None)
    parser.add_argument("--head_end", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE = args.device
    batch_size = args.batch_size
    ALPHA = args.alpha
    MODEL_PATH = args.model_path
    model_short = MODEL_PATH.split("/")[-1]

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
    print(f"Model: {model.__class__.__name__} | layers={total_layers} | query heads={n_heads}")

    layer_start = 0 if args.layer_start is None else args.layer_start
    layer_end = total_layers if args.layer_end is None else args.layer_end
    head_start = 0 if args.head_start is None else args.head_start
    head_end = n_heads if args.head_end is None else args.head_end
    sharded = (
        args.layer_start is not None or args.layer_end is not None
        or args.head_start is not None or args.head_end is not None
    )
    if not (0 <= layer_start < layer_end <= total_layers):
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) for total_layers={total_layers}"
        )

    print(f"Loading IFEval val samples (n_val={args.n_val}, seed={args.seed})")
    val_samples, _ = load_ifeval_samples(n_val=args.n_val, n_test=0, seed=args.seed)
    print(f"Loaded {len(val_samples)} val samples")

    pasta_output_dir = os.path.join(args.output_dir, model_short, "pasta")
    os.makedirs(pasta_output_dir, exist_ok=True)
    if sharded:
        suffix = f"shard_{layer_start}_{layer_end}_h{head_start}_{head_end}"
        results_path = os.path.join(
            pasta_output_dir, f"layer_head_selection_results.{suffix}.json"
        )
    else:
        results_path = os.path.join(pasta_output_dir, "layer_head_selection_results.json")
    print(f"Writing results to {results_path}")

    results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        n_done = sum(len(v) for v in results.values())
        print(f"Resuming — {n_done} (layer, head) entries already scored")

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not args.enable_thinking:
        template_kwargs["enable_thinking"] = False

    formatted_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["full_input"]}], **template_kwargs
        )
        for s in val_samples
    ]
    instruction_blocks = [s["instruction_block"] for s in val_samples]

    score_key = (
        "instruction_level_strict_acc"
        if args.score_metric == "instruction_level"
        else "prompt_level_strict_acc"
    )

    for layer in range(layer_start, layer_end):
        for head in range(head_start, head_end):
            if str(layer) in results and str(head) in results[str(layer)]:
                continue

            t0 = time.time()
            head_config = {f"{layer}": [head]}
            pasta = Qwen3PASTA(
                model=model,
                tokenizer=tokenizer,
                head_config=head_config,
                alpha=ALPHA,
                scale_position="exclude",
            )

            generations: list[str] = []
            for i in range(0, len(val_samples), batch_size):
                batch_texts = formatted_texts[i:i + batch_size]
                batch_blocks = instruction_blocks[i:i + batch_size]

                inputs, offset_mapping = pasta.inputs_from_batch(batch_texts)
                inputs = inputs.to(DEVICE)

                with pasta.apply_steering(
                    model=model,
                    strings=batch_texts,
                    substrings=[batch_blocks],
                    model_input=inputs,
                    offsets_mapping=offset_mapping,
                ) as steered_model:
                    out = steered_model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens_generated,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                out = out[:, inputs.input_ids.shape[1]:]
                generations.extend(
                    tokenizer.batch_decode(out, skip_special_tokens=True)
                )

            eval_results = evaluate_generations(generations, val_samples)
            agg = compute_aggregate_metrics(eval_results, n_bootstrap=200)
            score = agg[score_key]

            dt = time.time() - t0
            print(
                f"layer={layer:>3} head={head:>3} {args.score_metric}={score:.3f} "
                f"prompt_strict={agg['prompt_level_strict_acc']:.3f} ({dt:.1f}s)"
            )

            results.setdefault(str(layer), {})[str(head)] = {
                "score": score,
                "prompt_level_strict_acc": agg["prompt_level_strict_acc"],
                "instruction_level_strict_acc": agg["instruction_level_strict_acc"],
                "n_samples": agg["n_samples"],
                "n_instructions": agg["n_instructions"],
                "time": dt,
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            del pasta
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Done. Total {time.time() - start_time:.1f}s. Results at {results_path}")


if __name__ == "__main__":
    main()
