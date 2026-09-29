"""PASTA per-(layer, head) scoring sweep for Qwen3.

Mirrors src/pasta.py but adds the small set of changes needed to make pastalib
work with Qwen3:

  1. Qwen3ForCausalLM is registered in PASTA's setup_model.
  2. The attention-module path and attention_mask arg index are added to the
     two class-level dicts.
  3. The size-lying override (which keeps downstream HF code from seeing the
     per-head-expanded mask) is extended to qwen3.
  4. The model is loaded with attn_implementation="eager" so the additive
     attention_mask actually reaches Qwen3Attention.forward.
  5. The chat template is applied with enable_thinking=False by default so
     the val generations stay short.

Output JSON shape is identical to src/pasta.py — same {layer: {head: {score, time}}}
structure, written to:
    {output_dir}/{model_short}/pasta/layer_head_selection_results.json
or, when a shard range is given:
    {output_dir}/{model_short}/pasta/layer_head_selection_results.shard_{a}_{b}.json
so multiple machines can sweep different layer ranges in parallel for the
same task without racing on the same JSON. Merge the shards by hand or with a
small dict-update script before feeding into pasta_head_selection.py.
"""
from __future__ import annotations

import functools
import gc
import json
import os
import random
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import transformers
from pastalib.pasta import PASTA
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import DATASET_CHOICES, DATASET_CONFIGS, get_dataset
from evaluation_utils import evaluate_generations
from model_interaction_utils import tokenize_instructions


class Qwen3PASTA(PASTA):
    """PASTA subclass that knows about Qwen3ForCausalLM.

    Why a subclass instead of monkey-patching pastalib globally: the size-lying
    branch in edit_multisection_attention is a literal `if model_name in [...]`,
    so we override that one method (verbatim copy + qwen3 added to the list)
    plus setup_model. ATTN_MODULE_NAME / ATTENTION_MASK_ARGIDX are class-level
    dicts so we update those in-place at import time.
    """

    def setup_model(self, model):
        cls_name = model.__class__.__name__
        if cls_name == "Qwen3ForCausalLM":
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
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output_dir", type=str, default="results_pasta")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Task to score heads on",
    )
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--baseline_generations_dir", type=str, default=None)
    parser.add_argument("--answer_file_name", type=str, default="new_answer.txt")
    parser.add_argument("--steer_to_pt", action="store_true")
    parser.add_argument("--cache_harmless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--use_fluency", action="store_true")
    parser.add_argument("--use_relevance", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Apply Qwen3 chat template with thinking enabled (default: disabled)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=None,
        help="Inclusive start layer for sharded sweeps (default: 0)",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=None,
        help="Exclusive end layer for sharded sweeps (default: total_layers)",
    )
    parser.add_argument(
        "--head_start",
        type=int,
        default=None,
        help="Inclusive start head (default: 0). Mainly for smoke tests.",
    )
    parser.add_argument(
        "--head_end",
        type=int,
        default=None,
        help="Exclusive end head (default: n_heads). Mainly for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = args.dataset
    MODEL_PATH = args.model_path
    DEVICE = args.device
    batch_size = args.batch_size
    ALPHA = args.alpha

    cfg = DATASET_CONFIGS[dataset]
    max_tokens_generated = cfg["max_tokens_generated"]
    N_INST_TRAIN = cfg["n_inst_train"]
    N_INST_TEST = cfg["n_inst_test"]
    N_INST_VAL = cfg["n_inst_val"]
    eval_type = cfg["eval_type"]
    apply_chat_template = cfg.get("apply_chat_template", True)

    model_short = MODEL_PATH.split("/")[-1]
    output_dir = os.path.join(args.output_dir, model_short)
    baseline_generations_dir = args.baseline_generations_dir or output_dir

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

    tokenize_instructions_fn = functools.partial(
        tokenize_instructions,
        tokenizer=tokenizer,
        model_name=MODEL_PATH,
        apply_chat_template=apply_chat_template,
    )

    total_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Model: {model.__class__.__name__} | layers={total_layers} | query heads={n_heads}")

    layer_start = 0 if args.layer_start is None else args.layer_start
    layer_end = total_layers if args.layer_end is None else args.layer_end
    sharded = args.layer_start is not None or args.layer_end is not None
    if not (0 <= layer_start < layer_end <= total_layers):
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) for total_layers={total_layers}"
        )

    print("Loading dataset")
    (
        harmful_inst_train,
        harmless_inst_train,
        steer_dataset,
        steer_val_dataset,
        steer_questions,
        steer_answers,
        steer_alt_answers,
        steer_val_questions,
        steer_val_answers,
        steer_val_alt_answers,
        eval_args,
        steer_prompt,
    ) = get_dataset(
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        answer_file_name=args.answer_file_name,
        n_inst_train=N_INST_TRAIN,
        n_inst_test=N_INST_TEST,
        n_inst_val=N_INST_VAL,
        steer_to_pt=args.steer_to_pt,
        tokenizer=tokenizer,
        model=model,
        baseline_generations_dir=baseline_generations_dir,
        cache_harmless=args.cache_harmless,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_tokens_generated=max_tokens_generated,
        batch_size=batch_size,
    )

    pasta_output_dir = os.path.join(output_dir, "pasta")
    os.makedirs(pasta_output_dir, exist_ok=True)
    if sharded:
        results_path = os.path.join(
            pasta_output_dir,
            f"layer_head_selection_results.shard_{layer_start}_{layer_end}.json",
        )
    else:
        results_path = os.path.join(pasta_output_dir, "layer_head_selection_results.json")
    print(f"Writing results to {results_path}")

    results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        print(f"Resuming — {sum(len(v) for v in results.values())} (layer, head) entries already scored")

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not args.enable_thinking:
        template_kwargs["enable_thinking"] = False

    head_start = 0 if args.head_start is None else args.head_start
    head_end = n_heads if args.head_end is None else args.head_end

    for layer in range(layer_start, layer_end):
        for head in range(head_start, head_end):
            if str(layer) in results and str(head) in results[str(layer)]:
                continue

            head_t0 = time.time()
            head_config = {f"{layer}": [head]}
            pasta = Qwen3PASTA(
                model=model,
                tokenizer=tokenizer,
                head_config=head_config,
                alpha=ALPHA,
                scale_position="exclude",
            )

            def _to_user_str(sample):
                if isinstance(sample, str):
                    return sample
                user_parts = [m["content"] for m in sample if m.get("role") == "user"]
                return "\n\n".join(user_parts)

            emphasized_texts = [steer_prompt]
            texts = [steer_prompt + _to_user_str(sample) for sample in steer_val_dataset]
            messages = [{"role": "user", "content": text} for text in texts]
            if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
                texts = [
                    tokenizer.apply_chat_template([msg], **template_kwargs)
                    for msg in messages
                ]

            val_steer_generations = []
            for i in range(0, len(steer_val_dataset), batch_size):
                batch_texts = texts[i:i + batch_size]
                emphasized_texts = [steer_prompt] * len(batch_texts)

                inputs, offset_mapping = pasta.inputs_from_batch(batch_texts)
                inputs = inputs.to(DEVICE)

                with pasta.apply_steering(
                    model=model,
                    strings=batch_texts,
                    substrings=emphasized_texts,
                    model_input=inputs,
                    offsets_mapping=offset_mapping,
                ) as steered_model:
                    out = steered_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens_generated,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                out = out[:, inputs.input_ids.shape[1]:]
                val_steer_generations.extend(
                    tokenizer.batch_decode(out, skip_special_tokens=True)
                )

            val_results = evaluate_generations(
                generations=val_steer_generations,
                dataset=dataset,
                eval_type=eval_type,
                eval_args=eval_args,
                questions=steer_val_questions,
                answers=steer_val_answers,
                alt_answers=steer_val_alt_answers,
                is_validation=True,
                use_fluency=args.use_fluency,
                use_relevance=args.use_relevance,
                steer_dataset=steer_val_dataset,
            )

            head_dt = time.time() - head_t0
            score = val_results["score"]
            print(
                f"layer={layer:>3} head={head:>3} score={score:.3f} "
                f"({head_dt:.1f}s)"
            )

            results.setdefault(str(layer), {})[str(head)] = {
                "score": score,
                "time": head_dt,
            }
            if args.use_fluency:
                results[str(layer)][str(head)]["fluency"] = val_results["fluency"]

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            del pasta
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Done. Total {time.time() - start_time:.1f}s. Results at {results_path}")


if __name__ == "__main__":
    main()
