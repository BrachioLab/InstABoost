import os
import gc
import re
import functools
from typing import Any, List, Callable, Tuple

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from jaxtyping import Int

from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformers import AutoTokenizer


def _normalize_instruction_messages(instruction):
    if isinstance(instruction, list):
        if not all(isinstance(message, dict) for message in instruction):
            raise TypeError("Chat instructions must be lists of message dicts")
        return instruction
    if isinstance(instruction, str):
        return [{"role": "user", "content": instruction}]
    raise TypeError(f"Unsupported instruction type: {type(instruction)!r}")


def apply_steer_prompt(instruction: Any, steer_prompt: str):
    if not steer_prompt:
        return instruction
    if isinstance(instruction, str):
        return steer_prompt + instruction
    if isinstance(instruction, list):
        normalized_messages = _normalize_instruction_messages(instruction)
        updated_messages = [dict(message) for message in normalized_messages]
        for message in updated_messages:
            if message.get("role") == "user":
                message["content"] = steer_prompt + message["content"]
                return updated_messages
        if not updated_messages:
            raise ValueError("Chat instructions cannot be empty")
        updated_messages[0]["content"] = steer_prompt + updated_messages[0]["content"]
        return updated_messages
    raise TypeError(f"Unsupported instruction type: {type(instruction)!r}")


def _get_stop_token_ids(model: HookedTransformer) -> set[int]:
    explicit_stop = getattr(model, "generation_stop_token_ids", None)
    if explicit_stop is not None:
        if isinstance(explicit_stop, int):
            explicit_stop = [explicit_stop]
        return {int(token_id) for token_id in explicit_stop if token_id is not None}

    ids: set[int] = set()
    eos = getattr(model.tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        ids.update(int(token_id) for token_id in eos if token_id is not None)
    elif eos is not None:
        ids.add(int(eos))

    # Many chat models (e.g. Llama-3-Instruct, Qwen3) ship multiple stop tokens
    # via GenerationConfig — typically both <eos> and <end_of_turn>/<eot>. The
    # tokenizer alone may only expose one of them, so merge what HF declares.
    try:
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig.from_pretrained(model.cfg.model_name)
        extra = gen_cfg.eos_token_id
        if isinstance(extra, (list, tuple)):
            ids.update(int(token_id) for token_id in extra if token_id is not None)
        elif extra is not None:
            ids.add(int(extra))
    except Exception:
        pass

    return ids


def _sample_next_tokens(
    logits: torch.Tensor,
    generation_config: dict | None,
) -> torch.Tensor:
    if not generation_config or not generation_config.get("do_sample", False):
        return logits.argmax(dim=-1)

    sampling_logits = logits.to(torch.float32)
    temperature = float(generation_config.get("temperature", 1.0))
    if temperature > 0 and temperature != 1.0:
        sampling_logits = sampling_logits / temperature

    top_k = int(generation_config.get("top_k", 0) or 0)
    if top_k > 0 and top_k < sampling_logits.shape[-1]:
        top_values, _ = torch.topk(sampling_logits, top_k, dim=-1)
        kth_values = top_values[:, -1].unsqueeze(-1)
        sampling_logits = torch.where(
            sampling_logits < kth_values,
            torch.full_like(sampling_logits, float("-inf")),
            sampling_logits,
        )

    top_p = float(generation_config.get("top_p", 1.0))
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(sampling_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        sampling_logits = torch.full_like(sampling_logits, float("-inf"))
        sampling_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)

    probs = torch.softmax(sampling_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _parse_gpt_oss_final_response(text: str) -> str:
    matches = re.findall(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if matches:
        return matches[0].strip()

    commentary_matches = re.findall(
        r"<\|channel\|>commentary<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if commentary_matches:
        return commentary_matches[0].strip()

    cleaned = re.sub(r"<\|[^>]+\|>", "", text)
    return cleaned.strip()


def _is_probably_degenerate_gpt_oss_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "<|" in stripped:
        return True
    if stripped.count("?") >= 4:
        return True
    if stripped.startswith(("The user", "The correct", "Chat")):
        return True
    if stripped in {"The", "We", "Ok?", "The?", "We?"}:
        return True
    nonspace = sum(not ch.isspace() for ch in stripped)
    alpha = sum(ch.isalpha() for ch in stripped)
    weird = sum(ch in "?*#[]{}|<>…" for ch in stripped)
    if nonspace >= 8 and alpha / max(nonspace, 1) < 0.35 and weird >= 2:
        return True
    if stripped.startswith(("**", "##", "|")) and alpha / max(nonspace, 1) < 0.45:
        return True
    return False


def _truncate_gpt_oss_degenerate_tail(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text.strip()

    kept_lines = []
    content_chars = 0
    for line in lines:
        stripped = line.strip()
        if content_chars >= 120 and _is_probably_degenerate_gpt_oss_line(stripped):
            break
        kept_lines.append(line)
        if stripped:
            content_chars += len(stripped)

    return "\n".join(kept_lines).strip()


def _decode_generated_tokens(model: HookedTransformer, generated_tokens: List[torch.Tensor]) -> List[str]:
    if model.cfg.model_name.startswith("openai/gpt-oss-"):
        decoded = [
            model.tokenizer.decode(tokens, skip_special_tokens=False)
            for tokens in generated_tokens
        ]
        return [_truncate_gpt_oss_degenerate_tail(_parse_gpt_oss_final_response(text)) for text in decoded]
    return model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def instaboost_pattern_hook_slice_renorm(
    pattern: torch.Tensor,
    hook: HookPoint,
    prompt_key_mask: torch.Tensor,
    multiplier: float = 3.0,
):
    """Instaboost hook matching the original ifeval_instr_last reference impl:
    multiply attention probabilities on prompt-key tokens by ``multiplier``,
    then L1-renormalise the row.

    Equivalent in pure math to ``prompt_attention_ablation_hook`` but uses
    exactly ``pattern[..., span] *= multiplier`` followed by
    ``F.normalize(pattern, p=1, dim=-1)``. This avoids the small floating-point
    drift introduced by the alternate formulation (``pattern * scale``,
    ``/ scaled_sum * original_sum``) which compounds across 32 layers and can
    flip greedy-decoded tokens. Used only by the IFEval runner.
    """
    if multiplier == 1.0:
        return pattern

    mask = prompt_key_mask.to(device=pattern.device, dtype=pattern.dtype)
    if mask.shape[-1] < pattern.shape[-1]:
        pad_width = pattern.shape[-1] - mask.shape[-1]
        mask = torch.nn.functional.pad(mask, (0, pad_width))
    elif mask.shape[-1] > pattern.shape[-1]:
        mask = mask[:, : pattern.shape[-1]]

    scale = 1 + (multiplier - 1) * mask[:, None, None, :]
    pattern = pattern * scale
    return torch.nn.functional.normalize(pattern, p=1, dim=-1)


def prompt_attention_ablation_hook(
    pattern: torch.Tensor,
    hook: HookPoint,
    prompt_key_mask: torch.Tensor,
    multiplier: float = 3.0,
):
    if multiplier == 1.0:
        return pattern

    mask = prompt_key_mask.to(device=pattern.device, dtype=pattern.dtype)
    if mask.shape[-1] < pattern.shape[-1]:
        pad_width = pattern.shape[-1] - mask.shape[-1]
        mask = torch.nn.functional.pad(mask, (0, pad_width))
    elif mask.shape[-1] > pattern.shape[-1]:
        mask = mask[:, :pattern.shape[-1]]

    scale = 1 + (multiplier - 1) * mask[:, None, None, :]
    original_visible_mass = pattern.sum(dim=-1, keepdim=True)
    scaled_pattern = pattern * scale
    scaled_visible_mass = scaled_pattern.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(
        scaled_visible_mass > 0,
        scaled_visible_mass,
        torch.ones_like(scaled_visible_mass),
    )
    scaled_pattern = scaled_pattern / safe_denominator
    return scaled_pattern * original_visible_mass


def spotlight_attn_scores_hook(
    attn_scores: torch.Tensor,
    hook: HookPoint,
    prompt_key_mask: torch.Tensor,
    target_attention: float,
):
    if target_attention <= 0:
        return attn_scores

    mask = prompt_key_mask.to(device=attn_scores.device, dtype=attn_scores.dtype)
    if mask.shape[-1] < attn_scores.shape[-1]:
        pad_width = attn_scores.shape[-1] - mask.shape[-1]
        mask = torch.nn.functional.pad(mask, (0, pad_width))
    elif mask.shape[-1] > attn_scores.shape[-1]:
        mask = mask[:, :attn_scores.shape[-1]]

    if not mask.any():
        return attn_scores

    attn_scores_float = attn_scores.to(torch.float32)
    mask_float = mask.to(torch.float32)
    attn_probs = attn_scores_float.softmax(dim=-1)
    current_attn = (attn_probs * mask_float[:, None, None, :]).sum(dim=-1)
    proportion = current_attn.clamp(min=0.0, max=1.0)
    boost_needed = proportion < target_attention

    if not boost_needed.any():
        return attn_scores

    bias = torch.zeros_like(proportion)
    bias[boost_needed] = torch.log(
        torch.full_like(proportion[boost_needed], target_attention) / (proportion[boost_needed] + 1e-8)
    )
    updated_scores = attn_scores_float + bias.unsqueeze(-1) * mask_float[:, None, None, :]
    return updated_scores.to(attn_scores.dtype)

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    return_attention_mask: bool = False,
) -> Int[Tensor, 'batch_size seq_len']:
    # Apply the chat template using the tokenizer's built-in method
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        formatted_instructions = [
            tokenizer.apply_chat_template(
                _normalize_instruction_messages(instruction),
                **chat_template_kwargs,
            )
            for instruction in instructions
        ]
        add_special_tokens = False
    else:
        print(f"Either apply_chat_template set to False or no chat template available. Using basic tokenization.")
        formatted_instructions = [
            instruction if isinstance(instruction, str) else instruction[-1]["content"]
            for instruction in instructions
        ]
        add_special_tokens = True

    encodings = tokenizer(
        formatted_instructions,
        padding=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    if return_attention_mask:
        return encodings.input_ids, encodings.attention_mask
    return encodings.input_ids

def tokenize_instructions_harmful(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    return_attention_mask: bool = False,
) -> Int[Tensor, 'batch_size seq_len']:
    # Apply the chat template using the tokenizer's built-in method
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        formatted_instructions = [
            tokenizer.apply_chat_template(
                _normalize_instruction_messages(instruction),
                **chat_template_kwargs,
            ) + "Sure,"
            for instruction in instructions
        ]
    else:
        print(f"Either apply_chat_template set to False or no chat template available. Using basic tokenization.")
        formatted_instructions = [
            instruction if isinstance(instruction, str) else instruction[-1]["content"]
            for instruction in instructions
        ]
    
    encodings = tokenizer(
        formatted_instructions,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if return_attention_mask:
        return encodings.input_ids, encodings.attention_mask
    return encodings.input_ids

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    cache_dir: str = None,
    prompt_key_mask: torch.Tensor | None = None,
    attention_pattern_multiplier: float | None = None,
    spotlight_target_attention: float | None = None,
    attention_mask: torch.Tensor | None = None,
    instaboost_legacy_renorm: bool = False,
    disable_kv_cache: bool = False,
) -> List[str]:
    device = model.cfg.device
    batch_size = toks.shape[0]
    prompt_toks = toks.to(device)
    stop_token_ids = _get_stop_token_ids(model)
    eos_token_id = model.tokenizer.eos_token_id
    generation_config = getattr(model, "generation_config_overrides", None)

    generated_toks = torch.zeros(
        (batch_size, max_tokens_generated),
        dtype=torch.long,
        device=device,
    )
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
    gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_kv_cache = (
        None
        if disable_kv_cache
        else HookedTransformerKeyValueCache.init_cache(model.cfg, device, batch_size)
    )
    # When KV cache is disabled we re-feed the full sequence every step (matches
    # the original ifeval_instr_last reference loop). ``running_toks`` tracks the
    # full prompt + generated tokens so the model can recompute attention from
    # scratch each step with hooks applied.
    running_toks = prompt_toks.clone() if disable_kv_cache else None
    current_toks = prompt_toks
    current_attention_mask = attention_mask.to(device) if attention_mask is not None else None

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for i in range(max_tokens_generated):
        if not active_sequences.any():
            break

        current_hooks = fwd_hooks
        if prompt_key_mask is not None and attention_pattern_multiplier is not None:
            instaboost_hook_fn = (
                instaboost_pattern_hook_slice_renorm
                if instaboost_legacy_renorm
                else prompt_attention_ablation_hook
            )
            prompt_attention_hook = functools.partial(
                instaboost_hook_fn,
                prompt_key_mask=prompt_key_mask,
                multiplier=attention_pattern_multiplier,
            )
            current_hooks = current_hooks + [
                (utils.get_act_name("pattern", layer), prompt_attention_hook)
                for layer in range(model.cfg.n_layers)
            ]
        if prompt_key_mask is not None and spotlight_target_attention is not None:
            spotlight_hook = functools.partial(
                spotlight_attn_scores_hook,
                prompt_key_mask=prompt_key_mask,
                target_attention=spotlight_target_attention,
            )
            current_hooks = current_hooks + [
                (utils.get_act_name("attn_scores", layer), spotlight_hook)
                for layer in range(model.cfg.n_layers)
            ]

        with model.hooks(fwd_hooks=current_hooks):
            with torch.no_grad():
                if disable_kv_cache:
                    model_kwargs = {}
                    if current_attention_mask is not None:
                        model_kwargs["attention_mask"] = current_attention_mask
                    logits = model(running_toks, return_type="logits", **model_kwargs)
                else:
                    model_kwargs = {"past_kv_cache": past_kv_cache}
                    if current_attention_mask is not None:
                        # TransformerLens expects attention_mask to match the current
                        # token chunk shape when using KV cache.
                        model_kwargs["attention_mask"] = current_attention_mask[:, -current_toks.shape[1]:]
                    logits = model(current_toks, return_type="logits", **model_kwargs)

        next_tokens = _sample_next_tokens(logits[:, -1, :], generation_config)
        active_sequences = active_sequences & ~torch.isin(
            next_tokens,
            torch.tensor(list(stop_token_ids), device=next_tokens.device),
        )
        gen_lengths = gen_lengths + active_sequences.long()
        generated_toks[active_sequences, i] = next_tokens[active_sequences]

        current_toks = next_tokens.unsqueeze(-1)
        current_toks[~active_sequences] = eos_token_id
        if disable_kv_cache:
            running_toks = torch.cat((running_toks, current_toks), dim=-1)
        if current_attention_mask is not None:
            next_attention = active_sequences.to(dtype=current_attention_mask.dtype).unsqueeze(-1)
            current_attention_mask = torch.cat((current_attention_mask, next_attention), dim=-1)

    generated_tokens = []
    for b in range(batch_size):
        seq_len = gen_lengths[b].item()
        if seq_len > 0:
            generated_tokens.append(generated_toks[b, :seq_len])
        else:
            generated_tokens.append(torch.tensor([], dtype=torch.long, device=device))

    return _decode_generated_tokens(model, generated_tokens)


def _generate_without_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
) -> List[str]:
    batch_size = toks.shape[0]
    device = model.cfg.device
    prompt_toks = toks.to(device)
    stop_token_ids = _get_stop_token_ids(model)
    eos_token_id = model.tokenizer.eos_token_id
    generation_config = getattr(model, "generation_config_overrides", None)

    generated_toks = torch.zeros(
        (batch_size, max_tokens_generated),
        dtype=torch.long,
        device=device,
    )
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
    gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, batch_size)
    current_toks = prompt_toks

    for i in range(max_tokens_generated):
        if not active_sequences.any():
            break

        with torch.no_grad():
            logits = model(
                current_toks,
                return_type="logits",
                past_kv_cache=past_kv_cache,
            )

        next_tokens = _sample_next_tokens(logits[:, -1, :], generation_config)
        active_sequences = active_sequences & ~torch.isin(
            next_tokens,
            torch.tensor(list(stop_token_ids), device=next_tokens.device),
        )
        gen_lengths = gen_lengths + active_sequences.long()
        generated_toks[active_sequences, i] = next_tokens[active_sequences]

        current_toks = next_tokens.unsqueeze(-1)
        current_toks[~active_sequences] = eos_token_id

    generated_tokens = []
    for b in range(batch_size):
        seq_len = gen_lengths[b].item()
        if seq_len > 0:
            generated_tokens.append(generated_toks[b, :seq_len])
        else:
            generated_tokens.append(torch.tensor([], dtype=torch.long, device=device))

    return _decode_generated_tokens(model, generated_tokens)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
    cache_dir: str = None,
    prompt_spans: List[Tuple[int, int, int]] | None = None,
    attention_pattern_multiplier: float | None = None,
    spotlight_target_attention: float | None = None,
    instaboost_legacy_renorm: bool = False,
    disable_kv_cache: bool = False,
) -> List[str]:

    if attention_pattern_multiplier is not None and spotlight_target_attention is not None:
        raise ValueError("Prompt-attention and spotlight interventions cannot be applied at the same time")

    # assert batch_size == 1, "Batch size must be 1 for now"
    generations = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating completions"):
        batch_instructions = instructions[i:i+batch_size]
        need_hooked_generation = bool(fwd_hooks) or (
            prompt_spans is not None and (
                attention_pattern_multiplier is not None or spotlight_target_attention is not None
            )
        )
        if need_hooked_generation:
            toks, attention_mask = tokenize_instructions_fn(
                instructions=batch_instructions,
                return_attention_mask=True,
            )
        else:
            toks = tokenize_instructions_fn(instructions=batch_instructions)
            attention_mask = None
        batch_prompt_spans = prompt_spans[i:i+batch_size] if prompt_spans is not None else None
        prompt_key_mask = None
        if batch_prompt_spans is not None:
            prompt_key_mask = torch.zeros((len(batch_prompt_spans), toks.shape[1]), dtype=torch.float32)
            for batch_idx, (start_idx, end_idx, token_count) in enumerate(batch_prompt_spans):
                pad_len = max(0, toks.shape[1] - token_count)
                start = max(0, pad_len + start_idx)
                end = min(toks.shape[1], pad_len + end_idx)
                if start < end:
                    prompt_key_mask[batch_idx, start:end] = 1.0

        has_padding_in_batch = (
            attention_mask is not None
            and len(batch_instructions) > 1
            and bool((attention_mask == 0).any().item())
        )
        # Hooked batched generation with padded prompts can leak padding context
        # across samples under KV-cache decoding. Isolate those batches per sample.
        if need_hooked_generation and has_padding_in_batch:
            generation = []
            for local_idx in range(len(batch_instructions)):
                # Re-tokenize single sample so we remove batch padding entirely.
                # Using sliced tensors from a padded batch leaks left-pad tokens
                # into the single-sample generation path and can hurt metrics.
                single_toks, single_attention_mask = tokenize_instructions_fn(
                    instructions=[batch_instructions[local_idx]],
                    return_attention_mask=True,
                )
                single_prompt_key_mask = None
                if batch_prompt_spans is not None:
                    start_idx, end_idx, token_count = batch_prompt_spans[local_idx]
                    single_prompt_key_mask = torch.zeros((1, single_toks.shape[1]), dtype=torch.float32)
                    pad_len = max(0, single_toks.shape[1] - token_count)
                    start = max(0, pad_len + start_idx)
                    end = min(single_toks.shape[1], pad_len + end_idx)
                    if start < end:
                        single_prompt_key_mask[0, start:end] = 1.0
                single_generation = _generate_with_hooks(
                    model,
                    single_toks,
                    max_tokens_generated=max_tokens_generated,
                    fwd_hooks=fwd_hooks if fwd_hooks else [],
                    cache_dir=(
                        os.path.join(cache_dir, f"samples_{i + local_idx}-{i + local_idx + 1}")
                        if cache_dir is not None
                        else None
                    ),
                    prompt_key_mask=single_prompt_key_mask,
                    attention_pattern_multiplier=attention_pattern_multiplier,
                    spotlight_target_attention=spotlight_target_attention,
                    attention_mask=single_attention_mask,
                    instaboost_legacy_renorm=instaboost_legacy_renorm,
                    disable_kv_cache=disable_kv_cache,
                )
                generation.extend(single_generation)
        elif fwd_hooks:
            generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
                cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None,
                prompt_key_mask=prompt_key_mask,
                attention_pattern_multiplier=attention_pattern_multiplier,
                spotlight_target_attention=spotlight_target_attention,
                attention_mask=attention_mask,
                instaboost_legacy_renorm=instaboost_legacy_renorm,
                disable_kv_cache=disable_kv_cache,
            )
        elif prompt_key_mask is not None and (
            attention_pattern_multiplier is not None or spotlight_target_attention is not None
        ):
            generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=[],
                cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None,
                prompt_key_mask=prompt_key_mask,
                attention_pattern_multiplier=attention_pattern_multiplier,
                spotlight_target_attention=spotlight_target_attention,
                attention_mask=attention_mask,
                instaboost_legacy_renorm=instaboost_legacy_renorm,
                disable_kv_cache=disable_kv_cache,
            )
        else:
            generation = _generate_without_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
            )
        generations.extend(generation)
        if need_hooked_generation:
            torch.cuda.empty_cache()

    return generations

def get_hiddens(
    model: HookedTransformer,
    harmful_inst: List[str],
    harmless_inst: List[str],
    layer: int,
    pos: int,
    tokenize_instructions_fn: Callable,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get hidden activations for harmful and harmless instructions.

    Args:
        model: The transformer model
        harmful_inst: List of harmful instructions
        harmless_inst: List of harmless instructions
        layer: Layer to get activations from
        pos: Position in sequence to get activations from
        tokenize_instructions_fn: Function to tokenize instructions
        batch_size: Batch size for processing

    Returns:
        Tuple of (harmful_activations, harmless_activations) tensors
    """
    resid_pres_harmful = []
    resid_pres_harmless = []
    target_hook_name = utils.get_act_name("resid_pre", layer)
    for i in tqdm(range(0, len(harmful_inst), batch_size), desc="Getting hidden states"):
        #print(f"Batch {i//batch_size} of {len(harmful_inst)//batch_size}")
        gc.collect(); torch.cuda.empty_cache()
        harmful_toks = tokenize_instructions_fn(instructions=harmful_inst[i:i+batch_size])
        harmless_toks = tokenize_instructions_fn(instructions=harmless_inst[i:i+batch_size])

        assert (harmful_toks[:, -1] != model.tokenizer.eos_token_id).all(), "Right padding tokens found in harmful instructions"
        assert (harmless_toks[:, -1] != model.tokenizer.eos_token_id).all(), "Right padding tokens found in harmless instructions"

        with torch.no_grad():
            harmful_logits_batch, harmful_cache_batch = model.run_with_cache(
                harmful_toks,
                names_filter=lambda hook_name: hook_name == target_hook_name,
            )
        resid_pres_harmful.append(harmful_cache_batch['resid_pre', layer][:, pos, :].cpu())
        del harmful_toks, harmful_logits_batch, harmful_cache_batch
        gc.collect(); torch.cuda.empty_cache()

        with torch.no_grad():
            harmless_logits_batch, harmless_cache_batch = model.run_with_cache(
                harmless_toks,
                names_filter=lambda hook_name: hook_name == target_hook_name,
            )
        resid_pres_harmless.append(harmless_cache_batch['resid_pre', layer][:, pos, :].cpu())
        del harmless_toks, harmless_logits_batch, harmless_cache_batch
        gc.collect(); torch.cuda.empty_cache()

    resid_pres_harmful = torch.cat(resid_pres_harmful)
    resid_pres_harmless = torch.cat(resid_pres_harmless)

    print(resid_pres_harmful.shape)
    print(resid_pres_harmless.shape)

    return resid_pres_harmful, resid_pres_harmless
