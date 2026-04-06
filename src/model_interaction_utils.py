import os
import gc
import re
import functools
from typing import List, Callable, Tuple

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


def _get_stop_token_ids(model: HookedTransformer) -> set[int]:
    stop_token_ids = getattr(model, "generation_stop_token_ids", None)
    if stop_token_ids is None:
        stop_token_ids = [model.tokenizer.eos_token_id]
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    return {int(token_id) for token_id in stop_token_ids if token_id is not None}


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
    attention_mask: torch.Tensor | None = None,
) -> List[str]:
    device = model.cfg.device
    batch_size = toks.shape[0]
    prompt_toks = toks.to(device)
    stop_token_ids = _get_stop_token_ids(model)
    eos_token_id = model.tokenizer.eos_token_id

    generated_toks = torch.zeros(
        (batch_size, max_tokens_generated),
        dtype=torch.long,
        device=device,
    )
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
    gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, batch_size)
    current_toks = prompt_toks

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for i in range(max_tokens_generated):
        if not active_sequences.any():
            break

        current_hooks = fwd_hooks
        if prompt_key_mask is not None and attention_pattern_multiplier is not None:
            prompt_attention_hook = functools.partial(
                prompt_attention_ablation_hook,
                prompt_key_mask=prompt_key_mask,
                multiplier=attention_pattern_multiplier,
            )
            current_hooks = current_hooks + [
                (utils.get_act_name("pattern", layer), prompt_attention_hook)
                for layer in range(model.cfg.n_layers)
            ]

        with model.hooks(fwd_hooks=current_hooks):
            with torch.no_grad():
                logits = model(
                    current_toks,
                    return_type="logits",
                    past_kv_cache=past_kv_cache,
                )

        next_tokens = logits[:, -1, :].argmax(dim=-1)
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

        next_tokens = logits[:, -1, :].argmax(dim=-1)
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
) -> List[str]:

    # assert batch_size == 1, "Batch size must be 1 for now"
    generations = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating completions"):
        #print("Prompt:", instructions[i])
        need_attention_mask = bool(fwd_hooks) or (
            prompt_spans is not None and attention_pattern_multiplier is not None
        )
        if need_attention_mask:
            toks, attention_mask = tokenize_instructions_fn(
                instructions=instructions[i:i+batch_size],
                return_attention_mask=True,
            )
        else:
            toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
            attention_mask = None
        prompt_key_mask = None
        if prompt_spans is not None:
            batch_prompt_spans = prompt_spans[i:i+batch_size]
            prompt_key_mask = torch.zeros((len(batch_prompt_spans), toks.shape[1]), dtype=torch.float32)
            for batch_idx, (start_idx, end_idx, token_count) in enumerate(batch_prompt_spans):
                pad_len = toks.shape[1] - token_count
                prompt_key_mask[batch_idx, pad_len + start_idx:pad_len + end_idx] = 1.0
        if fwd_hooks:
            generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
                cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None,
                prompt_key_mask=prompt_key_mask,
                attention_pattern_multiplier=attention_pattern_multiplier,
                attention_mask=attention_mask,
            )
        elif prompt_key_mask is not None and attention_pattern_multiplier is not None:
            generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=[],
                cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None,
                prompt_key_mask=prompt_key_mask,
                attention_pattern_multiplier=attention_pattern_multiplier,
                attention_mask=attention_mask,
            )
        else:
            generation = _generate_without_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
            )
        generations.extend(generation)
        if fwd_hooks:
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
