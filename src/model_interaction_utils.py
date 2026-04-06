import os
import gc
import re
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

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
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

    return tokenizer(formatted_instructions, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

def tokenize_instructions_harmful(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
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
    
    return tokenizer(formatted_instructions, padding=True, return_tensors="pt", add_special_tokens=False).input_ids

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    cache_dir: str = None,
) -> List[str]:
    batch_size = toks.shape[0]
    all_toks = torch.zeros((batch_size, toks.shape[1] + max_tokens_generated), dtype=torch.long, device="cuda")
    all_toks[:, :toks.shape[1]] = toks.to("cuda")
    stop_token_ids = _get_stop_token_ids(model)
    
    # Track which sequences are still generating
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=toks.device)
    gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=toks.device)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for i in range(max_tokens_generated):
        if not active_sequences.any():
            break
            
        with model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                logits = model(
                    all_toks[:, :-max_tokens_generated + i].to("cuda"),
                    return_type="logits",
                )
                
            #if cache_dir is not None and i == 0:
            #    # Only save the last token cache
            #    cache = cache.to("cpu")
            #    for k in cache.cache_dict:
            #        cache.cache_dict[k] = cache.cache_dict[k][:, -1, :]
            #    np.savez(os.path.join(cache_dir, f"token_{i}.npz"), cache.cache_dict)
            
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            
            # Update active sequences and generation lengths
            if i == 0:
                active_sequences = active_sequences.to(next_tokens.device)
                gen_lengths = gen_lengths.to(next_tokens.device)

            eos_mask = torch.isin(
                next_tokens,
                torch.tensor(list(stop_token_ids), device=next_tokens.device),
            )
            active_sequences = active_sequences & ~eos_mask
            gen_lengths = gen_lengths + active_sequences.long()
            
            # Only update tokens for active sequences
            all_toks[active_sequences, -max_tokens_generated+i] = next_tokens[active_sequences]

    # Decode only the generated portion for each sequence
    generated_tokens = []
    for b in range(batch_size):
        seq_len = gen_lengths[b].item()
        if seq_len > 0:
            generated_tokens.append(all_toks[b, toks.shape[1]:toks.shape[1] + seq_len])
        else:
            generated_tokens.append(torch.tensor([], dtype=torch.long, device=toks.device))
            
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
    cache_dir: str = None
) -> List[str]:

    # assert batch_size == 1, "Batch size must be 1 for now"
    generations = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating completions"):
        #print("Prompt:", instructions[i])
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        if fwd_hooks:
            generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
                cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None
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
