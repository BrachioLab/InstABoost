import os
import gc
from typing import List, Callable, Tuple

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from jaxtyping import Int

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
) -> Int[Tensor, 'batch_size seq_len']:
    # Convert each instruction into a chat message format
    messages = [{"role": "user", "content": instruction} for instruction in instructions]
    
    # Apply the chat template using the tokenizer's built-in method
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_instructions = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=add_generation_prompt) for msg in messages]
        add_special_tokens = False
    else:
        print(f"Either apply_chat_template set to False or no chat template available. Using basic tokenization.")
        formatted_instructions = [msg["content"] for msg in messages]
        add_special_tokens = True

    return tokenizer(formatted_instructions, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

def tokenize_instructions_harmful(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    add_generation_prompt: bool = True,
    apply_chat_template: bool = True,
) -> Int[Tensor, 'batch_size seq_len']:
    # Convert each instruction into a chat message format
    messages = [{"role": "user", "content": instruction} for instruction in instructions]
    
    # Apply the chat template using the tokenizer's built-in method
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_instructions = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=add_generation_prompt) + "Sure," for msg in messages]
    else:
        print(f"Either apply_chat_template set to False or no chat template available. Using basic tokenization.")
        formatted_instructions = [msg["content"] for msg in messages]
    
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
    
    # Track which sequences are still generating
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=toks.device)
    gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=toks.device)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for i in range(max_tokens_generated):
        if not active_sequences.any():
            break
            
        with model.hooks(fwd_hooks=fwd_hooks):
            names = lambda hook_name: 'resid' in hook_name
            with torch.no_grad():
                logits, cache = model.run_with_cache(all_toks[:, :-max_tokens_generated + i].to("cuda"), names_filter=names)
                
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

            eos_mask = (next_tokens == model.tokenizer.eos_token_id)
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
            
    return model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

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
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
            cache_dir=os.path.join(cache_dir, f"samples_{i}-{i+batch_size}") if cache_dir is not None else None
        )
        generations.extend(generation)
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
    for i in tqdm(range(0, len(harmful_inst), batch_size), desc="Getting hidden states"):
        #print(f"Batch {i//batch_size} of {len(harmful_inst)//batch_size}")
        gc.collect(); torch.cuda.empty_cache()
        harmful_toks = tokenize_instructions_fn(instructions=harmful_inst[i:i+batch_size])
        harmless_toks = tokenize_instructions_fn(instructions=harmless_inst[i:i+batch_size])

        assert (harmful_toks[:, -1] != model.tokenizer.eos_token_id).all(), "Right padding tokens found in harmful instructions"
        assert (harmless_toks[:, -1] != model.tokenizer.eos_token_id).all(), "Right padding tokens found in harmless instructions"

        harmful_logits_batch, harmful_cache_batch = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        resid_pres_harmful.append(harmful_cache_batch['resid_pre', layer][:, pos, :].cpu())
        del harmful_toks, harmful_logits_batch, harmful_cache_batch
        gc.collect(); torch.cuda.empty_cache()

        harmless_logits_batch, harmless_cache_batch = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        resid_pres_harmless.append(harmless_cache_batch['resid_pre', layer][:, pos, :].cpu())
        del harmless_toks, harmless_logits_batch, harmless_cache_batch
        gc.collect(); torch.cuda.empty_cache()

    resid_pres_harmful = torch.cat(resid_pres_harmful)
    resid_pres_harmless = torch.cat(resid_pres_harmless)

    print(resid_pres_harmful.shape)
    print(resid_pres_harmless.shape)

    return resid_pres_harmful, resid_pres_harmless
