from __future__ import annotations

import json
import random
from typing import Any

import requests
from datasets import load_dataset

INSTRUCTION_HEADER = "Your response should follow the instructions below:"
MS_JSONL_URL = (
    "https://raw.githubusercontent.com/microsoft/llm-steer-instruct/"
    "refs/heads/main/data/ifeval_wo_instructions.jsonl"
)

_INSTRUCTIONS_REGISTRY = None


def clean_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (kwargs or {}).items() if value is not None}


def get_instructions_registry():
    global _INSTRUCTIONS_REGISTRY
    if _INSTRUCTIONS_REGISTRY is None:
        try:
            from ifeval_scripts import instructions_registry as imported_registry
        except ImportError as error:
            raise ImportError(
                "IFEval checkers could not be imported. Install IFEval extras: "
                "absl-py, langdetect, nltk, immutabledict."
            ) from error
        _INSTRUCTIONS_REGISTRY = imported_registry
    return _INSTRUCTIONS_REGISTRY


def _build_instruction_text(instruction_id: str, kwargs: dict[str, Any] | None) -> str | None:
    instructions_registry = get_instructions_registry()
    checker_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
    if checker_cls is None:
        return None

    checker = checker_cls(instruction_id)
    cleaned_kwargs = clean_kwargs(kwargs)
    try:
        return checker.build_description(**cleaned_kwargs)
    except TypeError:
        # Some checkers have no kwargs.
        return checker.build_description()


def build_instruction_block(instruction_texts: list[str]) -> str:
    bullets = "\n".join(f"- {text}" for text in instruction_texts)
    return f"{INSTRUCTION_HEADER}\n{bullets}"


def load_ifeval_samples(
    n_val: int = 100,
    n_test: int = 400,
    seed: int = 42,
    ms_jsonl_url: str = MS_JSONL_URL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build instruction-last IFEval samples by merging:
    - google/IFEval (instruction metadata)
    - microsoft/llm-steer-instruct JSONL (instruction-free base question)
    """
    response = requests.get(ms_jsonl_url, timeout=(10, 60))
    response.raise_for_status()
    ms_rows: dict[int, dict[str, Any]] = {}
    for line in response.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ms_rows[row["key"]] = row

    ds = load_dataset("google/IFEval", split="train")

    samples: list[dict[str, Any]] = []
    for row in ds:
        key = row["key"]
        if key not in ms_rows:
            continue

        base_question = ms_rows[key]["model_output"]
        if not base_question.strip():
            continue

        valid_ids: list[str] = []
        valid_kwargs: list[dict[str, Any]] = []
        instruction_texts: list[str] = []

        for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            text = _build_instruction_text(instruction_id, kwargs)
            if text is None:
                continue
            valid_ids.append(instruction_id)
            valid_kwargs.append(kwargs)
            instruction_texts.append(text)

        if not instruction_texts:
            continue

        instruction_block = build_instruction_block(instruction_texts)
        full_input = base_question + "\n\n" + instruction_block

        samples.append(
            {
                "key": key,
                "original_prompt": row["prompt"],
                "base_question": base_question,
                "instruction_id_list": valid_ids,
                "kwargs_list": valid_kwargs,
                "instruction_texts": instruction_texts,
                "instruction_block": instruction_block,
                "full_input": full_input,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    val_samples = samples[:n_val]
    test_samples = samples[n_val : n_val + n_test]
    return val_samples, test_samples


def instruction_token_span(
    tokenizer: Any,
    full_input: str,
    instruction_block: str,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
) -> tuple[int, int, int]:
    """
    Map an instruction substring to its token span in the rendered chat input.

    Uses the tokenizer's offset mapping to find the exact tokens corresponding
    to `instruction_block` after the chat template has been applied. Returns
    `(token_start, token_end, total_token_count)`, where `total_token_count`
    matches the unpadded length produced by `tokenize_instructions` for the
    same `full_input` and chat-template kwargs (so consumers can correct for
    left padding inside batched generation).
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not expose apply_chat_template.")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "instruction_token_span requires a fast tokenizer for offset mapping."
        )

    chat_template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = enable_thinking

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_input}],
            **chat_template_kwargs,
        )
    except TypeError:
        chat_template_kwargs.pop("reasoning_effort", None)
        chat_template_kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_input}],
            **chat_template_kwargs,
        )

    char_start = rendered.find(instruction_block)
    if char_start < 0:
        raise ValueError("instruction_block not found in rendered prompt.")
    char_end = char_start + len(instruction_block)

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    token_indices = [
        i for i, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not token_indices:
        raise ValueError("No tokens mapped to instruction_block span.")

    token_start = token_indices[0]
    token_end = token_indices[-1] + 1
    token_count = len(encoded["input_ids"])
    return token_start, token_end, token_count


def get_instruction_prompt_spans(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    include_header: bool = True,
) -> list[tuple[int, int, int]]:
    """
    Build per-sample prompt spans for attention steering using offset-mapping
    on the rendered chat input.

    By default the boost region covers the full instruction block (header +
    bullets), matching the original instruction-last reference implementation.
    Set `include_header=False` to boost only the bullet items.
    """
    spans: list[tuple[int, int, int]] = []
    header_prefix = f"{INSTRUCTION_HEADER}\n"
    for sample in samples:
        instruction_block = sample.get("instruction_block", "")
        search_text = instruction_block
        if not include_header and instruction_block.startswith(header_prefix):
            search_text = instruction_block[len(header_prefix):]

        spans.append(
            instruction_token_span(
                tokenizer=tokenizer,
                full_input=sample["full_input"],
                instruction_block=search_text,
                add_generation_prompt=add_generation_prompt,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
        )
    return spans
