"""Minimal LLM-judge client used by prompts.py.

Provides ``APIModel`` and ``PromptedLLM`` with the same interface and prompt
framing as the wrapper previously used for the Gemini judges: the judge
instructions are followed by "FINAL ANSWER:" guidance, the input is passed as
"Text input: ...", decoding is greedy, and the final answer is extracted from
the response with the same set of patterns.

Requires ``GOOGLE_API_KEY`` (Gemini is accessed through its OpenAI-compatible
endpoint) and the ``openai`` package.
"""
from __future__ import annotations

import os
import re
import time

from openai import OpenAI

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class APIModel:
    def __init__(self, model_name: str, max_attempts: int = 5, delay_seconds: float = 2.0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set for the LLM judge")
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.client = OpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 10000) -> str:
        last_exception = None
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    n=1,
                )
                return response.choices[0].message.content or ""
            except Exception as error:  # noqa: BLE001 - retry on any API error
                last_exception = error
                if attempt < self.max_attempts - 1:
                    print(
                        f"Encountered {error.__class__.__name__} on attempt "
                        f"{attempt + 1}/{self.max_attempts}. Waiting {self.delay_seconds} seconds before retrying..."
                    )
                    time.sleep(self.delay_seconds)
        print(f"API call failed after {self.max_attempts} attempts for model {self.model_name}.")
        raise last_exception


_ANSWER_PATTERNS = [
    ("\\[ \\boxed{", r"\[ \\boxed{(.*)}"),
    ("**FINAL ANSWER:**", r"\*\*FINAL ANS.*:\*\*(.*)(?:<|$)"),
    ("*FINAL ANSWER:*", r"\*FINAL ANS.*:(.*)(?:<|$)"),
    ("**Final Answer:**", r"\*\*Final Ans.*:\*\*(.*)(?:<|$)"),
    ("*Final Answer:*", r"\*Final Ans.*:(.*)(?:<|$)"),
    ("Final answer:", r"Final ans.*:(.*)(?:<|$)"),
    ("*Final answer:*", r"\*Final ans.*:(.*)(?:<|$)"),
    ("**Final answer:**", r"\*\*Final ans.*:\*\*(.*)(?:<|$)"),
    ("**Answer:**", r"\*\*Answer:\*\*(.*)(?:<|$)"),
    ("*Answer:*", r"\*Answer:\*(.*)(?:<|$)"),
    ("**Answer**:", r"\*\*Answer\*\*:(.*)(?:<|$)"),
    ("*Answer*:", r"\*Answer\*:(.*)(?:<|$)"),
    ("FINAL ANSWER:", r"FINAL ANSWER:(.*)(?:<|$)"),
]


def extract_final_answer(output: str) -> str:
    try:
        pred = output.strip()
        for marker, pattern in _ANSWER_PATTERNS:
            if marker in output:
                pred = re.findall(pattern, output, re.DOTALL)[-1].strip()
                break
        if "```json" in pred:
            pred = re.findall(r"```json(.*?)```", pred, re.DOTALL)[-1]
        if "```" in pred:
            pred = re.sub(r"```", "", pred).strip()
        if "<|eot_id|>" in pred:
            pred = re.sub(r"<\|eot_id\|>", "", pred).strip()
        if "\\text{" in pred:
            pred = re.findall(r"\\text{(.*?)}", pred, re.DOTALL)[-1].strip()
        return pred
    except Exception:  # noqa: BLE001 - fall back to the raw output on parse failure
        return output.strip()


class PromptedLLM:
    def __init__(self, model: APIModel, prompt: str):
        self.model = model
        self.prompt = prompt

    def forward(self, str_input: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                        + "\nAt the end of your response, output just the answer after 'FINAL ANSWER:'.",
                    },
                    {"type": "text", "text": f"Text input: {str_input}"},
                ],
            }
        ]
        output = self.model.chat(messages, temperature=0.0)
        return extract_final_answer(output)
