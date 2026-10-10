"""Dependency-free helpers for completion-only supervised fine tuning."""

from __future__ import annotations

from typing import Any, Sequence


def build_completion_only_sample(
    prompt_ids: Sequence[int], completion_ids: Sequence[int], eos_token_id: int, max_length: int
) -> dict[str, list[int]] | None:
    """Return a fully masked prompt plus complete SQL/EOS, or None if it is too long.

    Neither schema/prompt nor SQL is truncated. A discarded example can never
    silently produce all -100 labels or teach a partial SQL statement.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if not completion_ids:
        raise ValueError("SQL completion must contain at least one token")
    prompt = list(prompt_ids)
    completion = list(completion_ids) + [eos_token_id]
    input_ids = prompt + completion
    if len(input_ids) > max_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt) + completion,
    }


def encode_sql_completion(
    tokenizer: Any, prompt: str, completion: str, max_length: int
) -> dict[str, list[int]] | None:
    if not isinstance(prompt, str) or not isinstance(completion, str) or not completion.strip():
        raise ValueError("Prompt must be a string and SQL completion must be nonempty")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(completion.strip(), add_special_tokens=False)["input_ids"]
    return build_completion_only_sample(prompt_ids, sql_ids, tokenizer.eos_token_id, max_length)
