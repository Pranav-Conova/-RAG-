"""Structured LLM output with thinking control.

Kept as a thin wrapper for existing call sites; `LLM.structured` now supports `thinking`
directly (default: settings.llm_thinking, off).
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def structured(
    llm,
    messages: list[dict],
    schema: type[T],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking: bool = True,
    retries: int = 1,
) -> T:
    return llm.structured(messages, schema, model=model, max_tokens=max_tokens,
                          retries=retries, thinking=thinking)
