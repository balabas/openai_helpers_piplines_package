"""Fake OpenAI-compatible streaming chat backend for pipeline tests.

The real pipeline wraps ``client.chat.completions`` and consumes a *stream* of
chunk objects (``stream=True`` is forced internally).  These helpers build that
chunk stream from high-level specs so each exception path can be triggered
deterministically without a live model.

A "turn" is one streamed response: a list of chunk dicts.  ``FakeChatCompletions``
serves scripted turns in order, one per ``create()`` call — which is exactly how
a multi-round tool loop or a retry drives several model calls.
"""
from __future__ import annotations

import json
from typing import Any


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    response_id: str = "chatcmpl-fake",
    model: str = "fake-model",
    created: int = 0,
) -> dict[str, Any]:
    """Build one streaming chunk in OpenAI delta format."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk: dict[str, Any] = {
        "id": response_id,
        "model": model,
        "created": created,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def text_turn(
    text: str,
    *,
    finish_reason: str = "stop",
    reasoning: str | None = None,
    n_chunks: int = 1,
    usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """A streamed assistant text answer, optionally split across ``n_chunks``."""
    chunks: list[dict[str, Any]] = []
    if reasoning:
        chunks.append(_chunk(reasoning=reasoning))
    if n_chunks <= 1 or len(text) <= 1:
        parts = [text]
    else:
        size = max(1, len(text) // n_chunks)
        parts = [text[i : i + size] for i in range(0, len(text), size)]
    for part in parts:
        chunks.append(_chunk(content=part))
    chunks.append(
        _chunk(
            content="",
            finish_reason=finish_reason,
            usage=usage or {"prompt_tokens": 1, "completion_tokens": 1},
        )
    )
    return chunks


def tool_call_turn(
    name: str,
    arguments: dict[str, Any] | str,
    *,
    call_id: str = "call_1",
    content: str = "",
    finish_reason: str = "tool_calls",
) -> list[dict[str, Any]]:
    """A streamed assistant turn requesting a single tool call."""
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    tool_calls = [
        {
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
    ]
    return [
        _chunk(content=content, tool_calls=tool_calls),
        _chunk(content="", finish_reason=finish_reason),
    ]


def loop_turn(numbers: int = 800) -> list[dict[str, Any]]:
    """A streamed turn whose content is a degenerate numeric counting loop.

    Triggers ``LoopGuardPipeline`` during streaming (numeric-list detector).
    """
    body = ", ".join(str(i) for i in range(numbers)) + ", "
    return [_chunk(content=body), _chunk(content="", finish_reason="stop")]


class FakeChatCompletions:
    """Sync ``chat.completions`` stand-in serving scripted streamed turns.

    Each ``create()`` returns the next turn as a plain list; the pipeline
    consumes it through its sync-iterator path.
    """

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        if self._index >= len(self._turns):
            raise AssertionError(
                f"FakeChatCompletions ran out of scripted turns after {self._index} calls"
            )
        turn = self._turns[self._index]
        self._index += 1
        return list(turn)


class FakeProviderError(Exception):
    """Mimics an OpenAI SDK error: carries a parsed ``body`` like the real one."""

    def __init__(self, message: str, *, body: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.body = body


class FailingChatCompletions:
    """Backend whose ``create()`` always raises, for error-surface tests."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.error


class AsyncFakeChatCompletions:
    """Async ``chat.completions`` stand-in exercising the ``__aiter__`` path."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._index >= len(self._turns):
            raise AssertionError("AsyncFakeChatCompletions ran out of scripted turns")
        turn = self._turns[self._index]
        self._index += 1

        async def _gen() -> Any:
            for chunk in turn:
                yield chunk

        return _gen()
