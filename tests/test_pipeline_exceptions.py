"""Exception-correctness tests for every pipeline error path.

Each path that can raise, retry, or degrade is driven through a fake streaming
backend (see ``fake_chat``) so the behaviour is deterministic and offline.

The tests are plain ``def test_*`` functions that run their coroutine via
``asyncio.run`` — no ``pytest-asyncio`` required.  Run with ``pytest tests`` or
directly: ``python tests/test_pipeline_exceptions.py``.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import sys
import tempfile

import httpx
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # fake_chat
sys.path.insert(0, _ROOT)   # package root

from fake_chat import (  # noqa: E402
    AsyncFakeChatCompletions,
    FakeChatCompletions,
    FakeProviderError,
    loop_turn,
    text_turn,
    tool_call_turn,
)
from openai_helpers_piplines_package import (  # noqa: E402
    EmptyAssistantOutputError,
    JsonFixPipeline,
    LoggerPipeline,
    LoopGuardPipeline,
    PipelineDebugStage,
    PipelineRequestError,
    StructuredOutputRepairExhaustedError,
    ToolPipeline,
    ToolIterationLimitExceededError,
    attempt,
    chat_session,
    classify_pipeline_error,
    pipelined_chat,
)
from openai import APIConnectionError, OpenAI  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _actions(result):
    """All ``level/action`` strings in the trace."""
    return [f"{e.level}/{e.action}" for e in (result.trace or [])]


def _live_client_and_model():
    if os.environ.get("OPENAI_HELPERS_LIVE_DEBUG_TESTS") != "1":
        pytest.skip("set OPENAI_HELPERS_LIVE_DEBUG_TESTS=1 to run live debug-injection tests")

    base_url = os.environ.get("OPENAI_HELPERS_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed-for-local-server")
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=30.0)
    model = os.environ.get("OPENAI_HELPERS_MODEL")
    if not model:
        try:
            models = client.models.list()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"live model server is not available: {type(exc).__name__}: {exc}")
        model = models.data[0].id if models.data else None
    if not model:
        pytest.skip("no live model available")
    return client, model


# --------------------------------------------------------------------------- #
# Baseline / happy paths
# --------------------------------------------------------------------------- #

def test_plain_text_response():
    backend = FakeChatCompletions([text_turn("hello world")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    assert result is not None
    assert result.response["choices"][0]["message"]["content"] == "hello world"
    assert result.parsed is None


def test_result_exposes_final_response_usage():
    usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    backend = FakeChatCompletions([text_turn("hello world", usage=usage)])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    assert result.usage == usage
    assert result.run_usage == usage


def test_result_exposes_aggregated_run_usage():
    backend = FakeChatCompletions(
        [
            tool_call_turn("add", {"a": 1, "b": 2}),
            text_turn(
                "done",
                usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            ),
        ]
    )
    backend._turns[0][-1]["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add"}],
            tool_sources=[{"add": lambda arguments: {"sum": arguments["a"] + arguments["b"]}}],
        )

    result = _run(go())
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
    assert result.run_usage == {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}


def test_attempt_is_root_package_helper():
    import openai_helpers_piplines_package as package

    assert package.attempt is attempt


def test_pipelined_chat_accepts_pipelines_keyword():
    backend = FakeChatCompletions([text_turn('{"x": 1}')])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline()])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "json"}], schema_dict={"x": int})

    result = _run(go())
    assert result.parsed == {"x": 1}


def test_pipelined_chat_rejects_pipelines_with_legacy_alias():
    backend = FakeChatCompletions([text_turn("hello")])

    try:
        pipelined_chat(backend, pipelines=[], layers=[])
    except ValueError as exc:
        assert "pipelines=" in str(exc)
        assert "legacy compatibility alias" in str(exc)
    else:
        raise AssertionError("expected ValueError when both pipeline keywords are passed")


def test_chat_session_accepts_initial_role_content():
    backend = FakeChatCompletions([text_turn("ok")])
    chat = pipelined_chat(backend, pipelines=[])
    session = chat_session(
        chat,
        role_content={"system": "use json"},
        model="demo-model",
    )

    async def go():
        return await session.step(role_content={"user": "say ok"})

    result = _run(go())

    assert result.response["choices"][0]["message"]["content"] == "ok"
    assert session.default_params == {"model": "demo-model"}
    assert backend.calls[0]["model"] == "demo-model"
    assert "role_content" not in backend.calls[0]
    assert backend.calls[0]["messages"] == [
        {"role": "system", "content": "use json"},
        {"role": "user", "content": "say ok"},
    ]


def test_chat_session_accepts_role_content_lists():
    backend = FakeChatCompletions([text_turn("ok")])
    chat = pipelined_chat(backend, pipelines=[])
    session = chat_session(
        chat,
        role_content=[
            {"system": "use json"},
            {"developer": "stay compact"},
        ],
    )

    async def go():
        return await session.step(
            role_content=[
                {"user": "say ok"},
            ],
        )

    _run(go())

    assert backend.calls[0]["messages"] == [
        {"role": "system", "content": "use json"},
        {"role": "developer", "content": "stay compact"},
        {"role": "user", "content": "say ok"},
    ]


def test_chat_session_rejects_invalid_role_content_list_items():
    backend = FakeChatCompletions([text_turn("ok")])
    chat = pipelined_chat(backend, pipelines=[])

    try:
        chat_session(chat, role_content=[{"system": "ok"}, "bad"])
    except TypeError as exc:
        assert "role_content list items must be dicts" in str(exc)
    else:
        raise AssertionError("expected TypeError for invalid role_content list item")


def test_async_backend_aiter_path():
    backend = AsyncFakeChatCompletions([text_turn("async answer", n_chunks=3)])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "async answer"


def test_reasoning_then_content_streamed():
    backend = FakeChatCompletions([text_turn("final", reasoning="let me think")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    msg = result.response["choices"][0]["message"]
    assert msg["content"] == "final"
    assert msg.get("reasoning_content") == "let me think"


# --------------------------------------------------------------------------- #
# JSON / schema pipeline
# --------------------------------------------------------------------------- #

def test_schema_success():
    backend = FakeChatCompletions([text_turn('{"x": 5}')])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "json"}],
            schema_dict={"x": int},
            return_trace=True,
        )

    result = _run(go())
    assert result.parsed == {"x": 5}
    assert "validation/parsed" in _actions(result)


def test_schema_invalid_then_retry_succeeds():
    backend = FakeChatCompletions([text_turn("not json at all"), text_turn('{"x": 7}')])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "json"}],
            schema_dict={"x": int},
            return_trace=True,
        )

    result = _run(go())
    assert result.parsed == {"x": 7}
    assert "validation/json_retry" in _actions(result)
    assert len(backend.calls) == 2


def test_schema_exhausted_raises_after_retries():
    # Model can't produce JSON -> exhausted repair must now raise instead of
    # fabricating a fallback value.
    backend = FakeChatCompletions([text_turn("still not json"), text_turn("still not json repair")])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            model="weak-model",
            messages=[{"role": "user", "content": "give me json"}],
            temperature=0.2,
            schema_dict={"x": int},
        )

    try:
        _run(go())
    except StructuredOutputRepairExhaustedError as exc:
        assert "Structured output repair exhausted" in str(exc)
        assert exc.attempts == 1
        assert classify_pipeline_error(exc) == PipelineRequestError.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED
        assert len(backend.calls) == 2
    else:
        raise AssertionError("expected ValueError after exhausted structured-output retries")


def test_empty_assistant_output_is_classified():
    backend = FakeChatCompletions([text_turn("")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "say something"}])

    try:
        _run(go())
    except EmptyAssistantOutputError as exc:
        assert str(exc) == "Empty assistant output"
        assert classify_pipeline_error(exc) == PipelineRequestError.EMPTY_ASSISTANT_OUTPUT
    else:
        raise AssertionError("expected ValueError for empty assistant output")


def _raise_at(stage, error, seen=None):
    def _debug(context):
        if context.get("debug_stage") == stage:
            if seen is not None:
                seen.append(context)
            return error
        return None

    return _debug


def test_debug_stage_can_raise_empty_assistant_output_at_package_site():
    seen = []
    backend = FakeChatCompletions([text_turn("")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            debug_generate_exception=_raise_at(
                PipelineDebugStage.EMPTY_ASSISTANT_OUTPUT,
                EmptyAssistantOutputError(),
                seen,
            ),
        )

    try:
        _run(go())
    except EmptyAssistantOutputError as exc:
        assert classify_pipeline_error(exc) == PipelineRequestError.EMPTY_ASSISTANT_OUTPUT
        assert len(backend.calls) == 1
        assert seen and seen[0]["debug_stage"] == PipelineDebugStage.EMPTY_ASSISTANT_OUTPUT
    else:
        raise AssertionError("expected debug-generated EmptyAssistantOutputError")


def test_debug_stage_can_raise_structured_repair_exhausted_at_package_site():
    seen = []
    backend = FakeChatCompletions([text_turn("not json"), text_turn("still not json")])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            schema_dict={"x": int},
            debug_generate_exception=_raise_at(
                PipelineDebugStage.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED,
                StructuredOutputRepairExhaustedError(attempts=1),
                seen,
            ),
        )

    try:
        _run(go())
    except StructuredOutputRepairExhaustedError as exc:
        assert exc.attempts == 1
        assert classify_pipeline_error(exc) == PipelineRequestError.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED
        assert len(backend.calls) == 2
        assert seen and seen[0]["debug_stage"] == PipelineDebugStage.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED
    else:
        raise AssertionError("expected debug-generated StructuredOutputRepairExhaustedError")


def test_inline_toolcall_text_with_schema_raises_after_repair_exhaustion():
    # Inline tool-call text is treated as invalid structured output and now
    # raises once the repair budget is exhausted.
    inline = "<tool_call> <function=add> <parameter=a> 3 </parameter> </function> </tool_call>"
    backend = FakeChatCompletions([text_turn(inline), text_turn(inline), text_turn(inline)])
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(), JsonFixPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            model="pseudo-xml",
            messages=[{"role": "user", "content": "add then json"}],
            tool_sources=[{"add": _add}],
            schema_dict={"answer": str},
        )

    try:
        _run(go())
    except StructuredOutputRepairExhaustedError as exc:
        assert "Structured output repair exhausted" in str(exc)
        assert exc.attempts == 2
        assert classify_pipeline_error(exc) == PipelineRequestError.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED
        assert len(backend.calls) == 3
    else:
        raise AssertionError("expected ValueError after repair exhaustion")


def test_schema_validation_error_is_retried():
    # Valid JSON, but wrong type for a required int field -> pydantic ValidationError,
    # which must be caught by the (ValueError-based) retry, not escape.
    backend = FakeChatCompletions([text_turn('{"x": "abc"}'), text_turn('{"x": 3}')])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "json"}],
            schema_dict={"x": int},
            return_trace=True,
        )

    result = _run(go())
    assert result.parsed == {"x": 3}
    assert "validation/json_retry" in _actions(result)


def test_schema_dict_without_jsonfix_raises():
    backend = FakeChatCompletions([text_turn("x")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "hi"}],
            schema_dict={"x": int},
        )

    try:
        _run(go())
    except ValueError as exc:
        assert "JsonFixPipeline" in str(exc)
    else:
        raise AssertionError("expected ValueError without JsonFixPipeline")


# --------------------------------------------------------------------------- #
# Tool pipeline
# --------------------------------------------------------------------------- #

def _add(a: int, b: int) -> dict:
    return {"sum": a + b}


def test_tool_call_then_final_answer():
    backend = FakeChatCompletions(
        [tool_call_turn("add", {"a": 1, "b": 2}), text_turn("the sum is 3")]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add"}],
            tool_sources=[{"add": _add}],
            return_trace=True,
        )

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "the sum is 3"
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].result == '{"sum": 3}'


def test_unknown_tool_is_graceful():
    backend = FakeChatCompletions(
        [tool_call_turn("subtract", {"a": 1, "b": 2}), text_turn("done")]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            tool_sources=[{"add": _add}],
        )

    result = _run(go())
    assert "Unknown tool" in result.tool_executions[0].result


def test_handler_exception_is_captured():
    def boom(a: int) -> dict:
        raise RuntimeError("kaboom")

    backend = FakeChatCompletions(
        [tool_call_turn("boom", {"a": 1}), text_turn("recovered")]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            tool_sources=[{"boom": boom}],
        )

    result = _run(go())
    assert "Tool execution failed" in result.tool_executions[0].result
    assert result.response["choices"][0]["message"]["content"] == "recovered"


def test_nonserializable_tool_result_is_stringified_not_failed():
    def now(a: int) -> dict:
        return {"ts": datetime.datetime(2026, 6, 28)}

    backend = FakeChatCompletions(
        [tool_call_turn("now", {"a": 1}), text_turn("ok")]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            tool_sources=[{"now": now}],
        )

    result = _run(go())
    out = result.tool_executions[0].result
    assert "Tool execution failed" not in out
    assert "2026-06-28" in out


def test_tool_sources_without_toolpipeline_raises():
    backend = FakeChatCompletions([text_turn("x")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "x"}],
            tool_sources=[{"add": _add}],
        )

    try:
        _run(go())
    except ValueError as exc:
        assert "ToolPipeline" in str(exc)
    else:
        raise AssertionError("expected ValueError without ToolPipeline")


def test_truncated_tool_result_returns_result_not_none():
    # The regression: finish_reason='length' after a tool round with retries
    # exhausted used to fall off the loop and return None.
    backend = FakeChatCompletions(
        [
            tool_call_turn("add", {"a": 1, "b": 2}),
            text_turn("", finish_reason="length"),
            text_turn("final answer"),
        ]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add"}],
            tool_sources=[{"add": _add}],
            return_trace=True,
        )

    result = _run(go())
    assert result is not None
    assert result.response["choices"][0]["message"]["content"] == "final answer"
    assert "llm_call/length_retry" in _actions(result)


def test_broken_empty_tool_result_heats_up_then_recovers():
    # Empty output after a tool call -> heat up temperature + "continue" nudge,
    # then recover (ported empty-streak heat-up from test_agent2 run_tool_loop).
    backend = FakeChatCompletions(
        [
            tool_call_turn("add", {"a": 1, "b": 2}),
            text_turn("", finish_reason="stop"),   # executions present but empty -> broken
            text_turn("final answer"),
        ]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add"}],
            tool_sources=[{"add": _add}],
            return_trace=True,
        )

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "final answer"
    assert "tool/empty_heat_up" in _actions(result)
    # first retry from greedy/unset heats to 0.1 (schedule: 0.1 * 2**(n-1))
    assert backend.calls[-1]["temperature"] == 0.1


def test_loop_guard_retry_escalates_temperature_no_injection():
    # Ported run_loop_guard: on loop -> discard partial, raise temperature,
    # re-roll with NO injected prompt. From greedy/unset the first retry -> 0.1.
    backend = FakeChatCompletions([loop_turn(), text_turn("clean answer")])
    chat = pipelined_chat(backend, pipelines=[LoopGuardPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "count"}],
            return_trace=True,
        )

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "clean answer"
    assert "loop_guard/retry" in _actions(result)
    # retry re-rolled at heated temperature (unset -> 0.1), and added NO message
    assert backend.calls[-1]["temperature"] == 0.1
    assert [m["role"] for m in backend.calls[-1]["messages"]] == ["user"]


def test_truncated_tool_result_boosts_max_tokens():
    # Ported run_tool_loop: truncation (finish_reason=length) -> boost max_tokens
    # and retry, no prompt injection.
    backend = FakeChatCompletions(
        [
            tool_call_turn("add", {"a": 1, "b": 2}),
            text_turn("", finish_reason="length"),   # truncated
            text_turn("done"),
        ]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add"}],
            tool_sources=[{"add": _add}],
            max_tokens=1000,
            return_trace=True,
        )

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "done"
    assert "llm_call/length_retry" in _actions(result)
    assert backend.calls[-1]["max_tokens"] == 1000 + 5000


def test_schema_appended_as_user_hint_not_system():
    # Ported run_validation: schema goes onto an existing message as a text hint,
    # never as a new system message.
    backend = FakeChatCompletions([text_turn('{"x": 1}')])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "give me json"}],
            schema_dict={"x": int},
        )

    result = _run(go())
    assert result.parsed == {"x": 1}
    sent = backend.calls[0]["messages"]
    assert all(m["role"] != "system" for m in sent)            # no system injected
    assert "Output format schema" in sent[0]["content"]        # appended to the user message
    assert "give me json" in sent[0]["content"]


def test_tool_iteration_safety_limit_raises_runtimeerror():
    # 100 consecutive tool-call rounds with no final text -> internal safety limit.
    turns = [tool_call_turn("add", {"a": 1, "b": 1}) for _ in range(100)]
    backend = FakeChatCompletions(turns)
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "loop"}],
            tool_sources=[{"add": _add}],
        )

    try:
        _run(go())
    except ToolIterationLimitExceededError as exc:
        assert "safety limit" in str(exc)
        assert exc.limit == 20
        assert classify_pipeline_error(exc) == PipelineRequestError.TOOL_ITERATION_LIMIT_EXCEEDED
    else:
        raise AssertionError("expected RuntimeError at tool iteration safety limit")


def test_debug_stage_can_raise_tool_iteration_limit_at_package_site():
    seen = []
    backend = FakeChatCompletions([tool_call_turn("add", {"a": 1, "b": 1}) for _ in range(20)])
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "loop"}],
            tool_sources=[{"add": _add}],
            debug_generate_exception=_raise_at(
                PipelineDebugStage.TOOL_ITERATION_LIMIT_EXCEEDED,
                ToolIterationLimitExceededError(limit=20),
                seen,
            ),
        )

    try:
        _run(go())
    except ToolIterationLimitExceededError as exc:
        assert exc.limit == 20
        assert classify_pipeline_error(exc) == PipelineRequestError.TOOL_ITERATION_LIMIT_EXCEEDED
        assert len(backend.calls) == 20
        assert seen and seen[0]["debug_stage"] == PipelineDebugStage.TOOL_ITERATION_LIMIT_EXCEEDED
    else:
        raise AssertionError("expected debug-generated ToolIterationLimitExceededError")


# --------------------------------------------------------------------------- #
# Loop-guard pipeline
# --------------------------------------------------------------------------- #

def test_loop_detected_then_retry_recovers():
    backend = FakeChatCompletions([loop_turn(), text_turn("clean answer")])
    chat = pipelined_chat(backend, pipelines=[LoopGuardPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "count"}],
            return_trace=True,
        )

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "clean answer"
    assert "loop_guard/retry" in _actions(result)


def _system_only_at_front(messages):
    """True if no `system` message appears after a non-system message."""
    seen_non_system = False
    for m in messages:
        if m.get("role") == "system":
            if seen_non_system:
                return False
        else:
            seen_non_system = True
    return True


def test_json_repair_does_not_inject_midconversation_system():
    # Regression: a JSON-repair retry must NOT append a `role: system` message
    # mid-conversation -- strict chat templates (Qwen, gpt-oss) reject it with
    # HTTP 400 ("System message must be at the beginning").
    backend = FakeChatCompletions(
        [
            tool_call_turn("add", {"a": 3, "b": 4}),
            text_turn("not json"),        # triggers JSON repair (tools then disabled)
            text_turn('{"answer": "7"}'),
        ]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(), JsonFixPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add then json"}],   # no system from the user
            tool_sources=[{"add": _add}],
            schema_dict={"answer": str},
            return_trace=True,
        )

    result = _run(go())
    assert result.parsed == {"answer": "7"}
    for sent in backend.calls:
        assert _system_only_at_front(sent["messages"]), sent["messages"]


def test_loop_retry_does_not_inject_midconversation_system():
    backend = FakeChatCompletions([loop_turn(), text_turn("short answer")])
    chat = pipelined_chat(backend, pipelines=[LoopGuardPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "count"}],
        )

    _run(go())
    for sent in backend.calls:
        assert _system_only_at_front(sent["messages"]), sent["messages"]


def test_request_failure_surfaces_params_messages_and_provider_text():
    # A failing create() should raise PipelineRequestError carrying the request
    # context (params + message history) and the provider's own error text,
    # instead of letting the raw SDK error/traceback bubble up.
    err = FakeProviderError(
        "Error code: 400",
        body={
            "error": {
                "code": 400,
                "message": "System message must be at the beginning.",
                "type": "invalid_request_error",
            }
        },
    )
    backend = FakeChatCompletions([text_turn("ok")])
    chat = pipelined_chat(backend, pipelines=[ToolPipeline()])

    async def go():
        return await chat.create(
            model="demo-model",
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "compute 3 + 4 then return JSON"},
            ],
            temperature=0.1,
            tool_sources=[{"add": _add}],
            debug_generate_exception=err,
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        text = str(exc)
        assert "System message must be at the beginning." in text   # provider text
        assert exc.request_stage == "tool_generation"
        assert classify_pipeline_error(exc) == PipelineRequestError.TOOL_REQUEST_FAILED
        assert "model='demo-model'" in text                          # params shown
        assert "temperature=0.1" in text
        assert "[0] system:" in text and "[1] user:" in text         # message history shown
        assert exc.params["model"] == "demo-model"                   # programmatic access
        assert len(exc.messages) == 2
        assert exc.original is err
    else:
        raise AssertionError("expected PipelineRequestError")


def test_connection_error_surfaces_underlying_transport_cause():
    backend = FakeChatCompletions([text_turn("ok")])
    chat = pipelined_chat(backend, pipelines=[])

    async def go():
        return await chat.create(
            model="demo-model",
            messages=[{"role": "user", "content": "hi"}],
            debug_generate_exception=APIConnectionError(
                message="Connection error.",
                request=httpx.Request("GET", "http://127.0.0.1:1/v1/chat/completions"),
            ),
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        text = str(exc)
        assert "Connection error." in text
        assert exc.request_stage == "chat_generation"
        assert classify_pipeline_error(exc) == PipelineRequestError.CHAT_REQUEST_FAILED
    else:
        raise AssertionError("expected PipelineRequestError")


def test_live_client_debug_exception_reaches_chat_request_stage():
    client, model = _live_client_and_model()
    chat = pipelined_chat(client.chat.completions, pipelines=[])

    async def go():
        return await chat.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            debug_generate_exception=APIConnectionError(
                message="Connection error.",
                request=httpx.Request("GET", "http://127.0.0.1:1/v1/chat/completions"),
            ),
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        assert exc.request_stage == "chat_generation"
        assert classify_pipeline_error(exc) == PipelineRequestError.CHAT_REQUEST_FAILED
        assert exc.params["model"] == model
        assert exc.messages == [{"role": "user", "content": "hi"}]
    else:
        raise AssertionError("expected live-client debug PipelineRequestError")


def test_live_client_debug_exception_reaches_tool_request_stage():
    client, model = _live_client_and_model()
    chat = pipelined_chat(client.chat.completions, pipelines=[ToolPipeline()])
    err = FakeProviderError(
        "Error code: 400",
        body={"error": {"code": 400, "message": "live tool debug failure", "type": "invalid_request_error"}},
    )

    async def go():
        return await chat.create(
            model=model,
            messages=[{"role": "user", "content": "use tool"}],
            tool_sources=[{"add": _add}],
            debug_generate_exception=err,
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        assert exc.request_stage == "tool_generation"
        assert classify_pipeline_error(exc) == PipelineRequestError.TOOL_REQUEST_FAILED
        assert exc.original is err
        assert exc.params["model"] == model
        assert len(exc.messages) == 1
    else:
        raise AssertionError("expected live-client debug PipelineRequestError")


def test_request_failure_during_json_repair_is_labelled():
    backend = FakeChatCompletions([text_turn("not json at all")])
    chat = pipelined_chat(backend, pipelines=[JsonFixPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            model="demo-model",
            messages=[{"role": "user", "content": "return json"}],
            schema_dict={"x": int},
            debug_generate_exception=lambda kwargs: (
                FakeProviderError(
                    "Error code: 400",
                    body={"error": {"code": 400, "message": "repair request failed", "type": "invalid_request_error"}},
                )
                if kwargs.get("debug_stage") == PipelineDebugStage.JSON_REPAIR_REQUEST_FAILED
                else None
            ),
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        assert exc.request_stage == "json_repair"
        assert classify_pipeline_error(exc) == PipelineRequestError.JSON_REPAIR_REQUEST_FAILED
        assert "repair request failed" in str(exc)
        assert len(backend.calls) == 1
    else:
        raise AssertionError("expected PipelineRequestError")


def test_genuine_bug_propagates_with_context_note():
    # A non-provider error (a real bug) must keep its own type and traceback,
    # NOT be wrapped as PipelineRequestError -- but carry the request context
    # as an attached note so it stays debuggable.
    class BugBackend:
        def create(self, **kwargs):
            raise TypeError("genuine bug in backend")

    chat = pipelined_chat(BugBackend(), pipelines=[])

    async def go():
        return await chat.create(
            model="buggy-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

    try:
        _run(go())
    except TypeError as exc:
        assert not isinstance(exc, PipelineRequestError)        # not swallowed by the umbrella
        notes = "\n".join(getattr(exc, "__notes__", []))
        assert "request context" in notes                      # context attached
        assert "model='buggy-model'" in notes
        assert "[0] user:" in notes
    else:
        raise AssertionError("expected the raw TypeError to propagate")


def test_genuine_bug_is_not_caught_by_attempt_umbrella():
    # Confirms the `attempt(coro)` pattern: a bug raises straight through it,
    # while a PipelineRequestError would be returned.
    class BugBackend:
        def create(self, **kwargs):
            raise KeyError("missing thing")

    chat = pipelined_chat(BugBackend(), pipelines=[])

    async def go():
        return await attempt(chat.create(model="m", messages=[{"role": "user", "content": "x"}]))

    try:
        _run(go())
    except KeyError:
        pass                                                   # raised straight through attempt
    else:
        raise AssertionError("expected KeyError to propagate past attempt()")


def test_loop_unrepaired_is_labelled_not_passed():
    backend = FakeChatCompletions([loop_turn(), loop_turn(), loop_turn(), loop_turn()])
    chat = pipelined_chat(backend, pipelines=[LoopGuardPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "count"}],
            return_trace=True,
        )

    result = _run(go())
    actions = _actions(result)
    assert "loop_guard/unrepaired" in actions
    assert actions.count("loop_guard/retry") == 3
    assert result.response["choices"][0]["finish_reason"] == "loop_guard"


# --------------------------------------------------------------------------- #
# Combined + logger
# --------------------------------------------------------------------------- #

def test_combined_tool_then_structured_output():
    backend = FakeChatCompletions(
        [tool_call_turn("add", {"a": 1, "b": 2}), text_turn('{"answer": "3"}')]
    )
    chat = pipelined_chat(backend, pipelines=[ToolPipeline(), LoopGuardPipeline(), JsonFixPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "add then json"}],
            tool_sources=[{"add": _add}],
            schema_dict={"answer": str},
            return_trace=True,
        )

    result = _run(go())
    assert result.parsed == {"answer": "3"}
    assert len(result.tool_executions) == 1


def test_logger_writes_without_raising():
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    try:
        backend = FakeChatCompletions(
            [tool_call_turn("add", {"a": 2, "b": 2}), text_turn("four", reasoning="adding")]
        )
        logger = LoggerPipeline(path=path)
        chat = pipelined_chat(backend, pipelines=[logger, ToolPipeline()])

        async def go():
            return await chat.create(
                messages=[{"role": "user", "content": "add"}],
                tool_sources=[{"add": _add}],
                return_trace=True,
            )

        result = _run(go())
        logger.close()
        assert result.response["choices"][0]["message"]["content"] == "four"
        log_text = open(path, encoding="utf-8").read()
        assert "request #1" in log_text
        assert "TOOL-RESULT" in log_text
    finally:
        os.remove(path)


if __name__ == "__main__":
    failures = []
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
