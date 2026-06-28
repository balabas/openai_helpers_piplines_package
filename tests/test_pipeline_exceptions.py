"""Exception-correctness tests for every pipeline layer.

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

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # fake_chat
sys.path.insert(0, _ROOT)   # package root

from fake_chat import (  # noqa: E402
    AsyncFakeChatCompletions,
    FailingChatCompletions,
    FakeChatCompletions,
    FakeProviderError,
    loop_turn,
    text_turn,
    tool_call_turn,
)
from openai_helpers_piplines_package import (  # noqa: E402
    JsonFixPipeline,
    LoggerPipeline,
    LoopGuardPipeline,
    ToolPipeline,
    with_pipelines,
)
from pipelines.chat import PipelineRequestError  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _actions(result):
    """All ``level/action`` strings in the trace."""
    return [f"{e.level}/{e.action}" for e in (result.trace or [])]


# --------------------------------------------------------------------------- #
# Baseline / happy paths
# --------------------------------------------------------------------------- #

def test_plain_text_response():
    backend = FakeChatCompletions([text_turn("hello world")])
    chat = with_pipelines(backend, layers=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    assert result is not None
    assert result.response["choices"][0]["message"]["content"] == "hello world"
    assert result.parsed is None


def test_async_backend_aiter_path():
    backend = AsyncFakeChatCompletions([text_turn("async answer", n_chunks=3)])
    chat = with_pipelines(backend, layers=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    assert result.response["choices"][0]["message"]["content"] == "async answer"


def test_reasoning_then_content_streamed():
    backend = FakeChatCompletions([text_turn("final", reasoning="let me think")])
    chat = with_pipelines(backend, layers=[])

    async def go():
        return await chat.create(messages=[{"role": "user", "content": "hi"}])

    result = _run(go())
    msg = result.response["choices"][0]["message"]
    assert msg["content"] == "final"
    assert msg.get("reasoning_content") == "let me think"


# --------------------------------------------------------------------------- #
# JSON / schema layer
# --------------------------------------------------------------------------- #

def test_schema_success():
    backend = FakeChatCompletions([text_turn('{"x": 5}')])
    chat = with_pipelines(backend, layers=[JsonFixPipeline()])

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
    chat = with_pipelines(backend, layers=[JsonFixPipeline(max_retries=1)])

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


def test_schema_exhausted_falls_back_to_model_construct():
    # Model can't produce JSON -> exhausted repair returns a graceful fallback
    # instead of raising. The fallback preserves the original text payload.
    backend = FakeChatCompletions([text_turn("still not json"), text_turn("still not json repair")])
    chat = with_pipelines(backend, layers=[JsonFixPipeline(max_retries=0)])

    async def go():
        return await chat.create(
            model="weak-model",
            messages=[{"role": "user", "content": "give me json"}],
            temperature=0.2,
            schema_dict={"x": int},
        )

    result = _run(go())
    assert result.parsed == {"x": "still not json"}
    assert result.response["choices"][0]["message"]["content"] == "still not json repair"
    assert len(backend.calls) == 2


def test_inline_toolcall_text_with_schema_falls_back_gracefully():
    # Inline tool-call text is treated as invalid structured output and repaired
    # with the same graceful fallback path used for other JSON failures.
    inline = "<tool_call> <function=add> <parameter=a> 3 </parameter> </function> </tool_call>"
    backend = FakeChatCompletions([text_turn(inline), text_turn(inline), text_turn(inline)])
    chat = with_pipelines(backend, layers=[ToolPipeline(), JsonFixPipeline(max_retries=1)])

    async def go():
        return await chat.create(
            model="pseudo-xml",
            messages=[{"role": "user", "content": "add then json"}],
            tool_sources=[{"add": _add}],
            schema_dict={"answer": str},
        )

    result = _run(go())
    assert result.parsed == {"answer": inline}
    assert result.response["choices"][0]["message"]["content"] == inline
    assert len(backend.calls) == 3


def test_schema_validation_error_is_retried():
    # Valid JSON, but wrong type for a required int field -> pydantic ValidationError,
    # which must be caught by the (ValueError-based) retry, not escape.
    backend = FakeChatCompletions([text_turn('{"x": "abc"}'), text_turn('{"x": 3}')])
    chat = with_pipelines(backend, layers=[JsonFixPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[])

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
        raise AssertionError("expected ValueError without JsonFixPipeline layer")


# --------------------------------------------------------------------------- #
# Tool layer
# --------------------------------------------------------------------------- #

def _add(a: int, b: int) -> dict:
    return {"sum": a + b}


def test_tool_call_then_final_answer():
    backend = FakeChatCompletions(
        [tool_call_turn("add", {"a": 1, "b": 2}), text_turn("the sum is 3")]
    )
    chat = with_pipelines(backend, layers=[ToolPipeline()])

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
    chat = with_pipelines(backend, layers=[ToolPipeline()])

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
    chat = with_pipelines(backend, layers=[ToolPipeline()])

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
    chat = with_pipelines(backend, layers=[ToolPipeline()])

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
    chat = with_pipelines(backend, layers=[])

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
        raise AssertionError("expected ValueError without ToolPipeline layer")


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
    chat = with_pipelines(backend, layers=[ToolPipeline(max_retries=0)])

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
    chat = with_pipelines(backend, layers=[ToolPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[LoopGuardPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[ToolPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[JsonFixPipeline()])

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
    chat = with_pipelines(backend, layers=[ToolPipeline()])

    async def go():
        return await chat.create(
            messages=[{"role": "user", "content": "loop"}],
            tool_sources=[{"add": _add}],
        )

    try:
        _run(go())
    except RuntimeError as exc:
        assert "safety limit" in str(exc)
    else:
        raise AssertionError("expected RuntimeError at tool iteration safety limit")


# --------------------------------------------------------------------------- #
# Loop-guard layer
# --------------------------------------------------------------------------- #

def test_loop_detected_then_retry_recovers():
    backend = FakeChatCompletions([loop_turn(), text_turn("clean answer")])
    chat = with_pipelines(backend, layers=[LoopGuardPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[ToolPipeline(), JsonFixPipeline(max_retries=1)])

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
    chat = with_pipelines(backend, layers=[LoopGuardPipeline(max_retries=1)])

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
    backend = FailingChatCompletions(err)
    chat = with_pipelines(backend, layers=[ToolPipeline()])

    async def go():
        return await chat.create(
            model="demo-model",
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "compute 3 + 4 then return JSON"},
            ],
            temperature=0.1,
            tool_sources=[{"add": _add}],
        )

    try:
        _run(go())
    except PipelineRequestError as exc:
        text = str(exc)
        assert "System message must be at the beginning." in text   # provider text
        assert "model='demo-model'" in text                          # params shown
        assert "temperature=0.1" in text
        assert "[0] system:" in text and "[1] user:" in text         # message history shown
        assert exc.params["model"] == "demo-model"                   # programmatic access
        assert len(exc.messages) == 2
        assert exc.original is err
    else:
        raise AssertionError("expected PipelineRequestError")


def test_genuine_bug_propagates_with_context_note():
    # A non-provider error (a real bug) must keep its own type and traceback,
    # NOT be wrapped as PipelineRequestError -- but carry the request context
    # as an attached note so it stays debuggable.
    class BugBackend:
        def create(self, **kwargs):
            raise TypeError("genuine bug in backend")

    chat = with_pipelines(BugBackend(), layers=[])

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

    chat = with_pipelines(BugBackend(), layers=[])

    async def attempt(coro):
        try:
            return await coro
        except PipelineRequestError as error:
            return error

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
    chat = with_pipelines(backend, layers=[LoopGuardPipeline(max_retries=0)])

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
    chat = with_pipelines(backend, layers=[ToolPipeline(), LoopGuardPipeline(), JsonFixPipeline()])

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
        chat = with_pipelines(backend, layers=[logger, ToolPipeline()])

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
