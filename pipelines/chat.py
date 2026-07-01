"""OpenAI-compatible chat.completions wrapper with pipeline semantics."""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, TypeVar

from .json_fix import JsonFixPipeline, StructuredOutputRequest
from .logger import LoggerPipeline
from .loop_guard import LoopGuardPipeline
from .tool import ToolExecution, ToolHandler, ToolPipeline


class _RequestStage(StrEnum):
    """Internal request-generation stages stored on PipelineRequestError.request_stage."""

    CHAT_GENERATION = "chat_generation"
    STRUCTURED_GENERATION = "structured_generation"
    TOOL_GENERATION = "tool_generation"
    JSON_REPAIR = "json_repair"


class PipelineErrorKind(StrEnum):
    """Package-defined expected error branches for application code."""

    CHAT_REQUEST_FAILED = "chat_request_failed"
    TOOL_REQUEST_FAILED = "tool_request_failed"
    JSON_REPAIR_REQUEST_FAILED = "json_repair_request_failed"
    EMPTY_ASSISTANT_OUTPUT = "empty_assistant_output"
    STRUCTURED_OUTPUT_REPAIR_EXHAUSTED = "structured_output_repair_exhausted"
    TOOL_ITERATION_LIMIT_EXCEEDED = "tool_iteration_limit_exceeded"


class PipelineDebugStage(StrEnum):
    """Debug insertion points accepted by debug_generate_exception callables."""

    CHAT_COMPLETIONS_CREATE = "chat_completions_create"
    JSON_REPAIR_REQUEST_FAILED = "json_repair_request_failed"
    EMPTY_ASSISTANT_OUTPUT = "empty_assistant_output"
    STRUCTURED_OUTPUT_REPAIR_EXHAUSTED = "structured_output_repair_exhausted"
    TOOL_ITERATION_LIMIT_EXCEEDED = "tool_iteration_limit_exceeded"


EMPTY_ASSISTANT_OUTPUT_MESSAGE = "Empty assistant output"
STRUCTURED_OUTPUT_REPAIR_EXHAUSTED_PREFIX = "Structured output repair exhausted"
TOOL_ITERATION_LIMIT_PREFIX = "Exceeded internal tool iteration safety limit"

_T = TypeVar("_T")


def _request_error_kind_for_stage(stage: _RequestStage | str | None) -> PipelineErrorKind:
    stage = stage or _RequestStage.CHAT_GENERATION
    if stage == _RequestStage.TOOL_GENERATION:
        return PipelineErrorKind.TOOL_REQUEST_FAILED
    if stage == _RequestStage.JSON_REPAIR:
        return PipelineErrorKind.JSON_REPAIR_REQUEST_FAILED
    return PipelineErrorKind.CHAT_REQUEST_FAILED


class PipelineRequestError(Exception):
    """Catchable wrapper for provider/transport errors."""

    CHAT_REQUEST_FAILED = PipelineErrorKind.CHAT_REQUEST_FAILED
    TOOL_REQUEST_FAILED = PipelineErrorKind.TOOL_REQUEST_FAILED
    JSON_REPAIR_REQUEST_FAILED = PipelineErrorKind.JSON_REPAIR_REQUEST_FAILED
    EMPTY_ASSISTANT_OUTPUT = PipelineErrorKind.EMPTY_ASSISTANT_OUTPUT
    STRUCTURED_OUTPUT_REPAIR_EXHAUSTED = PipelineErrorKind.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED
    TOOL_ITERATION_LIMIT_EXCEEDED = PipelineErrorKind.TOOL_ITERATION_LIMIT_EXCEEDED

    def __init__(
        self,
        original: Exception,
        request: dict[str, Any],
        *,
        request_stage: _RequestStage | str | None = None,
    ) -> None:
        self.original = original
        self.request_stage = request_stage
        self.error_kind = _request_error_kind_for_stage(request_stage)
        self.params = {key: value for key, value in request.items() if key != "messages"}
        self.messages: list[dict[str, Any]] = list(request.get("messages") or [])
        super().__init__(self._format())

    def _render_traceback_(self) -> list[str]:
        return f"{type(self).__name__}: {self}".split("\n")

    def _format(self) -> str:
        lines = [f"chat.completions.create failed: {_provider_error_text(self.original)}", ""]
        if self.request_stage:
            lines.extend([f"Request stage: {self.request_stage}", ""])
        lines.extend(
            [
                f"Request params: {_format_params(self.params)}",
                f"Messages ({len(self.messages)}):",
                *_format_messages(self.messages),
                "",
                "Inspect the full request on the exception: .params, .messages, .original",
            ]
        )
        return "\n".join(lines)


class EmptyAssistantOutputError(ValueError):
    """Raised when the assistant returns no text and no tool call."""

    error_kind = PipelineErrorKind.EMPTY_ASSISTANT_OUTPUT

    def __init__(self) -> None:
        super().__init__(EMPTY_ASSISTANT_OUTPUT_MESSAGE)


class StructuredOutputRepairExhaustedError(ValueError):
    """Raised when structured-output repair retries are exhausted."""

    error_kind = PipelineErrorKind.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED

    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(f"{STRUCTURED_OUTPUT_REPAIR_EXHAUSTED_PREFIX} after {attempts} attempts")


class ToolIterationLimitExceededError(RuntimeError):
    """Raised when tool-call rounds exceed the internal safety limit."""

    error_kind = PipelineErrorKind.TOOL_ITERATION_LIMIT_EXCEEDED

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"{TOOL_ITERATION_LIMIT_PREFIX}: {limit}")


def classify_pipeline_error(error: BaseException) -> PipelineErrorKind | None:
    """Return the package-defined expected error kind, or None for non-pipeline bugs."""

    if isinstance(error, PipelineRequestError):
        return error.error_kind

    kind = getattr(error, "error_kind", None)
    if isinstance(kind, PipelineErrorKind):
        return kind

    text = str(error)
    if isinstance(error, ValueError):
        if text == EMPTY_ASSISTANT_OUTPUT_MESSAGE:
            return PipelineErrorKind.EMPTY_ASSISTANT_OUTPUT
        if text.startswith(STRUCTURED_OUTPUT_REPAIR_EXHAUSTED_PREFIX):
            return PipelineErrorKind.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED

    if isinstance(error, RuntimeError) and text.startswith(TOOL_ITERATION_LIMIT_PREFIX):
        return PipelineErrorKind.TOOL_ITERATION_LIMIT_EXCEEDED

    return None


async def attempt(coro: Awaitable[_T]) -> _T | BaseException:
    """Return expected pipeline errors as values; let genuine bugs raise."""

    try:
        return await coro
    except PipelineRequestError as error:
        return error
    except (ValueError, RuntimeError) as error:
        if classify_pipeline_error(error) is not None:
            return error
        raise


def _debug_exception(
    debug_generate_exception: Any,
    context: dict[str, Any],
    *,
    callable_only: bool = False,
) -> BaseException | None:
    if debug_generate_exception is None:
        return None
    if callable_only and isinstance(debug_generate_exception, BaseException):
        return None
    if callable(debug_generate_exception) and not isinstance(debug_generate_exception, BaseException):
        debug_generate_exception = debug_generate_exception(context)
    if debug_generate_exception is None:
        return None
    if not isinstance(debug_generate_exception, BaseException):
        raise TypeError("debug_generate_exception must be an Exception, callable returning one, or None")
    return debug_generate_exception


def _raise_debug_exception(
    debug_generate_exception: Any,
    context: dict[str, Any],
    *,
    callable_only: bool = False,
) -> None:
    exc = _debug_exception(debug_generate_exception, context, callable_only=callable_only)
    if exc is not None:
        raise exc


def _is_provider_error(exc: Exception) -> bool:
    if any(hasattr(exc, attr) for attr in ("status_code", "response", "body")):
        return True
    root_module = type(exc).__module__.split(".", 1)[0]
    return root_module in {"openai", "httpx", "anthropic", "aiohttp", "requests", "httpcore"}


def _request_context_note(request: dict[str, Any], request_stage: str | None = None) -> str:
    params = {key: value for key, value in request.items() if key != "messages"}
    messages = request.get("messages") or []
    lines = [
        "[pipeline] request context for the error above:",
    ]
    if request_stage:
        lines.append(f"  stage: {request_stage}")
    lines.extend(
        [
            f"  params: {_format_params(params)}",
            f"  messages ({len(messages)}):",
            *(f"  {line}" for line in _format_messages(messages)),
        ]
    )
    return "\n".join(lines)


def _provider_error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str) and error:
            return error
        if body.get("message"):
            return str(body["message"])
    response = getattr(exc, "response", None)
    if response is not None:
        getter = getattr(response, "json", None)
        if callable(getter):
            try:
                data = getter()
            except Exception:
                data = None
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict) and error.get("message"):
                    return str(error["message"])
        text = getattr(response, "text", None)
        if text:
            return str(text)
    text = str(exc) or type(exc).__name__
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        cause_text = str(cause).strip()
        if cause_text and cause_text not in text:
            return f"{text} (caused by {type(cause).__name__}: {cause_text})"
    return text


def _format_params(params: dict[str, Any]) -> str:
    shown: list[str] = []
    for key, value in params.items():
        if key == "tools" and isinstance(value, list):
            names = [
                tool.get("function", {}).get("name", tool) if isinstance(tool, dict) else tool
                for tool in value
            ]
            shown.append(f"tools={names!r}")
        else:
            shown.append(f"{key}={value!r}")
    return ", ".join(shown) if shown else "(none)"


def _format_messages(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role", "?")
        content = message.get("content")
        text = content if isinstance(content, str) else "" if content is None else str(content)
        extra = ""
        tool_calls = message.get("tool_calls")
        if tool_calls:
            names = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
            extra = f"  tool_calls={names!r}"
        if "\n" in text:
            lines.append(f"  [{index}] {role}:{extra}")
            lines.extend(f"        {line}" for line in text.splitlines())
        else:
            lines.append(f"  [{index}] {role}: {text!r}{extra}")
    return lines


def _escalate(base: float | None, retry_number: int) -> float:
    base_value = base if (base and base > 0) else 0.0
    if base_value <= 0:
        return round(0.1 * (2 ** (retry_number - 1)), 4)
    return round(base_value * (2 ** (retry_number - 1)), 4)


def _response_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Expected dict-like response, got {type(value).__name__}")


def _message_from_response(response: Any) -> dict[str, Any]:
    data = _response_dict(response)
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("chat response did not contain choices")
    choice = choices[0]
    if hasattr(choice, "model_dump"):
        choice = choice.model_dump(mode="json")
    if not isinstance(choice, dict):
        raise TypeError("chat response choice must be dict-like")
    message = choice.get("message") or {}
    if hasattr(message, "model_dump"):
        message = message.model_dump(mode="json")
    if not isinstance(message, dict):
        raise TypeError("chat response choice message must be dict-like")
    return dict(message)


def _finish_reason(response: Any) -> str | None:
    data = _response_dict(response)
    choices = data.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    if isinstance(choice, dict):
        return choice.get("finish_reason")
    return getattr(choice, "finish_reason", None)


def _usage(response: Any) -> dict[str, Any]:
    data = _response_dict(response)
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else {}


def _sum_usage(responses: list[Any]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for response in responses:
        _add_usage(totals, _usage(response))
    return totals


def _add_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + value
        elif isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                _add_usage(nested, value)


def _text_and_tool_calls(response: Any) -> tuple[str, list[dict[str, Any]]]:
    message = _message_from_response(response)
    text = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    normalized: list[dict[str, Any]] = []
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump(mode="json")
        if not isinstance(fn, dict):
            continue
        normalized.append(
            {
                "id": str(call.get("id") or f"tool_call_{idx}"),
                "type": call.get("type", "function"),
                "function": {
                    "name": str(fn.get("name") or ""),
                    "arguments": fn.get("arguments", "{}"),
                },
            }
        )
    return text, [call for call in normalized if call["function"]["name"]]


def _is_empty_generation(text: str, tool_calls: list[dict[str, Any]]) -> bool:
    return not text.strip() and not tool_calls


def _json_response_format(schema_json: str, *, name: str = "DynamicSchema") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": json.loads(schema_json),
            "strict": True,
        },
    }


@dataclass(slots=True)
class PipelineEvent:
    level: str
    action: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelinedChatCompletionResult:
    response: Any
    parsed: dict[str, Any] | None = None
    trace: list[PipelineEvent] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[Any] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)

    @property
    def usage(self) -> dict[str, Any]:
        """Token usage for the final response."""

        return _usage(self.response)

    @property
    def run_usage(self) -> dict[str, Any]:
        """Aggregated token usage across every model response in this run."""

        return _sum_usage(self.raw_responses or [self.response])


@dataclass(slots=True)
class _Generation:
    response: Any
    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    parsed: dict[str, Any] | None = None
    loop_reason: str | None = None
    loop_scope: str | None = None
    events: list[PipelineEvent] = field(default_factory=list)


class PipelinedChatCompletions:
    _MAX_TOOL_ITERATIONS = 20
    _MAX_EMPTY_STREAK = 3
    _MAX_TOKENS_STEP = 5000
    _DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        chat_completions: Any,
        *,
        pipelines: Sequence[Any] | None = None,
        layers: Sequence[Any] | None = None,
    ) -> None:
        if pipelines is not None and layers is not None:
            raise ValueError("Pass pipelines= only once; the legacy compatibility alias cannot be combined with it")
        self.chat_completions = chat_completions
        configured_pipelines = pipelines if pipelines is not None else layers
        self.pipelines = self._normalize_pipelines(configured_pipelines)
        self.layers = self.pipelines

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any] | None = None,
        tool_sources: Sequence[Any] | Any | None = None,
        on_clarify: Callable[[str], Any] | None = None,
        return_trace: bool = False,
        **chat_kwargs: Any,
    ) -> PipelinedChatCompletionResult:
        trace: list[PipelineEvent] = [
            PipelineEvent("pipelined_chat", "start", {"pipelines": [type(pipeline).__name__ for pipeline in self.pipelines]})
        ]
        current_messages = [dict(message) for message in messages]

        json_pipeline = self._pipeline_of_type(JsonFixPipeline)
        logger_pipeline = self._pipeline_of_type(LoggerPipeline)
        loop_pipeline = self._pipeline_of_type(LoopGuardPipeline)
        tool_pipeline = self._pipeline_of_type(ToolPipeline)

        if schema_dict is not None and json_pipeline is None:
            raise ValueError("schema_dict requires JsonFixPipeline in configured pipelines")
        if tool_sources is not None and tool_pipeline is None:
            raise ValueError("tool_sources requires ToolPipeline in configured pipelines")

        structured: StructuredOutputRequest | None = None
        if schema_dict is not None and json_pipeline is not None:
            structured = json_pipeline.build_request(messages=current_messages, schema_dict=schema_dict)
            current_messages = structured.messages
            trace.append(PipelineEvent("validation", "schema_prepared", {"schema_keys": list(schema_dict.keys())}))

        tool_schemas: list[dict[str, Any]] | None = None
        tool_registry: dict[str, ToolHandler] = {}
        if tool_sources is not None and tool_pipeline is not None:
            tool_schemas, tool_registry = await self._resolve_tool_sources(tool_sources)
            trace.append(PipelineEvent("tool", "sources_resolved", {"count": len(tool_registry)}))

        base_request_kwargs = dict(chat_kwargs)
        debug_generate_exception = base_request_kwargs.get("debug_generate_exception")

        if tool_schemas is None:
            generation = await self._generate_one(
                messages=current_messages,
                request_kwargs=base_request_kwargs,
                json_pipeline=json_pipeline,
                structured=structured,
                logger_pipeline=logger_pipeline,
                loop_pipeline=loop_pipeline,
                request_stage=(
                    _RequestStage.STRUCTURED_GENERATION
                    if structured is not None
                    else _RequestStage.CHAT_GENERATION
                ),
            )
            trace.extend(generation.events)
            if _is_empty_generation(generation.text, generation.tool_calls):
                _raise_debug_exception(
                    debug_generate_exception,
                    {
                        "debug_stage": PipelineDebugStage.EMPTY_ASSISTANT_OUTPUT,
                        "messages": current_messages,
                        "request_stage": (
                            _RequestStage.STRUCTURED_GENERATION
                            if structured is not None
                            else _RequestStage.CHAT_GENERATION
                        ),
                        "generation": generation,
                    },
                    callable_only=True,
                )
                raise EmptyAssistantOutputError()

            raw_responses = [generation.response]
            parsed = generation.parsed
            if structured is not None and parsed is None:
                trace.append(PipelineEvent("validation", "json_retry", {"error": "heuristic validation failed"}))
                parsed, repair_response = await self._repair_structured_output(
                    generation=generation,
                    json_pipeline=json_pipeline,
                    structured=structured,
                    request_kwargs=base_request_kwargs,
                    logger_pipeline=logger_pipeline,
                    loop_pipeline=loop_pipeline,
                )
                trace.append(PipelineEvent("validation", "parsed", {"keys": list(parsed.keys())}))
                response = repair_response or generation.response
                if repair_response is not None:
                    raw_responses.append(repair_response)
            else:
                response = generation.response

            trace.append(
                PipelineEvent(
                    "pipelined_chat",
                    "finished",
                    {"tool_rounds": 0, "loop_retries": 0, "json_retries": 0},
                )
            )
            self._log_events(logger_pipeline, trace)
            self._log_end(logger_pipeline, response)
            return PipelinedChatCompletionResult(
                response=response,
                parsed=parsed,
                trace=trace if return_trace else None,
                messages=current_messages,
                raw_responses=raw_responses,
                tool_executions=[],
            )

        raw_responses: list[Any] = []
        tool_executions: list[ToolExecution] = []
        empty_streak = 0
        base_temperature = base_request_kwargs.get("temperature")
        max_tokens_step = self._MAX_TOKENS_STEP
        for round_index in range(self._MAX_TOOL_ITERATIONS):
            generation = await self._generate_one(
                messages=current_messages,
                request_kwargs=base_request_kwargs,
                json_pipeline=json_pipeline,
                structured=structured,
                logger_pipeline=logger_pipeline,
                loop_pipeline=loop_pipeline,
                tool_schemas=tool_schemas,
                request_stage=(
                    _RequestStage.TOOL_GENERATION
                    if tool_schemas
                    else _RequestStage.CHAT_GENERATION
                ),
            )
            raw_responses.append(generation.response)
            trace.extend(generation.events)

            if generation.loop_reason:
                trace.append(PipelineEvent("loop_guard", "detected", {"reason": generation.loop_reason, "scope": generation.loop_scope}))

            if _is_empty_generation(generation.text, generation.tool_calls):
                empty_streak += 1
                trace.append(PipelineEvent("tool", "empty_output", {"streak": empty_streak}))
                if empty_streak >= self._MAX_EMPTY_STREAK:
                    response = generation.response
                    break
                base_request_kwargs["temperature"] = _escalate(base_temperature, empty_streak)
                current_messages.append({"role": "user", "content": "continue, check tool call correctness"})
                trace.append(PipelineEvent("tool", "empty_heat_up", {"temperature": base_request_kwargs["temperature"]}))
                continue

            empty_streak = 0

            if generation.tool_calls:
                round_executions = await self._execute_tool_calls(
                    generation.tool_calls,
                    tool_registry,
                    on_clarify=on_clarify,
                )
                tool_executions.extend(round_executions)
                self._log_tool_results(logger_pipeline, round_executions)
                current_messages.extend(
                    [
                        self._assistant_message_from_generation(generation),
                        *tool_pipeline.tool_messages(round_executions),
                    ]
                )
                trace.append(
                    PipelineEvent(
                        "tool",
                        "executed",
                        {"names": [execution.name for execution in round_executions], "iteration": round_index},
                    )
                )
                continue

            response = generation.response
            parsed = generation.parsed
            if structured is not None and parsed is None:
                trace.append(PipelineEvent("validation", "json_retry", {"error": "heuristic validation failed"}))
                parsed, repair_response = await self._repair_structured_output(
                    generation=generation,
                    json_pipeline=json_pipeline,
                    structured=structured,
                    request_kwargs=base_request_kwargs,
                    logger_pipeline=logger_pipeline,
                    loop_pipeline=loop_pipeline,
                )
                trace.append(PipelineEvent("validation", "parsed", {"keys": list(parsed.keys())}))
                response = repair_response or response
                if repair_response is not None:
                    raw_responses.append(repair_response)
            trace.append(
                PipelineEvent(
                    "pipelined_chat",
                    "finished",
                    {
                        "tool_rounds": round_index + 1,
                        "loop_retries": 0,
                        "json_retries": 0,
                    },
                )
            )
            self._log_events(logger_pipeline, trace)
            self._log_end(logger_pipeline, response)
            return PipelinedChatCompletionResult(
                response=response,
                parsed=parsed,
                trace=trace if return_trace else None,
                messages=current_messages,
                raw_responses=raw_responses,
                tool_executions=tool_executions,
            )

        _raise_debug_exception(
            debug_generate_exception,
            {
                "debug_stage": PipelineDebugStage.TOOL_ITERATION_LIMIT_EXCEEDED,
                "messages": current_messages,
                "request_stage": _RequestStage.TOOL_GENERATION,
                "limit": self._MAX_TOOL_ITERATIONS,
            },
            callable_only=True,
        )
        raise ToolIterationLimitExceededError(self._MAX_TOOL_ITERATIONS)

    async def _repair_structured_output(
        self,
        *,
        generation: _Generation,
        json_pipeline: JsonFixPipeline | None,
        structured: StructuredOutputRequest,
        request_kwargs: dict[str, Any],
        logger_pipeline: LoggerPipeline | None,
        loop_pipeline: LoopGuardPipeline | None,
    ) -> tuple[dict[str, Any], Any | None]:
        if json_pipeline is None:
            raise ValueError("schema_dict requires JsonFixPipeline in configured pipelines")

        fit_messages = [
            {
                "role": "user",
                "content": json_pipeline.build_fit_prompt(schema_json=structured.schema_json, raw_text=generation.text),
            }
        ]
        fit_kwargs = dict(request_kwargs)
        fit_kwargs.pop("tools", None)
        fit_kwargs.pop("tool_choice", None)
        fit_kwargs["response_format"] = _json_response_format(structured.schema_json, name=structured.model_cls.__name__)

        last_response: Any | None = None
        for retry_index in range(json_pipeline.max_retries + 1):
            repaired = await self._generate_one(
                messages=fit_messages,
                request_kwargs=fit_kwargs,
                json_pipeline=None,
                structured=structured,
                logger_pipeline=logger_pipeline,
                loop_pipeline=loop_pipeline,
                force_response_format=fit_kwargs["response_format"],
                request_stage=_RequestStage.JSON_REPAIR,
            )
            last_response = repaired.response
            try:
                parsed = json_pipeline.parse(repaired.text, structured.model_cls)
            except Exception:
                parsed = None
            if parsed is not None:
                return parsed, repaired.response
            if retry_index < json_pipeline.max_retries:
                fit_messages.extend(
                    json_pipeline.build_retry_messages(
                        previous_text=repaired.text,
                        error=ValueError("Previous JSON failed validation"),
                        schema_json=structured.schema_json,
                    )
                )
        _raise_debug_exception(
            request_kwargs.get("debug_generate_exception"),
            {
                "debug_stage": PipelineDebugStage.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED,
                "messages": fit_messages,
                "request_stage": _RequestStage.JSON_REPAIR,
                "attempts": json_pipeline.max_retries + 1,
            },
            callable_only=True,
        )
        raise StructuredOutputRepairExhaustedError(json_pipeline.max_retries + 1)

    def _construct_fallback(self, model_cls: Any, text: str) -> dict[str, Any]:
        fields = getattr(model_cls, "model_fields", {})
        if fields:
            first_field = next(iter(fields))
            return {first_field: text}
        return {"text": text}

    async def _generate_one(
        self,
        *,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        json_pipeline: JsonFixPipeline | None,
        structured: StructuredOutputRequest | None,
        logger_pipeline: LoggerPipeline | None,
        loop_pipeline: LoopGuardPipeline | None,
        tool_schemas: list[dict[str, Any]] | None = None,
        force_response_format: dict[str, Any] | None = None,
        request_stage: _RequestStage | str = _RequestStage.CHAT_GENERATION,
    ) -> _Generation:
        base_messages = [dict(message) for message in messages]
        if structured is not None and json_pipeline is not None:
            base_messages = json_pipeline.append_schema_hint(base_messages, structured.schema_json)

        base_temperature = request_kwargs.get("temperature")
        base_max_tokens = int(request_kwargs.get("max_tokens") or self._DEFAULT_MAX_TOKENS)
        loop_retries = 0
        max_tokens_boost = 0
        events: list[PipelineEvent] = []

        while True:
            temp = base_temperature if loop_retries == 0 else _escalate(base_temperature, loop_retries)
            current_kwargs = dict(request_kwargs)
            debug_generate_exception = current_kwargs.pop("debug_generate_exception", None)
            current_kwargs["messages"] = [dict(message) for message in base_messages]
            current_kwargs["stream"] = True
            current_kwargs["stream_options"] = {"include_usage": True}
            current_kwargs["max_tokens"] = base_max_tokens + max_tokens_boost
            if loop_retries > 0 or "temperature" in current_kwargs:
                current_kwargs["temperature"] = temp
            elif temp is not None:
                current_kwargs["temperature"] = temp
            if tool_schemas:
                current_kwargs["tools"] = tool_schemas
            response_format = force_response_format
            if response_format is None and structured is not None and not tool_schemas:
                response_format = _json_response_format(structured.schema_json, name=structured.model_cls.__name__)
            if response_format is not None:
                current_kwargs["response_format"] = response_format

            if logger_pipeline is not None:
                logger_pipeline.logger.log_request(current_kwargs)

            try:
                response = await self._call_create(
                    current_kwargs,
                    request_stage=request_stage,
                    debug_generate_exception=debug_generate_exception,
                )
            except PipelineRequestError as exc:
                response_format_type = current_kwargs.get("response_format", {}).get("type")
                error_text = str(exc).lower()
                if response_format_type == "json_schema" and ("json_schema" in error_text or "response_format" in error_text):
                    fallback_kwargs = dict(current_kwargs)
                    fallback_kwargs["response_format"] = {"type": "json_object"}
                    if logger_pipeline is not None:
                        logger_pipeline.logger.log_request(fallback_kwargs)
                    response = await self._call_create(
                        fallback_kwargs,
                        request_stage=request_stage,
                        debug_generate_exception=debug_generate_exception,
                    )
                    current_kwargs = fallback_kwargs
                else:
                    raise
            generation = await self._consume_stream(response, logger_pipeline, loop_pipeline)
            if structured is not None and json_pipeline is not None and not generation.tool_calls:
                try:
                    generation.parsed = json_pipeline.parse(generation.text, structured.model_cls)
                except Exception:
                    generation.parsed = None
                else:
                    generation.events.append(PipelineEvent("validation", "parsed", {"keys": list(generation.parsed.keys())}))
            events.extend(generation.events)

            if generation.loop_reason and loop_retries < 3:
                loop_retries += 1
                events.append(
                    PipelineEvent(
                        "loop_guard",
                        "retry",
                        {"reason": generation.loop_reason, "temperature": _escalate(base_temperature, loop_retries)},
                    )
                )
                continue
            if generation.loop_reason:
                events.append(PipelineEvent("loop_guard", "unrepaired", {"reason": generation.loop_reason}))
            if generation.finish_reason == "length" and max_tokens_boost // self._MAX_TOKENS_STEP < 3:
                max_tokens_boost += self._MAX_TOKENS_STEP
                events.append(PipelineEvent("llm_call", "length_retry", {"max_tokens": base_max_tokens + max_tokens_boost}))
                continue
            generation.events = events
            return generation

    async def _call_create(
        self,
        kwargs: dict[str, Any],
        *,
        request_stage: _RequestStage | str | None = None,
        debug_generate_exception: Any = None,
    ) -> Any:
        create = getattr(self.chat_completions, "create", self.chat_completions)
        try:
            _raise_debug_exception(
                debug_generate_exception,
                {
                    **kwargs,
                    "debug_stage": (
                        PipelineDebugStage.JSON_REPAIR_REQUEST_FAILED
                        if request_stage == _RequestStage.JSON_REPAIR
                        else PipelineDebugStage.CHAT_COMPLETIONS_CREATE
                    ),
                    "request_stage": request_stage,
                },
            )
            value = create(**kwargs)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            if _is_provider_error(exc):
                raise PipelineRequestError(exc, kwargs, request_stage=request_stage) from None
            if hasattr(exc, "add_note"):
                exc.add_note(_request_context_note(kwargs, request_stage))
            raise
        return value

    async def _consume_stream(
        self,
        stream: Any,
        logger_pipeline: LoggerPipeline | None,
        loop_pipeline: LoopGuardPipeline | None,
    ) -> _Generation:
        chunks: list[dict[str, Any]] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        content_accum = ""
        reasoning_accum = ""
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        loop_reason: str | None = None
        loop_scope: str | None = None
        response_id: str | None = None
        model: str | None = None
        created: int | None = None
        usage: dict[str, Any] | None = None

        async for chunk in self._aiter_stream(stream):
            data = self._to_plain(chunk)
            if not isinstance(data, dict):
                continue
            chunks.append(data)
            response_id = data.get("id") or response_id
            model = data.get("model") or model
            created = data.get("created") or created
            if isinstance(data.get("usage"), dict):
                usage = data["usage"]

            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                choice = self._to_plain(choice)
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or choice.get("message") or {}
            delta = self._to_plain(delta)
            if not isinstance(delta, dict):
                continue

            reasoning = delta.get("reasoning_content") or delta.get("thinking") or ""
            if reasoning:
                reasoning_text = str(reasoning)
                reasoning_parts.append(reasoning_text)
                reasoning_accum += reasoning_text
                if logger_pipeline is not None:
                    logger_pipeline.logger.log_output_chunk(reasoning_text, is_thinking=True)
                if loop_pipeline is not None:
                    loop_reason = loop_pipeline.check(reasoning_accum)
                    if loop_reason:
                        loop_scope = "thought"
                        await self._close_stream(stream)
                        break

            content = delta.get("content") or ""
            if content:
                content_text = str(content)
                content_parts.append(content_text)
                content_accum += content_text
                if logger_pipeline is not None:
                    logger_pipeline.logger.log_output_chunk(content_text, is_thinking=False)
                if loop_pipeline is not None:
                    loop_reason = loop_pipeline.check(content_accum)
                    if loop_reason:
                        loop_scope = "message"
                        await self._close_stream(stream)
                        break

            for tool_call in delta.get("tool_calls") or []:
                self._merge_tool_call_delta(tool_call_parts, tool_call)

            if loop_reason:
                break

        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        reasoning_text = "".join(reasoning_parts)
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        tool_calls = self._final_tool_calls(tool_call_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
            if logger_pipeline is not None:
                logger_pipeline.logger.log_tool_calls(tool_calls)

        response: dict[str, Any] = {
            "id": response_id or "chatcmpl-stream",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "loop_guard" if loop_reason else finish_reason,
                }
            ],
        }
        if usage is not None:
            response["usage"] = usage
        response["_stream_chunks"] = chunks
        if loop_reason:
            response["_loop_reason"] = loop_reason
            response["_loop_scope"] = loop_scope
        return _Generation(
            response=response,
            text="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=response["choices"][0]["finish_reason"],
            loop_reason=loop_reason,
            loop_scope=loop_scope,
        )

    async def _close_stream(self, stream: Any) -> None:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            value = aclose()
            if inspect.isawaitable(value):
                await value
            return
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    async def _aiter_stream(self, stream: Any) -> AsyncIterator[Any]:
        if hasattr(stream, "__aiter__"):
            async for item in stream:
                yield item
            return
        if hasattr(stream, "__iter__"):
            iterator = iter(stream)
            sentinel = object()
            while True:
                item = await asyncio.to_thread(self._next_or_sentinel, iterator, sentinel)
                if item is sentinel:
                    break
                yield item
            return
        raise TypeError(f"Expected streaming response, got {type(stream).__name__}")

    def _next_or_sentinel(self, iterator: Any, sentinel: Any) -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    def _merge_tool_call_delta(self, tool_call_parts: dict[int, dict[str, Any]], tool_call: Any) -> None:
        tool_call = self._to_plain(tool_call)
        if not isinstance(tool_call, dict):
            return
        index = int(tool_call.get("index", len(tool_call_parts)))
        current = tool_call_parts.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tool_call.get("id"):
            current["id"] = str(tool_call["id"])
        if tool_call.get("type"):
            current["type"] = tool_call["type"]
        function = self._to_plain(tool_call.get("function") or {})
        if not isinstance(function, dict):
            return
        if function.get("name"):
            current["function"]["name"] += str(function["name"])
        if function.get("arguments"):
            current["function"]["arguments"] += str(function["arguments"])

    def _final_tool_calls(self, tool_call_parts: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index in sorted(tool_call_parts):
            tool_call = tool_call_parts[index]
            if not tool_call.get("id"):
                tool_call["id"] = f"tool_call_{index}"
            result.append(tool_call)
        return result

    def _normalize_pipelines(self, pipelines: Sequence[Any] | None) -> tuple[Any, ...]:
        if pipelines is None:
            return ()
        normalized: list[Any] = []
        for pipeline in pipelines:
            if isinstance(pipeline, str):
                pipeline = self._pipeline_from_name(pipeline)
            if not isinstance(pipeline, (ToolPipeline, LoopGuardPipeline, JsonFixPipeline, LoggerPipeline)):
                raise TypeError(f"Unsupported pipeline: {pipeline!r}")
            normalized.append(pipeline)
        return tuple(normalized)

    def _pipeline_from_name(self, name: str) -> Any:
        key = name.strip().lower()
        if key in {"tool", "tools"}:
            return ToolPipeline()
        if key in {"loop", "loop_guard"}:
            return LoopGuardPipeline()
        if key in {"json", "json_fix", "validation"}:
            return JsonFixPipeline()
        if key in {"logger", "logging", "log"}:
            return LoggerPipeline()
        raise ValueError(f"Unknown pipeline: {name!r}")

    def _pipeline_of_type(self, pipeline_type: type) -> Any | None:
        for pipeline in self.pipelines:
            if isinstance(pipeline, pipeline_type):
                return pipeline
        return None

    async def _resolve_tool_sources(self, tool_sources: Sequence[Any] | Any) -> tuple[list[dict[str, Any]], dict[str, ToolHandler]]:
        sources = list(tool_sources) if isinstance(tool_sources, (list, tuple)) else [tool_sources]
        schemas: list[dict[str, Any]] = []
        registry: dict[str, ToolHandler] = {}
        for source in sources:
            if isinstance(source, dict):
                for name, handler in source.items():
                    if not callable(handler):
                        continue
                    registry[name] = self._wrap_callable(handler)
                    schemas.append(self._schema_from_callable(name, handler))
                continue

            if hasattr(source, "list_tools") and hasattr(source, "call_tool"):
                mcp_schemas, mcp_registry = await self._resolve_mcp_like_source(source)
                schemas.extend(mcp_schemas)
                registry.update(mcp_registry)
                continue

            if callable(source):
                name = getattr(source, "__name__", "tool")
                registry[name] = self._wrap_callable(source)
                schemas.append(self._schema_from_callable(name, source))
                continue

            raise TypeError(f"Unsupported tool source: {source!r}")
        return schemas, registry

    async def _resolve_mcp_like_source(self, source: Any) -> tuple[list[dict[str, Any]], dict[str, ToolHandler]]:
        listed = source.list_tools()
        if inspect.isawaitable(listed):
            listed = await listed
        tools = getattr(listed, "tools", listed)
        schemas: list[dict[str, Any]] = []
        registry: dict[str, ToolHandler] = {}
        for tool in tools or []:
            name = self._get_value(tool, "name")
            if not name:
                continue
            description = self._get_value(tool, "description", "")
            parameters = (
                self._get_value(tool, "inputSchema")
                or self._get_value(tool, "input_schema")
                or self._get_value(tool, "parameters")
                or {"type": "object", "properties": {}}
            )
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description or "",
                        "parameters": parameters,
                    },
                }
            )
            registry[name] = self._mcp_handler(source, name)
        return schemas, registry

    def _mcp_handler(self, source: Any, name: str) -> ToolHandler:
        async def handler(arguments: dict[str, Any]) -> Any:
            value = source.call_tool(name, arguments)
            if inspect.isawaitable(value):
                value = await value
            return self._mcp_result_to_plain_value(value)

        return handler

    def _mcp_result_to_plain_value(self, value: Any) -> Any:
        content = self._get_value(value, "content")
        if content is None:
            return value
        if isinstance(content, list):
            parts = []
            for item in content:
                text = self._get_value(item, "text")
                if text is not None:
                    parts.append(text)
                else:
                    parts.append(item)
            return "\n".join(parts) if all(isinstance(part, str) for part in parts) else parts
        return content

    def _wrap_callable(self, handler: Callable[..., Any]) -> ToolHandler:
        signature = inspect.signature(handler)
        parameters = list(signature.parameters.values())
        pass_dict = (
            len(parameters) == 1
            and parameters[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

        def wrapped(arguments: dict[str, Any]) -> Any:
            if pass_dict:
                return handler(arguments)
            return handler(**arguments)

        return wrapped

    def _schema_from_callable(self, name: str, handler: Callable[..., Any]) -> dict[str, Any]:
        signature = inspect.signature(handler)
        properties: dict[str, Any] = {}
        required: list[str] = []
        parameters = list(signature.parameters.values())
        pass_dict = (
            len(parameters) == 1
            and parameters[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if pass_dict:
            param = parameters[0]
            if param.name not in {"arguments", "args", "kwargs"}:
                properties[param.name] = self._json_type(param.annotation)
                if param.default is inspect.Parameter.empty:
                    required.append(param.name)
        else:
            for param in parameters:
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                properties[param.name] = self._json_type(param.annotation)
                if param.default is inspect.Parameter.empty:
                    required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": inspect.getdoc(handler) or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": pass_dict and not properties,
                },
            },
        }

    def _json_type(self, annotation: Any) -> dict[str, Any]:
        if annotation is inspect.Signature.empty:
            return {"type": "string"}
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            args = getattr(annotation, "__args__", (str,))
            return {"type": "array", "items": self._json_type(args[0])}
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is dict:
            return {"type": "object"}
        return {"type": "string"}

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_registry: dict[str, ToolHandler],
        *,
        on_clarify: Callable[[str], Any] | None,
    ) -> list[ToolExecution]:
        executions: list[ToolExecution] = []
        for call in tool_calls:
            function = call["function"]
            name = function["name"]
            arguments = self._parse_arguments(function.get("arguments"))
            handler = tool_registry.get(name)
            if name == "request_clarification" and on_clarify is not None:
                question = str(arguments.get("question") or arguments.get("prompt") or "")
                value = on_clarify(question)
                if inspect.isawaitable(value):
                    value = await value
                result = self._stringify_result(value)
            elif handler is None:
                result = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
            else:
                try:
                    value = handler(arguments)
                    if inspect.isawaitable(value):
                        value = await value
                    result = self._stringify_result(value)
                except Exception as exc:
                    result = json.dumps({"error": f"Tool execution failed: {exc}"}, ensure_ascii=False)

            executions.append(
                ToolExecution(
                    tool_call_id=call["id"],
                    name=name,
                    arguments=arguments,
                    result=result,
                )
            )
        return executions

    def _assistant_message_from_generation(self, generation: _Generation) -> dict[str, Any]:
        return self._assistant_message_from_response(generation.response)

    def _assistant_message_from_response(self, response: Any) -> dict[str, Any]:
        return dict(_message_from_response(response))

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": arguments}

    def _stringify_result(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _get_value(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _log_tool_results(self, logger_pipeline: LoggerPipeline | None, executions: list[ToolExecution]) -> None:
        if logger_pipeline is None or not logger_pipeline.log_tool_results:
            return
        for execution in executions:
            logger_pipeline.logger.log_tool_result(execution.name, execution.arguments, execution.result)

    def _log_events(self, logger_pipeline: LoggerPipeline | None, events: list[PipelineEvent]) -> None:
        if logger_pipeline is None or not logger_pipeline.log_events:
            return
        for event in events:
            logger_pipeline.logger.log_event(event.level, event.action, event.detail)

    def _log_end(self, logger_pipeline: LoggerPipeline | None, response: Any) -> None:
        if logger_pipeline is None:
            return
        usage = _usage(response)
        logger_pipeline.logger.log_end(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            done_reason=_finish_reason(response) or "",
        )

    def _to_plain(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "dict"):
            return value.dict()
        return value


def pipelined_chat(
    chat_completions: Any,
    *,
    pipelines: Sequence[Any] | None = None,
    layers: Sequence[Any] | None = None,
) -> PipelinedChatCompletions:
    return PipelinedChatCompletions(chat_completions, pipelines=pipelines, layers=layers)
