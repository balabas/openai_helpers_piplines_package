"""Tool-call pipeline: normalization, execution, and message construction.

Responsibility
--------------
Handles everything between the model returning a ``tool_calls`` response and
the next inference round:

1. **Normalize** raw tool-call objects from the streamed response into a
   canonical dict format (providers differ in how they serialize arguments).
2. **Execute** each tool call by dispatching to the caller-supplied handler,
   supporting both sync and async handlers transparently.
3. **Construct** the ``assistant`` + ``tool`` messages to append to the
   conversation so the model can see the tool results.

This pipeline is stateless; the same ``ToolPipeline`` instance is reused
across all calls in a session.

Usage
-----
Constructed by the caller and passed to ``with_pipelines(layers=[...])``.
``PipelinedChatCompletions.create()`` (``pipelines/chat.py``) drives the
tool-call rounds and calls these methods (``openai_tools``,
``execute_tool_calls``, ``assistant_message``, ``tool_messages``) each round.
No imports from the rest of this package.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


#: Type alias for a tool handler — sync or async callable that receives the
#: argument dict and returns any JSON-serialisable value (or a string).
ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(slots=True)
class ToolExecution:
    """Result of a single tool invocation.

    Attributes
    ----------
    tool_call_id:
        ID from the model's tool-call request; echoed back in the tool message.
    name:
        Tool function name.
    arguments:
        Parsed argument dict as passed to the handler.
    result:
        Serialized handler return value (always a string, JSON-encoded when
        the handler returned a non-string).
    """

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: str


class ToolPipeline:
    """Stateless helper for the tool-call round-trip driven by the orchestrator.

    ``PipelinedChatCompletions.create()`` calls these methods during each
    tool-call round; the same instance is reused across all calls in a session.
    """

    def __init__(self, *, max_retries: int = 1) -> None:
        self.max_retries = max_retries

    def normalize_tool_calls(
        self, tool_calls: list[Any] | None
    ) -> list[dict[str, Any]]:
        """Normalise raw tool-call objects into a consistent ``{id, type, function}`` shape.

        Providers and streaming reassembly produce slightly different formats
        (``SimpleNamespace`` objects from streaming, dicts from non-streaming,
        sometimes with ``name``/``arguments`` at the top level instead of
        inside ``function``).  This method collapses all variants.

        Entries without a ``name`` are silently dropped — they represent
        incomplete streaming chunks that should not be executed.
        """
        normalized: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls or []):
            function = self._get_value(call, "function") or {}
            name = self._get_value(function, "name") or self._get_value(call, "name")
            if not name:
                continue
            arguments = self._get_value(
                function,
                "arguments",
                self._get_value(call, "arguments", {}),
            )
            normalized.append(
                {
                    "id": str(self._get_value(call, "id") or f"tool_call_{idx}"),
                    "type": self._get_value(call, "type", "function"),
                    "function": {
                        "name": str(name),
                        "arguments": self._parse_arguments(arguments),
                    },
                }
            )
        return normalized

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_registry: dict[str, ToolHandler],
    ) -> list[ToolExecution]:
        """Dispatch each tool call to its handler and collect results.

        Unknown tool names produce an ``{"error": "Unknown tool: …"}`` result
        rather than raising, so the model can see and react to the error.
        Handler exceptions are similarly captured as JSON error results.
        Both sync and async handlers are supported via ``inspect.isawaitable``.
        """
        executions: list[ToolExecution] = []
        for call in self.normalize_tool_calls(tool_calls):
            function  = call["function"]
            name      = function["name"]
            arguments = function["arguments"]
            handler   = tool_registry.get(name)
            if handler is None:
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

    def assistant_message(
        self, content: str, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the ``assistant`` message that records the model's tool-call requests.

        Arguments are serialized back to JSON strings because the OpenAI
        message format requires ``function.arguments`` to be a string, not a
        dict — even though we parse them into dicts for handler dispatch.
        """
        serialized = []
        for tc in self.normalize_tool_calls(tool_calls):
            tc = dict(tc)
            fn = dict(tc.get("function") or {})
            args = fn.get("arguments")
            fn["arguments"] = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else (args or "{}")
            tc["function"] = fn
            serialized.append(tc)
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": serialized,
        }

    def tool_messages(self, executions: list[ToolExecution]) -> list[dict[str, Any]]:
        """Build the ``tool`` role messages that carry handler results back to the model."""
        return [
            {
                "role": "tool",
                "tool_call_id": execution.tool_call_id,
                "name": execution.name,
                "content": execution.result,
            }
            for execution in executions
        ]

    def openai_tools(
        self, tool_schemas: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Pass tool schemas through unchanged; returns ``None`` when empty.

        The ``None`` return is intentional — the orchestrator checks truthiness
        to decide whether to include a ``tools`` key in the request.
        """
        if not tool_schemas:
            return None
        return [dict(schema) for schema in tool_schemas]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_value(self, obj: Any, key: str, default: Any = None) -> Any:
        """Read ``key`` from either a dict-like object or an attribute object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        """Coerce tool-call arguments to a dict regardless of how they arrived.

        Providers may send arguments as a JSON string, a pre-parsed dict, or
        occasionally something else entirely.  ``{"raw": …}`` / ``{"value": …}``
        wrappers let the handler receive *something* rather than crashing.
        """
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
        """Convert a handler return value to a string for the tool message.

        ``default=str`` keeps a successful handler that returns a
        non-JSON-native value (datetime, Decimal, custom object) from being
        misreported as a tool execution failure.
        """
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)
