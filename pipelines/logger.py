"""Logging layer for OpenAI-compatible pipeline runs."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _message_code(message: dict[str, Any]) -> str:
    raw = json.dumps(message, sort_keys=True, ensure_ascii=False, default=str)
    return "M" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6].upper()


class PipelineLogger:
    """File logger with context tags and repeated-message deduplication.

    Log format:

    - request messages are logged once and later referenced by short codes
    - request parameters are logged separately from messages
    - response text, tool calls, tool results, trace events, and run end are visible
    - ``set_context()``, ``save_state()``, and ``restore_state()`` are available
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("w", encoding="utf-8")
        self._call_num = 0
        self._start = 0.0
        self._levels: list[str] = []
        self._seen_messages: set[str] = set()
        self._end_written = False
        self._in_thinking = False
        self._first_chunk_written = False

    def set_context(
        self,
        phase: str = "",
        step: str = "",
        sub_phase: str = "",
        attempt: int | None = None,
    ) -> None:
        if phase:
            self._levels = [phase]
        if step:
            self._levels = self._levels[:1] + [step]
        if sub_phase:
            base = 1
            if len(self._levels) > 1 and self._levels[1].startswith("step:"):
                base = 2
            self._levels = self._levels[:base] + [sub_phase]
        if attempt is not None:
            self._levels = [level for level in self._levels if not level.isdigit()]
            self._levels.append(str(attempt))

    def log_request(self, body: dict[str, Any], measured_ctx: int | None = None) -> None:
        self._call_num += 1
        self._start = time.monotonic()
        self._end_written = False
        self._in_thinking = False
        self._first_chunk_written = False
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ctx_info = f" [measured_ctx={measured_ctx}]" if measured_ctx is not None else ""
        self._write(f"\n{ts} [INFO] pipeline.chat: request #{self._call_num}{ctx_info}{self._context_tag()}\n")

        for message in body.get("messages", []) or []:
            self._write_message(message)

        display_body = dict(body)
        if display_body.get("tools"):
            display_body["tools"] = [
                tool.get("function", {}).get("name", tool)
                if isinstance(tool, dict)
                else tool
                for tool in display_body["tools"]
            ]

        body_without_messages = {key: value for key, value in display_body.items() if key != "messages"}
        params_json = json.dumps(body_without_messages, indent=2, ensure_ascii=False, default=str)
        for line in params_json.splitlines():
            self._write(f"|REQ-PRM|{line}\n")
        self._write("\n")
        self._file.flush()

    def log_output_chunk(self, text: str, *, is_thinking: bool = False) -> None:
        if not text:
            return
        if is_thinking and not self._in_thinking:
            self._write("\n[THINKING]\n")
            self._in_thinking = True
        elif not is_thinking and self._in_thinking:
            self._write("\n[MESSAGE]\n")
            self._in_thinking = False
        elif not is_thinking and not self._first_chunk_written:
            self._write("\n[MESSAGE]\n")
        self._first_chunk_written = True
        self._write(text)
        self._file.flush()

    def log_tool_result(self, name: str, arguments: dict[str, Any], result: str) -> None:
        self._write_json_lines(
            "TOOL-RESULT",
            {"name": name, "arguments": arguments, "result": self._parse_json_or_text(result)},
        )

    def log_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        if tool_calls:
            self.log_info("[TOOL_CALLS]")
            self._write_json_lines("TOOL-CALL", tool_calls)

    def log_event(self, level: str, action: str, detail: dict[str, Any] | None = None) -> None:
        self._write_json_lines("EVENT", {"level": level, "action": action, "detail": detail or {}})

    def log_info(self, text: str) -> None:
        for line in str(text).splitlines():
            if line:
                self._write(f"|INFO| {line}\n")
        self._file.flush()

    def log_end(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        done_reason: str = "",
    ) -> None:
        if self._end_written:
            return
        self._end_written = True
        elapsed = time.monotonic() - self._start
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_info = f" [prompt_tokens={prompt_tokens}]" if prompt_tokens else ""
        completion_info = f" [completion_tokens={completion_tokens}]" if completion_tokens else ""
        reason_info = f" [done_reason={done_reason}]" if done_reason else ""
        self._write(
            f"\n{ts} [INFO] pipeline.chat: response_end #{self._call_num} "
            f"({elapsed:.1f}s){prompt_info}{completion_info}{reason_info}{self._context_tag()}\n"
        )
        self._file.flush()

    def save_state(self) -> dict[str, Any]:
        return {
            "call_num": self._call_num,
            "levels": list(self._levels),
            "seen_messages": set(self._seen_messages),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._call_num = state["call_num"]
        self._levels = list(state["levels"])
        self._seen_messages |= set(state["seen_messages"])

    def close(self) -> None:
        self._file.close()

    def _context_tag(self) -> str:
        if not self._levels:
            return ""
        return f" [{':'.join(self._levels)}]"

    def _write_message(self, message: dict[str, Any]) -> None:
        code = _message_code(message)
        if code in self._seen_messages:
            self._write(f"|REQ-MESSAGES| <{code}>\n")
            return
        self._seen_messages.add(code)
        message_json = json.dumps(message, indent=2, ensure_ascii=False, default=str)
        lines = message_json.splitlines()
        if not lines:
            self._write(f"|REQ-MESSAGES| {code}:## {{}}\n")
            return
        self._write(f"|REQ-MESSAGES| {code}:## {lines[0]}\n")
        for line in lines[1:]:
            self._write(f"|REQ-MESSAGES|{line}\n")

    def _write_json_lines(self, prefix: str, value: Any) -> None:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        for line in text.splitlines():
            self._write(f"|{prefix}|{line}\n")
        self._file.flush()

    def _parse_json_or_text(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return text

    def _write(self, text: str) -> None:
        self._file.write(text)


class LoggerPipeline:
    """Layer object that enables request/response/event logging."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        logger: PipelineLogger | None = None,
        log_events: bool = True,
        log_tool_results: bool = True,
    ) -> None:
        if logger is None:
            logger = PipelineLogger(path or os.environ.get("PIPELINE_LOG", "pipeline.log"))
        self.logger = logger
        self.log_events = log_events
        self.log_tool_results = log_tool_results

    def set_context(self, **kwargs: Any) -> None:
        self.logger.set_context(**kwargs)

    def close(self) -> None:
        self.logger.close()
