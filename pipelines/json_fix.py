"""Structured-output helpers for schema hints, heuristic repair, and fit prompts."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from helpers.pydantic_helper import dict_to_pydantic_schema

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\":
                i += 1
                if i < len(text):
                    out.append(text[i])
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            elif ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 1
                continue
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _balanced_blocks(text: str) -> list[str]:
    """Return balanced JSON-like blocks in the order they appear."""
    blocks: list[str] = []
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            if not stack:
                start = idx
            stack.append("}" if ch == "{" else "]")
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                blocks.append(text[start : idx + 1])
                start = None
    return blocks


def _extract_last_balanced_json(text: str) -> str:
    blocks = _balanced_blocks(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _normalize_json_like(text: str) -> str:
    text = _THINK_RE.sub("", text)
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    text = _strip_json_comments(text)
    text = text.strip()
    text = _extract_last_balanced_json(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text.strip()


def _loads_best_effort(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"Expected JSON output as text, got {type(text).__name__}")

    candidates: list[str] = []
    raw = text.strip()
    if raw:
        candidates.append(raw)
    cleaned = _normalize_json_like(text)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    if not candidates:
        raise ValueError("No JSON found in model output: response was empty")

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    pythonish = cleaned
    pythonish = re.sub(r"\btrue\b", "True", pythonish)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish)
    pythonish = re.sub(r"\bnull\b", "None", pythonish)
    try:
        return ast.literal_eval(pythonish)
    except Exception as exc:
        raise ValueError(
            "Unable to repair JSON output. "
            f"Extracted candidate: {_preview(cleaned)!r}. "
            f"Raw output preview: {_preview(text)!r}"
        ) from exc


def _flat_dict_value_type(model_cls: type[BaseModel]) -> type | None:
    fields = getattr(model_cls, "model_fields", {})
    if not fields:
        return None
    types = {info.annotation for info in fields.values()}
    if len(types) == 1:
        candidate = next(iter(types))
        if candidate in {int, float, str, bool}:
            return candidate
    return None


@dataclass(slots=True)
class StructuredOutputRequest:
    """Compatibility container used by older call sites."""

    messages: list[dict[str, Any]]
    model_cls: type[BaseModel]
    schema_json: str


class JsonFixPipeline:
    """Helpers for schema hints, heuristic parsing, and fit prompts."""

    def __init__(self, *, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def build_request(
        self,
        *,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
    ) -> StructuredOutputRequest:
        model_cls = dict_to_pydantic_schema(schema_dict)
        schema_json = json.dumps(model_cls.model_json_schema(), ensure_ascii=False, indent=2)
        return StructuredOutputRequest(
            messages=self.append_schema_hint(messages, schema_json),
            model_cls=model_cls,
            schema_json=schema_json,
        )

    def append_schema_hint(self, messages: list[dict[str, Any]], schema_json: str) -> list[dict[str, Any]]:
        prepared = [dict(message) for message in messages]
        hint = f"\nOutput format schema:\n```json\n{schema_json}\n```"
        if not prepared:
            return [{"role": "user", "content": hint.lstrip()}]
        index = len(prepared) - 1
        base = prepared[index].get("content")
        base = base if isinstance(base, str) else "" if base is None else str(base)
        prepared[index]["content"] = base + hint
        return prepared

    def heuristic_fix(self, text: str) -> str:
        return _normalize_json_like(text)

    def parse(self, text: str, model_cls: type[BaseModel]) -> dict[str, Any]:
        parsed = _loads_best_effort(self.heuristic_fix(text))
        fields = list(getattr(model_cls, "model_fields", {}).keys())
        if isinstance(parsed, list) and len(fields) == 1:
            parsed = {fields[0]: parsed}
        elif isinstance(parsed, dict) and len(fields) == 2 and len(parsed) == 1:
            only_key, only_value = next(iter(parsed.items()))
            if isinstance(only_value, (list, tuple)) and len(only_value) == 2:
                parsed = {fields[0]: only_value[0], fields[1]: only_value[1]}
        if isinstance(parsed, dict):
            value_type = _flat_dict_value_type(model_cls)
            if value_type is not None and any(not isinstance(v, value_type) for v in parsed.values()):
                bad = [key for key, value in parsed.items() if not isinstance(value, value_type)]
                raise ValueError(f"Expected all values to be {value_type.__name__}, got wrong type for: {bad[:3]}")
            return model_cls.model_validate(parsed, strict=False).model_dump()
        return model_cls.model_validate(parsed, strict=False).model_dump()

    def build_fit_prompt(self, *, schema_json: str, raw_text: str) -> str:
        return (
            "Extract the response as JSON from the reply below.\n"
            "Choose the last JSON block if there are multiple.\n"
            "Correct invalid JSON, preserve the semantic content exactly, "
            "and return only JSON that matches the schema.\n\n"
            f"Schema:\n{schema_json}\n\n"
            f"Reply:\n{raw_text}"
        )

    def build_retry_messages(
        self,
        previous_text: str,
        error: Exception,
        schema_json: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "Previous JSON failed validation. "
                    "Repair ONLY the JSON. Do NOT add or remove semantic entities.\n"
                    f"Schema:\n{schema_json}\n\n"
                    f"Previous output:\n{previous_text}\n\n"
                    f"Validation error: {error}"
                ),
            }
        ]
