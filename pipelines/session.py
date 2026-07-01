"""Stateful chat session built on top of Pipelined Chat."""
from __future__ import annotations

from typing import Any

from .chat import PipelinedChatCompletionResult

RoleContent = dict[str, Any] | list[dict[str, Any]]


class ChatMessages(list[dict[str, Any]]):
    """List of chat messages with role_content helpers."""

    def append_role_content(self, role_content: RoleContent) -> None:
        """Append ``{role: content}`` entries, or a list of those dicts."""
        self.extend_role_content(role_content)

    def extend_role_content(self, role_content: RoleContent) -> None:
        """Extend with ``{role: content}`` entries, or a list of those dicts."""
        if isinstance(role_content, list):
            for item in role_content:
                self._extend_role_content_dict(item)
            return
        if not isinstance(role_content, dict):
            raise TypeError("role_content must be a {role: content} dict or a list of those dicts")
        self._extend_role_content_dict(role_content)

    def copy_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy safe to pass to a chat completion call."""
        return [dict(message) for message in self]

    def _extend_role_content_dict(self, item: Any) -> None:
        if not isinstance(item, dict):
            raise TypeError("role_content list items must be dicts")
        for role, content in item.items():
            self.append({"role": str(role), "content": content})


class ChatSession:
    """Stateful dialogue wrapper with default create parameters."""

    def __init__(
        self,
        chat: Any,
        *,
        role_content: RoleContent | None = None,
        **default_params: Any,
    ) -> None:
        self.chat = chat
        self.default_params = dict(default_params)
        self.messages = ChatMessages()
        if role_content is not None:
            self.messages.append_role_content(role_content)
        self.last_result: PipelinedChatCompletionResult | None = None

    async def step(
        self,
        *,
        role_content: RoleContent | None = None,
        auto_append: bool = True,
        **override_params: Any,
    ) -> PipelinedChatCompletionResult:
        """Run one dialogue step.

        Parameters
        ----------
        role_content:
            Optional ``{role: content}`` dict, or a list of those dicts, to
            append before the model call.
        auto_append:
            When True, append the final assistant answer to ``session.messages``
            after the model call succeeds. This does not control
            ``role_content``; pre-step role content is appended before the call.
        **override_params:
            Per-step chat parameters. These override ``default_params`` and are
            forwarded to the wrapped chat object.
        """
        if role_content is not None:
            self.messages.append_role_content(role_content)
        params = {**self.default_params, **override_params}
        result = await self.chat.create(
            messages=self.messages.copy_messages(),
            **params,
        )
        self.last_result = result
        if auto_append:
            self.append_result(result)
        return result

    def append_result(self, result: PipelinedChatCompletionResult | None = None) -> None:
        """Append the final assistant message from a result to session history."""
        selected = result or self.last_result
        if selected is None:
            raise ValueError("No result available to append")
        message = self._assistant_message_from_result(selected)
        self.messages.append(message)

    def clear(self) -> None:
        """Clear messages and last result, keeping default params."""
        self.messages.clear()
        self.last_result = None

    def _assistant_message_from_result(self, result: PipelinedChatCompletionResult) -> dict[str, Any]:
        response = result.response
        if hasattr(response, "model_dump"):
            response = response.model_dump(mode="json")
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict-like response, got {type(response).__name__}")
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("Result response does not contain choices")
        choice = choices[0]
        if hasattr(choice, "model_dump"):
            choice = choice.model_dump(mode="json")
        if not isinstance(choice, dict):
            raise TypeError("Result choice must be dict-like")
        message = choice.get("message") or {}
        if hasattr(message, "model_dump"):
            message = message.model_dump(mode="json")
        if not isinstance(message, dict):
            raise TypeError("Result message must be dict-like")
        return dict(message)


def chat_session(
    chat: Any,
    *,
    role_content: RoleContent | None = None,
    **default_params: Any,
) -> ChatSession:
    """Create a stateful chat session around Pipelined Chat."""
    return ChatSession(chat, role_content=role_content, **default_params)
