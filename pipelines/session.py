"""Stateful chat session layer built on top of pipelined chat completions."""
from __future__ import annotations

from typing import Any

from .chat import PipelinedChatCompletionResult


class ChatMessages(list[dict[str, Any]]):
    """List of chat messages with an explicit role-content shorthand helper."""

    def append_role_content(self, role_content: dict[str, Any]) -> None:
        """Append one or more ``{"role": "content"}`` shorthand messages."""
        self.extend_role_content(role_content)

    def extend_role_content(self, role_content: dict[str, Any]) -> None:
        """Extend with shorthand messages preserving mapping insertion order."""
        for role, content in role_content.items():
            self.append({"role": str(role), "content": content})

    def copy_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy safe to pass to a chat completion call."""
        return [dict(message) for message in self]


class ChatSession:
    """Stateful dialogue wrapper with default create parameters."""

    def __init__(self, chat: Any, **default_params: Any) -> None:
        self.chat = chat
        self.default_params = dict(default_params)
        self.messages = ChatMessages()
        self.last_result: PipelinedChatCompletionResult | None = None

    async def step(
        self,
        *,
        role_content: dict[str, Any] | None = None,
        auto_append: bool = True,
        **override_params: Any,
    ) -> PipelinedChatCompletionResult:
        """Run one dialogue step.

        Parameters
        ----------
        role_content:
            Optional shorthand messages to append before the model call. For
            example ``{"user": "Continue"}`` becomes
            ``{"role": "user", "content": "Continue"}``.
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


def chat_session(chat: Any, **default_params: Any) -> ChatSession:
    """Create a stateful chat session around a pipelined chat object."""
    return ChatSession(chat, **default_params)
