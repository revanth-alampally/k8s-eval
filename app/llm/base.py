"""Provider-neutral LLM interface.

The agent talks only to these types. Swapping OpenAI for Anthropic, Gemini or a fake
means writing one adapter, with no change to orchestration, prompts or tool handling.

The response type deliberately exposes only ``content`` and ``tool_calls``. Providers
that emit reasoning traces must drop them in their adapter: chain-of-thought never
enters this layer, so it cannot leak into an API response by accident.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A model's request to run one tool. Arguments are unvalidated at this point."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str | None = None
    # Set on assistant messages that requested tools.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set on tool messages, tying the result back to the request.
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    """A tool as advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = "unknown"
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """One method, so an adapter is small enough to be obviously correct."""

    name: str = "unknown"

    @abstractmethod
    async def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Produce the next assistant turn.

        Implementations must raise ``LLMUnavailableError`` for transport, auth and rate
        limit failures, so the agent can report a failure instead of inventing an answer.
        """
