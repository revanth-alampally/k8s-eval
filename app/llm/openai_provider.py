"""OpenAI adapter.

The only file in the codebase that knows OpenAI's wire format. Everything the agent
does is expressed in the neutral types from ``app.llm.base``, so adding Anthropic or
Gemini means writing a sibling of this file and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.errors import LLMUnavailableError
from app.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)
from app.observability.instrumentation import track_operation


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float = 0.0,
    ) -> None:
        # Imported lazily so the package is only required when this provider is used.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._temperature = temperature

    async def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from openai import OpenAIError

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_message(message) for message in messages],
            "temperature": self._temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = [_to_openai_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        with track_operation("llm.generate", provider=self.name, model=self._model) as log:
            try:
                completion = await self._client.chat.completions.create(**payload)
            except OpenAIError as exc:
                # Never let a provider failure become a fabricated answer.
                raise LLMUnavailableError(
                    "The language model provider is unavailable.",
                    provider=self.name,
                    error_type=type(exc).__name__,
                ) from exc

            choice = completion.choices[0]
            log["finish_reason"] = choice.finish_reason
            if completion.usage is not None:
                log["total_tokens"] = completion.usage.total_tokens

        # Only content and tool calls are read. Any reasoning field the API returns is
        # deliberately dropped here so chain-of-thought cannot reach the agent.
        return LLMResponse(
            content=choice.message.content,
            tool_calls=[_from_openai_tool_call(call) for call in choice.message.tool_calls or []],
            model=completion.model,
            finish_reason=choice.finish_reason,
        )


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }

    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _from_openai_tool_call(call: Any) -> ToolCall:
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        # Malformed arguments are left empty on purpose: schema validation will reject
        # the call and the error is fed back to the model, which is the same path a
        # hallucinated argument takes.
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(id=call.id, name=call.function.name, arguments=arguments)
