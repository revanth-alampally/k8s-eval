"""Bridge the provider-neutral LLM contract into LangChain's chat-model API."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

from app.llm.base import LLMProvider, Message, Role, ToolCall, ToolDefinition


class ProviderChatModel(BaseChatModel):
    """A LangChain model that delegates every turn to :class:`LLMProvider`.

    Provider adapters remain the only vendor-aware code. This bridge translates
    LangChain's messages/tool schemas into the small neutral protocol the application
    already owns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: LLMProvider
    _tools: list[ToolDefinition] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return f"provider:{self.provider.name}"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **_: Any,
    ) -> ProviderChatModel:
        bound = self.model_copy(deep=False)
        bound._tools = [_tool_definition(tool) for tool in tools]
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous fallback for LangChain callers outside an event loop."""
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=None, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        response = await self.provider.generate(
            messages=[_provider_message(message) for message in messages],
            tools=self._tools,
        )
        tool_calls = [
            {
                "name": call.name,
                "args": call.arguments,
                "id": call.id,
                "type": "tool_call",
            }
            for call in response.tool_calls
        ]
        message = AIMessage(content=response.content or "", tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_definition(tool: Any) -> ToolDefinition:
    if isinstance(tool, dict):
        function = tool.get("function", tool)
        return ToolDefinition(
            name=str(function["name"]),
            description=str(function.get("description", "")),
            input_schema=dict(function.get("parameters", {})),
        )

    schema = tool.args_schema.model_json_schema() if tool.args_schema is not None else {}
    return ToolDefinition(name=tool.name, description=tool.description or "", input_schema=schema)


def _provider_message(message: BaseMessage) -> Message:
    if isinstance(message, SystemMessage):
        return Message(role=Role.SYSTEM, content=_content(message.content))
    if isinstance(message, HumanMessage):
        return Message(role=Role.USER, content=_content(message.content))
    if isinstance(message, ToolMessage):
        return Message(
            role=Role.TOOL,
            content=_content(message.content),
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
    if isinstance(message, AIMessage):
        return Message(
            role=Role.ASSISTANT,
            content=_content(message.content),
            tool_calls=[
                ToolCall(
                    id=str(call["id"]),
                    name=str(call["name"]),
                    arguments=dict(call.get("args", {})),
                )
                for call in message.tool_calls
            ],
        )
    return Message(role=Role.ASSISTANT, content=_content(message.content))


def _content(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return str(content)
