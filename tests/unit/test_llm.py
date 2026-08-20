"""Provider contract: adapters must speak the neutral types, never Kubernetes."""

from __future__ import annotations

import inspect

import pytest

from app.config import LLMProviderName, Settings
from app.errors import LLMUnavailableError
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolDefinition
from app.llm.factory import build_provider
from app.llm.fake import FakeLLMProvider, HeuristicLLMProvider, ScriptedLLMProvider


def test_generate_signature_is_the_whole_contract() -> None:
    params = set(inspect.signature(LLMProvider.generate).parameters)
    assert params == {"self", "messages", "tools", "temperature", "max_tokens"}
    assert "client" not in params
    assert "kubectl" not in params


def test_llm_response_cannot_carry_chain_of_thought() -> None:
    assert set(LLMResponse.model_fields) == {
        "content",
        "tool_calls",
        "model",
        "finish_reason",
    }


async def test_scripted_provider_is_the_fake_used_in_tests() -> None:
    fake: LLMProvider = FakeLLMProvider(
        [LLMResponse(content="ok", model="scripted", finish_reason="stop")]
    )
    response = await fake.generate(messages=[Message(role=Role.USER, content="hi")])
    assert response.content == "ok"
    assert not response.wants_tools


async def test_scripted_provider_records_offered_tools() -> None:
    provider = ScriptedLLMProvider(
        [LLMResponse(content="ok", model="scripted", finish_reason="stop")]
    )
    tool = ToolDefinition(name="list_pods", description="list", input_schema={"type": "object"})
    await provider.generate(
        messages=[Message(role=Role.USER, content="hi")],
        tools=[tool],
    )
    assert provider.tools_offered[0][0].name == "list_pods"


def test_factory_defaults_to_the_heuristic_fake(settings: Settings) -> None:
    provider = build_provider(settings)
    assert isinstance(provider, HeuristicLLMProvider)


def test_factory_refuses_openai_without_a_key(settings: Settings) -> None:
    configured = settings.model_copy(update={"llm_provider": LLMProviderName.OPENAI})
    with pytest.raises(LLMUnavailableError):
        build_provider(configured)


def test_tool_call_arguments_default_to_a_dict() -> None:
    call = ToolCall(id="1", name="list_pods")
    assert call.arguments == {}
