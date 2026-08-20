"""Provider selection."""

from __future__ import annotations

from app.config import LLMProviderName, Settings
from app.errors import LLMUnavailableError
from app.llm.base import LLMProvider
from app.llm.fake import HeuristicLLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the configured provider.

    Defaults to the heuristic fake so the service is runnable, and testable end to end,
    with no API key and no network access.
    """
    if settings.llm_provider is LLMProviderName.OPENAI:
        if settings.llm_api_key is None:
            raise LLMUnavailableError(
                "KAGENT_LLM_API_KEY is required when the OpenAI provider is selected.",
                provider="openai",
            )
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout_seconds=float(settings.llm_timeout_seconds),
            temperature=settings.llm_temperature,
        )

    namespace = settings.default_namespace
    return HeuristicLLMProvider(default_namespace=namespace)
