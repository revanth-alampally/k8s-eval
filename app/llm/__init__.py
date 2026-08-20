"""Provider-neutral LLM access.

The agent depends on ``LLMProvider`` only. Vendor specifics live in adapters here.
"""

from app.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)
from app.llm.factory import build_provider
from app.llm.fake import FakeLLMProvider, HeuristicLLMProvider, ScriptedLLMProvider

__all__ = [
    "FakeLLMProvider",
    "HeuristicLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "Role",
    "ScriptedLLMProvider",
    "ToolCall",
    "ToolDefinition",
    "build_provider",
]
