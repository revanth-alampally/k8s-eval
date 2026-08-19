"""Tool contract and shared guards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import MutationDisabledError, NamespaceNotAllowedError
from app.tools.schemas import ToolInput


def require_allowed_namespace(namespace: str, settings: Settings) -> None:
    """Fail closed on any namespace outside the configured allowlist.

    Checked inside every tool rather than once at the edge, so a new caller or a future
    code path cannot reach the cluster without passing this gate.
    """
    if not settings.is_namespace_allowed(namespace):
        raise NamespaceNotAllowedError(
            f"Namespace '{namespace}' is not accessible to this agent.",
            namespace=namespace,
            allowed_namespaces=list(settings.allowed_namespaces),
        )


def require_mutations_enabled(tool_name: str, settings: Settings) -> None:
    """Second line of defence behind the registry's read-only filter."""
    if settings.read_only_mode:
        raise MutationDisabledError(
            f"'{tool_name}' changes cluster state and the agent is in read-only mode.",
            tool=tool_name,
        )


@dataclass(frozen=True)
class ToolSpec:
    """Everything needed to describe, validate and run one tool.

    ``mutating`` is a static property of the tool rather than something inferred from
    its arguments at call time. That is what makes the confirmation gate reliable: the
    orchestrator can tell whether an operation changes state before running it.
    """

    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[..., BaseModel]
    mutating: bool = False

    def json_schema(self) -> dict[str, Any]:
        """Argument schema, for later use as an LLM tool definition."""
        return self.input_model.model_json_schema()
