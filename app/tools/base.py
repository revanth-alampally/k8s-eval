"""Tool contract and shared guards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import MutationDisabledError, NamespaceNotAllowedError, PermissionDeniedError
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


def require_mutating_tool_allowed(tool_name: str, settings: Settings) -> None:
    """Only explicitly configured mutation names may cross the executor boundary."""
    if tool_name not in settings.allowed_mutating_tools:
        raise PermissionDeniedError(
            f"Mutating tool '{tool_name}' is not allowlisted.",
            tool=tool_name,
        )


class ToolMetadata(BaseModel):
    """The public description of a tool: what the model is offered, and what the API
    can list. Carries no handler, so it is safe to serialise anywhere."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    requires_confirmation: bool


@dataclass(frozen=True)
class ToolSpec:
    """Everything needed to describe, validate and run one tool.

    ``read_only`` and ``requires_confirmation`` are static properties of the tool rather
    than something inferred from its arguments at call time. That is what makes the
    confirmation gate reliable: the orchestrator knows whether an operation changes
    state before it runs, without parsing anything.
    """

    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[..., BaseModel]
    read_only: bool = True
    requires_confirmation: bool = False

    @property
    def mutating(self) -> bool:
        return not self.read_only

    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema(),
            read_only=self.read_only,
            requires_confirmation=self.requires_confirmation,
        )
