"""The tool registry.

This module is the complete, reviewable answer to "what can the agent do to my
cluster?". Six entries, five of them read-only. There is deliberately no tool that
accepts a command string, and no escape hatch that reaches the Kubernetes API directly.

Adding a capability means adding a ``ToolSpec`` here, which makes capability growth a
visible change in a diff rather than something that emerges from a cleverer prompt.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import ToolNotFoundError
from app.tools.base import ToolMetadata, ToolSpec
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.diagnose import diagnose_pod
from app.tools.k8s.mutate import restart_deployment
from app.tools.k8s.read import (
    describe_pod,
    get_pod,
    get_pod_logs,
    list_deployments,
    list_pods,
)
from app.tools.schemas import (
    DescribePodInput,
    DiagnosePodInput,
    GetPodInput,
    GetPodLogsInput,
    ListDeploymentsInput,
    ListPodsInput,
    RestartDeploymentInput,
    parse_arguments,
)

_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_pods",
        description=(
            "List all pods in a namespace with phase, readiness, restart count and a "
            "health verdict. Use this for 'what pods are running' and 'are any pods "
            "unhealthy'."
        ),
        input_model=ListPodsInput,
        handler=list_pods,
    ),
    ToolSpec(
        name="get_pod",
        description=(
            "Get one pod's current state, including per-container status, images and "
            "restart counts."
        ),
        input_model=GetPodInput,
        handler=get_pod,
    ),
    ToolSpec(
        name="describe_pod",
        description=(
            "Get a pod together with its recent events. Use this to diagnose a pod that "
            "never started, such as an image pull failure, where no logs exist."
        ),
        input_model=DescribePodInput,
        handler=describe_pod,
    ),
    ToolSpec(
        name="diagnose_pod",
        description=(
            "Gather all available evidence about one pod in a single call: phase, "
            "container states, restart counts, unmet conditions, recent warning events "
            "and recent logs. Use this for 'why is this pod failing?'. It returns facts "
            "only and draws no conclusion; the explanation is yours to form, and must "
            "rest solely on the signals returned."
        ),
        input_model=DiagnosePodInput,
        handler=diagnose_pod,
    ),
    ToolSpec(
        name="get_pod_logs",
        description=(
            "Read the tail of a container's log. Set previous=true to read the log of a "
            "crashed container, which is required to explain a CrashLoopBackOff."
        ),
        input_model=GetPodLogsInput,
        handler=get_pod_logs,
    ),
    ToolSpec(
        name="list_deployments",
        description="List deployments in a namespace with desired and available replica counts.",
        input_model=ListDeploymentsInput,
        handler=list_deployments,
    ),
    ToolSpec(
        name="restart_deployment",
        description=(
            "Trigger a rolling restart of a deployment. This CHANGES CLUSTER STATE and "
            "requires explicit user confirmation before it runs."
        ),
        input_model=RestartDeploymentInput,
        handler=restart_deployment,
        read_only=False,
        requires_confirmation=True,
    ),
)


def build_registry(settings: Settings) -> Mapping[str, ToolSpec]:
    """Return the tools available under the current configuration.

    In read-only mode mutating tools are not registered at all, so they are absent from
    the schemas offered to the model. A tool the model cannot see is one it cannot try
    to talk its way into.
    """
    return {
        spec.name: spec for spec in _TOOL_SPECS if not (settings.read_only_mode and spec.mutating)
    }


def get_tool(name: str, settings: Settings) -> ToolSpec:
    registry = build_registry(settings)
    try:
        return registry[name]
    except KeyError:
        raise ToolNotFoundError(
            f"No tool named '{name}'.",
            tool=name,
            available_tools=sorted(registry),
        ) from None


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    client: KubernetesClient,
    settings: Settings,
) -> BaseModel:
    """Validate arguments and run a tool by name.

    The single entry point the agent layer will use. Argument validation happens here,
    before the handler is reached, so an invalid call costs nothing and touches nothing.
    """
    spec = get_tool(name, settings)
    args = parse_arguments(spec.input_model, **dict(arguments))
    result = spec.handler(**args.model_dump(), client=client, settings=settings)
    return result


def tool_metadata(settings: Settings) -> list[ToolMetadata]:
    """Describe the available tools, for the agent's tool definitions and for callers."""
    return [spec.metadata() for spec in build_registry(settings).values()]
