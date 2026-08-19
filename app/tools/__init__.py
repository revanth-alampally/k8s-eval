"""Deterministic tool layer.

Every fact the agent states about the cluster originates here. This package contains no
LLM code: tools are ordinary, unit-testable Python functions over the Kubernetes API,
each with a Pydantic argument model that is validated before anything is executed.

There is no tool that takes a command string, and no tool that shells out.
"""

from app.tools.base import ToolSpec
from app.tools.k8s.client import KubernetesClient, KubernetesClientProvider
from app.tools.k8s.diagnose import diagnose_pod
from app.tools.k8s.mutate import restart_deployment
from app.tools.k8s.read import (
    describe_pod,
    get_pod,
    get_pod_logs,
    list_deployments,
    list_pods,
)
from app.tools.registry import build_registry, execute_tool, get_tool, tool_schemas

__all__ = [
    "KubernetesClient",
    "KubernetesClientProvider",
    "ToolSpec",
    "build_registry",
    "describe_pod",
    "diagnose_pod",
    "execute_tool",
    "get_pod",
    "get_pod_logs",
    "get_tool",
    "list_deployments",
    "list_pods",
    "restart_deployment",
    "tool_schemas",
]
