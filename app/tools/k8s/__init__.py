"""Deterministic Kubernetes access. No LLM code lives below this point."""

from app.tools.k8s.client import KubernetesClient, KubernetesClientProvider
from app.tools.k8s.mutate import restart_deployment
from app.tools.k8s.read import (
    describe_pod,
    get_pod,
    get_pod_logs,
    list_deployments,
    list_pods,
)

__all__ = [
    "KubernetesClient",
    "KubernetesClientProvider",
    "describe_pod",
    "get_pod",
    "get_pod_logs",
    "list_deployments",
    "list_pods",
    "restart_deployment",
]
