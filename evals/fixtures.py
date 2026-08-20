"""Hermetic typed fixtures for evaluation runs; never imports a Kubernetes client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from app.errors import ClusterTimeoutError, ClusterUnavailableError, ResourceNotFoundError
from app.knowledge.service import KnowledgeHit
from app.tools.k8s.models import (
    DeploymentListResult,
    DeploymentSummary,
    DiagnosticSignal,
    PodDetail,
    PodDiagnosis,
    PodListResult,
    PodLogsResult,
    PodSummary,
    SignalSeverity,
    SignalSource,
)

NAMESPACE = "ai-agent-demo"


class FixtureKnowledge:
    def search(self, query: str) -> list[KnowledgeHit]:
        return [
            KnowledgeHit(
                content="ImagePullBackOff means Kubernetes could not pull the configured image.",
                source_path="README.md",
                chunk_index=0,
            )
        ]


class FixtureToolExecutor:
    """Small deterministic executor that records calls and forbids mutations."""

    def __init__(self, fixture: str = "mixed_cluster") -> None:
        self.fixture = fixture
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mutation_executed = False

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
        values = dict(arguments)
        self.calls.append((name, values))
        if self.fixture == "cluster_unavailable":
            raise ClusterUnavailableError("Fixture Kubernetes API is unavailable.")
        if self.fixture == "cluster_timeout":
            raise ClusterTimeoutError("Fixture Kubernetes API timed out.")
        if name == "restart_deployment":
            self.mutation_executed = True
            raise AssertionError("Evaluation fixtures never execute mutations.")
        if name == "list_pods":
            return _pod_list(values["namespace"])
        if name == "get_pod":
            return _pod_detail(values["namespace"], values["pod_name"])
        if name == "diagnose_pod":
            return _diagnosis(values["namespace"], values["pod_name"])
        if name == "get_pod_logs":
            return PodLogsResult(
                namespace=values["namespace"],
                pod_name=values["pod_name"],
                container=values.get("container"),
                previous=bool(values.get("previous", False)),
                tail_lines=int(values.get("tail_lines", 100)),
                line_count=1,
                truncated=False,
                content="fixture log line",
            )
        if name == "list_deployments":
            return DeploymentListResult(
                namespace=values["namespace"],
                total=2,
                unavailable_count=1,
                deployments=[
                    DeploymentSummary(
                        name="nginx-good",
                        namespace=values["namespace"],
                        available=True,
                        replicas_desired=2,
                        replicas_ready=2,
                        replicas_available=2,
                        replicas_updated=2,
                    ),
                    DeploymentSummary(
                        name="nginx-missing",
                        namespace=values["namespace"],
                        available=False,
                        replicas_desired=1,
                        replicas_ready=0,
                        replicas_available=0,
                        replicas_updated=0,
                    ),
                ],
            )
        raise AssertionError(f"Fixture does not allow tool {name}.")


def _pod_list(namespace: str) -> PodListResult:
    pods = [
        _summary("nginx-good-abc123", namespace, "Running", True),
        _summary("redis-abc123", namespace, "Running", True),
        _summary("nginx-crash-def456", namespace, "CrashLoopBackOff", False),
        _summary("nginx-missing-ghi789", namespace, "ImagePullBackOff", False),
    ]
    return PodListResult(
        namespace=namespace,
        total=len(pods),
        unhealthy_count=2,
        pods=pods,
        unhealthy_pods=["nginx-crash-def456", "nginx-missing-ghi789"],
    )


def _summary(name: str, namespace: str, status: str, healthy: bool) -> PodSummary:
    return PodSummary(
        name=name,
        namespace=namespace,
        phase="Running" if healthy else "Pending",
        healthy=healthy,
        status_reason=None if healthy else status,
        containers_ready=1 if healthy else 0,
        containers_total=1,
        restart_count=5 if status == "CrashLoopBackOff" else 0,
    )


def _pod_detail(namespace: str, pod_name: str) -> PodDetail:
    if pod_name.startswith("ghost"):
        raise ResourceNotFoundError(f"pod '{pod_name}' was not found.", name=pod_name)
    summary = next(pod for pod in _pod_list(namespace).pods if pod.name == pod_name)
    return PodDetail(**summary.model_dump())


def _diagnosis(namespace: str, pod_name: str) -> PodDiagnosis:
    if pod_name.startswith("ghost"):
        raise ResourceNotFoundError(f"pod '{pod_name}' was not found.", name=pod_name)
    reason = "CrashLoopBackOff" if "crash" in pod_name else "ImagePullBackOff"
    return PodDiagnosis(
        pod=pod_name,
        namespace=namespace,
        status=reason,
        healthy=False,
        phase="Pending",
        containers_ready=0,
        containers_total=1,
        restart_count=5 if reason == "CrashLoopBackOff" else 0,
        logs_available=reason != "ImagePullBackOff",
        signals=[
            DiagnosticSignal(
                source=SignalSource.CONTAINER_STATE,
                reason=reason,
                evidence=f"Fixture evidence: {reason}.",
                severity=SignalSeverity.WARNING,
            )
        ],
    )
