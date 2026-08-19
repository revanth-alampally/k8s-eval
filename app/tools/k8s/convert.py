"""Conversion from Kubernetes API objects to the typed models.

Kept separate from the tool functions so the mapping -- especially the health rule --
can be unit tested against hand-built API objects with no client involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.tools.k8s.models import (
    ContainerState,
    ContainerStatusSummary,
    DeploymentSummary,
    EventSummary,
    PodDetail,
    PodSummary,
    ResourceCondition,
)

# Terminated with this reason is a successful exit, not a failure.
_SUCCESSFUL_TERMINATION = "Completed"


def container_summary(status: Any) -> ContainerStatusSummary:
    state = getattr(status, "state", None)
    running = getattr(state, "running", None)
    waiting = getattr(state, "waiting", None)
    terminated = getattr(state, "terminated", None)

    if running is not None:
        kind = ContainerState.RUNNING
    elif waiting is not None:
        kind = ContainerState.WAITING
    elif terminated is not None:
        kind = ContainerState.TERMINATED
    else:
        kind = ContainerState.UNKNOWN

    source = waiting or terminated
    return ContainerStatusSummary(
        name=str(getattr(status, "name", "")),
        image=str(getattr(status, "image", "")),
        ready=bool(getattr(status, "ready", False)),
        restart_count=int(getattr(status, "restart_count", 0) or 0),
        state=kind,
        reason=getattr(source, "reason", None),
        message=getattr(source, "message", None),
        exit_code=getattr(terminated, "exit_code", None),
        started_at=getattr(running, "started_at", None) or getattr(terminated, "started_at", None),
        finished_at=getattr(terminated, "finished_at", None),
    )


def _status_reason(pod: Any, statuses: list[Any]) -> str | None:
    """Most specific failure reason available, preferring container-level detail."""
    for status in statuses:
        state = getattr(status, "state", None)
        waiting = getattr(state, "waiting", None)
        if waiting is not None and getattr(waiting, "reason", None):
            return str(waiting.reason)
    for status in statuses:
        state = getattr(status, "state", None)
        terminated = getattr(state, "terminated", None)
        reason = getattr(terminated, "reason", None)
        if reason and reason != _SUCCESSFUL_TERMINATION:
            return str(reason)
    pod_reason = getattr(getattr(pod, "status", None), "reason", None)
    return str(pod_reason) if pod_reason else None


def pod_summary(pod: Any) -> PodSummary:
    metadata = getattr(pod, "metadata", None)
    spec = getattr(pod, "spec", None)
    status = getattr(pod, "status", None)

    container_statuses = list(getattr(status, "container_statuses", None) or [])
    spec_containers = list(getattr(spec, "containers", None) or [])

    ready = sum(1 for item in container_statuses if getattr(item, "ready", False))
    total = len(spec_containers) or len(container_statuses)
    phase = str(getattr(status, "phase", None) or "Unknown")

    return PodSummary(
        name=str(getattr(metadata, "name", "") or ""),
        namespace=str(getattr(metadata, "namespace", "") or ""),
        phase=phase,
        healthy=_is_healthy(phase, ready, total),
        status_reason=_status_reason(pod, container_statuses),
        containers_ready=ready,
        containers_total=total,
        restart_count=sum(
            int(getattr(item, "restart_count", 0) or 0) for item in container_statuses
        ),
        node_name=getattr(spec, "node_name", None),
        pod_ip=getattr(status, "pod_ip", None),
        created_at=getattr(metadata, "creation_timestamp", None),
        images=[str(container.image) for container in spec_containers if container.image],
    )


def _is_healthy(phase: str, ready: int, total: int) -> bool:
    if phase == "Succeeded":
        return True
    return phase == "Running" and total > 0 and ready == total


def pod_detail(pod: Any) -> PodDetail:
    summary = pod_summary(pod)
    metadata = getattr(pod, "metadata", None)
    spec = getattr(pod, "spec", None)
    status = getattr(pod, "status", None)

    return PodDetail(
        **summary.model_dump(),
        labels={str(k): str(v) for k, v in (getattr(metadata, "labels", None) or {}).items()},
        service_account=getattr(spec, "service_account_name", None),
        owner=_owner(metadata),
        containers=[
            container_summary(item) for item in getattr(status, "container_statuses", None) or []
        ],
        conditions=[
            ResourceCondition(
                type=str(condition.type),
                status=str(condition.status),
                reason=getattr(condition, "reason", None),
                message=getattr(condition, "message", None),
                last_transition_time=getattr(condition, "last_transition_time", None),
            )
            for condition in getattr(status, "conditions", None) or []
        ],
    )


def _owner(metadata: Any) -> str | None:
    for reference in getattr(metadata, "owner_references", None) or []:
        if getattr(reference, "controller", False):
            return f"{reference.kind}/{reference.name}"
    return None


def event_summary(event: Any) -> EventSummary:
    return EventSummary(
        type=str(getattr(event, "type", None) or "Normal"),
        reason=getattr(event, "reason", None),
        message=getattr(event, "message", None),
        count=int(getattr(event, "count", 1) or 1),
        first_seen=getattr(event, "first_timestamp", None),
        last_seen=event_timestamp(event),
    )


def event_timestamp(event: Any) -> datetime | None:
    """Best available time for an event; the fields populated vary by event source."""
    return (
        getattr(event, "last_timestamp", None)
        or getattr(event, "event_time", None)
        or getattr(event, "first_timestamp", None)
    )


def event_sort_key(event: Any) -> datetime:
    return event_timestamp(event) or datetime.min.replace(tzinfo=UTC)


def deployment_summary(deployment: Any) -> DeploymentSummary:
    metadata = getattr(deployment, "metadata", None)
    spec = getattr(deployment, "spec", None)
    status = getattr(deployment, "status", None)

    desired = int(getattr(spec, "replicas", 0) or 0)
    available = int(getattr(status, "available_replicas", 0) or 0)
    template_spec = getattr(getattr(spec, "template", None), "spec", None)
    containers = list(getattr(template_spec, "containers", None) or [])

    return DeploymentSummary(
        name=str(getattr(metadata, "name", "") or ""),
        namespace=str(getattr(metadata, "namespace", "") or ""),
        available=desired > 0 and available >= desired,
        replicas_desired=desired,
        replicas_ready=int(getattr(status, "ready_replicas", 0) or 0),
        replicas_available=available,
        replicas_updated=int(getattr(status, "updated_replicas", 0) or 0),
        images=[str(container.image) for container in containers if container.image],
        created_at=getattr(metadata, "creation_timestamp", None),
        conditions=[
            ResourceCondition(
                type=str(condition.type),
                status=str(condition.status),
                reason=getattr(condition, "reason", None),
                message=getattr(condition, "message", None),
                last_transition_time=getattr(condition, "last_transition_time", None),
            )
            for condition in getattr(status, "conditions", None) or []
        ],
    )
