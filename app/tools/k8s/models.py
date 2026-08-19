"""Typed projections of Kubernetes objects.

These models are a deliberately narrow view of the underlying resources. What is left
out matters as much as what is included: container environment variables, volumes,
image pull secrets, service account tokens and annotations are all excluded, so secret
material cannot reach the model's context by way of a tool result. Adding a field here
is the moment to ask whether it can carry a secret.

Health is decided here, in Python, rather than left to the model to infer from raw
status text -- so "is this pod unhealthy?" has one testable answer.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ContainerState(StrEnum):
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class ContainerStatusSummary(BaseModel):
    name: str
    image: str
    ready: bool
    restart_count: int
    state: ContainerState
    # For a waiting container this is e.g. ImagePullBackOff / CrashLoopBackOff; for a
    # terminated one, Error / Completed / OOMKilled.
    reason: str | None = None
    message: str | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ResourceCondition(BaseModel):
    type: str
    status: str
    reason: str | None = None
    message: str | None = None
    last_transition_time: datetime | None = None


class PodSummary(BaseModel):
    name: str
    namespace: str
    phase: str
    healthy: bool = Field(
        description="True only when the pod is Running with every container ready, "
        "or has Succeeded."
    )
    status_reason: str | None = Field(
        default=None,
        description="Most specific reason available: a container's waiting or "
        "terminated reason if there is one, otherwise the pod-level reason.",
    )
    containers_ready: int
    containers_total: int
    restart_count: int
    node_name: str | None = None
    pod_ip: str | None = None
    created_at: datetime | None = None
    images: list[str] = Field(default_factory=list)


class PodDetail(PodSummary):
    labels: dict[str, str] = Field(default_factory=dict)
    service_account: str | None = None
    owner: str | None = Field(
        default=None, description="Controller reference, e.g. 'ReplicaSet/x'."
    )
    containers: list[ContainerStatusSummary] = Field(default_factory=list)
    conditions: list[ResourceCondition] = Field(default_factory=list)


class PodListResult(BaseModel):
    namespace: str
    total: int
    unhealthy_count: int
    pods: list[PodSummary] = Field(default_factory=list)
    unhealthy_pods: list[str] = Field(
        default_factory=list,
        description="Names of pods failing the health check, for direct use in answers.",
    )


class EventSummary(BaseModel):
    type: str
    reason: str | None = None
    message: str | None = None
    count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class PodDescription(BaseModel):
    """Pod detail plus its recent events.

    Events are the only evidence available for failures that happen before a container
    starts (image pull errors, scheduling failures), where logs do not exist.
    """

    pod: PodDetail
    events: list[EventSummary] = Field(default_factory=list)


class PodLogsResult(BaseModel):
    namespace: str
    pod_name: str
    container: str | None = None
    previous: bool = False
    tail_lines: int
    line_count: int
    truncated: bool = Field(
        description="True when the log was cut off by the line or byte limit, so the "
        "caller knows the output is partial."
    )
    content: str


class DeploymentSummary(BaseModel):
    name: str
    namespace: str
    available: bool = Field(description="True when every desired replica is available.")
    replicas_desired: int
    replicas_ready: int
    replicas_available: int
    replicas_updated: int
    images: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    conditions: list[ResourceCondition] = Field(default_factory=list)


class DeploymentListResult(BaseModel):
    namespace: str
    total: int
    unavailable_count: int
    deployments: list[DeploymentSummary] = Field(default_factory=list)


class RestartDeploymentResult(BaseModel):
    namespace: str
    deployment_name: str
    restarted_at: datetime
    replicas_desired: int
    message: str = Field(
        description="Plain statement of what changed, suitable for reporting verbatim."
    )
