"""Builders for Kubernetes API objects used in tests.

Real ``kubernetes.client`` model classes rather than stubs, so a typo in a field name
in the conversion layer fails the test instead of silently reading ``None`` off a mock.
The scenarios mirror the workloads in ``k8s/``: a healthy pod, a CrashLoopBackOff pod
and an ImagePullBackOff pod.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from kubernetes.client import (
    V1Container,
    V1ContainerState,
    V1ContainerStateRunning,
    V1ContainerStateTerminated,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1Deployment,
    V1DeploymentSpec,
    V1DeploymentStatus,
    V1LabelSelector,
    V1ObjectMeta,
    V1Pod,
    V1PodSpec,
    V1PodStatus,
    V1PodTemplateSpec,
)

NAMESPACE = "ai-agent-demo"
NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


def healthy_pod(name: str = "nginx-good-abc123", image: str = "nginx:1.27-alpine") -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name, namespace=NAMESPACE, labels={"app": "nginx-good"}, creation_timestamp=NOW
        ),
        spec=V1PodSpec(
            containers=[V1Container(name="nginx", image=image)],
            node_name="hello-control-plane",
            service_account_name="default",
        ),
        status=V1PodStatus(
            phase="Running",
            pod_ip="10.244.0.6",
            container_statuses=[
                V1ContainerStatus(
                    name="nginx",
                    image=image,
                    image_id="docker.io/library/nginx@sha256:abc",
                    ready=True,
                    restart_count=0,
                    state=V1ContainerState(running=V1ContainerStateRunning(started_at=NOW)),
                )
            ],
        ),
    )


def crashloop_pod(name: str = "nginx-crash-def456") -> V1Pod:
    """Container has crashed and is waiting to be restarted: logs exist, but only for
    the previous container instance."""
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name, namespace=NAMESPACE, labels={"app": "nginx-crash"}, creation_timestamp=NOW
        ),
        spec=V1PodSpec(containers=[V1Container(name="nginx", image="nginx:1.27-alpine")]),
        status=V1PodStatus(
            phase="Running",
            container_statuses=[
                V1ContainerStatus(
                    name="nginx",
                    image="nginx:1.27-alpine",
                    image_id="docker.io/library/nginx@sha256:abc",
                    ready=False,
                    restart_count=5,
                    state=V1ContainerState(
                        waiting=V1ContainerStateWaiting(
                            reason="CrashLoopBackOff",
                            message="back-off 2m40s restarting failed container",
                        )
                    ),
                    last_state=V1ContainerState(
                        terminated=V1ContainerStateTerminated(
                            exit_code=1, reason="Error", started_at=NOW, finished_at=NOW
                        )
                    ),
                )
            ],
        ),
    )


def image_pull_pod(name: str = "nginx-missing-ghi789") -> V1Pod:
    """Never started a container, so it has no logs at all -- only events."""
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name, namespace=NAMESPACE, labels={"app": "nginx-missing"}, creation_timestamp=NOW
        ),
        spec=V1PodSpec(containers=[V1Container(name="nginx", image="nginx:1.27-does-not-exist")]),
        status=V1PodStatus(
            phase="Pending",
            container_statuses=[
                V1ContainerStatus(
                    name="nginx",
                    image="nginx:1.27-does-not-exist",
                    image_id="",
                    ready=False,
                    restart_count=0,
                    state=V1ContainerState(
                        waiting=V1ContainerStateWaiting(
                            reason="ImagePullBackOff",
                            message='Back-off pulling image "nginx:1.27-does-not-exist"',
                        )
                    ),
                )
            ],
        ),
    )


def log_response(content: str | bytes) -> MagicMock:
    """Stand-in for the raw urllib3 response returned when _preload_content=False.

    The tools read ``.data`` and then release the connection, so the double mirrors both.
    """
    response = MagicMock(name="HTTPResponse")
    response.data = content.encode("utf-8") if isinstance(content, str) else content
    return response


def deployment(
    name: str = "nginx-good",
    desired: int = 2,
    available: int = 2,
    image: str = "nginx:1.27-alpine",
) -> V1Deployment:
    return V1Deployment(
        metadata=V1ObjectMeta(name=name, namespace=NAMESPACE, creation_timestamp=NOW),
        spec=V1DeploymentSpec(
            replicas=desired,
            selector=V1LabelSelector(match_labels={"app": name}),
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels={"app": name}),
                spec=V1PodSpec(containers=[V1Container(name="nginx", image=image)]),
            ),
        ),
        status=V1DeploymentStatus(
            replicas=desired,
            ready_replicas=available,
            available_replicas=available,
            updated_replicas=desired,
        ),
    )
