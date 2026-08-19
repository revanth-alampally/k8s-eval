"""Read-tool behaviour against a mocked Kubernetes API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from kubernetes.client import CoreV1Event, V1ObjectMeta, V1ObjectReference, V1PodList
from kubernetes.client.exceptions import ApiException

from app.config import Settings
from app.errors import ErrorCode, LogsUnavailableError, ToolArgumentError
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.models import ContainerState
from app.tools.k8s.read import (
    LOG_LIMIT_BYTES,
    describe_pod,
    get_pod,
    get_pod_logs,
    list_deployments,
    list_pods,
)
from tests.unit.factories import (
    NAMESPACE,
    NOW,
    crashloop_pod,
    deployment,
    healthy_pod,
    image_pull_pod,
    log_response,
)


def test_list_pods_separates_healthy_from_unhealthy(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.return_value = V1PodList(
        items=[healthy_pod(), crashloop_pod(), image_pull_pod()]
    )

    result = list_pods(NAMESPACE, client=k8s_client, settings=settings)

    assert result.total == 3
    assert result.unhealthy_count == 2
    assert result.unhealthy_pods == ["nginx-crash-def456", "nginx-missing-ghi789"]
    assert [pod.healthy for pod in result.pods] == [True, False, False]


def test_list_pods_reports_the_specific_failure_reason(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.return_value = V1PodList(items=[crashloop_pod(), image_pull_pod()])

    reasons = {
        pod.name: pod.status_reason
        for pod in list_pods(NAMESPACE, client=k8s_client, settings=settings).pods
    }

    assert reasons["nginx-crash-def456"] == "CrashLoopBackOff"
    assert reasons["nginx-missing-ghi789"] == "ImagePullBackOff"


def test_list_pods_applies_a_request_timeout(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.return_value = V1PodList(items=[])

    list_pods(NAMESPACE, client=k8s_client, settings=settings)

    kwargs = core_api.list_namespaced_pod.call_args.kwargs
    assert kwargs["_request_timeout"] == 5.0
    assert kwargs["timeout_seconds"] == 5


def test_get_pod_projects_container_detail(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = crashloop_pod()

    pod = get_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)

    assert pod.healthy is False
    assert pod.restart_count == 5
    assert pod.labels == {"app": "nginx-crash"}
    assert len(pod.containers) == 1
    container = pod.containers[0]
    assert container.state is ContainerState.WAITING
    assert container.reason == "CrashLoopBackOff"


def test_get_pod_detail_excludes_secret_bearing_fields(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Environment variables, volumes and annotations must never reach a tool result."""
    core_api.read_namespaced_pod.return_value = healthy_pod()

    payload = get_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)
    fields = set(payload.model_dump())

    assert not fields & {"env", "environment", "volumes", "annotations", "image_pull_secrets"}


def test_describe_pod_returns_events_sorted_newest_first(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(
        items=[
            _event("Pulling", datetime(2026, 8, 19, 12, 0, tzinfo=UTC)),
            _event("Failed", datetime(2026, 8, 19, 12, 5, tzinfo=UTC)),
        ]
    )

    description = describe_pod(
        NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings
    )

    assert [event.reason for event in description.events] == ["Failed", "Pulling"]
    assert description.pod.status_reason == "ImagePullBackOff"


def test_describe_pod_scopes_events_to_the_requested_pod(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])

    describe_pod(NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings)

    kwargs = core_api.list_namespaced_event.call_args.kwargs
    assert kwargs["field_selector"] == "involvedObject.name=nginx-missing-ghi789"
    assert kwargs["namespace"] == NAMESPACE


def test_get_pod_logs_forwards_container_and_limits(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod_log.return_value = log_response("line one\nline two\n")

    result = get_pod_logs(
        NAMESPACE,
        "nginx-crash-def456",
        container="nginx",
        tail_lines=50,
        previous=True,
        client=k8s_client,
        settings=settings,
    )

    kwargs = core_api.read_namespaced_pod_log.call_args.kwargs
    assert kwargs["container"] == "nginx"
    assert kwargs["tail_lines"] == 50
    assert kwargs["previous"] is True
    assert kwargs["limit_bytes"] == LOG_LIMIT_BYTES
    assert kwargs["_request_timeout"] == 5.0
    assert result.line_count == 2
    assert result.truncated is False


def test_get_pod_logs_reads_the_raw_response_body(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Regression guard: with the client's default deserialisation this endpoint
    returns the *repr* of the bytes, so the whole log arrives as one "b'...'" line."""
    response = log_response("first\nsecond\nthird\n")
    core_api.read_namespaced_pod_log.return_value = response

    result = get_pod_logs(
        NAMESPACE, "nginx-crash-def456", tail_lines=10, client=k8s_client, settings=settings
    )

    assert core_api.read_namespaced_pod_log.call_args.kwargs["_preload_content"] is False
    assert result.line_count == 3
    assert not result.content.startswith("b'")
    response.release_conn.assert_called_once()


def test_get_pod_logs_survives_invalid_utf8_output(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod_log.return_value = log_response(b"ok\n\xff\xfe binary noise\n")

    result = get_pod_logs(
        NAMESPACE, "nginx-crash-def456", tail_lines=10, client=k8s_client, settings=settings
    )

    assert result.line_count == 2


def test_get_pod_logs_reports_logs_unavailable_when_no_container_ever_started(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """An ImagePullBackOff pod has no log. That must not look like a bad argument, or
    the agent will retry instead of falling back to describe_pod."""
    exc = ApiException(status=400, reason="Bad Request")
    exc.body = json.dumps(
        {"message": 'container "nginx" in pod "nginx-missing" is waiting to start'}
    )
    core_api.read_namespaced_pod_log.side_effect = exc

    with pytest.raises(LogsUnavailableError) as caught:
        get_pod_logs(NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings)

    assert caught.value.code is ErrorCode.LOGS_UNAVAILABLE


def test_get_pod_logs_clamps_tail_lines_to_the_configured_maximum(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod_log.return_value = log_response("")

    result = get_pod_logs(
        NAMESPACE, "nginx-good-abc123", tail_lines=1500, client=k8s_client, settings=settings
    )

    assert core_api.read_namespaced_pod_log.call_args.kwargs["tail_lines"] == settings.max_log_lines
    assert result.tail_lines == settings.max_log_lines


def test_get_pod_logs_rejects_a_tail_beyond_the_absolute_bound(
    k8s_client: KubernetesClient, settings: Settings
) -> None:
    with pytest.raises(ToolArgumentError):
        get_pod_logs(
            NAMESPACE, "nginx-good-abc123", tail_lines=10_000, client=k8s_client, settings=settings
        )


def test_get_pod_logs_flags_truncation(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod_log.return_value = log_response(
        "\n".join(f"line {i}" for i in range(10))
    )

    result = get_pod_logs(
        NAMESPACE, "nginx-good-abc123", tail_lines=10, client=k8s_client, settings=settings
    )

    assert result.truncated is True


def test_list_deployments_reports_availability(
    k8s_client: KubernetesClient, apps_api: MagicMock, settings: Settings
) -> None:
    apps_api.list_namespaced_deployment.return_value = MagicMock(
        items=[
            deployment("nginx-good", desired=2, available=2),
            deployment("nginx-missing", desired=1, available=0),
        ]
    )

    result = list_deployments(NAMESPACE, client=k8s_client, settings=settings)

    assert result.total == 2
    assert result.unavailable_count == 1
    assert [item.available for item in result.deployments] == [True, False]
    assert result.deployments[0].images == ["nginx:1.27-alpine"]


def _event(reason: str, when: datetime) -> CoreV1Event:
    return CoreV1Event(
        metadata=V1ObjectMeta(name=f"nginx-missing.{reason}", namespace=NAMESPACE),
        involved_object=V1ObjectReference(name="nginx-missing-ghi789"),
        type="Warning",
        reason=reason,
        message=f"{reason} happened",
        count=1,
        first_timestamp=NOW,
        last_timestamp=when,
    )
