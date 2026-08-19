"""Every failure mode must arrive as a typed application error.

No ``ApiException`` or transport exception may escape the tools package: the agent has
to be able to distinguish "the pod does not exist" (a fact it can report) from "the API
did not answer" (a situation where it must not answer at all).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import MaxRetryError, ReadTimeoutError

from app.config import Settings
from app.errors import (
    AppError,
    ClusterTimeoutError,
    ClusterUnavailableError,
    ErrorCode,
    NamespaceNotAllowedError,
    PermissionDeniedError,
    ResourceNotFoundError,
    ToolArgumentError,
    ToolExecutionError,
)
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.read import get_pod, list_pods
from tests.unit.factories import NAMESPACE


def _api_exception(status: int, message: str, reason: str = "Failure") -> ApiException:
    exc = ApiException(status=status, reason=reason)
    exc.body = json.dumps({"kind": "Status", "status": "Failure", "message": message})
    return exc


@pytest.mark.parametrize(
    ("status", "expected", "code"),
    [
        (404, ResourceNotFoundError, ErrorCode.RESOURCE_NOT_FOUND),
        (401, PermissionDeniedError, ErrorCode.PERMISSION_DENIED),
        (403, PermissionDeniedError, ErrorCode.PERMISSION_DENIED),
        (408, ClusterTimeoutError, ErrorCode.CLUSTER_TIMEOUT),
        (504, ClusterTimeoutError, ErrorCode.CLUSTER_TIMEOUT),
        (400, ToolArgumentError, ErrorCode.TOOL_ARGUMENT_INVALID),
        (429, ClusterUnavailableError, ErrorCode.CLUSTER_UNAVAILABLE),
        (500, ClusterUnavailableError, ErrorCode.CLUSTER_UNAVAILABLE),
        (503, ClusterUnavailableError, ErrorCode.CLUSTER_UNAVAILABLE),
        (409, ToolExecutionError, ErrorCode.TOOL_EXECUTION_FAILED),
    ],
)
def test_api_status_maps_to_typed_error(
    k8s_client: KubernetesClient,
    core_api: MagicMock,
    settings: Settings,
    status: int,
    expected: type[AppError],
    code: ErrorCode,
) -> None:
    core_api.read_namespaced_pod.side_effect = _api_exception(status, "boom")

    with pytest.raises(expected) as caught:
        get_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    assert caught.value.code is code


def test_not_found_error_names_the_resource(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.side_effect = _api_exception(404, 'pods "missing" not found')

    with pytest.raises(ResourceNotFoundError) as caught:
        get_pod(NAMESPACE, "missing", client=k8s_client, settings=settings)

    assert "missing" in caught.value.message
    assert caught.value.details["namespace"] == NAMESPACE
    assert caught.value.details["kubernetes_status"] == 404


def test_read_timeout_maps_to_cluster_timeout(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.side_effect = ReadTimeoutError(None, "/api", "timed out")  # type: ignore[arg-type]

    with pytest.raises(ClusterTimeoutError):
        list_pods(NAMESPACE, client=k8s_client, settings=settings)


def test_connection_failure_maps_to_cluster_unavailable(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.side_effect = MaxRetryError(None, "/api", "connection refused")  # type: ignore[arg-type]

    with pytest.raises(ClusterUnavailableError) as caught:
        list_pods(NAMESPACE, client=k8s_client, settings=settings)

    assert caught.value.details["reason"] == "MaxRetryError"


def test_error_details_never_include_response_headers(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Headers can carry bearer tokens, so only the parsed message is propagated."""
    exc = _api_exception(403, "pods is forbidden: User cannot list resource")
    exc.headers = {"Authorization": "Bearer super-secret-token"}  # type: ignore[assignment]
    core_api.read_namespaced_pod.side_effect = exc

    with pytest.raises(PermissionDeniedError) as caught:
        get_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    serialised = json.dumps(caught.value.to_response().model_dump(mode="json"))
    assert "super-secret-token" not in serialised
    assert "forbidden" in serialised


def test_unparseable_error_body_does_not_crash_translation(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    exc = ApiException(status=500, reason="Internal Server Error")
    exc.body = "<html>gateway error</html>"
    core_api.read_namespaced_pod.side_effect = exc

    with pytest.raises(ClusterUnavailableError):
        get_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)


def test_namespace_outside_the_allowlist_is_refused_before_any_api_call(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    with pytest.raises(NamespaceNotAllowedError) as caught:
        list_pods("kube-system", client=k8s_client, settings=settings)

    core_api.list_namespaced_pod.assert_not_called()
    assert caught.value.details["allowed_namespaces"] == [NAMESPACE]
