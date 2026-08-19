"""Argument validation is the boundary between an untrusted caller and the cluster."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.errors import ToolArgumentError
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.read import get_pod, list_pods
from tests.unit.factories import NAMESPACE, healthy_pod

# Names that must never reach the Kubernetes API. Kubernetes would reject most of them
# too, but the point is that they fail here, before any call is made.
HOSTILE_NAMES = [
    "nginx; rm -rf /",
    "../../etc/passwd",
    "nginx pod",
    "NGINX",
    "nginx$(whoami)",
    "nginx,involvedObject.namespace=kube-system",
    "nginx&watch=true",
    "-leading-dash",
    "trailing-dash-",
    "",
    "a" * 254,
]


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_invalid_pod_names_are_rejected_without_calling_the_api(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings, name: str
) -> None:
    with pytest.raises(ToolArgumentError):
        get_pod(NAMESPACE, name, client=k8s_client, settings=settings)

    core_api.read_namespaced_pod.assert_not_called()


@pytest.mark.parametrize("namespace", ["kube system", "Default", "ns/../other", "a" * 64])
def test_invalid_namespaces_are_rejected(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings, namespace: str
) -> None:
    with pytest.raises(ToolArgumentError):
        list_pods(namespace, client=k8s_client, settings=settings)

    core_api.list_namespaced_pod.assert_not_called()


def test_validation_error_reports_the_offending_field(
    k8s_client: KubernetesClient, settings: Settings
) -> None:
    with pytest.raises(ToolArgumentError) as caught:
        get_pod(NAMESPACE, "Not Valid", client=k8s_client, settings=settings)

    fields = {error["field"] for error in caught.value.details["errors"]}
    assert fields == {"pod_name"}


def test_names_are_whitespace_trimmed_rather_than_rejected(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = healthy_pod()

    get_pod(NAMESPACE, "  nginx-good-abc123  ", client=k8s_client, settings=settings)

    assert core_api.read_namespaced_pod.call_args.kwargs["name"] == "nginx-good-abc123"
