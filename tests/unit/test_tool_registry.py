"""Registry behaviour, including the mutation guards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kubernetes.client import V1PodList

from app.config import Settings
from app.errors import (
    MutationDisabledError,
    PermissionDeniedError,
    ToolArgumentError,
    ToolNotFoundError,
)
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.models import PodListResult, RestartDeploymentResult
from app.tools.k8s.mutate import RESTART_ANNOTATION, restart_deployment
from app.tools.registry import build_registry, execute_tool, get_tool, tool_metadata
from tests.unit.factories import NAMESPACE, deployment, healthy_pod

EXPECTED_TOOLS = {
    "list_pods",
    "get_pod",
    "describe_pod",
    "diagnose_pod",
    "get_pod_logs",
    "list_deployments",
    "restart_deployment",
}


def test_registry_exposes_exactly_the_intended_tools(settings: Settings) -> None:
    assert set(build_registry(settings)) == EXPECTED_TOOLS


def test_no_tool_accepts_a_free_form_command(settings: Settings) -> None:
    """Guards the core design rule: no shell, no kubectl passthrough, no eval.

    A future tool that takes a ``command``/``script``/``query`` string would let the
    model do arbitrary things through a single approved entry point, so it must not be
    possible to add one without this test failing.
    """
    forbidden_argument_names = {"command", "cmd", "script", "shell", "exec", "query", "args"}

    for spec in build_registry(settings).values():
        assert not set(spec.input_model.model_fields) & forbidden_argument_names
        assert not any(word in spec.name for word in ("exec", "shell", "kubectl", "run", "apply"))


def test_only_restart_deployment_is_mutating(settings: Settings) -> None:
    mutating = {name for name, spec in build_registry(settings).items() if spec.mutating}

    assert mutating == {"restart_deployment"}


def test_read_only_mode_removes_mutating_tools_entirely(settings: Settings) -> None:
    read_only = settings.model_copy(update={"read_only_mode": True})

    registry = build_registry(read_only)

    assert "restart_deployment" not in registry
    assert set(registry) == EXPECTED_TOOLS - {"restart_deployment"}
    with pytest.raises(ToolNotFoundError):
        get_tool("restart_deployment", read_only)


def test_restart_deployment_refuses_even_when_called_directly_in_read_only_mode(
    k8s_client: KubernetesClient, apps_api: MagicMock, settings: Settings
) -> None:
    """Defence in depth: bypassing the registry must not bypass the guard."""
    read_only = settings.model_copy(update={"read_only_mode": True})

    with pytest.raises(MutationDisabledError):
        restart_deployment("ai-agent-demo", "nginx-good", client=k8s_client, settings=read_only)

    apps_api.patch_namespaced_deployment.assert_not_called()


def test_restart_deployment_refuses_when_not_in_mutation_allowlist(
    k8s_client: KubernetesClient, apps_api: MagicMock, settings: Settings
) -> None:
    blocked = settings.model_copy(update={"allowed_mutating_tools": []})

    with pytest.raises(PermissionDeniedError):
        restart_deployment("ai-agent-demo", "nginx-good", client=k8s_client, settings=blocked)

    apps_api.patch_namespaced_deployment.assert_not_called()


def test_restart_deployment_patches_the_restart_annotation(
    k8s_client: KubernetesClient, apps_api: MagicMock, settings: Settings
) -> None:
    apps_api.patch_namespaced_deployment.return_value = deployment("nginx-good", desired=2)

    result = restart_deployment(NAMESPACE, "nginx-good", client=k8s_client, settings=settings)

    kwargs = apps_api.patch_namespaced_deployment.call_args.kwargs
    annotations = kwargs["body"]["spec"]["template"]["metadata"]["annotations"]
    assert RESTART_ANNOTATION in annotations
    assert kwargs["name"] == "nginx-good"
    assert kwargs["namespace"] == NAMESPACE
    assert isinstance(result, RestartDeploymentResult)
    assert result.replicas_desired == 2


def test_restart_deployment_does_not_delete_anything(
    k8s_client: KubernetesClient, apps_api: MagicMock, core_api: MagicMock, settings: Settings
) -> None:
    apps_api.patch_namespaced_deployment.return_value = deployment("nginx-good")

    restart_deployment(NAMESPACE, "nginx-good", client=k8s_client, settings=settings)

    apps_api.delete_namespaced_deployment.assert_not_called()
    core_api.delete_namespaced_pod.assert_not_called()


def test_execute_tool_dispatches_and_returns_a_typed_result(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.list_namespaced_pod.return_value = V1PodList(items=[healthy_pod()])

    result = execute_tool(
        "list_pods", {"namespace": NAMESPACE}, client=k8s_client, settings=settings
    )

    assert isinstance(result, PodListResult)
    assert result.total == 1


def test_execute_tool_rejects_unknown_arguments(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """A hallucinated argument fails closed rather than being silently ignored."""
    with pytest.raises(ToolArgumentError):
        execute_tool(
            "list_pods",
            {"namespace": NAMESPACE, "force": True},
            client=k8s_client,
            settings=settings,
        )

    core_api.list_namespaced_pod.assert_not_called()


def test_execute_tool_rejects_missing_arguments(
    k8s_client: KubernetesClient, settings: Settings
) -> None:
    with pytest.raises(ToolArgumentError):
        execute_tool("get_pod", {"namespace": NAMESPACE}, client=k8s_client, settings=settings)


def test_execute_tool_rejects_an_unknown_tool(
    k8s_client: KubernetesClient, settings: Settings
) -> None:
    with pytest.raises(ToolNotFoundError) as caught:
        execute_tool("kubectl", {"command": "get pods"}, client=k8s_client, settings=settings)

    assert "list_pods" in caught.value.details["available_tools"]


def test_tool_metadata_is_json_serialisable_and_flags_mutation(settings: Settings) -> None:
    by_name = {item.name: item for item in tool_metadata(settings)}

    restart = by_name["restart_deployment"]
    assert restart.read_only is False
    assert restart.requires_confirmation is True
    assert "deployment_name" in restart.input_schema["properties"]

    listing = by_name["list_pods"]
    assert listing.read_only is True
    assert listing.requires_confirmation is False
    assert "namespace" in listing.input_schema["properties"]
