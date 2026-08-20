"""Mutating Kubernetes tools.

Everything here changes cluster state. Two properties are non-negotiable:

* the tool is declared ``mutating=True`` in the registry, so the orchestrator can
  require an explicit confirmation before it ever runs;
* it re-checks ``read_only_mode`` itself rather than trusting the registry to have
  filtered it out, because a guard that exists in only one place is one refactor away
  from not existing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.config import Settings
from app.observability.instrumentation import track_operation
from app.tools.base import (
    require_allowed_namespace,
    require_mutating_tool_allowed,
    require_mutations_enabled,
)
from app.tools.k8s.client import KubernetesClient, translate_api_errors
from app.tools.k8s.models import RestartDeploymentResult
from app.tools.schemas import RestartDeploymentInput, parse_arguments

RESTART_ANNOTATION = "kubectl.kubernetes.io/restartedAt"


def restart_deployment(
    namespace: str,
    deployment_name: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> RestartDeploymentResult:
    """Trigger a rolling restart of a deployment. MUTATING.

    Implemented the same way ``kubectl rollout restart`` does it: stamp a timestamp
    annotation onto the pod template, which changes the template hash and causes the
    controller to roll pods. Nothing is deleted, and the operation is a no-op if the
    deployment does not exist (it fails with resource_not_found instead).
    """
    args = parse_arguments(
        RestartDeploymentInput, namespace=namespace, deployment_name=deployment_name
    )
    require_allowed_namespace(args.namespace, settings)
    require_mutations_enabled("restart_deployment", settings)
    require_mutating_tool_allowed("restart_deployment", settings)

    restarted_at = datetime.now(UTC)
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {RESTART_ANNOTATION: restarted_at.strftime("%Y-%m-%dT%H:%M:%SZ")}
                }
            }
        }
    }

    with (
        track_operation(
            "tool.restart_deployment",
            tool="restart_deployment",
            mutating=True,
            namespace=args.namespace,
            deployment_name=args.deployment_name,
        ),
        translate_api_errors(
            resource="deployment", name=args.deployment_name, namespace=args.namespace
        ),
    ):
        updated = client.apps.patch_namespaced_deployment(
            name=args.deployment_name,
            namespace=args.namespace,
            body=patch,
            _request_timeout=client.timeout_seconds,
        )

    desired = int(getattr(getattr(updated, "spec", None), "replicas", 0) or 0)
    return RestartDeploymentResult(
        namespace=args.namespace,
        deployment_name=args.deployment_name,
        restarted_at=restarted_at,
        replicas_desired=desired,
        message=(
            f"Triggered a rolling restart of deployment '{args.deployment_name}' in namespace "
            f"'{args.namespace}'. {desired} replica(s) will be replaced gradually."
        ),
    )
