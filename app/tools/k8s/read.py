"""Read-only Kubernetes tools.

These never change cluster state and are safe for the agent to run without asking.
Every function follows the same shape: validate arguments, check the namespace
allowlist, call exactly one Kubernetes API with a timeout, convert the response to a
typed model, and log the outcome without the payload.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.errors import LogsUnavailableError
from app.observability.instrumentation import track_operation
from app.tools.base import require_allowed_namespace
from app.tools.k8s.client import KubernetesClient, translate_api_errors
from app.tools.k8s.convert import (
    deployment_summary,
    event_sort_key,
    event_summary,
    pod_detail,
    pod_summary,
)
from app.tools.k8s.models import (
    DeploymentListResult,
    PodDescription,
    PodDetail,
    PodListResult,
    PodLogsResult,
)
from app.tools.schemas import (
    DescribePodInput,
    GetPodInput,
    GetPodLogsInput,
    ListDeploymentsInput,
    ListPodsInput,
    parse_arguments,
)

# Bounds a single log response regardless of line count, so one chatty container cannot
# blow up the response or, later, the model's context window.
LOG_LIMIT_BYTES = 256 * 1024

# Recent events are the useful ones; older entries are noise for diagnosis.
MAX_EVENTS = 20


def list_pods(
    namespace: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> PodListResult:
    """List every pod in a namespace with a computed health verdict for each."""
    args = parse_arguments(ListPodsInput, namespace=namespace)
    require_allowed_namespace(args.namespace, settings)

    with track_operation("tool.list_pods", tool="list_pods", namespace=args.namespace) as log:
        with translate_api_errors(resource="pods", namespace=args.namespace):
            response = client.core.list_namespaced_pod(
                namespace=args.namespace,
                timeout_seconds=int(client.timeout_seconds),
                _request_timeout=client.timeout_seconds,
            )

        pods = [pod_summary(item) for item in response.items or []]
        unhealthy = [pod.name for pod in pods if not pod.healthy]
        log["pod_count"] = len(pods)
        log["unhealthy_count"] = len(unhealthy)

    return PodListResult(
        namespace=args.namespace,
        total=len(pods),
        unhealthy_count=len(unhealthy),
        pods=pods,
        unhealthy_pods=unhealthy,
    )


def get_pod(
    namespace: str,
    pod_name: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> PodDetail:
    """Fetch one pod, including per-container state and restart counts."""
    args = parse_arguments(GetPodInput, namespace=namespace, pod_name=pod_name)
    require_allowed_namespace(args.namespace, settings)

    with track_operation(
        "tool.get_pod", tool="get_pod", namespace=args.namespace, pod_name=args.pod_name
    ) as log:
        with translate_api_errors(resource="pod", name=args.pod_name, namespace=args.namespace):
            pod = client.core.read_namespaced_pod(
                name=args.pod_name,
                namespace=args.namespace,
                _request_timeout=client.timeout_seconds,
            )

        detail = pod_detail(pod)
        log["phase"] = detail.phase
        log["healthy"] = detail.healthy

    return detail


def describe_pod(
    namespace: str,
    pod_name: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> PodDescription:
    """Fetch a pod together with its recent events.

    The events are the point: for failures that occur before a container starts, such as
    an image that cannot be pulled, there are no logs at all and the event stream is the
    only evidence of what went wrong.
    """
    args = parse_arguments(DescribePodInput, namespace=namespace, pod_name=pod_name)
    require_allowed_namespace(args.namespace, settings)

    with track_operation(
        "tool.describe_pod", tool="describe_pod", namespace=args.namespace, pod_name=args.pod_name
    ) as log:
        with translate_api_errors(resource="pod", name=args.pod_name, namespace=args.namespace):
            pod = client.core.read_namespaced_pod(
                name=args.pod_name,
                namespace=args.namespace,
                _request_timeout=client.timeout_seconds,
            )

        ordered = list_pod_events(args.pod_name, namespace=args.namespace, client=client)

        detail = pod_detail(pod)
        log["phase"] = detail.phase
        log["event_count"] = len(ordered)

    return PodDescription(pod=detail, events=[event_summary(item) for item in ordered])


def list_pod_events(
    pod_name: str,
    *,
    namespace: str,
    client: KubernetesClient,
    warnings_only: bool = False,
    limit: int = MAX_EVENTS,
) -> list[Any]:
    """Recent events for a pod, newest first.

    An internal helper shared by describe_pod and diagnose_pod rather than a registered
    tool; callers are responsible for having validated the names already.
    """
    # A field selector over an already-validated name; the name pattern cannot contain
    # the ',' or '=' that would be needed to inject another selector.
    selector = f"involvedObject.name={pod_name}"
    if warnings_only:
        selector = f"{selector},type=Warning"

    with translate_api_errors(resource="events", name=pod_name, namespace=namespace):
        events = client.core.list_namespaced_event(
            namespace=namespace,
            field_selector=selector,
            timeout_seconds=int(client.timeout_seconds),
            _request_timeout=client.timeout_seconds,
        )

    return sorted(events.items or [], key=event_sort_key, reverse=True)[:limit]


def get_pod_logs(
    namespace: str,
    pod_name: str,
    container: str | None = None,
    tail_lines: int = 100,
    *,
    previous: bool = False,
    client: KubernetesClient,
    settings: Settings,
) -> PodLogsResult:
    """Read the tail of a container's log.

    ``previous=True`` reads the log of the last terminated container, which is the only
    way to see why a CrashLoopBackOff pod died -- the current container has usually not
    produced output yet.
    """
    args = parse_arguments(
        GetPodLogsInput,
        namespace=namespace,
        pod_name=pod_name,
        container=container,
        tail_lines=tail_lines,
        previous=previous,
    )
    require_allowed_namespace(args.namespace, settings)

    effective_tail = min(args.tail_lines, settings.max_log_lines)

    with track_operation(
        "tool.get_pod_logs",
        tool="get_pod_logs",
        namespace=args.namespace,
        pod_name=args.pod_name,
        container=args.container,
        previous=args.previous,
    ) as log:
        with translate_api_errors(
            resource="pod logs",
            name=args.pod_name,
            namespace=args.namespace,
            # Names and limits are already schema-validated, so a 400 here means the
            # container has no readable log (never started, or the previous instance is
            # gone), not that the caller passed something malformed.
            bad_request_error=LogsUnavailableError,
        ):
            # _preload_content=False is required for correctness, not performance: the
            # generated client declares this endpoint as returning `str` and coerces the
            # response body with str(), which turns the log into a "b'...'" repr. Reading
            # the raw response and decoding it ourselves is the only way to get the text.
            response = client.core.read_namespaced_pod_log(
                name=args.pod_name,
                namespace=args.namespace,
                container=args.container,
                tail_lines=effective_tail,
                previous=args.previous,
                limit_bytes=LOG_LIMIT_BYTES,
                timestamps=False,
                _request_timeout=client.timeout_seconds,
                _preload_content=False,
            )
            try:
                text = _decode_log(response.data)
            finally:
                response.release_conn()

        lines = text.splitlines()
        # Only the line count is logged: log content can contain anything the workload
        # printed, including credentials, so it never enters our own logs.
        log["line_count"] = len(lines)

    return PodLogsResult(
        namespace=args.namespace,
        pod_name=args.pod_name,
        container=args.container,
        previous=args.previous,
        tail_lines=effective_tail,
        line_count=len(lines),
        truncated=len(lines) >= effective_tail or len(text.encode("utf-8")) >= LOG_LIMIT_BYTES,
        content=text,
    )


def _decode_log(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, bytes | bytearray):
        # Container output is arbitrary bytes and may not be valid UTF-8.
        return content.decode("utf-8", errors="replace")
    return str(content)


def list_deployments(
    namespace: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> DeploymentListResult:
    """List deployments in a namespace with their replica availability."""
    args = parse_arguments(ListDeploymentsInput, namespace=namespace)
    require_allowed_namespace(args.namespace, settings)

    with track_operation(
        "tool.list_deployments", tool="list_deployments", namespace=args.namespace
    ) as log:
        with translate_api_errors(resource="deployments", namespace=args.namespace):
            response = client.apps.list_namespaced_deployment(
                namespace=args.namespace,
                timeout_seconds=int(client.timeout_seconds),
                _request_timeout=client.timeout_seconds,
            )

        deployments = [deployment_summary(item) for item in response.items or []]
        unavailable = [item for item in deployments if not item.available]
        log["deployment_count"] = len(deployments)
        log["unavailable_count"] = len(unavailable)

    return DeploymentListResult(
        namespace=args.namespace,
        total=len(deployments),
        unavailable_count=len(unavailable),
        deployments=deployments,
    )
