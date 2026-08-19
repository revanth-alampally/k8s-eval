"""Evidence collection for a failing pod.

``diagnose_pod`` is a composite read-only tool: it runs the same deterministic queries a
human would run by hand -- pod state, conditions, warning events, logs -- and returns
what it found as a flat list of signals.

It deliberately does **not** explain anything. There is no cause, no recommendation, no
ranking of likelihood. Every ``evidence`` string is quoted from the cluster, and the
``source`` on each signal says where it came from, so a later reader can tell the
cluster's own account (an event) from the workload's (a log line).

That boundary is the point. If this function guessed at causes, those guesses would be
indistinguishable from facts by the time they reached the model, and a wrong guess would
be laundered into a confident answer. Tools establish facts; reasoning happens elsewhere,
and only over what is recorded here.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.errors import LogsUnavailableError, ResourceNotFoundError
from app.observability.instrumentation import track_operation
from app.tools.base import require_allowed_namespace
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.convert import event_summary
from app.tools.k8s.models import (
    ContainerState,
    ContainerStatusSummary,
    DiagnosticSignal,
    PodDetail,
    PodDiagnosis,
    SignalSeverity,
    SignalSource,
)
from app.tools.k8s.read import get_pod, get_pod_logs, list_pod_events
from app.tools.schemas import DiagnosePodInput, parse_arguments

# A diagnosis wants the tail where the failure is, not the whole log.
DIAGNOSTIC_LOG_LINES = 25
# Bounds a single evidence string so one chatty container cannot dominate the payload.
MAX_EVIDENCE_CHARS = 1500
MAX_WARNING_EVENTS = 10
# Sidecar-heavy pods exist; cap the number of log reads per diagnosis.
MAX_LOG_CONTAINERS = 3

# Terminating with this reason is a normal exit, not a fault.
_CLEAN_EXIT = "Completed"

# kubelet returns this with HTTP 200 when a container's logs have been garbage collected.
_LOG_PLACEHOLDER_PREFIX = "unable to retrieve container logs"


def diagnose_pod(
    namespace: str,
    pod_name: str,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> PodDiagnosis:
    """Collect every deterministic signal available about a pod. Read-only.

    Raises ``ResourceNotFoundError`` if the pod does not exist -- that is a fact the
    caller must not paper over. Missing logs, by contrast, are recorded as a signal
    rather than raised, because "this container never produced output" is itself
    evidence and the rest of the diagnosis is still valid.
    """
    args = parse_arguments(DiagnosePodInput, namespace=namespace, pod_name=pod_name)
    require_allowed_namespace(args.namespace, settings)

    with track_operation(
        "tool.diagnose_pod",
        tool="diagnose_pod",
        namespace=args.namespace,
        pod_name=args.pod_name,
    ) as log:
        detail = get_pod(args.namespace, args.pod_name, client=client, settings=settings)
        events = list_pod_events(
            args.pod_name,
            namespace=args.namespace,
            client=client,
            warnings_only=True,
            limit=MAX_WARNING_EVENTS,
        )

        signals: list[DiagnosticSignal] = [_phase_signal(detail)]
        signals.extend(_condition_signals(detail))
        signals.extend(_container_signals(detail))
        signals.extend(_restart_signals(detail))
        signals.extend(_event_signals(events))

        log_signals, logs_available = _log_signals(detail, client=client, settings=settings)
        signals.extend(log_signals)

        log["signal_count"] = len(signals)
        log["warning_signals"] = sum(
            1 for signal in signals if signal.severity is SignalSeverity.WARNING
        )
        log["logs_available"] = logs_available

    return PodDiagnosis(
        pod=detail.name,
        namespace=detail.namespace,
        # Mirrors kubectl's STATUS column so the headline matches what an operator would
        # see in a terminal.
        status=detail.status_reason or detail.phase,
        healthy=detail.healthy,
        phase=detail.phase,
        containers_ready=detail.containers_ready,
        containers_total=detail.containers_total,
        restart_count=detail.restart_count,
        logs_available=logs_available,
        signals=signals,
    )


def _phase_signal(detail: PodDetail) -> DiagnosticSignal:
    return DiagnosticSignal(
        source=SignalSource.POD_PHASE,
        reason=detail.phase,
        evidence=(
            f"Pod is in phase {detail.phase} with "
            f"{detail.containers_ready}/{detail.containers_total} containers ready."
        ),
        severity=SignalSeverity.INFO if detail.healthy else SignalSeverity.WARNING,
    )


def _condition_signals(detail: PodDetail) -> list[DiagnosticSignal]:
    """Only conditions that are not satisfied; a True condition is not evidence."""
    signals = []
    for condition in detail.conditions:
        if condition.status == "True":
            continue
        detail_text = f": {condition.message}" if condition.message else "."
        signals.append(
            DiagnosticSignal(
                source=SignalSource.POD_CONDITION,
                reason=condition.reason or condition.type,
                evidence=f"Condition {condition.type} is {condition.status}{detail_text}",
                severity=SignalSeverity.WARNING,
                observed_at=condition.last_transition_time,
            )
        )
    return signals


def _container_signals(detail: PodDetail) -> list[DiagnosticSignal]:
    return [_container_signal(container) for container in detail.containers]


def _container_signal(container: ContainerStatusSummary) -> DiagnosticSignal:
    if container.state is ContainerState.WAITING:
        reason = container.reason or "Waiting"
        evidence = f"Container '{container.name}' is waiting: {reason}."
        if container.message:
            evidence = f"{evidence} {container.message}"
        severity = SignalSeverity.WARNING
    elif container.state is ContainerState.TERMINATED:
        reason = container.reason or "Terminated"
        evidence = (
            f"Container '{container.name}' terminated with exit code "
            f"{container.exit_code} ({reason})."
        )
        if container.message:
            evidence = f"{evidence} {container.message}"
        severity = SignalSeverity.INFO if reason == _CLEAN_EXIT else SignalSeverity.WARNING
    elif container.state is ContainerState.RUNNING:
        reason = "Running"
        readiness = "ready" if container.ready else "not ready"
        evidence = (
            f"Container '{container.name}' is running and {readiness}, image {container.image}."
        )
        severity = SignalSeverity.INFO if container.ready else SignalSeverity.WARNING
    else:
        reason = "Unknown"
        evidence = f"Container '{container.name}' has no reported state."
        severity = SignalSeverity.WARNING

    return DiagnosticSignal(
        source=SignalSource.CONTAINER_STATE,
        reason=reason,
        evidence=_clip(evidence),
        container=container.name,
        severity=severity,
        observed_at=container.finished_at or container.started_at,
    )


def _restart_signals(detail: PodDetail) -> list[DiagnosticSignal]:
    """Restarts are reported separately from state: a container can be running now and
    still have restarted repeatedly, which the current state alone would hide."""
    return [
        DiagnosticSignal(
            source=SignalSource.RESTART_COUNT,
            reason="ContainerRestarted",
            evidence=(
                f"Container '{container.name}' has restarted {container.restart_count} time(s)."
            ),
            container=container.name,
            severity=SignalSeverity.WARNING,
        )
        for container in detail.containers
        if container.restart_count > 0
    ]


def _event_signals(events: list[Any]) -> list[DiagnosticSignal]:
    signals = []
    for raw in events:
        event = event_summary(raw)
        occurrences = f" (seen {event.count} times)" if event.count > 1 else ""
        signals.append(
            DiagnosticSignal(
                source=SignalSource.WARNING_EVENT,
                reason=event.reason or "Warning",
                evidence=_clip(f"{event.message or ''}{occurrences}".strip()),
                severity=SignalSeverity.WARNING,
                observed_at=event.last_seen,
            )
        )
    return signals


def _log_signals(
    detail: PodDetail,
    *,
    client: KubernetesClient,
    settings: Settings,
) -> tuple[list[DiagnosticSignal], bool]:
    """Read what logs are available, recording absence as evidence rather than failing."""
    signals: list[DiagnosticSignal] = []
    any_available = False

    for container in detail.containers[:MAX_LOG_CONTAINERS]:
        # A container that crashed has its useful output in the *previous* instance; the
        # current one has usually not run yet.
        prefer_previous = container.restart_count > 0 or container.state is ContainerState.WAITING
        signal = _read_log(
            detail,
            container,
            prefer_previous=prefer_previous,
            client=client,
            settings=settings,
        )
        signals.append(signal)
        if signal.source is SignalSource.CONTAINER_LOGS:
            any_available = True

    return signals, any_available


def _read_log(
    detail: PodDetail,
    container: ContainerStatusSummary,
    *,
    prefer_previous: bool,
    client: KubernetesClient,
    settings: Settings,
) -> DiagnosticSignal:
    attempts = [prefer_previous, not prefer_previous] if prefer_previous else [False]
    note: str | None = None

    for previous in attempts:
        try:
            result = get_pod_logs(
                detail.namespace,
                detail.name,
                container=container.name,
                tail_lines=DIAGNOSTIC_LOG_LINES,
                previous=previous,
                client=client,
                settings=settings,
            )
        except (LogsUnavailableError, ResourceNotFoundError) as exc:
            note = exc.message
            continue

        text = result.content.strip()
        if not text:
            note = f"Container '{container.name}' produced no output."
            continue
        if _is_kubelet_placeholder(text):
            note = text
            continue

        return DiagnosticSignal(
            source=SignalSource.CONTAINER_LOGS,
            reason="PreviousContainerLogs" if previous else "ContainerLogs",
            evidence=_clip(text),
            container=container.name,
            # Log content is evidence, not a verdict: severity stays neutral because
            # nothing here has established that the log shows a problem.
            severity=SignalSeverity.INFO,
        )

    return DiagnosticSignal(
        source=SignalSource.LOGS_UNAVAILABLE,
        reason="NoLogsAvailable",
        evidence=note or f"Container '{container.name}' has produced no log output.",
        container=container.name,
        severity=SignalSeverity.WARNING,
    )


def _is_kubelet_placeholder(text: str) -> bool:
    """Detect kubelet's stand-in text for a log it can no longer serve.

    Once a terminated container is garbage collected, kubelet answers HTTP 200 with
    "unable to retrieve container logs for <runtime>://<id>" rather than an error.
    Recording that as log content would claim evidence that does not exist, so it is
    reported as unavailable instead. If the wording ever changes the check simply stops
    matching and the text is surfaced verbatim, which is the safe direction to fail.
    """
    return text.lower().startswith(_LOG_PLACEHOLDER_PREFIX)


def _clip(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated)"
