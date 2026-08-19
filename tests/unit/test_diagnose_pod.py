"""diagnose_pod collects evidence and draws no conclusions.

The scenarios mirror the workloads in ``k8s/``: a healthy pod, a CrashLoopBackOff pod
whose reason is only in the previous container's log, and an ImagePullBackOff pod that
has no log at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from kubernetes.client import CoreV1Event, V1ObjectMeta, V1ObjectReference
from kubernetes.client.exceptions import ApiException

from app.config import Settings
from app.errors import ResourceNotFoundError
from app.tools.k8s.client import KubernetesClient
from app.tools.k8s.diagnose import DIAGNOSTIC_LOG_LINES, MAX_EVIDENCE_CHARS, diagnose_pod
from app.tools.k8s.models import PodDiagnosis, SignalSeverity, SignalSource
from tests.unit.factories import (
    NAMESPACE,
    NOW,
    crashloop_pod,
    healthy_pod,
    image_pull_pod,
    log_response,
)

CRASH_LOG = (
    "/docker-entrypoint.sh: Configuration complete; ready for start up\n"
    'nginx: [emerg] unknown directive "enable_turbo_mode" in /etc/nginx/nginx.conf:7\n'
)


def _warning_event(reason: str, message: str, count: int = 1) -> CoreV1Event:
    return CoreV1Event(
        metadata=V1ObjectMeta(name=f"evt.{reason}", namespace=NAMESPACE),
        involved_object=V1ObjectReference(name="pod"),
        type="Warning",
        reason=reason,
        message=message,
        count=count,
        first_timestamp=NOW,
        last_timestamp=datetime(2026, 8, 19, 12, 5, tzinfo=UTC),
    )


def _no_logs(message: str) -> ApiException:
    exc = ApiException(status=400, reason="Bad Request")
    exc.body = json.dumps({"message": message})
    return exc


def _sources(diagnosis: PodDiagnosis) -> set[SignalSource]:
    return {signal.source for signal in diagnosis.signals}


def _reasons(diagnosis: PodDiagnosis) -> set[str]:
    return {signal.reason for signal in diagnosis.signals}


# --- healthy pod ---------------------------------------------------------------


def test_healthy_pod_reports_running_with_no_warnings(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = healthy_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response('GET / HTTP/1.1" 200\n')

    diagnosis = diagnose_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    assert diagnosis.status == "Running"
    assert diagnosis.healthy is True
    assert diagnosis.containers_ready == 1
    assert diagnosis.logs_available is True
    assert not [s for s in diagnosis.signals if s.severity is SignalSeverity.WARNING]
    assert SignalSource.WARNING_EVENT not in _sources(diagnosis)
    assert SignalSource.RESTART_COUNT not in _sources(diagnosis)


def test_healthy_pod_reads_current_logs_not_previous(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = healthy_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response("serving\n")

    diagnose_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    kwargs = core_api.read_namespaced_pod_log.call_args.kwargs
    assert kwargs["previous"] is False
    assert kwargs["tail_lines"] == DIAGNOSTIC_LOG_LINES


# --- CrashLoopBackOff ----------------------------------------------------------


def test_crashloop_pod_collects_state_restarts_and_previous_logs(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = crashloop_pod()
    core_api.list_namespaced_event.return_value = MagicMock(
        items=[_warning_event("BackOff", "Back-off restarting failed container", count=7)]
    )
    core_api.read_namespaced_pod_log.return_value = log_response(CRASH_LOG)

    diagnosis = diagnose_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)

    assert diagnosis.status == "CrashLoopBackOff"
    assert diagnosis.healthy is False
    assert _sources(diagnosis) >= {
        SignalSource.POD_PHASE,
        SignalSource.CONTAINER_STATE,
        SignalSource.RESTART_COUNT,
        SignalSource.WARNING_EVENT,
        SignalSource.CONTAINER_LOGS,
    }

    restart = next(s for s in diagnosis.signals if s.source is SignalSource.RESTART_COUNT)
    assert "5 time(s)" in restart.evidence

    state = next(s for s in diagnosis.signals if s.source is SignalSource.CONTAINER_STATE)
    assert state.reason == "CrashLoopBackOff"


def test_crashloop_pod_reads_the_previous_container_log(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """The crash reason only exists in the terminated instance's log."""
    core_api.read_namespaced_pod.return_value = crashloop_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response(CRASH_LOG)

    diagnosis = diagnose_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)

    assert core_api.read_namespaced_pod_log.call_args.kwargs["previous"] is True
    logs = next(s for s in diagnosis.signals if s.source is SignalSource.CONTAINER_LOGS)
    assert logs.reason == "PreviousContainerLogs"
    assert "enable_turbo_mode" in logs.evidence


def test_crashloop_falls_back_to_current_log_when_previous_is_gone(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Terminated containers are garbage collected; the fallback keeps evidence flowing."""
    core_api.read_namespaced_pod.return_value = crashloop_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.side_effect = [
        _no_logs("previous terminated container not found"),
        log_response("starting up\n"),
    ]

    diagnosis = diagnose_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)

    logs = next(s for s in diagnosis.signals if s.source is SignalSource.CONTAINER_LOGS)
    assert logs.reason == "ContainerLogs"
    assert diagnosis.logs_available is True


# --- ImagePullBackOff ----------------------------------------------------------


def test_image_pull_pod_reports_event_evidence_and_no_logs(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(
        items=[
            _warning_event(
                "Failed",
                'Failed to pull image "nginx:1.27-does-not-exist": not found',
            )
        ]
    )
    core_api.read_namespaced_pod_log.side_effect = _no_logs(
        'container "nginx" in pod "nginx-missing" is waiting to start'
    )

    diagnosis = diagnose_pod(
        NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings
    )

    assert diagnosis.status == "ImagePullBackOff"
    assert diagnosis.logs_available is False
    assert SignalSource.LOGS_UNAVAILABLE in _sources(diagnosis)

    event = next(s for s in diagnosis.signals if s.source is SignalSource.WARNING_EVENT)
    assert "does-not-exist" in event.evidence
    assert event.reason == "Failed"


def test_image_pull_pod_only_queries_warning_events(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.side_effect = _no_logs("waiting to start")

    diagnose_pod(NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings)

    selector = core_api.list_namespaced_event.call_args.kwargs["field_selector"]
    assert selector == "involvedObject.name=nginx-missing-ghi789,type=Warning"


# --- pod not found -------------------------------------------------------------


def test_missing_pod_raises_rather_than_returning_empty_evidence(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """A diagnosis of a pod that does not exist must fail loudly. Returning an empty
    signal list would invite an answer about a pod that was never there."""
    exc = ApiException(status=404, reason="Not Found")
    exc.body = json.dumps({"message": 'pods "ghost" not found'})
    core_api.read_namespaced_pod.side_effect = exc

    with pytest.raises(ResourceNotFoundError):
        diagnose_pod(NAMESPACE, "ghost", client=k8s_client, settings=settings)

    core_api.list_namespaced_event.assert_not_called()


# --- logs unavailable ----------------------------------------------------------


def test_unavailable_logs_are_recorded_as_a_signal_not_an_error(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.side_effect = _no_logs(
        'container "nginx" in pod "nginx-missing" is waiting to start'
    )

    diagnosis = diagnose_pod(
        NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings
    )

    signal = next(s for s in diagnosis.signals if s.source is SignalSource.LOGS_UNAVAILABLE)
    assert signal.container == "nginx"
    assert "waiting to start" in signal.evidence
    assert diagnosis.logs_available is False
    # The rest of the evidence still arrived.
    assert SignalSource.CONTAINER_STATE in _sources(diagnosis)


def test_kubelet_placeholder_is_not_reported_as_log_content(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Once a crashed container is garbage collected, kubelet answers 200 with
    'unable to retrieve container logs for containerd://...'. Treating that as log
    content would claim evidence that does not exist."""
    placeholder = "unable to retrieve container logs for containerd://f41da0553ad8"
    core_api.read_namespaced_pod.return_value = crashloop_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response(placeholder)

    diagnosis = diagnose_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)

    assert diagnosis.logs_available is False
    signal = next(s for s in diagnosis.signals if s.source is SignalSource.LOGS_UNAVAILABLE)
    assert placeholder in signal.evidence
    assert SignalSource.CONTAINER_LOGS not in _sources(diagnosis)


def test_empty_log_output_counts_as_unavailable(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = healthy_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response("   \n")

    diagnosis = diagnose_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    assert diagnosis.logs_available is False
    assert SignalSource.LOGS_UNAVAILABLE in _sources(diagnosis)


# --- the separation of concerns itself -----------------------------------------


def test_diagnosis_contains_no_interpretation_fields(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    """Guards the design rule: this tool establishes facts, the LLM reasons over them.

    If a 'cause' or 'recommendation' field ever appears here, a guess would become
    indistinguishable from a fact by the time it reached the model.
    """
    core_api.read_namespaced_pod.return_value = crashloop_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response(CRASH_LOG)

    diagnosis = diagnose_pod(NAMESPACE, "nginx-crash-def456", client=k8s_client, settings=settings)
    payload = json.dumps(diagnosis.model_dump(mode="json"))

    forbidden = {
        "cause",
        "root_cause",
        "probable_cause",
        "explanation",
        "recommendation",
        "suggestion",
        "diagnosis_text",
        "summary",
        "conclusion",
    }
    assert not forbidden & set(diagnosis.model_dump())
    assert not forbidden & set(diagnosis.signals[0].model_dump())
    # Nothing in the payload asserts a cause; every reason is a cluster-issued label.
    assert "because" not in payload.lower()


def test_every_reason_comes_from_the_cluster_vocabulary(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = image_pull_pod()
    core_api.list_namespaced_event.return_value = MagicMock(
        items=[_warning_event("Failed", "Failed to pull image")]
    )
    core_api.read_namespaced_pod_log.side_effect = _no_logs("waiting to start")

    diagnosis = diagnose_pod(
        NAMESPACE, "nginx-missing-ghi789", client=k8s_client, settings=settings
    )

    assert _reasons(diagnosis) <= {
        "Pending",
        "ImagePullBackOff",
        "Failed",
        "NoLogsAvailable",
    }


def test_log_evidence_is_bounded(
    k8s_client: KubernetesClient, core_api: MagicMock, settings: Settings
) -> None:
    core_api.read_namespaced_pod.return_value = healthy_pod()
    core_api.list_namespaced_event.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = log_response("x" * 50_000)

    diagnosis = diagnose_pod(NAMESPACE, "nginx-good-abc123", client=k8s_client, settings=settings)

    logs = next(s for s in diagnosis.signals if s.source is SignalSource.CONTAINER_LOGS)
    assert len(logs.evidence) <= MAX_EVIDENCE_CHARS + len("... (truncated)")
