"""Kubernetes client construction and error translation.

Everything the rest of the application knows about talking to Kubernetes is here. Two
responsibilities:

1. Build API clients from configuration, without mutating the library's global default
   configuration (which would make the behaviour of one part of the process depend on
   what another part loaded first).
2. Translate every transport and API failure into the application error taxonomy, so no
   ``ApiException`` ever escapes this package. Callers upstream branch on ``ErrorCode``,
   never on HTTP status codes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from kubernetes import client as k8s
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from app.config import Settings
from app.errors import (
    AppError,
    ClusterTimeoutError,
    ClusterUnavailableError,
    PermissionDeniedError,
    ResourceNotFoundError,
    ToolArgumentError,
    ToolExecutionError,
)
from app.observability.logging import get_logger

_logger = get_logger(__name__)

# Readiness probes must not inherit the full request budget, or a hung API server turns
# every health check into a ten-second stall.
PING_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class KubernetesClient:
    """Thin, injectable holder for the API clients a tool may use.

    Deliberately narrow: Core (pods, logs, events), Apps (deployments) and Version. There
    is no generic ``ApiClient`` accessor, so a future tool cannot quietly reach for an
    API this layer has not vetted.
    """

    core: k8s.CoreV1Api
    apps: k8s.AppsV1Api
    version: k8s.VersionApi
    timeout_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings) -> KubernetesClient:
        configuration = _load_configuration(settings)
        api_client = k8s.ApiClient(configuration=configuration)
        return cls(
            core=k8s.CoreV1Api(api_client),
            apps=k8s.AppsV1Api(api_client),
            version=k8s.VersionApi(api_client),
            timeout_seconds=float(settings.kube_request_timeout_seconds),
        )

    def ping(self) -> str:
        """Return the cluster's git version, or raise a typed error. Used by readiness."""
        with translate_api_errors(resource="cluster version"):
            info = self.version.get_code(_request_timeout=PING_TIMEOUT_SECONDS)
        return str(info.git_version)


class KubernetesClientProvider:
    """Lazily builds and caches a single client for the process.

    Lazy rather than eager at startup: an unreachable cluster should surface as an
    unhealthy readiness probe and typed per-request errors, not as an API that refuses
    to boot at all.
    """

    def __init__(self, settings: Settings, client: KubernetesClient | None = None) -> None:
        self._settings = settings
        # An injected client short-circuits construction; the seam tests use to stay
        # hermetic, so nothing in the unit suite can reach a real cluster.
        self._client = client
        self._lock = threading.Lock()

    def get(self) -> KubernetesClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = KubernetesClient.from_settings(self._settings)
                    _logger.info(
                        "kubernetes.client_initialised",
                        context=self._settings.kube_context or "<current>",
                        timeout_seconds=self._settings.kube_request_timeout_seconds,
                    )
        return self._client


def _load_configuration(settings: Settings) -> k8s.Configuration:
    configuration = k8s.Configuration()
    try:
        if settings.kubeconfig_path or settings.kube_context:
            k8s_config.load_kube_config(
                config_file=str(settings.kubeconfig_path) if settings.kubeconfig_path else None,
                context=settings.kube_context or None,
                client_configuration=configuration,
            )
        else:
            try:
                k8s_config.load_incluster_config(client_configuration=configuration)
            except ConfigException:
                k8s_config.load_kube_config(client_configuration=configuration)
    except ConfigException as exc:
        raise ClusterUnavailableError(
            "Could not load Kubernetes configuration.",
            reason=str(exc),
        ) from exc
    return configuration


@contextmanager
def translate_api_errors(
    *,
    resource: str,
    name: str | None = None,
    namespace: str | None = None,
    bad_request_error: type[AppError] = ToolArgumentError,
) -> Iterator[None]:
    """Convert Kubernetes and transport failures into typed application errors.

    ``bad_request_error`` overrides the mapping for HTTP 400/422. Callers whose
    arguments are already fully validated can say what a rejection really means for
    them -- for logs it is "no log exists in this state", not "your arguments are wrong".
    """
    context: dict[str, Any] = {"resource": resource}
    if name is not None:
        context["name"] = name
    if namespace is not None:
        context["namespace"] = namespace

    try:
        yield
    except ApiException as exc:
        raise _from_api_exception(exc, context, bad_request_error) from exc
    except (Urllib3TimeoutError, TimeoutError) as exc:
        # Builtin TimeoutError subclasses OSError, so this must precede the OSError arm.
        raise ClusterTimeoutError(
            f"Timed out talking to the Kubernetes API while reading {resource}.",
            **context,
        ) from exc
    except MaxRetryError as exc:
        raise ClusterUnavailableError(
            "Could not reach the Kubernetes API server.",
            **context,
            reason=_safe_reason(exc),
        ) from exc
    except (Urllib3HTTPError, OSError) as exc:
        raise ClusterUnavailableError(
            "Connection to the Kubernetes API server failed.",
            **context,
            reason=_safe_reason(exc),
        ) from exc


def _from_api_exception(
    exc: ApiException,
    context: dict[str, Any],
    bad_request_error: type[AppError],
) -> Exception:
    http_status = int(exc.status or 0)
    detail = _api_message(exc)
    details: dict[str, Any] = {**context, "kubernetes_status": http_status}
    if detail:
        details["kubernetes_message"] = detail

    resource = context["resource"]
    name = context.get("name")
    namespace = context.get("namespace")
    located = f"{resource} '{name}'" if name else resource
    if namespace:
        located = f"{located} in namespace '{namespace}'"

    if http_status == 404:
        return ResourceNotFoundError(f"{located} was not found.", **details)
    if http_status in (401, 403):
        return PermissionDeniedError(
            f"Not authorised to access {located}.",
            **details,
        )
    if http_status in (408, 504):
        return ClusterTimeoutError(f"The Kubernetes API timed out reading {located}.", **details)
    if http_status in (400, 422):
        # The API rejected the request; arguments were already schema-validated, so what
        # this means is tool-specific.
        return bad_request_error(
            detail or f"The Kubernetes API rejected the request for {located}.",
            **details,
        )
    if http_status == 429 or http_status >= 500:
        return ClusterUnavailableError(
            f"The Kubernetes API is unavailable (status {http_status}).",
            **details,
        )
    return ToolExecutionError(f"Kubernetes API call for {located} failed.", **details)


def _api_message(exc: ApiException) -> str:
    """Extract only the human-readable message from an API error body.

    Deliberately narrow: the raw body and, above all, ``exc.headers`` are never
    propagated, since headers can carry authorization material.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, bytes | bytearray):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str) and body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return ""
        if isinstance(parsed, dict):
            message = parsed.get("message")
            if isinstance(message, str):
                return message
    reason = getattr(exc, "reason", None)
    return reason if isinstance(reason, str) else ""


def _safe_reason(exc: BaseException) -> str:
    """A short, non-sensitive description of a transport failure."""
    return type(exc).__name__
