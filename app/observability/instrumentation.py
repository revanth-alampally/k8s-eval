"""Latency and outcome instrumentation.

A single primitive, :func:`track_operation`, is used for HTTP requests, agent turns,
tool calls and Kubernetes API calls. Keeping one shape (``operation``, ``outcome``,
``duration_ms``) means a single log query can answer "what is slow?" and "what fails?"
across every layer, and makes it trivial to swap in Prometheus or OpenTelemetry later.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from app.context import get_correlation_id, get_trace_id
from app.observability.logging import get_logger
from app.observability.metrics import MetricsRegistry

_logger = get_logger(__name__)
_SAFE_EVENT_FIELDS = {
    "request_id",
    "trace_id",
    "tool",
    "outcome",
    "status",
    "error_code",
    "error_type",
    "provider",
    "duration_ms",
    "tool_calls",
    "model_turns",
    "mutating",
}


def emit_agent_event(metrics: MetricsRegistry, event: str, **fields: Any) -> None:
    """Emit only an explicit metadata allowlist; payloads can never enter telemetry."""
    safe = {key: value for key, value in fields.items() if key in _SAFE_EVENT_FIELDS}
    if request_id := get_correlation_id():
        safe.setdefault("request_id", request_id)
    if trace_id := get_trace_id():
        safe.setdefault("trace_id", trace_id)
    metrics.event(event, safe)
    _logger.info(event, **safe)


@contextmanager
def track_operation(
    operation: str,
    **fields: Any,
) -> Iterator[MutableMapping[str, Any]]:
    """Time a block and emit one log event when it finishes.

    Yields a mutable dict; anything added to it is included in the completion log,
    which lets callers record results only known part-way through (e.g. item counts).
    """
    extra: MutableMapping[str, Any] = {}
    started = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        _logger.warning(
            "operation.failed",
            operation=operation,
            outcome="error",
            error_type=type(exc).__name__,
            duration_ms=_elapsed_ms(started),
            **fields,
            **extra,
        )
        raise
    else:
        _logger.info(
            "operation.completed",
            operation=operation,
            outcome="ok",
            duration_ms=_elapsed_ms(started),
            **fields,
            **extra,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
