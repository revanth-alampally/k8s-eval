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

from app.observability.logging import get_logger

_logger = get_logger(__name__)


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
