"""Request-scoped context.

The correlation ID lives in a :class:`contextvars.ContextVar` so any layer -- route,
agent, tool, Kubernetes client -- can stamp it onto logs and error responses without
threading it through every function signature.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

CORRELATION_ID_HEADER = "X-Correlation-ID"
TRACE_ID_HEADER = "X-Trace-ID"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def set_correlation_id(value: str) -> None:
    _correlation_id.set(value)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_trace_id() -> str | None:
    return _trace_id.get()
