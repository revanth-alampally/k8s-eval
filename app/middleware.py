"""HTTP middleware."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.context import (
    CORRELATION_ID_HEADER,
    TRACE_ID_HEADER,
    new_correlation_id,
    new_trace_id,
    set_correlation_id,
    set_trace_id,
)
from app.observability.logging import get_logger

_logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to the request context, all its logs, and the response.

    An inbound ``X-Correlation-ID`` is honoured so a trace can span the caller and this
    service; otherwise one is generated.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or new_correlation_id()
        trace_id = request.headers.get(TRACE_ID_HEADER) or new_trace_id()
        set_correlation_id(correlation_id)
        set_trace_id(trace_id)

        with structlog.contextvars.bound_contextvars(
            correlation_id=correlation_id,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        ):
            started = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception:
                # The exception handler logs the detail; this records the timing.
                _logger.warning(
                    "http.request",
                    operation="http_request",
                    outcome="error",
                    duration_ms=_elapsed_ms(started),
                )
                raise

            _logger.info(
                "http.request",
                operation="http_request",
                outcome="ok",
                status_code=response.status_code,
                duration_ms=_elapsed_ms(started),
            )
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[TRACE_ID_HEADER] = trace_id
            return response


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
