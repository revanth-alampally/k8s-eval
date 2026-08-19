"""Structured logging setup.

Every log line is an event with key/value pairs rather than an interpolated string,
so logs stay greppable and machine-parseable. ``uvicorn`` and other stdlib loggers are
routed through the same structlog pipeline so the output format is uniform.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from app.config import LogFormat, Settings


def _static_fields(**fields: Any) -> Any:
    """Processor that stamps constant fields (service, environment) onto every event."""

    def processor(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> Any:
        event_dict.update(fields)
        return event_dict

    return processor


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    level = logging.getLevelNamesMapping()[settings.log_level]

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _static_fields(
            service=settings.service_name,
            environment=settings.environment.value,
        ),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_format is LogFormat.JSON
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Records coming from stdlib loggers have not been through the shared
        # processors yet, so they run here instead.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = []
        stdlib_logger.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
