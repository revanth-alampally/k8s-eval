"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import health
from app.config import Settings, get_settings
from app.errors import install_exception_handlers
from app.middleware import CorrelationIdMiddleware
from app.observability.logging import configure_logging, get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    _logger.info(
        "app.startup",
        version=__version__,
        allowed_namespaces=settings.allowed_namespaces,
        read_only_mode=settings.read_only_mode,
        require_confirmation=settings.require_confirmation,
        llm_model=settings.llm_model,
    )
    # Kubernetes and LLM clients are created here so their lifetime matches the app.
    yield
    _logger.info("app.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="AI Kubernetes Operations Agent",
        version=__version__,
        summary="Natural-language Kubernetes operations backed by deterministic tools.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(CorrelationIdMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)

    return app


app = create_app()
