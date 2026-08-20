"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import __version__
from app.api.routes import agent, health
from app.config import Settings, get_settings
from app.errors import install_exception_handlers
from app.knowledge.service import KnowledgeService
from app.llm.factory import build_provider
from app.middleware import CorrelationIdMiddleware
from app.observability.logging import configure_logging, get_logger
from app.tools.k8s.client import KubernetesClientProvider
from app.tools.registry import build_registry

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    registry = build_registry(settings)
    _logger.info(
        "app.startup",
        version=__version__,
        allowed_namespaces=settings.allowed_namespaces,
        read_only_mode=settings.read_only_mode,
        require_confirmation=settings.require_confirmation,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.llm_model,
        # Logged at startup so the enabled capability set is recorded in every run.
        tools=sorted(registry),
        mutating_tools=sorted(name for name, spec in registry.items() if spec.mutating),
    )
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
    # Lazily connected: an unreachable cluster is a readiness failure, not a boot failure.
    app.state.kubernetes = KubernetesClientProvider(settings)
    app.state.llm = build_provider(settings)
    app.state.knowledge = KnowledgeService(settings, repository_root=Path.cwd())

    app.add_middleware(CorrelationIdMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(agent.router)

    return app


app = create_app()
