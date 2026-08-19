"""Liveness and readiness endpoints.

``/health`` answers "is the process alive?" and must never depend on the cluster or the
LLM -- otherwise a Kubernetes hiccup would make an orchestrator restart a healthy API.
``/health/ready`` answers "can this instance serve traffic?" and is where dependency
probes belong.
"""

from __future__ import annotations

from enum import StrEnum

import anyio.to_thread
from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app import __version__
from app.api.deps import ClientProviderDep, SettingsDep
from app.errors import AppError

router = APIRouter(tags=["health"])


class CheckStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class DependencyCheck(BaseModel):
    status: CheckStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    status: CheckStatus
    service: str
    version: str
    environment: str


class ReadinessResponse(HealthResponse):
    checks: dict[str, DependencyCheck] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status=CheckStatus.OK,
        service=settings.service_name,
        version=__version__,
        environment=settings.environment.value,
    )


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(
    settings: SettingsDep,
    provider: ClientProviderDep,
    response: Response,
) -> ReadinessResponse:
    checks: dict[str, DependencyCheck] = {
        "config": DependencyCheck(
            status=CheckStatus.OK,
            detail=f"namespaces={','.join(settings.allowed_namespaces)}",
        ),
        "kubernetes": await _check_kubernetes(provider),
    }

    overall = _aggregate(checks)
    if overall is CheckStatus.UNAVAILABLE:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall,
        service=settings.service_name,
        version=__version__,
        environment=settings.environment.value,
        checks=checks,
    )


async def _check_kubernetes(provider: ClientProviderDep) -> DependencyCheck:
    """Probe the API server. The client is blocking, so it runs off the event loop."""
    try:
        version = await anyio.to_thread.run_sync(lambda: provider.get().ping())
    except AppError as exc:
        # Without a cluster this instance cannot answer anything truthfully, so it is
        # not ready -- better to be pulled from rotation than to serve empty answers.
        return DependencyCheck(status=CheckStatus.UNAVAILABLE, detail=exc.message)
    return DependencyCheck(status=CheckStatus.OK, detail=f"server {version}")


def _aggregate(checks: dict[str, DependencyCheck]) -> CheckStatus:
    statuses = {check.status for check in checks.values()}
    if CheckStatus.UNAVAILABLE in statuses:
        return CheckStatus.UNAVAILABLE
    if CheckStatus.DEGRADED in statuses:
        return CheckStatus.DEGRADED
    return CheckStatus.OK
