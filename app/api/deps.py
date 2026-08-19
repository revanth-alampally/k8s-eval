"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.tools.k8s.client import KubernetesClientProvider

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_client_provider(request: Request) -> KubernetesClientProvider:
    provider = request.app.state.kubernetes
    assert isinstance(provider, KubernetesClientProvider)
    return provider


ClientProviderDep = Annotated[KubernetesClientProvider, Depends(get_client_provider)]
