"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.confirmation import ConfirmationStore
from app.config import Settings, get_settings
from app.knowledge.service import KnowledgeService
from app.llm.base import LLMProvider
from app.observability.metrics import MetricsRegistry
from app.tools.k8s.client import KubernetesClientProvider

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_client_provider(request: Request) -> KubernetesClientProvider:
    provider = request.app.state.kubernetes
    assert isinstance(provider, KubernetesClientProvider)
    return provider


def get_llm_provider(request: Request) -> LLMProvider:
    provider = request.app.state.llm
    assert isinstance(provider, LLMProvider)
    return provider


def get_knowledge_service(request: Request) -> KnowledgeService:
    service = request.app.state.knowledge
    assert isinstance(service, KnowledgeService)
    return service


def get_confirmation_store(request: Request) -> ConfirmationStore:
    store = request.app.state.confirmations
    assert isinstance(store, ConfirmationStore)
    return store


def get_metrics(request: Request) -> MetricsRegistry:
    metrics = request.app.state.metrics
    assert isinstance(metrics, MetricsRegistry)
    return metrics


ClientProviderDep = Annotated[KubernetesClientProvider, Depends(get_client_provider)]
LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]
KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
ConfirmationStoreDep = Annotated[ConfirmationStore, Depends(get_confirmation_store)]
MetricsDep = Annotated[MetricsRegistry, Depends(get_metrics)]
