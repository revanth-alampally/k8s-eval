from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Environment, LogFormat, Settings, get_settings
from app.main import create_app
from app.tools.k8s.client import KubernetesClient, KubernetesClientProvider

DEMO_NAMESPACE = "ai-agent-demo"


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps a developer's local .env from leaking into tests.
    return Settings(
        environment=Environment.TEST,
        log_format=LogFormat.JSON,
        allowed_namespaces=[DEMO_NAMESPACE],
        max_log_lines=200,
        _env_file=None,
    )


@pytest.fixture
def core_api() -> MagicMock:
    return MagicMock(name="CoreV1Api")


@pytest.fixture
def apps_api() -> MagicMock:
    return MagicMock(name="AppsV1Api")


@pytest.fixture
def k8s_client(core_api: MagicMock, apps_api: MagicMock) -> KubernetesClient:
    """A KubernetesClient whose API surfaces are mocks.

    Nothing in the test suite reaches a real cluster; the tools are exercised entirely
    against the mocked client, with real ``kubernetes.client`` model objects as
    responses so the field access under test matches the actual API schema.
    """
    return KubernetesClient(
        core=core_api,
        apps=apps_api,
        version=MagicMock(name="VersionApi"),
        timeout_seconds=5.0,
    )


@pytest.fixture
def client(settings: Settings, k8s_client: KubernetesClient) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    # Inject the mocked cluster client so the readiness probe never dials a real API
    # server; the suite must pass identically with and without a cluster running.
    app.state.kubernetes = KubernetesClientProvider(settings, client=k8s_client)
    k8s_client.version.get_code.return_value = SimpleNamespace(git_version="v1.36.1")
    with TestClient(app) as test_client:
        yield test_client
