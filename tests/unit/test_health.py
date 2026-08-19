from __future__ import annotations

from fastapi.testclient import TestClient
from urllib3.exceptions import MaxRetryError

from app.context import CORRELATION_ID_HEADER
from app.tools.k8s.client import KubernetesClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "k8s-ops-agent"
    assert body["environment"] == "test"


def test_readiness_reports_cluster_connectivity(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["config"]["status"] == "ok"
    assert body["checks"]["kubernetes"]["status"] == "ok"
    assert "v1.36.1" in body["checks"]["kubernetes"]["detail"]


def test_readiness_fails_when_the_cluster_is_unreachable(
    client: TestClient, k8s_client: KubernetesClient
) -> None:
    """An instance that cannot reach the cluster cannot answer anything truthfully, so
    it must report itself unready rather than serve empty results."""
    k8s_client.version.get_code.side_effect = MaxRetryError(None, "/version")  # type: ignore[arg-type]

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checks"]["kubernetes"]["status"] == "unavailable"


def test_liveness_is_independent_of_the_cluster(
    client: TestClient, k8s_client: KubernetesClient
) -> None:
    """A cluster outage must not cause an orchestrator to restart a healthy process."""
    k8s_client.version.get_code.side_effect = MaxRetryError(None, "/version")  # type: ignore[arg-type]

    assert client.get("/health").status_code == 200


def test_correlation_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers[CORRELATION_ID_HEADER]


def test_correlation_id_is_echoed_when_supplied(client: TestClient) -> None:
    response = client.get("/health", headers={CORRELATION_ID_HEADER: "trace-123"})

    assert response.headers[CORRELATION_ID_HEADER] == "trace-123"


def test_unknown_route_returns_structured_error(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "resource_not_found"
    assert body["correlation_id"]
