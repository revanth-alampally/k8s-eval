from __future__ import annotations

from fastapi.testclient import TestClient

from app.context import CORRELATION_ID_HEADER


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "k8s-ops-agent"
    assert body["environment"] == "test"


def test_readiness_includes_checks(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["config"]["status"] == "ok"


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
