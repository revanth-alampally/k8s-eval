"""HTTP envelope for the agent: answer, request_id, tools_used, status. Nothing else."""

from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.context import CORRELATION_ID_HEADER
from app.knowledge.service import KnowledgeHit
from app.llm.base import LLMResponse, ToolCall
from app.llm.fake import ScriptedLLMProvider
from tests.unit.factories import NAMESPACE


def _call(name: str, **arguments: object) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=dict(arguments))],
        model="scripted",
        finish_reason="tool_calls",
    )


def _say(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="scripted", finish_reason="stop")


def test_agent_endpoint_returns_the_public_contract(client: TestClient) -> None:
    client.app.state.llm = ScriptedLLMProvider([_say("All reported pods are healthy.")])

    response = client.post(
        "/v1/agent",
        json={"message": "Are any pods unhealthy?"},
        headers={CORRELATION_ID_HEADER: "trace-agent-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Observed:\nAll reported pods are healthy.")
    assert "Interpretation:" in body["answer"]
    assert body["request_id"] == "trace-agent-1"
    assert body["status"] == "success"
    assert [item["tool"] for item in body["tools_used"]] == ["list_pods"]
    assert set(body) <= {"answer", "request_id", "status", "tools_used", "pending_confirmation"}
    assert body.get("pending_confirmation") is None
    assert response.headers[CORRELATION_ID_HEADER] == "trace-agent-1"


def test_agent_endpoint_records_tool_metadata_not_payloads(
    client: TestClient, core_api: object
) -> None:
    from kubernetes.client import V1PodList

    from tests.unit.factories import image_pull_pod

    core_api.list_namespaced_pod.return_value = V1PodList(items=[image_pull_pod()])  # type: ignore[attr-defined]
    client.app.state.llm = ScriptedLLMProvider(
        [
            _call("list_pods", namespace=NAMESPACE),
            _say("nginx-missing is unhealthy."),
        ]
    )

    response = client.post("/v1/agent", json={"message": "Are any pods unhealthy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["tools_used"][0]["tool"] == "list_pods"
    assert body["tools_used"][0]["outcome"] == "success"
    assert "unhealthy" in body["tools_used"][0]["summary"]
    # Raw pod objects never appear in the HTTP body.
    dumped = str(body)
    assert "container_statuses" not in dumped
    assert "nginx-missing-ghi789" not in dumped or "nginx-missing" in body["answer"]


def test_agent_endpoint_records_rag_metadata_not_retrieved_text(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    def search(query: str) -> list[KnowledgeHit]:
        assert query == "How do I diagnose image pulls?"
        return [
            KnowledgeHit(
                content="CANARY-RAG-CONTENT",
                source_path="README.md",
                chunk_index=0,
            )
        ]

    monkeypatch.setattr(client.app.state.knowledge, "search", search)
    client.app.state.llm = ScriptedLLMProvider(
        [
            _call("search_knowledge", query="How do I diagnose image pulls?"),
            _say("Consult the documented image-pull troubleshooting steps."),
        ]
    )

    response = client.post("/v1/agent", json={"message": "How do I diagnose image pulls?"})

    assert response.status_code == 200
    body = response.json()
    assert body["tools_used"][0]["tool"] == "search_knowledge"
    assert "CANARY-RAG-CONTENT" not in str(body)


def test_agent_endpoint_blocks_restart_without_touching_the_cluster(
    client: TestClient, apps_api: object
) -> None:
    client.app.state.llm = ScriptedLLMProvider(
        [_call("restart_deployment", namespace=NAMESPACE, deployment_name="nginx-good")]
    )

    response = client.post("/v1/agent", json={"message": "Restart the nginx-good deployment."})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "confirmation_required"
    assert body["pending_confirmation"]["tool"] == "restart_deployment"
    assert body["tools_used"][0]["outcome"] == "blocked"
    apps_api.patch_namespaced_deployment.assert_not_called()  # type: ignore[attr-defined]


def test_empty_message_is_a_structured_validation_error(client: TestClient) -> None:
    response = client.post("/v1/agent", json={"message": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
