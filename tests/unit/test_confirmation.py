from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.agent.confirmation import ConfirmationStore
from app.errors import ConfirmationInvalidError


def test_confirmation_is_bound_to_exact_action_and_single_use() -> None:
    store = ConfirmationStore(ttl_seconds=60)
    token, _ = store.issue(
        tool="restart_deployment",
        arguments={"namespace": "ai-agent-demo", "deployment_name": "nginx-good"},
        session_id="user-a",
        request_id="request-a",
    )

    action = store.consume(token=token, session_id="user-a")

    assert action.tool == "restart_deployment"
    assert action.arguments["deployment_name"] == "nginx-good"
    with pytest.raises(ConfirmationInvalidError):
        store.consume(token=token, session_id="user-a")


def test_confirmation_rejects_forged_cross_session_and_expired_tokens() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    clock = [now]
    store = ConfirmationStore(ttl_seconds=1, now=lambda: clock[0])
    token, _ = store.issue(
        tool="restart_deployment",
        arguments={"namespace": "ai-agent-demo", "deployment_name": "nginx-good"},
        session_id="user-a",
        request_id="request-a",
    )

    with pytest.raises(ConfirmationInvalidError):
        store.consume(token="forged-token-that-is-long-enough", session_id="user-a")
    with pytest.raises(ConfirmationInvalidError):
        store.consume(token=token, session_id="user-b")

    clock[0] = now + timedelta(seconds=2)
    with pytest.raises(ConfirmationInvalidError, match="expired"):
        store.consume(token=token, session_id="user-a")
