"""Short-lived, single-use capabilities for mutating tool execution."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from app.errors import ConfirmationInvalidError
from app.observability.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ConfirmedAction:
    tool: str
    arguments: dict[str, Any]
    request_id: str


@dataclass
class _PendingAction:
    tool: str
    arguments: dict[str, Any]
    session_digest: str
    request_id: str
    expires_at: datetime
    consumed: bool = False


class ConfirmationStore:
    """Process-local confirmation capabilities.

    The token is a 256-bit opaque random value. Only a SHA-256 digest is retained,
    preventing accidental token disclosure from authorisation state or audit logs.
    """

    def __init__(self, *, ttl_seconds: int, now: Callable[[], datetime] | None = None) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._now = now or (lambda: datetime.now(UTC))
        self._entries: dict[str, _PendingAction] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        session_id: str,
        request_id: str,
    ) -> tuple[str, datetime]:
        issued_at = self._now()
        expires_at = issued_at + self._ttl
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired(issued_at)
            self._entries[_digest(token)] = _PendingAction(
                tool=tool,
                arguments=_canonical_arguments(arguments),
                session_digest=_digest(session_id),
                request_id=request_id,
                expires_at=expires_at,
            )
        _audit(
            "issued", tool=tool, arguments=arguments, request_id=request_id, session_id=session_id
        )
        return token, expires_at

    def consume(self, *, token: str, session_id: str) -> ConfirmedAction:
        now = self._now()
        digest = _digest(token)
        with self._lock:
            pending = self._entries.get(digest)
            if pending is None:
                _audit("rejected_invalid", session_id=session_id)
                raise ConfirmationInvalidError("The confirmation token is invalid or has expired.")
            if pending.consumed:
                _audit("rejected_replayed", tool=pending.tool, session_id=session_id)
                raise ConfirmationInvalidError("The confirmation token has already been used.")
            if not secrets.compare_digest(pending.session_digest, _digest(session_id)):
                _audit("rejected_session", tool=pending.tool, session_id=session_id)
                raise ConfirmationInvalidError(
                    "The confirmation token does not belong to this session."
                )
            if now >= pending.expires_at:
                del self._entries[digest]
                _audit("rejected_expired", tool=pending.tool, session_id=session_id)
                raise ConfirmationInvalidError("The confirmation token has expired.")

            pending.consumed = True
            action = ConfirmedAction(
                tool=pending.tool,
                arguments=dict(pending.arguments),
                request_id=pending.request_id,
            )
        _audit(
            "confirmed",
            tool=action.tool,
            arguments=action.arguments,
            request_id=action.request_id,
            session_id=session_id,
        )
        return action

    def _purge_expired(self, now: datetime) -> None:
        self._entries = {
            digest: entry
            for digest, entry in self._entries.items()
            if entry.expires_at > now and not entry.consumed
        }


def _canonical_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return cast(dict[str, Any], json.loads(encoded))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _audit(
    outcome: str,
    *,
    tool: str | None = None,
    arguments: dict[str, Any] | None = None,
    request_id: str | None = None,
    session_id: str,
) -> None:
    fields: dict[str, Any] = {
        "outcome": outcome,
        "session_fingerprint": _digest(session_id)[:12],
    }
    if tool is not None:
        fields["tool"] = tool
    if request_id is not None:
        fields["origin_request_id"] = request_id
    if arguments is not None:
        fields["namespace"] = arguments.get("namespace")
        fields["resource_name"] = arguments.get("deployment_name")
    _logger.info("mutation.attempt", **fields)


def audit_mutation_execution(
    *,
    outcome: str,
    tool: str,
    arguments: dict[str, Any],
    request_id: str,
    session_id: str,
) -> None:
    _audit(
        outcome,
        tool=tool,
        arguments=arguments,
        request_id=request_id,
        session_id=session_id,
    )
