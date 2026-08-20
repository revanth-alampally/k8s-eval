"""Agent request/response contracts.

What is *not* here matters as much as what is. There is no field carrying the model's
reasoning, no field carrying the raw transcript, and no field carrying full tool output.
Callers get the answer plus an execution record they can audit; the working material
stays server-side and reachable only through the correlation ID in the logs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    SUCCESS = "success"
    # The model chose a mutating tool; nothing was executed.
    CONFIRMATION_REQUIRED = "confirmation_required"
    # The tool-call budget ran out before the model settled on an answer.
    INCOMPLETE = "incomplete"


class ToolOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    # Rejected before execution: unknown tool, or a mutation awaiting confirmation.
    BLOCKED = "blocked"


class ToolInvocation(BaseModel):
    """One line of the execution record: what was run, with what, and how it went."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    outcome: ToolOutcome
    duration_ms: float
    # A short factual descriptor of the result, never the result itself.
    summary: str | None = None
    error_code: str | None = None


class PendingConfirmation(BaseModel):
    """A mutating action the agent stopped short of performing."""

    tool: str
    arguments: dict[str, Any]
    description: str
    confirmation_token: str | None = None
    expires_at: datetime | None = None


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    confirmation_token: str | None = Field(default=None, min_length=20, max_length=512)


class AgentResponse(BaseModel):
    answer: str
    request_id: str
    trace_id: str = ""
    status: AgentStatus
    tools_used: list[ToolInvocation] = Field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None


class AgentResult(BaseModel):
    """The orchestrator's return value, mapped onto ``AgentResponse`` by the route."""

    answer: str
    status: AgentStatus
    tools_used: list[ToolInvocation] = Field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None
