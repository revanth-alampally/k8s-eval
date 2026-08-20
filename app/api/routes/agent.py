"""Natural-language Kubernetes operations.

The HTTP envelope is deliberately thin: a message in, an answer plus an execution
record out. Tool payloads, the model transcript and any chain-of-thought stay on this
side of the process, reachable from logs by ``request_id``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.confirmation import audit_mutation_execution
from app.agent.orchestrator import Agent
from app.agent.schemas import AgentRequest, AgentResponse, AgentStatus, ToolInvocation, ToolOutcome
from app.api.deps import (
    ClientProviderDep,
    ConfirmationStoreDep,
    KnowledgeServiceDep,
    LLMProviderDep,
    MetricsDep,
    SettingsDep,
)
from app.context import get_correlation_id, get_trace_id, new_correlation_id
from app.errors import ConfirmationInvalidError, InvalidRequestError
from app.observability.instrumentation import emit_agent_event
from app.tools.base import require_mutating_tool_allowed
from app.tools.registry import execute_tool, get_tool
from app.tools.schemas import parse_arguments

router = APIRouter(prefix="/v1", tags=["agent"])
SESSION_ID_HEADER = "X-KAgent-Session-ID"


@router.post("/agent", response_model=AgentResponse, summary="Ask the operations agent")
async def run_agent(
    request: Request,
    body: AgentRequest,
    settings: SettingsDep,
    kubernetes: ClientProviderDep,
    llm: LLMProviderDep,
    knowledge: KnowledgeServiceDep,
    confirmations: ConfirmationStoreDep,
    metrics: MetricsDep,
) -> AgentResponse:
    session_id = request.headers.get(SESSION_ID_HEADER)
    request_id = get_correlation_id() or new_correlation_id()
    trace_id = get_trace_id() or request_id
    started = time.perf_counter()
    emit_agent_event(metrics, "agent.request", request_id=request_id, trace_id=trace_id)
    metrics.increment("agent_requests_total")

    def execute(name: str, arguments: Mapping[str, Any]) -> BaseModel:
        client = kubernetes.get()
        return execute_tool(name, arguments, client=client, settings=settings)

    if body.confirmation_token is not None:
        if not _is_affirmative(body.message):
            metrics.increment("safety_denials_total", outcome="confirmation_invalid")
            raise ConfirmationInvalidError("Confirmation requests must explicitly say 'Yes'.")
        if not session_id:
            raise InvalidRequestError(f"{SESSION_ID_HEADER} is required for confirmation.")
        action = confirmations.consume(token=body.confirmation_token, session_id=session_id)
        # Re-check current registration, explicit allowlist and Pydantic input validation
        # before crossing into the existing Kubernetes executor.
        spec = get_tool(action.tool, settings)
        if not spec.mutating:
            raise ConfirmationInvalidError("The confirmation token does not represent a mutation.")
        require_mutating_tool_allowed(action.tool, settings)
        arguments = parse_arguments(spec.input_model, **action.arguments).model_dump(mode="json")
        try:
            result = await anyio.to_thread.run_sync(execute, action.tool, arguments)
        except Exception:
            audit_mutation_execution(
                outcome="execution_failed",
                tool=action.tool,
                arguments=arguments,
                request_id=action.request_id,
                session_id=session_id,
            )
            raise
        audit_mutation_execution(
            outcome="executed",
            tool=action.tool,
            arguments=arguments,
            request_id=action.request_id,
            session_id=session_id,
        )
        response = AgentResponse(
            answer=str(result.model_dump(mode="json").get("message", "Mutation completed.")),
            request_id=request_id,
            trace_id=trace_id,
            status=AgentStatus.SUCCESS,
            tools_used=[
                ToolInvocation(
                    tool=action.tool,
                    arguments=arguments,
                    outcome=ToolOutcome.SUCCESS,
                    duration_ms=0.0,
                    summary="confirmed mutation executed",
                )
            ],
        )
        _record_response(metrics, request_id, trace_id, response, started)
        return response

    result = await Agent(
        llm=llm,
        execute=execute,
        knowledge=knowledge,
        settings=settings,
        metrics=metrics,
    ).run(body.message)
    if result.pending_confirmation is not None:
        if not session_id:
            metrics.increment("safety_denials_total", outcome="missing_session")
            audit_mutation_execution(
                outcome="rejected_missing_session",
                tool=result.pending_confirmation.tool,
                arguments=result.pending_confirmation.arguments,
                request_id=request_id,
                session_id="<missing>",
            )
            raise InvalidRequestError(f"{SESSION_ID_HEADER} is required to request a mutation.")
        spec = get_tool(result.pending_confirmation.tool, settings)
        if not spec.mutating:
            raise ConfirmationInvalidError("Only mutating tools may request confirmation.")
        require_mutating_tool_allowed(spec.name, settings)
        arguments = parse_arguments(
            spec.input_model, **result.pending_confirmation.arguments
        ).model_dump(mode="json")
        token, expires_at = confirmations.issue(
            tool=spec.name,
            arguments=arguments,
            session_id=session_id,
            request_id=request_id,
        )
        result.pending_confirmation = result.pending_confirmation.model_copy(
            update={
                "arguments": arguments,
                "confirmation_token": token,
                "expires_at": expires_at,
            }
        )
    response = AgentResponse(
        answer=result.answer,
        request_id=request_id,
        trace_id=trace_id,
        status=result.status,
        tools_used=result.tools_used,
        pending_confirmation=result.pending_confirmation,
    )
    _record_response(metrics, request_id, trace_id, response, started)
    return response


def _is_affirmative(message: str) -> bool:
    return message.strip().lower().rstrip(".!") in {"yes", "confirm", "proceed"}


def _record_response(
    metrics: MetricsDep,
    request_id: str,
    trace_id: str,
    response: AgentResponse,
    started: float,
) -> None:
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    emit_agent_event(
        metrics,
        "agent.response",
        request_id=request_id,
        trace_id=trace_id,
        status=response.status.value,
        tool_calls=len(response.tools_used),
        duration_ms=duration_ms,
    )
    metrics.observe("agent_latency_seconds", duration_ms / 1000)
    metrics.observe("agent_tool_calls_per_request", float(len(response.tools_used)))
    if response.status.value != "success":
        metrics.increment("agent_errors_total", status=response.status.value)
