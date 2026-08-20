"""Natural-language Kubernetes operations.

The HTTP envelope is deliberately thin: a message in, an answer plus an execution
record out. Tool payloads, the model transcript and any chain-of-thought stay on this
side of the process, reachable from logs by ``request_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anyio.to_thread
from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.orchestrator import Agent
from app.agent.schemas import AgentRequest, AgentResponse
from app.api.deps import ClientProviderDep, KnowledgeServiceDep, LLMProviderDep, SettingsDep
from app.context import get_correlation_id, new_correlation_id
from app.tools.registry import execute_tool

router = APIRouter(prefix="/v1", tags=["agent"])


@router.post("/agent", response_model=AgentResponse, summary="Ask the operations agent")
async def run_agent(
    body: AgentRequest,
    settings: SettingsDep,
    kubernetes: ClientProviderDep,
    llm: LLMProviderDep,
    knowledge: KnowledgeServiceDep,
) -> AgentResponse:
    client = await anyio.to_thread.run_sync(kubernetes.get)

    def execute(name: str, arguments: Mapping[str, Any]) -> BaseModel:
        return execute_tool(name, arguments, client=client, settings=settings)

    result = await Agent(
        llm=llm,
        execute=execute,
        knowledge=knowledge,
        settings=settings,
    ).run(body.message)
    return AgentResponse(
        answer=result.answer,
        request_id=get_correlation_id() or new_correlation_id(),
        status=result.status,
        tools_used=result.tools_used,
        pending_confirmation=result.pending_confirmation,
    )
