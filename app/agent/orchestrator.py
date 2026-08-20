"""LangChain planning over safe typed tool adapters.

LangChain decides whether to answer, retrieve repository knowledge, or request a typed
tool. The adapters are deliberately thin: Kubernetes requests pass into the existing
Tool Executor, which remains the only component allowed to access the cluster.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol

import anyio.to_thread
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.agent.grounding import (
    INSUFFICIENT_EVIDENCE,
    classify_live_state_request,
    unsupported_resource,
    unsupported_resource_answer,
)
from app.agent.langchain_model import ProviderChatModel
from app.agent.prompts import system_prompt
from app.agent.schemas import (
    AgentResult,
    AgentStatus,
    PendingConfirmation,
    ToolInvocation,
    ToolOutcome,
)
from app.config import Settings
from app.errors import (
    AppError,
    ClusterTimeoutError,
    ClusterUnavailableError,
    LLMUnavailableError,
)
from app.knowledge.service import KnowledgeService
from app.llm.base import LLMProvider
from app.observability.instrumentation import track_operation
from app.tools.base import ToolSpec
from app.tools.registry import build_registry

_LOOP_LIMIT = (
    "I reached the tool-call limit before I could finish. "
    "I will not guess at cluster state I did not retrieve."
)


class _ConfirmationRequested(Exception):
    """Stops the LangChain graph immediately after a mutation was selected."""


class _ToolBudgetExceeded(Exception):
    """Stops the graph without permitting another tool execution."""


class ToolExecutor(Protocol):
    """Runs a named tool after the agent has authorised the call.

    Implementations are responsible for argument validation and for talking to
    Kubernetes. The agent never does either.
    """

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> BaseModel: ...


class Agent:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        execute: ToolExecutor,
        settings: Settings,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._llm = llm
        self._execute = execute
        self._settings = settings
        self._knowledge = knowledge

    async def run(self, message: str) -> AgentResult:
        registry = build_registry(self._settings)
        runtime = _ToolRuntime(
            registry=registry,
            execute=self._execute,
            knowledge=self._knowledge,
            settings=self._settings,
        )
        tools = runtime.tools()
        agent = create_agent(
            ProviderChatModel(provider=self._llm),
            tools,
            system_prompt=system_prompt(self._settings),
        )

        with track_operation(
            "agent.langchain_run",
            provider=self._llm.name,
            max_tool_calls=self._settings.max_tool_calls_per_request,
        ) as log:
            unsupported = unsupported_resource(message)
            if unsupported is not None:
                return AgentResult(
                    answer=unsupported_resource_answer(unsupported),
                    status=AgentStatus.SUCCESS,
                )

            live_request = classify_live_state_request(message)
            agent_message = message
            if live_request is not None:
                evidence = await runtime.acquire_required_evidence(
                    registry[live_request.required_tool]
                )
                if not runtime.has_successful_kubernetes_tool:
                    return AgentResult(
                        answer=INSUFFICIENT_EVIDENCE,
                        status=AgentStatus.INCOMPLETE,
                        tools_used=runtime.invocations,
                    )
                # The full payload is request-local: it is passed only to the planner
                # and never to logs or the HTTP response.
                agent_message = (
                    f"{message}\n\n"
                    f"Authoritative Kubernetes evidence from "
                    f"`{live_request.required_tool}`:\n{evidence}"
                )
            try:
                state = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": agent_message}]},
                    config={"recursion_limit": (self._settings.max_tool_calls_per_request * 2) + 3},
                )
            except Exception as exc:
                if isinstance(exc, _ConfirmationRequested) and runtime.pending is not None:
                    return AgentResult(
                        answer=_confirmation_answer(runtime.pending),
                        status=AgentStatus.CONFIRMATION_REQUIRED,
                        tools_used=runtime.invocations,
                        pending_confirmation=runtime.pending,
                    )
                if isinstance(exc, _ToolBudgetExceeded):
                    return AgentResult(
                        answer=_LOOP_LIMIT,
                        status=AgentStatus.INCOMPLETE,
                        tools_used=runtime.invocations,
                    )
                if type(exc).__name__ == "GraphRecursionError":
                    return AgentResult(
                        answer=_LOOP_LIMIT,
                        status=AgentStatus.INCOMPLETE,
                        tools_used=runtime.invocations,
                    )
                if isinstance(
                    exc,
                    (ClusterUnavailableError, ClusterTimeoutError, LLMUnavailableError),
                ):
                    raise
                raise LLMUnavailableError(
                    "The language model provider is unavailable.",
                    provider=self._llm.name,
                    error_type=type(exc).__name__,
                ) from exc

            if runtime.pending is not None:
                log["tool_calls"] = len(runtime.invocations)
                log["status"] = AgentStatus.CONFIRMATION_REQUIRED.value
                return AgentResult(
                    answer=_confirmation_answer(runtime.pending),
                    status=AgentStatus.CONFIRMATION_REQUIRED,
                    tools_used=runtime.invocations,
                    pending_confirmation=runtime.pending,
                )

            answer = _final_answer(state.get("messages", []))
            if live_request is not None:
                answer = _format_grounded_answer(answer)
            status = AgentStatus.INCOMPLETE if runtime.limit_reached else AgentStatus.SUCCESS
            log["tool_calls"] = len(runtime.invocations)
            log["status"] = status.value
            return AgentResult(answer=answer, status=status, tools_used=runtime.invocations)


class _ToolRuntime:
    """Per-request adapter state; never shared across requests."""

    def __init__(
        self,
        *,
        registry: Mapping[str, ToolSpec],
        execute: ToolExecutor,
        knowledge: KnowledgeService | None,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._execute = execute
        self._knowledge = knowledge
        self._settings = settings
        self.invocations: list[ToolInvocation] = []
        self.pending: PendingConfirmation | None = None
        self.limit_reached = False

    @property
    def has_successful_kubernetes_tool(self) -> bool:
        return any(
            invocation.tool != "search_knowledge" and invocation.outcome is ToolOutcome.SUCCESS
            for invocation in self.invocations
        )

    async def acquire_required_evidence(self, spec: ToolSpec) -> str:
        """Run the baseline read tool that policy requires for live state."""
        return await self._invoke_registered(
            spec,
            {"namespace": self._settings.default_namespace},
        )

    def tools(self) -> list[StructuredTool]:
        tools = [self._registered_tool(spec) for spec in self._registry.values()]
        if self._settings.rag_enabled and self._knowledge is not None:
            tools.append(
                StructuredTool.from_function(
                    coroutine=self._search_knowledge,
                    name="search_knowledge",
                    description=(
                        "Search repository-owned runbooks and documentation. Use this "
                        "for static guidance, never as evidence of current cluster state."
                    ),
                )
            )
        return tools

    def _registered_tool(self, spec: ToolSpec) -> StructuredTool:
        async def invoke(**arguments: Any) -> str:
            return await self._invoke_registered(spec, arguments)

        return StructuredTool.from_function(
            coroutine=invoke,
            name=spec.name,
            description=spec.description,
            args_schema=spec.input_model,
        )

    async def _invoke_registered(self, spec: ToolSpec, arguments: Mapping[str, Any]) -> str:
        if self.pending is not None or self.limit_reached:
            return _error_payload("agent_stopped", "No further tools may run for this request.")

        if spec.requires_confirmation and self._settings.require_confirmation:
            self.pending = PendingConfirmation(
                tool=spec.name,
                arguments=dict(arguments),
                description=_mutation_description(spec.name, arguments),
            )
            self.invocations.append(
                ToolInvocation(
                    tool=spec.name,
                    arguments=dict(arguments),
                    outcome=ToolOutcome.BLOCKED,
                    duration_ms=0.0,
                    summary="confirmation required; not executed",
                    error_code="confirmation_required",
                )
            )
            raise _ConfirmationRequested

        if len(self.invocations) >= self._settings.max_tool_calls_per_request:
            self.limit_reached = True
            raise _ToolBudgetExceeded

        started = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(self._execute, spec.name, arguments)
        except (ClusterUnavailableError, ClusterTimeoutError):
            raise
        except AppError as exc:
            duration = _elapsed_ms(started)
            self.invocations.append(
                ToolInvocation(
                    tool=spec.name,
                    arguments=dict(arguments),
                    outcome=ToolOutcome.ERROR,
                    duration_ms=duration,
                    summary=exc.message,
                    error_code=exc.code.value,
                )
            )
            return _error_payload(exc.code.value, exc.message)

        payload = result.model_dump(mode="json")
        self.invocations.append(
            ToolInvocation(
                tool=spec.name,
                arguments=dict(arguments),
                outcome=ToolOutcome.SUCCESS,
                duration_ms=_elapsed_ms(started),
                summary=summarise_result(spec.name, payload),
            )
        )
        return _json(payload)

    async def _search_knowledge(self, query: str) -> str:
        if self.pending is not None or self.limit_reached:
            return _error_payload("agent_stopped", "No further tools may run for this request.")
        if len(self.invocations) >= self._settings.max_tool_calls_per_request:
            self.limit_reached = True
            raise _ToolBudgetExceeded

        started = time.perf_counter()
        assert self._knowledge is not None
        hits = await anyio.to_thread.run_sync(self._knowledge.search, query)
        payload = {"results": [hit.model_dump(mode="json") for hit in hits]}
        self.invocations.append(
            ToolInvocation(
                tool="search_knowledge",
                arguments={"query": query},
                outcome=ToolOutcome.SUCCESS,
                duration_ms=_elapsed_ms(started),
                summary=f"{len(hits)} documentation chunks retrieved",
            )
        )
        return _json(payload)


def _confirmation_answer(pending: PendingConfirmation) -> str:
    return (
        f"{pending.description} This changes cluster state, so I have not done it. "
        "Confirm if you want me to proceed."
    )


def _mutation_description(tool: str, arguments: Mapping[str, Any]) -> str:
    if tool == "restart_deployment":
        name = arguments.get("deployment_name", "the deployment")
        namespace = arguments.get("namespace", "the namespace")
        return f"I would restart deployment '{name}' in namespace '{namespace}'."
    return f"I would run '{tool}' with {dict(arguments)}."


def summarise_result(tool: str, payload: Mapping[str, Any]) -> str:
    """A one-line factual descriptor. Never the payload itself."""
    if tool == "list_pods":
        return f"{payload.get('total', 0)} pods, {payload.get('unhealthy_count', 0)} unhealthy"
    if tool == "list_deployments":
        return (
            f"{payload.get('total', 0)} deployments, "
            f"{payload.get('unavailable_count', 0)} unavailable"
        )
    if tool == "get_pod":
        return f"phase={payload.get('phase')}, healthy={payload.get('healthy')}"
    if tool == "describe_pod":
        pod = payload.get("pod", {})
        events = payload.get("events", [])
        phase = pod.get("phase") if isinstance(pod, dict) else None
        return f"phase={phase}, events={len(events)}"
    if tool == "diagnose_pod":
        return (
            f"status={payload.get('status')}, "
            f"logs_available={payload.get('logs_available')}, "
            f"signals={len(payload.get('signals', []))}"
        )
    if tool == "get_pod_logs":
        return f"{payload.get('line_count', 0)} lines, truncated={payload.get('truncated')}"
    if tool == "restart_deployment":
        return str(payload.get("message", "restart triggered"))
    return tool


def _final_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            answer = message.content.strip()
            if answer:
                return answer
    return INSUFFICIENT_EVIDENCE


def _format_grounded_answer(answer: str) -> str:
    """Require readers to see facts and inference as separate categories."""
    if "Observed:" in answer and ("Likely cause:" in answer or "Interpretation:" in answer):
        return answer
    return (
        f"Observed:\n{answer}\n\nInterpretation:\nNo additional inference beyond the tool results."
    )


def _error_payload(code: str, message: str) -> str:
    return _json({"error": {"code": code, "message": message}})


def _json(value: object) -> str:
    return json.dumps(value, default=str)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
