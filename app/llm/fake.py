"""Fake providers for development and tests.

``ScriptedLLMProvider`` replays a fixed list of responses, which is what unit tests use:
the agent's behaviour becomes fully deterministic and no network call is possible.

``HeuristicLLMProvider`` is a keyword-driven stand-in so the whole service runs end to
end without an API key. It is not a language model and does not pretend to be one, but
it obeys the same contract the real prompt imposes: it only ever states what a tool
returned, and it reaches those facts by calling the same tools through the same
validation path.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from app.errors import LLMUnavailableError
from app.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)

# Matches hyphenated Kubernetes-style names in free text, e.g. "nginx-missing".
_NAME_PATTERN = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)+)\b")

_LOG_WORDS = ("log", "logs", "output", "stdout")
_DIAGNOSE_WORDS = ("why", "failing", "failed", "fail", "broken", "crash", "wrong", "diagnose")
_HEALTH_WORDS = ("unhealthy", "healthy", "problem", "issue", "ok")
_RESTART_WORDS = ("restart", "reboot", "roll", "bounce")
_DEPLOYMENT_WORDS = ("deployment", "deployments", "replica", "replicas")
_POD_WORDS = ("pod", "pods", "running", "cluster", "workload", "workloads")


class ScriptedLLMProvider(LLMProvider):
    """Replays pre-built responses in order, recording what it was asked."""

    name = "scripted"

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses = deque(responses)
        self.calls: list[list[Message]] = []
        self.tools_offered: list[list[ToolDefinition]] = []

    async def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_offered.append(list(tools))
        if not self._responses:
            raise LLMUnavailableError("Scripted provider ran out of responses.")
        return self._responses.popleft()


class HeuristicLLMProvider(LLMProvider):
    """Keyword-matching stand-in that drives the real tools."""

    name = "heuristic"

    def __init__(self, default_namespace: str = "ai-agent-demo") -> None:
        self._namespace = default_namespace

    async def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        question = _last_user_message(messages).lower()
        available = {tool.name for tool in tools}
        results = _tool_results(messages)
        target = _first_name(question)

        if not tools:
            # Tools withheld: the orchestrator wants a final answer now.
            return _answer(_summarise(results, question))

        if not results:
            return self._first_step(question, target, available)
        return self._next_step(question, target, available, results)

    def _first_step(self, question: str, target: str | None, available: set[str]) -> LLMResponse:
        if _mentions(question, _RESTART_WORDS) and target and "restart_deployment" in available:
            return _call(
                "restart_deployment",
                {"namespace": self._namespace, "deployment_name": target},
            )
        needs_pod = _mentions(question, _DIAGNOSE_WORDS) or _mentions(question, _LOG_WORDS)
        if needs_pod and target:
            # The user names a workload, not a pod. Resolve it to a real pod first
            # rather than guessing at the generated suffix.
            return _call("list_pods", {"namespace": self._namespace})
        if _mentions(question, _DEPLOYMENT_WORDS) and "list_deployments" in available:
            return _call("list_deployments", {"namespace": self._namespace})
        if _mentions(question, _POD_WORDS) or _mentions(question, _HEALTH_WORDS):
            return _call("list_pods", {"namespace": self._namespace})
        return _answer(
            "I can report on pods and deployments, read container logs, gather "
            "diagnostic evidence for a failing pod, and restart a deployment. "
            "What would you like to know?"
        )

    def _next_step(
        self,
        question: str,
        target: str | None,
        available: set[str],
        results: list[tuple[str, Any]],
    ) -> LLMResponse:
        seen = {name for name, _ in results}
        pod = _match_pod(results, target)

        if pod and "diagnose_pod" not in seen and _mentions(question, _DIAGNOSE_WORDS):
            return _call("diagnose_pod", {"namespace": self._namespace, "pod_name": pod})
        if pod and "get_pod_logs" not in seen and _mentions(question, _LOG_WORDS):
            return _call(
                "get_pod_logs",
                {"namespace": self._namespace, "pod_name": pod, "tail_lines": 25},
            )
        return _answer(_summarise(results, question))


def _call(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)],
        model="heuristic",
        finish_reason="tool_calls",
    )


def _answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="heuristic", finish_reason="stop")


def _mentions(text: str, words: Sequence[str]) -> bool:
    return any(word in text for word in words)


def _last_user_message(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role is Role.USER and message.content:
            return message.content
    return ""


def _tool_results(messages: Sequence[Message]) -> list[tuple[str, Any]]:
    results: list[tuple[str, Any]] = []
    for message in messages:
        if message.role is not Role.TOOL or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            payload = message.content
        results.append((message.name or "unknown", payload))
    return results


def _first_name(question: str) -> str | None:
    match = _NAME_PATTERN.search(question)
    return match.group(1) if match else None


def _match_pod(results: list[tuple[str, Any]], target: str | None) -> str | None:
    """Resolve a workload name the user typed to a real pod name from list_pods."""
    for name, payload in results:
        if name != "list_pods" or not isinstance(payload, dict):
            continue
        pods = [pod for pod in payload.get("pods", []) if isinstance(pod, dict)]
        if target:
            for pod in pods:
                if str(pod.get("name", "")).startswith(target):
                    return str(pod["name"])
        unhealthy = [pod for pod in pods if not pod.get("healthy", True)]
        if unhealthy:
            return str(unhealthy[0]["name"])
    return None


def _summarise(results: list[tuple[str, Any]], question: str) -> str:
    if not results:
        return "I do not have any cluster information for that yet."

    name, payload = results[-1]
    if isinstance(payload, dict) and "error" in payload:
        error = payload["error"]
        return f"I could not complete that: {error.get('message', 'the tool failed')}"
    if not isinstance(payload, dict):
        return str(payload)[:400]

    if name == "list_pods":
        return _summarise_pods(payload)
    if name == "diagnose_pod":
        return _summarise_diagnosis(payload)
    if name == "get_pod_logs":
        return _summarise_logs(payload)
    if name == "list_deployments":
        return _summarise_deployments(payload)
    if name == "restart_deployment":
        return str(payload.get("message", "The restart was triggered."))
    return f"Collected {name} results."


def _summarise_pods(payload: dict[str, Any]) -> str:
    total = payload.get("total", 0)
    unhealthy = payload.get("unhealthy_pods", [])
    namespace = payload.get("namespace", "")
    if not unhealthy:
        return f"All {total} pods in {namespace} are healthy."
    return f"{total} pods in {namespace}, {len(unhealthy)} unhealthy: {', '.join(unhealthy)}."


def _summarise_diagnosis(payload: dict[str, Any]) -> str:
    pod = payload.get("pod", "the pod")
    status = payload.get("status", "unknown")
    signals = [s for s in payload.get("signals", []) if s.get("severity") == "warning"]
    lines = [f"{pod} is in state {status}."]
    for signal in signals[:3]:
        lines.append(f"- {signal.get('reason')}: {signal.get('evidence', '')[:220]}")
    if not payload.get("logs_available", True):
        lines.append("No container logs exist, so no container ever started.")
    return "\n".join(lines)


def _summarise_logs(payload: dict[str, Any]) -> str:
    pod = payload.get("pod_name", "the pod")
    content = str(payload.get("content", "")).strip()
    tail = "\n".join(content.splitlines()[-8:])
    return f"Last lines from {pod}:\n{tail}" if tail else f"{pod} has produced no log output."


def _summarise_deployments(payload: dict[str, Any]) -> str:
    deployments = payload.get("deployments", [])
    unavailable = [d["name"] for d in deployments if not d.get("available", True)]
    total = payload.get("total", len(deployments))
    if not unavailable:
        return f"All {total} deployments are fully available."
    return f"{total} deployments, {len(unavailable)} not fully available: {', '.join(unavailable)}."


FakeLLMProvider = ScriptedLLMProvider
