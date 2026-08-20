"""Run fixture-backed AI evaluations and write a JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.agent.orchestrator import Agent
from app.config import Settings
from app.errors import AppError
from app.llm.base import LLMProvider, LLMResponse, Message, ToolDefinition
from app.llm.factory import build_provider
from app.llm.fake import HeuristicLLMProvider
from evals.fixtures import FixtureKnowledge, FixtureToolExecutor
from evals.schemas import EvalCase, EvalCaseResult, EvalRun, TrajectoryEvent
from evals.scoring import aggregate, score_case


class RecordingLLMProvider(LLMProvider):
    """Record timing/counts without requiring a provider to expose internal telemetry."""

    def __init__(self, provider: LLMProvider, events: list[TrajectoryEvent]) -> None:
        self._provider = provider
        self._events = events
        self.name = provider.name
        self.latency_ms = 0.0
        self.turns = 0

    async def generate(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._provider.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._events.append(
                TrajectoryEvent(
                    event="model_decision",
                    summary="tool_calls" if response.tool_calls else "final_response",
                    arguments={"tool_calls": [call.model_dump() for call in response.tool_calls]},
                )
            )
            return response
        finally:
            self.turns += 1
            self.latency_ms += (time.perf_counter() - started) * 1000


class RecordingFixtureExecutor:
    """Decorates fixture tools with visible call/result trajectory events."""

    def __init__(self, executor: FixtureToolExecutor, events: list[TrajectoryEvent]) -> None:
        self._executor = executor
        self._events = events

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
        values = dict(arguments)
        self._events.append(TrajectoryEvent(event="tool_call", tool=name, arguments=values))
        try:
            result = self._executor(name, values)
        except AppError as exc:
            self._events.append(
                TrajectoryEvent(
                    event="tool_result",
                    tool=name,
                    outcome=exc.code.value,
                    summary=exc.message,
                )
            )
            raise
        self._events.append(
            TrajectoryEvent(
                event="tool_result",
                tool=name,
                outcome="success",
                summary=type(result).__name__,
            )
        )
        return result


async def run_case(
    case: EvalCase,
    *,
    provider: LLMProvider,
    settings: Settings,
) -> EvalCaseResult:
    executor = FixtureToolExecutor(case.fixture)
    trajectory = [TrajectoryEvent(event="user_request", summary=case.user_input)]
    recording_provider = RecordingLLMProvider(provider, trajectory)
    recording_executor = RecordingFixtureExecutor(executor, trajectory)
    started = time.perf_counter()
    result = None
    error: AppError | None = None
    try:
        result = await Agent(
            llm=recording_provider,
            execute=recording_executor,
            settings=settings,
            knowledge=FixtureKnowledge(),  # type: ignore[arg-type]
        ).run(case.user_input)
    except AppError as exc:
        error = exc
    total_ms = (time.perf_counter() - started) * 1000
    tools = [] if result is None else [item.model_dump(mode="json") for item in result.tools_used]
    _append_unrecorded_invocations(trajectory, tools)
    answer = error.message if error is not None else (result.answer if result is not None else "")
    status = error.code.value if error is not None else (result.status.value if result else None)
    trajectory.append(
        TrajectoryEvent(
            event="final_response",
            outcome=status,
            summary="error" if error is not None else "answer",
        )
    )
    scores, failures = score_case(
        case,
        answer=answer,
        status=status,
        tools_used=tools,
        mutation_executed=executor.mutation_executed,
    )
    tool_latency = sum(float(item.get("duration_ms", 0.0)) for item in tools)
    return EvalCaseResult(
        id=case.id,
        scores=scores,
        passed=not failures,
        failures=failures,
        answer=answer,
        status=status,
        tools_used=tools,
        total_latency_ms=round(total_ms, 2),
        model_latency_ms=round(recording_provider.latency_ms, 2),
        tool_latency_ms=round(tool_latency, 2),
        model_turns=recording_provider.turns,
        tool_calls=len(tools),
        trajectory=trajectory,
    )


async def run_suite(
    cases: Sequence[EvalCase],
    *,
    provider: LLMProvider,
    settings: Settings,
    dataset: str,
) -> EvalRun:
    results = [await run_case(case, provider=provider, settings=settings) for case in cases]
    latency = {
        "average_total_latency_ms": _average([item.total_latency_ms for item in results]),
        "average_model_latency_ms": _average([item.model_latency_ms for item in results]),
        "average_tool_latency_ms": _average([item.tool_latency_ms for item in results]),
        "average_model_turns": _average([float(item.model_turns) for item in results]),
        "average_tool_calls": _average([float(item.tool_calls) for item in results]),
    }
    return EvalRun(
        dataset=dataset,
        provider=provider.name,
        cases=results,
        metrics=aggregate([item.scores for item in results], latency),
    )


def load_cases(path: Path) -> list[EvalCase]:
    cases = [EvalCase.model_validate_json(line) for line in path.read_text().splitlines() if line]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation dataset contains duplicate IDs.")
    return cases


def write_results(run: EvalRun, path: Path) -> None:
    path.write_text(json.dumps(run.model_dump(mode="json"), indent=2) + "\n")


def print_summary(run: EvalRun) -> None:
    print(f"Provider: {run.provider} | Cases: {len(run.cases)}")
    print("| Metric | Value |")
    print("| --- | ---: |")
    for name, value in run.metrics.items():
        print(f"| {name.replace('_', ' ')} | {value:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hermetic Kubernetes-agent evaluations.")
    parser.add_argument("--dataset", type=Path, default=Path("evals/k8s_ops_agent.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval-results.json"))
    parser.add_argument("--provider", choices=("fake", "configured"), default="fake")
    args = parser.parse_args()
    settings = Settings()
    provider = (
        HeuristicLLMProvider(default_namespace=settings.default_namespace)
        if args.provider == "fake"
        else build_provider(settings)
    )
    run = asyncio.run(
        run_suite(
            load_cases(args.dataset),
            provider=provider,
            settings=settings,
            dataset=str(args.dataset),
        )
    )
    write_results(run, args.output)
    print_summary(run)


def _average(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _append_unrecorded_invocations(
    trajectory: list[TrajectoryEvent], tools_used: Sequence[dict[str, Any]]
) -> None:
    """Capture adapter-blocked calls, such as confirmation-required mutations."""
    recorded = {
        (event.tool, json.dumps(event.arguments, sort_keys=True))
        for event in trajectory
        if event.event == "tool_call"
    }
    for invocation in tools_used:
        key = (str(invocation["tool"]), json.dumps(invocation.get("arguments", {}), sort_keys=True))
        if key in recorded:
            continue
        trajectory.extend(
            [
                TrajectoryEvent(
                    event="tool_call",
                    tool=key[0],
                    arguments=dict(invocation.get("arguments", {})),
                ),
                TrajectoryEvent(
                    event="tool_result",
                    tool=key[0],
                    outcome=str(invocation.get("outcome")),
                    summary=invocation.get("summary"),
                ),
            ]
        )


if __name__ == "__main__":
    main()
