"""Run fixture-backed AI evaluations and write a JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path

from app.agent.orchestrator import Agent
from app.config import Settings
from app.errors import AppError
from app.llm.base import LLMProvider, LLMResponse, Message, ToolDefinition
from app.llm.factory import build_provider
from app.llm.fake import HeuristicLLMProvider
from evals.fixtures import FixtureKnowledge, FixtureToolExecutor
from evals.schemas import EvalCase, EvalCaseResult, EvalRun
from evals.scoring import aggregate, score_case


class RecordingLLMProvider(LLMProvider):
    """Record timing/counts without requiring a provider to expose internal telemetry."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
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
            return await self._provider.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            self.turns += 1
            self.latency_ms += (time.perf_counter() - started) * 1000


async def run_case(
    case: EvalCase,
    *,
    provider: LLMProvider,
    settings: Settings,
) -> EvalCaseResult:
    executor = FixtureToolExecutor(case.fixture)
    recording_provider = RecordingLLMProvider(provider)
    started = time.perf_counter()
    result = None
    error: AppError | None = None
    try:
        result = await Agent(
            llm=recording_provider,
            execute=executor,
            settings=settings,
            knowledge=FixtureKnowledge(),  # type: ignore[arg-type]
        ).run(case.user_input)
    except AppError as exc:
        error = exc
    total_ms = (time.perf_counter() - started) * 1000
    tools = [] if result is None else [item.model_dump(mode="json") for item in result.tools_used]
    answer = error.message if error is not None else (result.answer if result is not None else "")
    status = error.code.value if error is not None else (result.status.value if result else None)
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


if __name__ == "__main__":
    main()
