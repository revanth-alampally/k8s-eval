from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Environment, Settings
from app.llm.base import LLMResponse
from app.llm.fake import ScriptedLLMProvider
from evals.fixtures import FixtureToolExecutor
from evals.runner import load_cases, run_suite, write_results
from evals.schemas import EvalCase
from evals.scoring import score_case


async def test_runner_uses_scripted_provider_and_serialises_results(tmp_path: Path) -> None:
    case = EvalCase(
        id="direct-001",
        user_input="Hello",
        expected_tools=[],
        expected_behavior="direct_answer",
    )
    settings = Settings(environment=Environment.TEST)
    provider = ScriptedLLMProvider([LLMResponse(content="Hello", finish_reason="stop")])

    run = await run_suite([case], provider=provider, settings=settings, dataset="fixture.jsonl")
    output = tmp_path / "eval-results.json"
    write_results(run, output)

    assert run.cases[0].passed is True
    assert run.cases[0].model_turns == 1
    assert json.loads(output.read_text())["provider"] == "scripted"


def test_fixture_executor_never_executes_mutations() -> None:
    executor = FixtureToolExecutor()

    with pytest.raises(AssertionError):
        executor(
            "restart_deployment", {"namespace": "ai-agent-demo", "deployment_name": "nginx-good"}
        )

    assert executor.mutation_executed is True


def test_scorer_detects_unnecessary_tools_and_hallucinated_terms() -> None:
    case = EvalCase(
        id="score-001",
        user_input="List pods",
        expected_tools=["list_pods"],
        expected_arguments={"namespace": "ai-agent-demo"},
        expected_behavior="grounded_answer",
        forbidden_answer_terms=["all pods healthy"],
    )
    scores, failures = score_case(
        case,
        answer="All pods healthy",
        status="success",
        tools_used=[
            {
                "tool": "list_pods",
                "arguments": {"namespace": "ai-agent-demo"},
                "outcome": "success",
            },
            {"tool": "get_pod", "arguments": {}, "outcome": "success"},
        ],
        mutation_executed=False,
    )

    assert scores.tool_call_precision == 0.0
    assert scores.hallucination == 1.0
    assert failures


def test_dataset_loads_and_has_unique_ids() -> None:
    cases = load_cases(Path("evals/k8s_ops_agent.jsonl"))

    assert len(cases) >= 25
    assert len({case.id for case in cases}) == len(cases)
