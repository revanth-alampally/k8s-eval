"""Versioned, serialisable contracts for evaluation input and output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    schema_version: str = "k8s-agent-eval/v1"
    id: str
    user_input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str
    fixture: str = "mixed_cluster"
    expected_status: str | None = None
    allowed_tools: list[str] | None = None
    required_answer_terms: list[str] = Field(default_factory=list)
    forbidden_answer_terms: list[str] = Field(default_factory=list)

    @property
    def permitted_tools(self) -> list[str]:
        return self.allowed_tools if self.allowed_tools is not None else self.expected_tools


class MetricScores(BaseModel):
    tool_selection_accuracy: float
    tool_argument_accuracy: float
    tool_call_precision: float
    task_success: float
    groundedness: float
    hallucination: float
    safety_violation: float


class EvalCaseResult(BaseModel):
    id: str
    scores: MetricScores
    passed: bool
    failures: list[str] = Field(default_factory=list)
    answer: str = ""
    status: str | None = None
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    total_latency_ms: float
    model_latency_ms: float
    tool_latency_ms: float
    model_turns: int
    tool_calls: int


class EvalRun(BaseModel):
    dataset: str
    provider: str
    cases: list[EvalCaseResult]
    metrics: dict[str, float]
