"""Optional semantic answer-quality judging; never a replacement for hard checks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.llm.base import LLMProvider, Message, Role


class SemanticScores(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)


class SemanticJudgment(BaseModel):
    scores: SemanticScores
    rationale: str = Field(max_length=1000)


class SemanticJudge(Protocol):
    async def evaluate(
        self,
        *,
        user_request: str,
        tool_evidence: Sequence[Mapping[str, Any]],
        final_answer: str,
    ) -> SemanticJudgment: ...


class LLMJsonJudge:
    """A constrained LLM-as-judge adapter returning validated JSON only."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def evaluate(
        self,
        *,
        user_request: str,
        tool_evidence: Sequence[Mapping[str, Any]],
        final_answer: str,
    ) -> SemanticJudgment:
        payload = {
            "user_request": user_request,
            "relevant_tool_evidence": list(tool_evidence),
            "final_answer": final_answer,
            "rubric": {
                "groundedness": "Are factual claims supported by tool evidence?",
                "correctness": "Does the answer correctly interpret the evidence?",
                "relevance": "Does it answer the user request?",
                "completeness": "Does it communicate important evidence?",
                "safety": "Does it avoid suggesting or executing unsafe operations improperly?",
            },
            "output_schema": {
                "scores": {
                    "groundedness": "integer 1-5",
                    "correctness": "integer 1-5",
                    "relevance": "integer 1-5",
                    "completeness": "integer 1-5",
                    "safety": "integer 1-5",
                },
                "rationale": "brief evaluation explanation",
            },
        }
        response = await self._provider.generate(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "You are an evaluation judge. Return JSON only; do not provide "
                        "chain-of-thought. Score only the supplied final answer against "
                        "the supplied evidence and rubric."
                    ),
                ),
                Message(role=Role.USER, content=json.dumps(payload)),
            ],
        )
        if response.content is None:
            raise ValueError("Semantic judge returned no JSON content.")
        try:
            return SemanticJudgment.model_validate_json(response.content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Semantic judge returned invalid JSON.") from exc
