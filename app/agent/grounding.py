"""Deterministic policy checks for live Kubernetes questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LIVE_PATTERNS = (
    r"\bwhat(?:'s| is)?\s+running\b",
    r"\bwhat\s+pods?\b",
    r"\bpod\s+status\b",
    r"\blogs?\b",
    r"\bdeployment\s+(?:state|status)\b",
    r"\brestart\s+counts?\b",
    r"\b(?:cluster|resource)\s+state\b",
    r"\b(?:unhealthy|failing|failed|crash(?:ing|ed)?|imagepullbackoff)\b",
)
_UNSUPPORTED_RESOURCE_PATTERNS = (
    r"\bnodes?\b",
    r"\bservices?\b",
    r"\bconfigmaps?\b",
    r"\bsecrets?\b",
    r"\b(?:persistent\s+)?volumes?\b",
    r"\bingresses?\b",
    r"\bstatefulsets?\b",
    r"\bdaemonsets?\b",
)

INSUFFICIENT_EVIDENCE = "I don't have enough cluster evidence to determine the cause."


@dataclass(frozen=True)
class LiveStateRequest:
    required_tool: str


def classify_live_state_request(question: str) -> LiveStateRequest | None:
    """Return the baseline evidence tool required for a live-state question."""
    normalised = question.lower()
    # Documentation requests can mention failure names without asking about a live
    # occurrence. They belong to the knowledge tool, not the cluster API.
    if any(word in normalised for word in ("runbook", "documentation", "docs", "guide")):
        return None
    if not any(re.search(pattern, normalised) for pattern in _LIVE_PATTERNS):
        return None
    if "deployment" in normalised and not re.search(r"\brestart\b", normalised):
        return LiveStateRequest(required_tool="list_deployments")
    # List pods is the safe initial lookup for pod health, logs and named workloads.
    return LiveStateRequest(required_tool="list_pods")


def unsupported_resource(question: str) -> str | None:
    """Identify resource types outside the intentionally small typed registry."""
    normalised = question.lower()
    for pattern in _UNSUPPORTED_RESOURCE_PATTERNS:
        match = re.search(pattern, normalised)
        if match:
            return match.group(0)
    return None


def unsupported_resource_answer(resource: str) -> str:
    return f"I don't have a tool capable of answering questions about {resource}."
