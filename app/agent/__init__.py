"""Agent orchestration (not implemented yet).

Planned contract for this package:

``schemas.py``
    ``AgentRequest`` / ``AgentResponse``. The response carries the natural-language
    answer plus the structured trace of tool calls it was derived from, so a caller can
    always audit which cluster facts backed the summary.

``planner.py``
    One turn of the loop: hand the question, conversation state and tool schemas to the
    LLM, and get back either a tool call or a final answer. The LLM only ever chooses
    tools and phrases results -- it never supplies cluster state.

``orchestrator.py``
    Drives the loop: plan -> validate arguments -> execute tool -> feed the result back,
    bounded by ``settings.max_tool_calls_per_request``. Intercepts mutating tools and
    raises ``ConfirmationRequiredError`` with a confirmation token instead of executing.

``prompts.py``
    System prompt. Its central rule: never state a fact about the cluster that did not
    come from a tool result; if no tool provided it, say so.
"""
