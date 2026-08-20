# AI evaluation corpus

This directory is intentionally separate from `tests/`.

- `tests/` contains deterministic pytest unit/integration tests for Python code,
  FastAPI contracts, Kubernetes-client error handling, validation, and confirmation
  capabilities.
- `evals/k8s_ops_agent.jsonl` is an AI behavior corpus. An evaluator supplies each
  `user_input` to an agent, then scores tool choice, arguments, confirmation behavior,
  grounding, and refusal behavior against the expected fields.

The JSONL corpus is not collected by pytest and is not a pre-commit requirement. It may
be run against a real model, so latency and output can vary. Keep any live-cluster or
real-model evaluator separately marked with pytest's existing `eval` marker.

Each record contains:

- `id`: stable scenario identifier.
- `user_input`: prompt sent to the agent.
- `expected_tools`: ordered typed tool names, or an empty list when no tool is allowed.
- `expected_arguments`: expected arguments for the first tool, when applicable.
- `expected_behavior`: scorer-friendly behavior label.

Optional v1 contract fields make a case executable and deterministic:

- `fixture`: `mixed_cluster` (default), `cluster_unavailable`, or `cluster_timeout`.
- `expected_status`: required result status/error code.
- `allowed_tools`: tool calls permitted in addition to required `expected_tools`.
- `required_answer_terms` / `forbidden_answer_terms`: deterministic answer markers.

## Running

```bash
make evals
# equivalent: .venv/bin/python -m evals.runner --provider fake
```

This writes ignored `eval-results.json` and prints aggregate tool-selection, argument,
precision, task-success, groundedness, hallucination, safety, and latency metrics.

Each per-case result also includes a `trajectory` containing only observable events:
`user_request`, `model_decision` (tool calls or final-response decision), `tool_call`,
`tool_result`, and `final_response`. It deliberately excludes private reasoning and raw
tool payloads. The scorer reports correct-first-tool, ordering, repeated/unnecessary
call penalties, stopped-when-sufficient, mutation-after-confirmation, and approximate
trajectory efficiency (`necessary_tool_calls / actual_tool_calls`).

Use `--provider configured` to evaluate the configured real provider:

```bash
KAGENT_LLM_PROVIDER=openai KAGENT_LLM_API_KEY=... \
  .venv/bin/python -m evals.runner --provider configured
```

Configured-provider evaluation may make LLM network calls, but it still uses only the
fixture executor: it cannot contact or mutate a Kubernetes cluster. Groundedness is a
deterministic contract score over fixture evidence/claim markers, not an LLM-judge
semantic guarantee.

Tool-result fixtures, not the evaluator prompt, must provide any simulated Kubernetes
state. A passing answer must remain grounded in those supplied results and must never
invent cluster facts.
