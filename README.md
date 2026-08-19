# AI Kubernetes Operations Agent

A natural-language operations assistant for a local [kind](https://kind.sigs.k8s.io/)
cluster. You ask *"are any pods unhealthy?"* or *"restart the nginx deployment"*; the
agent picks a tool, the tool talks to the Kubernetes API, and the agent summarises what
the tool actually returned.

## The one rule that shapes the design

**The LLM never invents cluster state.** It has two jobs: choose a tool, and phrase the
result. Every fact in an answer traces back to a tool result, and the response includes
the tool trace so you can check. Anything the model wants to know that no tool returned,
it must say it does not know.

Everything else follows from that:

| Principle | How it is enforced |
| --- | --- |
| State comes only from tools | The LLM sees tool output; it is never asked to recall cluster facts |
| Reads are automatic | Read tools are marked non-mutating and execute inside the agent loop |
| Mutations need confirmation | Mutating tools return a confirmation token instead of executing |
| Arguments are validated | Each tool has a Pydantic argument model; invalid args fail closed |
| Errors are structured | One error envelope with a stable `code`, at every layer |
| Requests are traceable | A correlation ID is bound to every log line and returned in the response |
| Behaviour is observable | Every tool call and HTTP request logs `operation`, `outcome`, `duration_ms` |

## Request flow

```
User
 └─ POST /v1/chat                    FastAPI: validate, assign correlation ID
     └─ Agent orchestrator           bounded loop, max_tool_calls_per_request
         ├─ LLM planner              chooses a tool + arguments (no cluster facts)
         ├─ Tool registry            resolves name, validates args, checks namespace
         │   └─ Kubernetes tools     deterministic calls to the Kubernetes API
         └─ LLM summariser           phrases the answer from tool results only
```

Mutating requests short-circuit: the orchestrator returns `confirmation_required` with a
token and a plain-English description of the change. The client re-submits the token to
`POST /v1/confirmations/{token}` to actually apply it.

## Layout

```
app/
├── main.py                  application factory: middleware, handlers, routers
├── config.py                Settings (pydantic-settings, KAGENT_ prefix)
├── context.py               correlation ID contextvar
├── errors.py                error taxonomy + FastAPI exception handlers
├── middleware.py            correlation ID + request timing
├── api/
│   ├── deps.py              shared dependencies (settings injection)
│   └── routes/
│       ├── health.py        liveness + readiness
│       ├── chat.py          POST /v1/chat                      (planned)
│       └── confirmations.py POST /v1/confirmations/{token}     (planned)
├── agent/                   planner, orchestrator, prompts     (planned)
├── tools/
│   ├── base.py              ToolSpec, namespace and mutation guards
│   ├── schemas.py           argument models + Kubernetes name validation
│   ├── registry.py          the complete list of what the agent can do
│   └── k8s/
│       ├── client.py        client construction + error translation
│       ├── models.py        typed, secret-free projections of API objects
│       ├── convert.py       API object -> model mapping, incl. the health rule
│       ├── read.py          list_pods, get_pod, describe_pod, get_pod_logs,
│       │                    list_deployments
│       ├── diagnose.py      diagnose_pod: evidence collection, no interpretation
│       └── mutate.py        restart_deployment (mutating)
└── observability/
    ├── logging.py           structlog setup, JSON or console
    └── instrumentation.py   track_operation: one timing primitive for all layers
tests/
├── unit/                    fast, no cluster, no LLM
├── integration/             against a live kind cluster       (planned)
└── evals/                   LLM behaviour: tool choice, no-hallucination (planned)

k8s/                         demo workloads, namespace ai-agent-demo
├── 00-namespace.yaml
├── nginx-good.yaml          healthy, 2 replicas, + Service
├── nginx-crash.yaml         CrashLoopBackOff via an unparseable nginx.conf
├── nginx-missing.yaml       ImagePullBackOff via a nonexistent image tag
├── redis.yaml               healthy, exec readiness probe, + Service
└── kustomization.yaml
scripts/                     demo-up.sh, demo-down.sh, demo-status.sh
```

### Why the layers are split this way

- **`tools/` has no LLM and `agent/` has no Kubernetes client.** The tool layer is
  ordinary Python you can unit test with a fake API client; the agent layer can be
  tested with a fake tool registry. Neither test needs both halves.
- **`config.py` owns the safety switches** (`read_only_mode`, `allowed_namespaces`,
  `require_confirmation`) so guardrails are one grep away, not scattered through
  handlers.
- **`errors.py` owns the taxonomy**, so a tool failure and a bad request produce the
  same envelope and the agent can reason about failures instead of parsing strings.
- **`observability/` exposes a single `track_operation`**, so "what is slow" and "what
  fails" are the same log query at every layer.

## Getting started

```bash
make install          # create .venv and install the project
cp .env.example .env  # then set KAGENT_LLM_API_KEY
make run              # http://127.0.0.1:8000/docs
```

Check it is alive:

```bash
curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/health/ready | python -m json.tool
```

`/health` is deliberately independent of the cluster and the LLM: a Kubernetes outage
should not make an orchestrator restart a healthy API process. Dependency probes belong
in `/health/ready`.

### Demo cluster

The manifests in `k8s/` create a namespace, `ai-agent-demo`, containing two healthy and
two broken workloads so there is always something real to diagnose.

```bash
make cluster-up   # only if you do not already have a kind cluster
make demo-up      # apply k8s/ and wait for both healthy and broken states
make demo-status  # what is healthy, what is not, and the evidence
make demo-down
```

| Workload | Expected state | Where the evidence lives |
| --- | --- | --- |
| `nginx-good` (2 replicas) | `Running` `2/2` | — |
| `redis` | `Running` `1/1` | — |
| `nginx-crash` | `CrashLoopBackOff` | container **logs** (`nginx: [emerg] unknown directive`) |
| `nginx-missing` | `ImagePullBackOff` | **events** only; no logs exist |

The last two fail in deliberately different ways. `nginx-crash` is a genuine nginx
container handed an unparseable config, so the reason is only in the logs of a container
that is no longer running — the agent has to ask for *previous* logs. `nginx-missing`
never started a container at all, so logs are empty and the answer is in the pod's
container statuses and Warning events. An agent that always reaches for logs will get
one of these right and the other wrong, which makes them a useful eval pair.

`make demo-up` refuses to run against a context that is not `kind-*`, since it creates
intentionally broken workloads. Override with `ALLOW_NON_KIND_CONTEXT=1` if needed.

## Configuration

All settings are environment variables prefixed with `KAGENT_` (see `.env.example`).
The ones that matter for safety:

- `KAGENT_ALLOWED_NAMESPACES` — hard allowlist; tools refuse anything outside it.
- `KAGENT_READ_ONLY_MODE` — kill switch; mutating tools are not registered at all.
- `KAGENT_REQUIRE_CONFIRMATION` — gate mutations behind an explicit token.
- `KAGENT_MAX_TOOL_CALLS_PER_REQUEST` — bounds the agent loop.

## Development

```bash
make test    # pytest
make lint    # ruff + mypy
make fmt     # ruff format
```

## The tools

Six tools, five read-only. This list is the complete answer to "what can the agent do
to my cluster?", and it lives in one file, `app/tools/registry.py`.

| Tool | Mutating | Notes |
| --- | --- | --- |
| `list_pods(namespace)` | no | Computes a health verdict per pod, plus a list of unhealthy names |
| `get_pod(namespace, pod_name)` | no | Per-container state, images, restart counts |
| `describe_pod(namespace, pod_name)` | no | Pod **plus recent events** — the only evidence when a container never started |
| `diagnose_pod(namespace, pod_name)` | no | All evidence about one pod in a single call, as structured signals |
| `get_pod_logs(namespace, pod_name, container, tail_lines)` | no | `previous=True` reads a crashed container's log |
| `list_deployments(namespace)` | no | Desired vs available replicas |
| `restart_deployment(namespace, deployment_name)` | **yes** | Rolling restart via the standard `restartedAt` annotation |

There is deliberately **no** tool that accepts a command string, and a test enforces
this: `test_no_tool_accepts_a_free_form_command` fails if any tool ever grows a
`command`, `script`, `query` or similar argument.

### `diagnose_pod`: evidence, not explanation

`diagnose_pod` runs the queries a human would run by hand — pod state, unmet conditions,
restart counts, warning events, logs — and returns them as a flat list of signals:

```json
{
  "pod": "nginx-missing-7575f48ccf-rftkj",
  "status": "ImagePullBackOff",
  "healthy": false,
  "logs_available": false,
  "signals": [
    {
      "source": "container_state",
      "reason": "ImagePullBackOff",
      "evidence": "Container 'nginx' is waiting: ImagePullBackOff. Back-off pulling image ...",
      "container": "nginx",
      "severity": "warning"
    }
  ]
}
```

It contains **no cause, no recommendation and no ranking of likelihood**. Every
`evidence` string is quoted from the cluster and every `reason` is a cluster-issued
label; `source` records where each fact came from, so the cluster's own account (an
event) stays distinguishable from the workload's (a log line). `severity` is a
mechanical mapping — a Warning event is `warning` — not a judgement.

That boundary is the whole point. If this function guessed at causes, the guess would be
indistinguishable from a fact by the time it reached the model, and a wrong guess would
be laundered into a confident answer. Tools establish facts; the LLM reasons over them,
and only over what is recorded here. `test_diagnosis_contains_no_interpretation_fields`
fails if a `cause`, `explanation` or `recommendation` field ever appears.

Two absences are themselves evidence. `logs_available: false` means no container ever
started, which is what separates an image-pull failure from a crash. And a pod that does
not exist raises `resource_not_found` rather than returning an empty signal list, since
empty evidence would invite an answer about a pod that was never there.

### Guarantees this layer provides

- **Arguments are validated before anything is called.** Every name is checked against
  the Kubernetes naming rules, so `nginx; rm -rf /` fails at the schema, not the API.
  `extra="forbid"` means a hallucinated argument is an error rather than being ignored.
- **Namespaces are allowlisted.** Checked inside every tool, not once at the edge.
- **Mutation is a static property.** `restart_deployment` is flagged `mutating=True`, so
  the confirmation gate can be applied before execution rather than inferred from a
  string. In `read_only_mode` it is not registered at all, *and* it re-checks the flag
  itself.
- **Failures are typed.** `resource_not_found`, `permission_denied`, `cluster_timeout`,
  `cluster_unavailable`, `tool_argument_invalid`, `logs_unavailable`,
  `namespace_not_allowed`. No `ApiException` escapes the package. The agent can tell
  "the pod does not exist" from "the API did not answer" — the second is a case where it
  must not answer at all.
- **Secrets stay out.** Output models exclude environment variables, volumes and
  annotations, and error details carry only the parsed API `message`, never response
  headers.
- **Everything is timed and logged.** Each call emits `operation`, `outcome` and
  `duration_ms`. Log *content* is never logged — only its line count.

## Status

Implemented: configuration, structured logging, correlation IDs, error taxonomy,
FastAPI skeleton, health endpoints (readiness probes the real cluster), and the full
Kubernetes tool layer including `diagnose_pod`, with 81 unit tests.

Next: agent loop → `/v1/chat` → confirmation flow for mutations → AI evals.
