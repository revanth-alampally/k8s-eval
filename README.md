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
├── tools/                   tool base class, registry, k8s tools (planned)
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

## Status

Implemented: configuration, structured logging, correlation IDs, error taxonomy,
FastAPI skeleton, health endpoints.

Next: Kubernetes client and read tools → tool registry → agent loop → `/v1/chat` →
confirmation flow for mutations → unit tests and evals.
