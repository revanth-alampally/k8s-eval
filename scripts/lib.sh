#!/usr/bin/env bash
# Shared helpers for the demo scripts. Sourced, not executed.

DEMO_NAMESPACE="${DEMO_NAMESPACE:-ai-agent-demo}"

require_tools() {
  local tool
  for tool in "$@"; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "error: '${tool}' not found on PATH" >&2
      exit 1
    fi
  done
}

# These manifests deliberately create broken workloads, so guard against them landing
# in a cluster that matters. Override with ALLOW_NON_KIND_CONTEXT=1 if you know better.
require_kind_context() {
  local context
  context="$(kubectl config current-context 2>/dev/null || true)"

  if [[ -z "${context}" ]]; then
    echo "error: no current kubectl context; is your cluster running?" >&2
    exit 1
  fi

  if [[ "${context}" != kind-* && "${ALLOW_NON_KIND_CONTEXT:-0}" != "1" ]]; then
    echo "error: refusing to act on context '${context}': not a kind cluster." >&2
    echo "       These manifests create intentionally broken workloads." >&2
    echo "       Set ALLOW_NON_KIND_CONTEXT=1 to override." >&2
    exit 1
  fi

  printf '%s' "${context}"
}

# Poll until a deployment's pods report an expected container "waiting" reason, e.g.
# CrashLoopBackOff. Used for workloads that are supposed to fail, where `rollout status`
# would simply hang until it timed out.
#   wait_for_waiting_reason <app-label> <reason-regex> [timeout-seconds]
wait_for_waiting_reason() {
  local app="$1" pattern="$2" timeout="${3:-120}"
  local deadline=$((SECONDS + timeout))
  local reason=""

  while ((SECONDS < deadline)); do
    reason="$(kubectl -n "${DEMO_NAMESPACE}" get pods -l "app=${app}" \
      -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' \
      2>/dev/null || true)"

    if [[ "${reason}" =~ ${pattern} ]]; then
      echo "  ${app}: ${reason}"
      return 0
    fi
    sleep 3
  done

  # Not fatal: the demo is still usable, the pod just has not backed off yet.
  echo "  ${app}: expected /${pattern}/ within ${timeout}s, currently '${reason:-<none>}'"
}
