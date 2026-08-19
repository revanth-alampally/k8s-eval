#!/usr/bin/env bash
# Human-readable snapshot of the demo namespace: what is healthy, what is not, and the
# evidence for the failures. This is roughly the ground truth an agent answer should be
# checked against.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh

require_tools kubectl

if ! kubectl get namespace "${DEMO_NAMESPACE}" >/dev/null 2>&1; then
  echo "namespace '${DEMO_NAMESPACE}' not found - run 'make demo-up' first" >&2
  exit 1
fi

echo "=== context: $(kubectl config current-context) / namespace: ${DEMO_NAMESPACE} ==="
echo
echo "--- deployments ---"
kubectl -n "${DEMO_NAMESPACE}" get deployments

echo
echo "--- pods ---"
kubectl -n "${DEMO_NAMESPACE}" get pods -o wide

echo
echo "--- unhealthy pods ---"
unhealthy="$(kubectl -n "${DEMO_NAMESPACE}" get pods --no-headers 2>/dev/null |
  awk '{split($2, ready, "/"); if (ready[1] != ready[2] || $3 != "Running") print "  " $1 "  " $3 "  restarts=" $4}')"

if [[ -z "${unhealthy}" ]]; then
  echo "  (none - expected nginx-crash and nginx-missing to be failing)"
else
  echo "${unhealthy}"
fi

echo
echo "--- why nginx-crash is failing (container logs) ---"
kubectl -n "${DEMO_NAMESPACE}" logs -l app=nginx-crash --tail=5 --previous 2>/dev/null ||
  kubectl -n "${DEMO_NAMESPACE}" logs -l app=nginx-crash --tail=5 2>/dev/null ||
  echo "  (no logs yet)"

echo
echo "--- why nginx-missing is failing (events, it has no logs) ---"
kubectl -n "${DEMO_NAMESPACE}" get events --field-selector type=Warning \
  --sort-by=.lastTimestamp -o custom-columns='REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message' \
  2>/dev/null | grep -i -E 'nginx-missing|pull' | tail -5 ||
  echo "  (no warning events yet)"
