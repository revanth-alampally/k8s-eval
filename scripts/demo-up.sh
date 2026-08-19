#!/usr/bin/env bash
# Apply the demo workloads and wait until both the healthy and the broken ones have
# settled, so `make demo-status` shows the interesting state immediately.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh

require_tools kubectl
context="$(require_kind_context)"

echo "applying k8s/ to context '${context}' (namespace: ${DEMO_NAMESPACE})"
kubectl apply -k k8s/
echo

echo "waiting for healthy workloads to roll out..."
for deployment in nginx-good redis; do
  kubectl -n "${DEMO_NAMESPACE}" rollout status "deployment/${deployment}" --timeout=120s
done
echo

# The broken deployments never roll out, so wait on the failure signal instead.
echo "waiting for broken workloads to report failure..."
wait_for_waiting_reason nginx-crash 'CrashLoopBackOff' 120
wait_for_waiting_reason nginx-missing 'ImagePullBackOff|ErrImagePull' 120
echo

exec scripts/demo-status.sh
