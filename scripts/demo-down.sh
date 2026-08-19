#!/usr/bin/env bash
# Remove every demo workload. Deleting the namespace removes the contents too, but
# deleting through kustomize keeps this honest if the namespace is ever managed
# elsewhere.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh

require_tools kubectl
context="$(require_kind_context)"

echo "deleting demo workloads from context '${context}' (namespace: ${DEMO_NAMESPACE})"
kubectl delete -k k8s/ --ignore-not-found=true --wait=true
echo "done"
