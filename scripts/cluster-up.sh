#!/usr/bin/env bash
# Optional convenience: create a kind cluster if you do not already have one.
# Workloads are NOT seeded here -- that is `make demo-up`, which applies k8s/.
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-k8s-ops-agent}"

if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
  echo "kind cluster '${CLUSTER_NAME}' already exists"
else
  kind create cluster --name "${CLUSTER_NAME}"
fi

kubectl config use-context "kind-${CLUSTER_NAME}"
kubectl get nodes

echo
echo "next: make demo-up"
