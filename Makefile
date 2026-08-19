.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python

DEMO_NAMESPACE ?= ai-agent-demo
export DEMO_NAMESPACE

.PHONY: help install hooks-install run test lint fmt check demo-up demo-down demo-status cluster-up cluster-down clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv, install the project, and activate the git hooks
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e '.[dev]'
	./scripts/install-hooks.sh

hooks-install: ## Activate the pre-commit hook for this clone
	./scripts/install-hooks.sh

run: ## Run the API with autoreload
	$(PY) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Lint and type-check
	$(PY) -m ruff check app tests
	$(PY) -m mypy app

fmt: ## Format and apply safe lint fixes
	$(PY) -m ruff format app tests
	$(PY) -m ruff check --fix app tests

check: ## Run everything the pre-commit hook runs, against the working tree
	./scripts/run-checks.sh

demo-up: ## Apply the demo workloads (healthy + intentionally broken)
	./scripts/demo-up.sh

demo-down: ## Remove the demo workloads and namespace
	./scripts/demo-down.sh

demo-status: ## Show what is healthy and unhealthy in the demo namespace
	./scripts/demo-status.sh

cluster-up: ## Create a kind cluster (only if you do not already have one)
	./scripts/cluster-up.sh

cluster-down: ## Delete the kind cluster created by cluster-up
	kind delete cluster --name $(or $(CLUSTER_NAME),k8s-ops-agent)

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
