#!/usr/bin/env bash
# Full verification: lint, formatting, type check, fast tests.
#
# Used by `make check` and by the pre-commit hook. Takes an optional directory to
# verify, which the hook uses to check the staged snapshot instead of the working tree.
# Output is suppressed unless a step fails, so a passing run stays quiet.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
target="${1:-${repo_root}}"

# Resolved against the real repo, not the target: a staged snapshot has no .venv.
PYTHON_BIN="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: no Python interpreter at ${PYTHON_BIN}" >&2
  echo "       run 'make install' first" >&2
  exit 1
fi

cd "${target}" || exit 1

failed=()

run() {
  local label="$1"
  shift
  local started=${SECONDS}
  local output
  printf '  %-14s' "${label}"
  if output="$("$@" 2>&1)"; then
    printf 'ok (%ss)\n' "$((SECONDS - started))"
  else
    printf 'FAILED\n'
    printf '%s\n' "${output}" | sed 's/^/    | /'
    failed+=("${label}")
  fi
}

run "ruff check" "${PYTHON_BIN}" -m ruff check app tests
run "ruff format" "${PYTHON_BIN}" -m ruff format --check app tests
run "mypy" "${PYTHON_BIN}" -m mypy app
# Markers excluded so that future tests needing a live cluster or a real model never
# become a prerequisite for committing.
run "pytest" "${PYTHON_BIN}" -m pytest -q -m "not cluster and not eval"

if ((${#failed[@]} > 0)); then
  echo
  echo "checks failed: ${failed[*]}" >&2
  exit 1
fi
