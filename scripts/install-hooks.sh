#!/usr/bin/env bash
# Activate the versioned hooks in .githooks/ for this clone.
#
# Symlinks into .git/hooks rather than setting core.hooksPath, so the repository's git
# configuration is left untouched and the hook still tracks the committed version of the
# script. Git hooks are per-clone by design, so this is run once after cloning.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
hooks_dir="$(git rev-parse --git-path hooks)"
mkdir -p "${hooks_dir}"

for hook in .githooks/*; do
  name="$(basename "${hook}")"
  chmod +x "${hook}"

  # Relative link so the repository stays portable if the checkout is moved.
  ln -sf "../../.githooks/${name}" "${hooks_dir}/${name}"
  if [[ ! -x "${hooks_dir}/${name}" ]]; then
    # Non-standard layout (worktree, separate git dir): fall back to an absolute link.
    ln -sf "$(pwd)/.githooks/${name}" "${hooks_dir}/${name}"
  fi

  echo "installed ${name} -> $(readlink "${hooks_dir}/${name}")"
done

echo
echo "Checks run on every 'git commit'. Bypass once with 'git commit --no-verify'."
