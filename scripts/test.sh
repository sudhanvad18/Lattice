#!/usr/bin/env bash
# Run the test suite (or ruff with --lint) against THIS checkout's sources,
# reusing the main checkout's virtualenv. Works from the repo root and from git worktrees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$(git -C "$ROOT" rev-parse --git-common-dir)/.." && pwd)"
PY="${LATTICE_PY:-$MAIN_ROOT/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "error: python not found at $PY (set LATTICE_PY)" >&2
  exit 2
fi

# The venv has an editable install pointing at the main checkout; put this checkout first.
export PYTHONPATH="$ROOT/packages/core:$ROOT/packages/api:$ROOT/packages/mcp${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

if [[ "${1:-}" == "--lint" ]]; then
  shift
  if ! "$PY" -m ruff --version >/dev/null 2>&1; then
    echo "ruff not installed in venv; skipping lint"
    exit 0
  fi
  if [[ $# -gt 0 ]]; then
    files=("$@")
  else
    files=()
    while IFS= read -r f; do files+=("$f"); done < <(
      { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | grep '\.py$' | sort -u
    )
  fi
  existing=()
  for f in ${files[@]+"${files[@]}"}; do [[ -f "$f" ]] && existing+=("$f"); done
  if [[ ${#existing[@]} -eq 0 ]]; then
    echo "no changed python files to lint"
    exit 0
  fi
  exec "$PY" -m ruff check "${existing[@]}"
fi

exec "$PY" -m pytest -o addopts= -p no:cacheprovider "${@:-tests}"
