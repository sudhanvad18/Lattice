#!/usr/bin/env bash
# Remove worker worktrees. Branches are kept unless --delete-branches is passed
# (only branches already merged into HEAD are deleted).
#
#   scripts/orchestrate/cleanup.sh [--delete-branches] [<task> ...]   # default: all
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELETE_BRANCHES=0
if [[ "${1:-}" == "--delete-branches" ]]; then DELETE_BRANCHES=1; shift; fi

tasks=("$@")
if [[ ${#tasks[@]} -eq 0 ]]; then
  shopt -s nullglob
  for d in "$ROOT/.worktrees"/*/; do tasks+=("$(basename "$d")"); done
fi

for task in ${tasks[@]+"${tasks[@]}"}; do
  wt="$ROOT/.worktrees/$task"
  if [[ -d "$wt" ]]; then
    git -C "$ROOT" worktree remove --force "$wt" && echo "removed worktree $task"
  fi
  if [[ $DELETE_BRANCHES -eq 1 ]]; then
    git -C "$ROOT" branch -d "agent/$task" 2>&1 | sed 's/^/  /'
  fi
  rm -f "$ROOT/.ai/logs/$task.pid"
done
git -C "$ROOT" worktree prune
