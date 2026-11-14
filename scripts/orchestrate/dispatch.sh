#!/usr/bin/env bash
# Launch headless Codex workers, one per task, each in its own git worktree.
#
#   scripts/orchestrate/dispatch.sh <task> [<task> ...]
#
# For each <task>: reads .ai/task-<task>.md, creates branch agent/<task> and worktree
# .worktrees/<task> from the current HEAD, runs `codex exec` there, then commits the result
# on the branch. Logs, exit codes and handoffs land in .ai/.
#
# Env: MAX_PARALLEL (default 3), CODEX_MODEL (optional), CODEX_ARGS (extra flags).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI="$ROOT/.ai"
LOGS="$AI/logs"
WT_DIR="$ROOT/.worktrees"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <task> [<task> ...]" >&2
  exit 2
fi
command -v codex >/dev/null || { echo "error: codex CLI not on PATH" >&2; exit 2; }
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "error: main checkout has uncommitted changes to tracked files; commit them first so workers branch from them" >&2
  exit 2
fi

mkdir -p "$LOGS" "$WT_DIR"
BASE_REF="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
BASE_SHA="$(git -C "$ROOT" rev-parse HEAD)"

for task in "$@"; do
  [[ -f "$AI/task-$task.md" ]] || { echo "error: missing spec $AI/task-$task.md" >&2; exit 2; }
done

run_worker() {
  local task="$1"
  local wt="$WT_DIR/$task"
  local branch="agent/$task"
  local log="$LOGS/$task.log"

  {
    echo "== $task: start $(date '+%F %T') base=$BASE_REF@${BASE_SHA:0:8}"
    if [[ -d "$wt" ]]; then
      echo "reusing existing worktree $wt"
    elif git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$ROOT" worktree add "$wt" "$branch"
    else
      git -C "$ROOT" worktree add -b "$branch" "$wt" "$BASE_SHA"
    fi

    mkdir -p "$wt/.ai"
    cp "$AI/task-$task.md" "$wt/.ai/task-$task.md"
    [[ -f "$AI/contracts.md" ]] && cp "$AI/contracts.md" "$wt/.ai/contracts.md"
    if [[ -d "$ROOT/packages/dashboard/node_modules" && ! -e "$wt/packages/dashboard/node_modules" ]]; then
      ln -s "$ROOT/packages/dashboard/node_modules" "$wt/packages/dashboard/node_modules"
    fi
  } >>"$log" 2>&1

  local prompt
  prompt="You are worker '$task' in a multi-agent build of the Lattice repo. Follow AGENTS.md (especially the Multi-Agent Orchestration Rules).
Your working directory is a git worktree on branch $branch. Your full task spec is below and also at .ai/task-$task.md. Shared contracts, if any, are at .ai/contracts.md.
When done, write .ai/handoff-$task.md as described in AGENTS.md.

$(cat "$AI/task-$task.md")"

  local model_args=()
  [[ -n "${CODEX_MODEL:-}" ]] && model_args=(-m "$CODEX_MODEL")

  # shellcheck disable=SC2086
  codex exec -C "$wt" -s workspace-write --color never \
    -o "$LOGS/$task.last.md" ${model_args[@]+"${model_args[@]}"} ${CODEX_ARGS:-} \
    - <<<"$prompt" >>"$log" 2>&1
  local code=$?

  {
    echo "== $task: codex exited $code at $(date '+%F %T')"
    [[ -f "$wt/.ai/handoff-$task.md" ]] && cp "$wt/.ai/handoff-$task.md" "$AI/handoff-$task.md"
    [[ -L "$wt/packages/dashboard/node_modules" ]] && rm "$wt/packages/dashboard/node_modules"
    git -C "$wt" add -A -- . ':!.ai'
    if git -C "$wt" diff --cached --quiet; then
      echo "no changes to commit"
    else
      git -C "$wt" commit -q --no-verify -m "agent($task): $(head -1 "$AI/task-$task.md" | sed 's/^#* *//')" \
        && echo "committed $(git -C "$wt" rev-parse --short HEAD) on $branch"
    fi
  } >>"$log" 2>&1

  echo "$code" >"$LOGS/$task.exit"
}

echo "dispatching $# task(s) from $BASE_REF@${BASE_SHA:0:8} (max $MAX_PARALLEL parallel)"
for task in "$@"; do
  rm -f "$LOGS/$task.exit"
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do sleep 5; done
  run_worker "$task" &
  echo "$!" >"$LOGS/$task.pid"
  echo "  launched $task (pid $!) -> .ai/logs/$task.log"
done

wait
echo "all workers finished:"
for task in "$@"; do
  echo "  $task: exit $(cat "$LOGS/$task.exit" 2>/dev/null || echo '?')"
done
