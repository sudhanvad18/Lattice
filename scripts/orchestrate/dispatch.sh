#!/usr/bin/env bash
# Launch headless coding workers, one per task, each in its own git worktree.
#
#   scripts/orchestrate/dispatch.sh <task>[:codex|:claude] [...]
#
# For each <task>: reads .ai/task-<task>.md, creates branch agent/<task> and worktree
# .worktrees/<task> from the current HEAD, runs the worker there (Codex by default, Claude Code
# with a ":claude" suffix or WORKER=claude), then commits the result on the branch.
# Logs, exit codes and handoffs land in .ai/.
#
# Env: MAX_PARALLEL (default 3), WORKER (default codex), CODEX_MODEL, CODEX_ARGS,
#      CLAUDE_MODEL, CLAUDE_ARGS.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI="$ROOT/.ai"
LOGS="$AI/logs"
WT_DIR="$ROOT/.worktrees"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <task>[:codex|:claude] [...]" >&2
  exit 2
fi

TASKS=()
WORKERS=()
for arg in "$@"; do
  task="${arg%%:*}"
  worker="${WORKER:-codex}"
  [[ "$arg" == *:* ]] && worker="${arg#*:}"
  case "$worker" in
    codex|claude) ;;
    *) echo "error: unknown worker '$worker' for $task (use codex or claude)" >&2; exit 2 ;;
  esac
  command -v "$worker" >/dev/null || { echo "error: $worker CLI not on PATH" >&2; exit 2; }
  TASKS+=("$task")
  WORKERS+=("$worker")
done
set -- "${TASKS[@]}"
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

run_codex() {
  local task="$1" wt="$2" prompt="$3" log="$4"
  local model_args=()
  [[ -n "${CODEX_MODEL:-}" ]] && model_args=(-m "$CODEX_MODEL")
  # shellcheck disable=SC2086
  codex exec -C "$wt" -s workspace-write --color never \
    -o "$LOGS/$task.last.md" ${model_args[@]+"${model_args[@]}"} ${CODEX_ARGS:-} \
    - <<<"$prompt" >>"$log" 2>&1
}

# Claude Code has no OS sandbox, so edits are confined to the worktree by permission mode and
# shell access is an allowlist: tests, linters, and read-only inspection. Git stays denied.
run_claude() {
  local task="$1" wt="$2" prompt="$3" log="$4"
  local model_args=()
  [[ -n "${CLAUDE_MODEL:-}" ]] && model_args=(--model "$CLAUDE_MODEL")
  # shellcheck disable=SC2086
  (cd "$wt" && claude -p --output-format text --permission-mode acceptEdits \
    --no-session-persistence \
    --allowedTools "Read Edit Write Glob Grep Bash(scripts/test.sh:*) Bash(ls:*) Bash(cat:*) Bash(rg:*) Bash(grep:*) Bash(find:*) Bash(head:*) Bash(tail:*) Bash(wc:*) Bash(npx tsc:*) Bash(npx eslint:*) Bash(npm run lint:*)" \
    --disallowedTools "Bash(git:*) Bash(curl:*) Bash(wget:*) Bash(pip:*) Bash(npm install:*) WebFetch WebSearch" \
    ${model_args[@]+"${model_args[@]}"} ${CLAUDE_ARGS:-} \
    <<<"$prompt" >"$LOGS/$task.last.md" 2>>"$log")
  local code=$?
  cat "$LOGS/$task.last.md" >>"$log"
  return $code
}

run_worker() {
  local task="$1"
  local worker="$2"
  local wt="$WT_DIR/$task"
  local branch="agent/$task"
  local log="$LOGS/$task.log"

  {
    echo "== $task: start $(date '+%F %T') base=$BASE_REF@${BASE_SHA:0:8} worker=$worker"
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

  "run_$worker" "$task" "$wt" "$prompt" "$log"
  local code=$?

  {
    echo "== $task: $worker exited $code at $(date '+%F %T')"
    [[ -f "$wt/.ai/handoff-$task.md" ]] && cp "$wt/.ai/handoff-$task.md" "$AI/handoff-$task.md"
    [[ -L "$wt/packages/dashboard/node_modules" ]] && rm "$wt/packages/dashboard/node_modules"
    git -C "$wt" add -A
    if git -C "$wt" diff --cached --quiet; then
      echo "no changes to commit"
    else
      git -C "$wt" commit -q --no-verify -m "agent($task/$worker): $(head -1 "$AI/task-$task.md" | sed 's/^#* *//')" \
        && echo "committed $(git -C "$wt" rev-parse --short HEAD) on $branch"
    fi
  } >>"$log" 2>&1

  echo "$code" >"$LOGS/$task.exit"
}

echo "dispatching $# task(s) from $BASE_REF@${BASE_SHA:0:8} (max $MAX_PARALLEL parallel)"
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  rm -f "$LOGS/$task.exit"
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do sleep 5; done
  run_worker "$task" "${WORKERS[$i]}" &
  echo "$!" >"$LOGS/$task.pid"
  echo "  launched $task with ${WORKERS[$i]} (pid $!) -> .ai/logs/$task.log"
done

wait
echo "all workers finished:"
for task in "$@"; do
  echo "  $task: exit $(cat "$LOGS/$task.exit" 2>/dev/null || echo '?')"
done
