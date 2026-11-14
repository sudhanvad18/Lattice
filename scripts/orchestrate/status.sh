#!/usr/bin/env bash
# Show the state of every dispatched worker: running/exit code, commits ahead, handoff, log tail.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="$ROOT/.ai/logs"
[[ -d "$LOGS" ]] || { echo "no workers dispatched yet"; exit 0; }

shopt -s nullglob
for pidfile in "$LOGS"/*.pid; do
  task="$(basename "$pidfile" .pid)"
  if [[ -f "$LOGS/$task.exit" ]]; then
    state="exited $(cat "$LOGS/$task.exit")"
  elif kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    state="running"
  else
    state="dead (no exit file)"
  fi
  ahead="$(git -C "$ROOT" rev-list --count "HEAD..agent/$task" 2>/dev/null || echo '-')"
  handoff="no"; [[ -f "$ROOT/.ai/handoff-$task.md" ]] && handoff="yes"
  echo "== $task: $state | commits ahead of HEAD: $ahead | handoff: $handoff"
  if git -C "$ROOT" show-ref --verify --quiet "refs/heads/agent/$task"; then
    git -C "$ROOT" diff --stat "HEAD...agent/$task" | tail -1 | sed 's/^/   /'
  fi
  tail -n "${TAIL:-3}" "$LOGS/$task.log" 2>/dev/null | sed 's/^/   | /'
done
