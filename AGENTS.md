# Lattice — Agent Instructions

Lattice is a multi-agent orchestration platform (LangGraph) with knowledge-graph-augmented RAG,
guardrails, and write-back. Monorepo layout:

- `packages/core/lattice/` — core library (agents, graph, ingestion, retrieval, guardrails, writeback, observability)
- `packages/api/lattice_api/` — FastAPI backend
- `packages/mcp/lattice_mcp/` — MCP server
- `packages/dashboard/` — Next.js 16 dashboard (read `packages/dashboard/AGENTS.md` before touching it)
- `tests/` — pytest suite, mirrors the package layout
- `deploy/`, `docker/`, `docker-compose.yml` — infrastructure

Current project state, open work, and which agent (Cursor, Codex, Claude Code, Grok) owns what
live in `HANDOFF.md`. Read it before starting anything that is not a dispatched task.

## Running tests

Always use the wrapper; it works from the main checkout and from any `.worktrees/*` checkout:

```bash
scripts/test.sh                      # full suite
scripts/test.sh tests/graph -q       # any pytest args
```

Lint: `scripts/test.sh --lint` runs ruff on files that are new since HEAD; they must be clean.
Existing files carry pre-existing lint debt: don't add new errors to the lines you write, and don't
reformat or "fix" untouched code (it causes merge conflicts between parallel workers).

## Conventions

- Python 3.11, Pydantic v2 models, `structlog` for logging, async agent code.
- New backends implement the existing abstract interfaces (e.g. `KnowledgeGraphBackend`, `WriteBackHandler`, `LLMProvider`) rather than adding parallel APIs.
- Optional heavy dependencies are imported lazily inside functions and degrade gracefully when missing.
- Every new module gets tests under the matching `tests/` subfolder. Tests must not need network access or running services; mock external clients.
- Do not add new third-party dependencies unless the task spec explicitly allows it (there is no network access in worker sandboxes).

# Multi-Agent Orchestration Rules

These apply when you are running as a worker (Codex or Claude Code) inside `.worktrees/<task>` on
an `agent/<task>` branch, and to cloud agents working on a `grok/<task>` branch.

- **Scope Isolation:** Only modify files explicitly listed in your assigned task spec. Never edit shared config files (`pyproject.toml`, `docker-compose.yml`, `.gitignore`, `AGENTS.md`, `packages/dashboard/package.json`) unless the spec instructs you to.
- **Contracts are fixed:** If the spec references a shared interface or contract file, implement against it exactly. If it seems wrong, note it in the handoff instead of changing it.
- **Verification:** Before finishing, run `scripts/test.sh` (full suite) and `scripts/test.sh --lint`. Fix every failure you introduced before exiting.
- **No git operations:** Do not commit, branch, rebase, or push. The orchestrator commits your work after you exit.
- **Handoff Report:** When complete, write `.ai/handoff-<task-name>.md` (inside your working directory) containing: files changed, what was implemented, test results (pass/fail counts), edge cases, and anything left undone or deviating from the spec.
