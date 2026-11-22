# Lattice handoff

Living status note shared by every agent working on this repo (Cursor, Codex, Claude Code,
Grok). Read this first, then `AGENTS.md` for the working rules. Whoever changes the state of the
project updates the relevant section here in the same PR.

Last updated: 2026-11-21 by Cursor (local orchestrator).

## What Lattice is

A multi-agent platform that turns a document and code corpus into a knowledge base and runs an
agent team over it. Core pieces, all under `packages/`:

- `core/lattice/ingestion`: parsers (markdown, PDF, code via tree-sitter, images via vision LLM),
  chunker, SHA-256 ingestion cache.
- `core/lattice/graph`: knowledge graph. Rule-based and AST extractors, a builder that merges
  repeat extractions, NetworkX backend (JSON snapshot persistence) and Neo4j backend, selected
  by `create_graph_backend()` from `LATTICE_GRAPH_BACKEND`. Community clustering, hub ("god")
  nodes, and surprising cross-community links.
- `core/lattice/retrieval`: ChromaDB vector store, hybrid retrieval, cross-encoder reranking.
- `core/lattice/agents`: LangGraph orchestrator with planner, researcher, writer, reviewer, and
  code agents; review/revise loop capped by `max_iterations`.
- `core/lattice/guardrails`: three-tier checks on artifacts (grounding against the KG, etc.).
- `core/lattice/evaluation`: benchmark engine and built-in tasks; results saved by `lattice eval`.
- `core/lattice/cli.py`: `lattice ingest | stats | search | graph | run | eval | serve | reset`.
- `api/lattice_api`: FastAPI app (`/tasks`, `/kg/*`, `/eval/results`, `/health`, websocket).
- `dashboard`: Next.js 16 + Tailwind v4 UI. Task submission, history, knowledge graph (3D/2D
  `react-force-graph` view with community colours, hub sizing, focus mode, insights panel), and
  evaluations.
- `mcp`: MCP server exposing Lattice tools.

Quick start: `lattice ingest examples/aerospace && lattice serve`, then `npm run dev` in
`packages/dashboard`. Tests: `scripts/test.sh` (360 passed, 1 skipped as of this note; the
skipped test needs a live Neo4j via `LATTICE_TEST_NEO4J_URI/USER/PASSWORD`).

## State of the repo

- Branch `main` only. The owner pushes to GitHub by hand; local `main` may be ahead of
  `origin/main`. If GitHub looks behind this note, ask the owner to push rather than redoing work.
- Recent work, newest first: eval endpoint and `lattice eval`; fix for an unbounded
  writer/reviewer revision loop; 3D/2D graph view; error boundary per dashboard tab; KG builder
  dedup across re-ingests; AST extractor wired into ingestion; Neo4j backend; graph API; CLI;
  Graphify-inspired features (tree-sitter, provenance labels, cache, clustering, image parsing).

## Who does what

| Agent | Where it runs | Use it for |
|---|---|---|
| Cursor (local) | Owner's machine | Orchestration of multi-file features: writes contracts and task specs, dispatches workers, merges, runs integration tests, inspects the UI. |
| Codex CLI workers | Local git worktrees, sandboxed, no network | Well-specified implementation tasks from `.ai/task-<name>.md`. Default worker. |
| Claude Code workers | Local git worktrees, tool allowlist, no git/network | Same contract as Codex; better for larger refactors, docs, and tasks needing long reasoning. Dispatch with `task:claude`. |
| Cursor cloud agents | Remote, own branch, open a PR | Self-contained tasks that need no local services: CI, docs, lint cleanup, tests. |
| Grok | xAI, reads GitHub and the local checkout | Coordinator and scout: backlog upkeep, PR review, CI triage, drift checks, daily summary, and the technology radar in `docs/agents/radar.md`. |

Rules that keep agents from colliding:

1. One owner per branch. Local workers use `agent/<task>`, cloud agents started by Grok use
   `grok/<task>`. Check the "In flight" table below and open PRs before starting work.
2. Nobody pushes to or rewrites `main` except the owner. Everything else lands by PR or by the
   local orchestrator's merge.
3. Shared files (`pyproject.toml`, `docker-compose.yml`, `packages/dashboard/package.json`,
   `AGENTS.md`, this file's "Who does what") change only in a PR that says so in its title.
4. Workers follow `AGENTS.md`: stay inside the files their spec owns, run `scripts/test.sh` and
   `scripts/test.sh --lint` before finishing, and write a handoff report.
5. Local dispatch is `scripts/orchestrate/dispatch.sh name1 name2:claude`, status with
   `status.sh`, cleanup with `cleanup.sh --delete-branches`. Workers cannot run git; the
   dispatcher commits their output.

## In flight

| Work | Owner | Branch / PR | Notes |
|---|---|---|---|
| (none) | | | |

## Open work, highest value first

Bugs and gaps:

1. `POST /chat` is called by the dashboard Chat tab but does not exist in the API.
2. The activity feed opens `/ws/activity`; the API only serves `/ws/{task_id}`, so "activity" is
   treated as a task id.
3. The orchestrator marks a task COMPLETED when it hits `max_iterations` with the latest review
   still rejected. Should be FAILED (or a distinct status); the eval "Completion" metric is
   inflated by this.
4. Entity resolution only merges exact (type, case-folded name) matches. Near duplicates such as
   "AMS-4928" and "AMS 4928" stay separate.
5. Neo4j backend is tested against a fake driver only; needs a run against a live instance
   (`docker compose up neo4j`). Its search matches entity names only.
6. `graspologic` is not installed, so Leiden clustering falls back to connected components.
7. `EvaluationEngine` builds its researcher without the knowledge base, so benchmarks do not
   exercise retrieval.

Product polish:

8. README is stale: claims 219 tests and 86% coverage, documents a non-existent
   `lattice init --template`. Rewrite around the CLI quick start and the dashboard.
9. No CI. Add a GitHub Actions workflow running `scripts/test.sh`, `--lint`, and the dashboard
   `tsc --noEmit`.
10. Lint debt: about 150 pre-existing ruff errors and two ESLint errors (`task-history.tsx`,
    `hooks/use-websocket.ts`). Fix in a dedicated cleanup PR, not mixed into feature work.
11. `benchmarks/` and `docs/` are mostly empty. A reproducible benchmark (real provider, fixed
    corpus, results committed) would back the project's claims.
12. In the 3D graph, labels on dimmed nodes stay visible during focus mode.

## Decisions worth knowing

- Graph entities are identified by (entity_type, name) at build time; relation identity is
  (source, target, relation_type). Extractor UUIDs are not stable across runs.
- The dashboard only shows demo data when the API is unreachable; an empty or failing API shows
  an empty state or an error, never fake data.
- The graph renderer is client-only (`next/dynamic`, `ssr: false`); shared constants live in
  `kg-style.ts` so server-rendered modules never import three.js.
