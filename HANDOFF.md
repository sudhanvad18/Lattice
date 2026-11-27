# Lattice handoff

Living status note shared by every agent working on this repo (Cursor, Codex, Claude Code,
Grok). Read this first, then `AGENTS.md` for the working rules. Whoever changes the state of the
project updates the relevant section here in the same PR.

Last updated: 2026-11-26 by Codex (temporary local orchestrator while Cursor is usage-limited).

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

Quick start: `python examples/mock_workflow.py` runs the real orchestrator with scripted model
responses (no model, no network). For the knowledge base: `lattice ingest examples/aerospace &&
lattice serve`, then `npm run dev` in `packages/dashboard`. Owner-facing docs: `README.md`,
`docs/development.md`, `docs/demo.md`. Tests: `scripts/test.sh -q` (405 passed, 1 skipped on 2026-11-24; the
skipped test needs a live Neo4j via `LATTICE_TEST_NEO4J_URI/USER/PASSWORD`).

## State of the repo

- `main` at `b74260e` is pushed to GitHub; local and `origin/main` agree. The nine local KB
  integration commits were pushed on 2026-11-24 with owner authorization.
- KB integration wave is complete: one process-owned KB, `/kb/status|ingest|reset`, structured
  provider errors and health checks, grounded `/chat`, KB-grounded evaluation, and the combined
  Ask dashboard. Contract: `.ai/contracts-integration.md` (local, gitignored).
- CI, root-manifest Docker installation fixes, `prometheus-client`, and the security dependency
  updates have landed. Do not dispatch these again from older backlog notes.
- Verification at `b74260e`: 405 passed, 1 skipped; dashboard `npx tsc --noEmit` and
  `npx eslint src` passed. `scripts/test.sh --lint` found no newly added Python files to check;
  this does not establish that existing lint debt is clean.
- Live Ollama smoke testing uses a temporary KB at `/private/tmp/lattice-smoke-20261124` and
  the installed `llama3.2:latest` model. The handoff's older claim that three models were already
  installed was stale: the model store was empty; only llama3.2 was downloaded this session.
- Ask streaming: Answer uses `POST /chat/stream` (SSE), with retrieval/model
  status, immediate text deltas, source metadata and a final cited response. `/chat` remains
  compatible. Native streaming is implemented for Ollama, Anthropic and OpenAI; custom providers
  can fall back to generate. Interrupted answers retain partial text with a notice; navigating
  away cancels the request. Agent-team task execution is unchanged by this Answer-mode feature.
  Contract: `.ai/contracts-chat-streaming.md`. Tests: 423 passed, 1 skipped; five client stream
  checks passed; dashboard TypeScript/ESLint passed. Live evidence: `.ai/ask-stream-smoke/`.
- Ingestion fix: cache keys include source path and content, preventing
  identical files from sharing chunk IDs/provenance. Default folder ingestion skips dependency
  and tool-cache directories, including node_modules. Legacy content-only cache entries are
  ignored and re-parsed without requiring a KB reset. Five regression tests were added;
  `scripts/test.sh -q`: 410 passed, 1 skipped; no new lint diagnostics relative to HEAD.
- Retried `/Users/sudhanva/Desktop/tester-automation-main` through the live API: 117 files,
  301 chunks, 120 entities, four relations; no duplicate-ID failure. The running temporary KB
  now contains this corpus instead of the earlier aerospace smoke corpus.
- Smoke evidence and screenshots: `.ai/smoke-20261124/`. Empty-KB gates, ingestion, cited
  Answer, task execution, and persistence passed functionally. Review correctness failed;
  see the prioritized backlog below. A one-task evaluation finished and displayed its report
  (535.8 s, seven iterations, reported pass). These smoke metrics are not a quality benchmark.

## Who does what

| Agent | Where it runs | Use it for |
|---|---|---|
| Codex orchestrator (temporary); Cursor when resumed | Owner's machine | Writes contracts and specs, dispatches Codex/Claude workers, reviews, trial-merges, runs integration tests, and inspects the UI. |
| Codex CLI workers | Local git worktrees, sandboxed, no network | Well-specified implementation tasks from `.ai/task-<name>.md`. Default worker. |
| Claude Code workers | Local git worktrees, tool allowlist, no git/network | Same contract as Codex; better for larger refactors, docs, and tasks needing long reasoning. Dispatch with `task:claude`. |
| Cursor cloud agents | Remote, own branch, open a PR | Self-contained tasks that need no local services: CI, docs, lint cleanup, tests. |
| Grok | xAI, reads GitHub and the local checkout | Coordinator and scout: backlog upkeep, PR review, CI triage, drift checks, daily summary, and the technology radar in `docs/agents/radar.md`. |

Rules that keep agents from colliding:

1. One owner per branch. Local workers use `agent/<task>`, cloud agents started by Grok use
   `grok/<task>`. Check the "In flight" table below and open PRs before starting work.
2. Push `main` only with owner authorization; never force-push or rewrite it. Worker output
   lands by PR or by the local orchestrator's reviewed merge.
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
| (no coding workers dispatched) | | | Two lifecycle specs prepared under `.ai/`; see below. |

## Open work, highest value first

1. **Review correctness and task lifecycle (Codex spec prepared).** Live Ollama returned a
   malformed/truncated rejection beginning with `"approved": false`; reviewer fallback approved
   it because `approve` matches the key `approved`. Fail closed on malformed verdicts and
   validate booleans. Also fix finalization: iteration exhaustion/rejected review and failed
   required subtasks must not produce completed. API polling stays pending during execution,
   and `created_at` is regenerated on each request. Prepared: `.ai/task-task-lifecycle.md`.
2. **Ask task monitoring (Claude spec prepared).** Inspection: Ask treats normalized completion
   of any agent step as whole-task completion, and a socket close can mark an unfinished run
   done. Reconcile against authoritative task state and clean up connections/timers. These
   disconnect/intermediate-event risks were inspected, not reproduced in the successful task
   smoke run. Prepared: `.ai/task-ask-lifecycle.md`. File ownership does not overlap item 1.
3. **Refresh owner-facing docs.** README/development/demo docs predate the KB wave. For example,
   development.md still describes retrieval-only chat and demo fallback. Document KB-first
   setup, Ask modes, evaluation, model installation, and ports (dashboard 3001, API 8000).
4. **Graph clustering and entity resolution.** `graspologic` is absent locally, so clustering
   falls back to connected components. Large code ingests can produce one dominant community.
   Exact type/case-folded-name merging leaves identifier variants such as AMS-4928/AMS 4928
   separate. Scope and measure each improvement before adding optional dependencies.
5. **Request-body limits for ingestion.** Follow up on `/kg/ingest` body limits from
   `docs/radar/2026-11-23.md`; recheck the installed FastAPI API before choosing an implementation.
6. **Scoped lint cleanup.** Existing Python debt remains; dashboard ESLint now passes locally.
   CI lint is currently advisory. Audit actual violations before assigning disjoint file sets.
7. **Live Neo4j validation.** Offline fake-driver tests exist; live integration remains skipped
   without configured Neo4j. Search currently matches names only.
8. **Reproducible quality benchmark.** Save corpus/revision/model/hardware/procedure and measured
   results; include a vector-only baseline. A smoke run is not a quality benchmark.
9. **Other scoped follow-ups.** Wire the configured embedding model into the vector store before
   embedding A/B trials; evaluate extraction changes separately. API authentication is still a
   placeholder; write-back handlers and approval gates are library components rather than an
   automatic API workflow. MCP 2.x migration remains open. Confirm scope before new features.

For the next two-worker wave, review/commit local orchestrator maintenance first (dispatcher
requires a clean tracked tree), then use:

```bash
MAX_PARALLEL=2 scripts/orchestrate/dispatch.sh task-lifecycle:codex ask-lifecycle:claude
```

No workers were launched during the smoke-test session. Workers must run the full test wrapper,
new-file lint, and dashboard checks where relevant, then write handoffs. Review changes and
trial-merge into a scratch worktree before landing on main. The dispatcher no longer bypasses
commit hooks with `--no-verify`; never restore that flag without owner authorization.

## Decisions worth knowing

- Graph entities are identified by (entity_type, name) at build time; relation identity is
  (source, target, relation_type). Extractor UUIDs are not stable across runs.
- One active KB per API process drives every feature. An empty KB refuses tasks/chat/evaluation
  with 409 `kb_empty`. Dashboard failures show real empty/error states; never substitute demo data.
- Local Ollama is the default. Missing/unreachable providers show actionable errors. Answer can
  fall back to explicitly labelled retrieval-only output; Agent team/evaluation require a provider.
- The graph renderer is client-only (`next/dynamic`, `ssr: false`); shared constants live in
  `kg-style.ts` so server-rendered modules never import three.js.
