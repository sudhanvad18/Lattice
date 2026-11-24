# Development and demonstration

## Install from the repository root

Use Python 3.11 or later. `packages/core/` does not contain a separate package manifest; install using the root `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,local]'
```

The `local` extra installs the Ollama client. `prometheus-client` is a project dependency, used by the metrics module. The optional MCP server also requires `python -m pip install 'mcp==1.30.0'`. That is the last 1.x release; `lattice_mcp.server` does not import on mcp 2.x.

For an existing environment moved from another checkout, reinstall the editable package or use a source-path override for local checks:

```bash
PYTHONPATH=packages/core:packages/api:packages/mcp python -m pytest
```

## Offline checks

The focused tests below use scripted providers and local fixtures. They do not require a running LLM, API keys, or model downloads:

```bash
python examples/mock_workflow.py
python -m pytest tests/agents/test_agents.py tests/agents/test_tool_loop.py \
  tests/writeback/test_writeback.py tests/api/test_api.py \
  tests/guardrails/test_guardrails.py
```

Use `python -m pytest` for the complete suite. The pytest configuration includes coverage reporting. Tests involving vector storage or parsing can require optional models or cached resources; record the environment alongside the results. `MockProvider` is injected through Python and is not a value accepted by `LATTICE_DEFAULT_PROVIDER`.

## Model-backed demo

Install and start Ollama, then download the model used by `demo.py`:

```bash
ollama serve
```

In a second terminal:

```bash
ollama pull llama3.2
source .venv/bin/activate
python demo.py "Write a concise summary of the demo knowledge base."
```

The demo seeds a small example knowledge graph and ChromaDB collection, runs the agent workflow, and prints artifacts, guardrail results, and a trace summary. These knowledge-base facts are demonstration fixtures. ChromaDB's embedding function can download its model on first use. This path requires local model resources and is separate from the fully scripted example.

## Run the API

Copy `.env.example` to `.env` if local configuration is needed. Settings use the `LATTICE_` prefix and load `.env` through Pydantic Settings.

```bash
uvicorn lattice_api.app:app --reload --host 127.0.0.1 --port 8000
```

`lattice serve` starts the same application, on `127.0.0.1:8000` by default, and sets `LATTICE_DATA_DIR` from the CLI `--data-dir` before the process starts.

Explore `http://localhost:8000/docs`. With Ollama running and its model available, submit a task:

```bash
curl -X POST http://localhost:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"task":"Write a short technical overview of a REST service.","max_iterations":4}'
```

Use the returned `task_id` with:

| Route | Purpose |
| --- | --- |
| `GET /tasks/{task_id}` | Status and artifact/review counts. |
| `GET /tasks/{task_id}/result` | Artifacts, review history, and audit trail. |
| `GET /tasks/{task_id}/audit` | Recorded audit events. |
| `WS /ws/{task_id}` | Task event stream. |
| `GET /kg/graph` | Graph nodes and edges. `limit` defaults to 500. |
| `GET /kg/stats` | Chunk, entity, and relation counts, plus the backend name. |
| `GET /kg/entities` | Entities, optionally filtered by `q`, `entity_type`, and `limit`. |
| `GET /kg/entities/{entity_id}` | One entity, its neighbors, and its relations. |
| `GET /kg/communities` | Communities, hub nodes, and surprising connections. |
| `POST /kg/ingest` | JSON body with `text` and optional `source`. Returns the chunk count and knowledge-base stats. Text shorter than 50 tokens stores no chunks. |
| `GET /eval/results` | Latest `lattice eval` report, or an empty payload when none is stored. |

`/kg/*` returns 503 when the knowledge base is not available. `GET /eval/results` reads `<data-dir>/eval/latest.json`.

API tasks use the knowledge base in `LATTICE_DATA_DIR` (default `data/`); populate it with `lattice ingest <dir>` first, or use the Python demo for a self-contained graph/vector context. Current API authentication is a placeholder; the local development service is not a production deployment recipe.

## Configuration

Consult `.env.example` and `packages/core/lattice/config.py` for the full set:

| Variable | Purpose |
| --- | --- |
| `LATTICE_DEFAULT_PROVIDER` | `ollama`, `anthropic`, or `openai`. |
| `LATTICE_OLLAMA_BASE_URL`, `LATTICE_OLLAMA_MODEL` | Local model endpoint and model name. |
| `LATTICE_ANTHROPIC_API_KEY`, `LATTICE_OPENAI_API_KEY` | Credentials when selecting a cloud provider. |
| `LATTICE_GRAPH_BACKEND` | `networkx` or `neo4j`. |
| `LATTICE_CHROMA_PERSIST_DIR` | Local ChromaDB storage location. |
| `LATTICE_GITHUB_TOKEN` | Optional GitHub write-back handler access. |
| `LATTICE_ENABLE_TRACING` | Optional tracing configuration. |
| `NEXT_PUBLIC_API_URL` | Dashboard API base URL; defaults to `http://localhost:8000`. |

## Dashboard

Use Node.js 20.9 or later with the current Next.js 16 dashboard:

```bash
cd packages/dashboard
npm ci
npm run dev
```

Open `http://localhost:3000`. `npm run build` builds the dashboard and `npm start` serves that build. Follow the dashboard's `AGENTS.md` before modifying Next.js code.

## Integration status

The dashboard is a UI prototype. These contracts need alignment before presenting a complete dashboard-to-agent workflow:

- Task submission sends `goal`, while the API requires `task`.
- The dashboard expects `id`/`goal`; the API returns `task_id`/`original_task`.
- Dashboard WebSocket paths differ from the API's `WS /ws/{task_id}` route.
- The dashboard chat request targets `POST /chat`, which the API does not provide. Graph (`/kg/*`) and evaluation (`/eval/results`) endpoints exist.
- Write-back handlers and approval gates exist as library components; the API task manager does not execute them automatically.
- API and MCP images install the project from the root `pyproject.toml`. Both copy `packages/core`, `packages/api`, and `packages/mcp`, because that manifest builds `lattice`, `lattice_api`, and `lattice_mcp` into one wheel. The MCP image pins `mcp==1.30.0`, the last 1.x release the current server starts with. Porting `lattice_mcp.server` to mcp 2.x is open work. The server is a stdio process, and the image still publishes port 8100. Helm charts and the dashboard image are unchanged and are not a deployment recipe.

Use the offline example or model-backed Python demo as the reproducible entry point while completing these integrations. `GET /kg/communities` returns community membership, hub nodes, and surprising connections. Relation provenance is a field on `/kg/graph` edges. The dashboard graph view draws both; the README states what that view shows.

## Record a useful demo

Show one task, its plan, a research artifact, the draft/review cycle, and the final artifact. A recording of `examples/mock_workflow.py` must be labeled **scripted offline demonstration**. For model-backed results, include the revision, provider/model, hardware, input knowledge base, and exact command. Report measured latency or quality only alongside saved evaluation output and the evaluation procedure.
