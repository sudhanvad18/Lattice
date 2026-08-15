# Lattice

**Autonomous AI Agent Platform with Full Observability**

A production-grade multi-agent system where specialized AI agents collaborate to research, write, review, and deliver — with write-back to real systems, human-in-the-loop approval, and complete traceability.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Lattice Platform                          │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  Dashboard  │   REST API   │  MCP Server  │   CLI (planned)    │
├─────────────┴──────────────┴──────────────┴────────────────────┤
│                     Agent Orchestrator (LangGraph)               │
├────────┬──────────┬─────────┬──────────┬───────────────────────┤
│Research│  Writer  │Reviewer │   Code   │      Ingestion        │
│ Agent  │  Agent   │  Agent  │  Agent   │       Agent           │
├────────┴──────────┴─────────┴──────────┴───────────────────────┤
│  Guardrails  │  Observability  │  Write-Back Engine  │  Eval   │
├──────────────┴─────────────────┴─────────────────────┴─────────┤
│   Knowledge Graph (NetworkX/Neo4j)  │  Vector Store (ChromaDB)  │
├─────────────────────────────────────┴───────────────────────────┤
│  Inference Providers: Ollama │ Anthropic │ OpenAI │ Mock        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Orchestration** | LangGraph-powered workflow: plan → research → write → review → iterate |
| **Write-Back** | Agents modify real systems (GitHub PRs, file system, knowledge graph) |
| **Human-in-the-Loop** | Configurable approval gates: always, on-low-confidence, or auto |
| **Guardrails** | Citation enforcement, confidence gating, hallucination detection |
| **Observability** | LangFuse traces, OpenTelemetry spans, Prometheus metrics |
| **Tool-Use Loop** | Agents autonomously call tools (search, KG query, custom) during execution |
| **Evaluation** | Benchmark engine measuring quality, citation accuracy, latency, iterations |
| **Domain Templates** | Pre-configured for aerospace, hardware validation, developer tools |
| **MCP Integration** | Exposes agents as IDE tools for Cursor, Claude Code, etc. |

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/sudhanvad18/Lattice.git
cd Lattice
python -m venv .venv && source .venv/bin/activate
pip install -e "packages/core[dev]"

# Run tests
pytest

# Start API server
uvicorn lattice_api.app:app --reload

# Start dashboard
cd packages/dashboard && npm install && npm run dev
```

### Docker (Full Stack)

```bash
docker compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:3000
# Grafana: http://localhost:3001
# Neo4j: http://localhost:7474
```

### Kubernetes

```bash
helm install lattice deploy/helm/lattice \
  --set redis.auth.password=<your-password> \
  --set neo4j.neo4j.password=<your-password> \
  --set secrets.anthropicApiKey=<key>
```

## Architecture

### Agent Workflow

```
User Task → Orchestrator → [Plan Subtasks]
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
         Researcher        Writer           Code Agent
         (RAG + KG)     (Generate)        (Code/Refactor)
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                           Reviewer
                     (LLM-as-Judge + Guardrails)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               APPROVED              REJECTED
                    │                     │
                    ▼                     ▼
            Write-Back Engine      Loop (max 3 revisions)
            (GitHub/FS/KG)
```

### Observability Stack

- **LangFuse**: Full call-chain traces (orchestrator → researcher → writer → reviewer)
- **OpenTelemetry**: Distributed tracing with OTLP export to Jaeger/Tempo
- **Prometheus**: Task throughput, agent utilization, LLM token usage, error rates
- **Structured Logging**: Every log line carries trace_id + span_id for correlation

### Guardrails (Pre-Write-Back)

1. **Citation Enforcement** — Factual claims must reference sources
2. **Confidence Gating** — Below threshold → system abstains instead of hallucinating
3. **Hallucination Detection** — Claims cross-referenced against Knowledge Graph entities

## Repository Structure

```
lattice/
├── packages/
│   ├── core/                    # Agent runtime + data layer
│   │   └── lattice/
│   │       ├── agents/          # LangGraph agents + orchestrator
│   │       ├── graph/           # Knowledge graph (NetworkX/Neo4j)
│   │       ├── retrieval/       # Vector store + cross-encoder reranker
│   │       ├── ingestion/       # Document parsing + chunking
│   │       ├── inference/       # LLM provider abstraction
│   │       ├── guardrails/      # Safety verification layer
│   │       ├── observability/   # Tracing + metrics + OTel
│   │       ├── writeback/       # Write-back engine + handlers
│   │       └── evaluation/      # Benchmark + citation utility
│   ├── api/                     # FastAPI backend + WebSocket
│   ├── mcp/                     # MCP server for IDE integration
│   └── dashboard/               # Next.js frontend
├── deploy/
│   ├── helm/                    # Kubernetes Helm chart
│   └── terraform/               # AWS infrastructure (EKS + RDS + ElastiCache)
├── docker/                      # Dockerfiles + observability configs
├── templates/                   # Domain-specific starter configs
│   ├── aerospace/
│   ├── hardware-validation/
│   └── developer-tools/
├── tests/                       # 219 tests, 86% coverage
├── docker-compose.yml           # Full stack in one command
└── mcp.json                     # MCP configuration for IDEs
```

## Benchmark Results

| Metric | Score |
|--------|-------|
| Task Completion Rate | 92% |
| Average Quality Score | 0.85 |
| Citation Accuracy | 0.88 |
| Average Latency | 12.4s |
| Revision Efficiency | 1.8 iterations avg |

*Measured on 6 evaluation tasks using Ollama (llama3.2) with the demo knowledge base.*

## Configuration

Environment variables (or `.env` file):

```env
# Inference
OLLAMA_HOST=http://localhost:11434
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...

# Storage
REDIS_URL=redis://localhost:6379/0
CHROMADB_HOST=localhost
NEO4J_URI=bolt://localhost:7687

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# Safety
LATTICE_APPROVAL_MODE=on_low_confidence  # always | on_low_confidence | never
```

## Domain Templates

Start with a pre-configured setup for your industry:

```bash
lattice init --template aerospace          # Engine maintenance + quality docs
lattice init --template hardware-validation # Test suites + failure tracking
lattice init --template developer-tools    # Services + APIs + runbooks
```

Each template provides:
- Entity/relation type definitions for the knowledge graph
- Tuned guardrail thresholds
- Agent system prompt specializations
- Chunking strategy optimized for document types

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| Inference | Ollama, Anthropic, OpenAI |
| Vector Store | ChromaDB |
| Knowledge Graph | NetworkX (dev), Neo4j (prod) |
| Reranking | PyTorch cross-encoder (fine-tunable) |
| API | FastAPI + WebSocket |
| Frontend | Next.js 16 + TypeScript + Tailwind |
| Observability | LangFuse + OpenTelemetry + Prometheus |
| Infrastructure | Docker Compose, Helm, Terraform (AWS) |
| Testing | pytest (219 tests, 86% coverage) |

## Contributing

```bash
# Setup dev environment
pip install -e "packages/core[dev]"

# Run tests with coverage
pytest --cov

# Type checking
mypy packages/core/

# Lint
ruff check packages/
```

## License

MIT
