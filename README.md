# Lattice

Autonomous multi-agent orchestration platform with knowledge-graph-augmented RAG.

Teams of specialized AI agents collaborate to execute multi-step tasks — research, write, code, analyze — then write results back to real systems (GitHub, knowledge graph, file system), with human-in-the-loop approval gates.

## Status

Under active development.

## Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/lattice.git
cd lattice
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,local]"

# Copy and configure environment
cp .env.example .env

# Install Ollama for local inference (macOS)
brew install ollama
ollama pull llama3.2

# Run the platform
python -m lattice_api
```

## Architecture

See [docs/architecture.md](docs/architecture.md)

## License

MIT
