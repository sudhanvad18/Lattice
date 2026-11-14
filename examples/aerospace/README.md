# Aerospace knowledge base walkthrough

This fictional TF-40 corpus demonstrates retrieval and graph extraction. Its
procedures and intervals are sample data, not approved maintenance instructions.
Run from the repository root after installing Lattice:

```bash
lattice ingest examples/aerospace
lattice stats
lattice search "Fan Blade"
lattice graph --communities
lattice graph --entity "Fan Blade"
lattice run "Summarize fan blade erosion inspection evidence and identify gaps" --provider ollama
```

Start Ollama and install your chosen model before running the agent team, or use
`--provider anthropic` / `--provider openai` with the corresponding
`LATTICE_ANTHROPIC_API_KEY` / `LATTICE_OPENAI_API_KEY` configured.

Use `lattice --data-dir /tmp/aerospace-demo ingest examples/aerospace` to select
another knowledge base. Repeat that global option for subsequent commands.
Re-ingestion reports cache hits for unchanged documents. `--no-cache` bypasses
parsed document reuse. `lattice reset --yes` clears stored knowledge and cache.
