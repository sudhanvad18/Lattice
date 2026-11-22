# Radar: standing scouting brief for Grok

Grok runs this continuously. The goal is to find models, repos, frameworks, datasets, and papers
that make the owner's projects stronger, and to propose new projects or applications worth
building. Favour a few well-argued finds over long lists; a week with nothing worth adopting
should say so.

## Projects in scope

| Project | Where | What it is | Where outside work could help |
|---|---|---|---|
| Lattice | `github.com/sudhanvad18/Lattice` (see `HANDOFF.md`) | Multi-agent LangGraph platform over a KG-augmented RAG knowledge base, with guardrails, write-back, CLI, API, and 3D dashboard. | Entity/relation extraction, entity resolution, graph RAG retrieval, embeddings and rerankers, agent frameworks and MCP, guardrails and evals, local inference, graph visualization, domain corpora. |
| MeanFlow PDE | Local only: `~/Desktop/Meanflow_MultiPDE` (no remote) | Research on one-step, observation-conditioned generative surrogates for paired PDE fields: improved MeanFlow trained with endpoint MSE, then raw Energy Score; ALCF Polaris training. | One-step generative models (MeanFlow, consistency/shortcut models, flow matching), probabilistic scoring rules, neural operators, PDE benchmarks and datasets, efficient UNet/transformer backbones. |

MeanFlow guard: read `CODEBASE_CONTEXT.md` and `RESEARCH_NOTES.md` beside the checkout. Never
write inside `masked-joint-mean-flows-pde/`: its release ledger is hash-verified and any added
file breaks publication runs. Radar output for this project goes in Lattice's `docs/radar/`
(clearly labelled) or in a message to the owner.

## Watchlist (seeds, not a closed list)

The names below are starting points; check whether each is still maintained before citing it.

- Knowledge graph construction: zero-shot NER/RE models (GLiNER, GLiREL family), LLM extraction
  pipelines, entity resolution/dedup libraries.
- Graph RAG: Microsoft GraphRAG, LightRAG, Graphiti, HippoRAG, neo4j-graphrag, LlamaIndex
  property graphs. Compare their retrieval strategies with Lattice's hybrid retriever.
- Retrieval models on Hugging Face: new embedding and reranker releases that run locally
  (BGE, GTE, Qwen embedding/reranker, nomic, jina, ColBERT-style late interaction). Track MTEB
  movement, size, and license.
- Agents: LangGraph releases and breaking changes, MCP servers and spec changes, agent SDKs,
  durable-execution and checkpointing patterns.
- Guardrails and evaluation: RAGAS, DeepEval, LLM-as-judge methods, safety classifiers, citation
  and grounding metrics that could replace Lattice's keyword-recall benchmark.
- Local inference: Ollama-servable models that fit a laptop, vLLM and llama.cpp features,
  structured-output support.
- Code understanding: tree-sitter grammars, code knowledge graphs, repo-level retrieval.
- Graph algorithms and viz: Leiden implementations that install cleanly on macOS, large-graph
  WebGL renderers.
- Scientific ML (MeanFlow): one-step and few-step generative models, scoring-rule training,
  neural operators, PDEBench / The Well style datasets, ensemble calibration metrics.

Sources: Hugging Face trending and new models, GitHub trending and release feeds, arXiv
(cs.CL, cs.IR, cs.LG, physics.comp-ph), framework changelogs, and X discussion from maintainers
and researchers in these areas.

## How to judge a find

Score each candidate on:

1. Fit: which concrete module or open problem does it improve? Name the file or the item in
   `HANDOFF.md` "Open work". No named fit, no recommendation.
2. Evidence: benchmark numbers, adoption, release recency, maintenance activity.
3. Cost: new dependency weight, GPU or network needs, API changes. Lattice must keep running
   offline on a laptop with Ollama.
4. License: permissive (MIT, Apache-2.0, BSD) preferred; flag anything else.
5. Test: the smallest experiment that would prove it out, ideally measurable with
   `lattice eval` or `scripts/test.sh`.

## Output

- Weekly digest as a PR adding `docs/radar/YYYY-MM-DD.md` (branch `grok/radar-YYYY-MM-DD`),
  using the template below. Check earlier digests and do not repeat an item unless something
  changed.
- Urgent alerts (a breaking change in a dependency we use, a security advisory, or a release
  that directly fixes an open item) go straight to the owner, without waiting for the digest.
- For anything rated "adopt", open a GitHub issue labelled `radar` whose body is a task spec in
  the `.ai/task-<name>.md` format (owned files, contract, acceptance tests), so it can be
  dispatched to a Codex, Claude Code, or cloud agent without rewriting.

Digest template:

```markdown
# Radar YYYY-MM-DD

## Adopt (worth a task now)
### <name> (<link>)
- What: one sentence.
- Fit: <project> / <module or open-work item>.
- Evidence: numbers, release date, license.
- Experiment: smallest test and how we measure it.
- Effort: S / M / L.

## Trial (promising, needs a spike)
## Watch (not ready, revisit)
## Long shots (new projects and applications)
## Dropped since last digest (and why)
```

## Long shots

Beyond incremental upgrades, propose each month at least one of:

- A new application of an existing project. Lattice is a domain-agnostic "corpus to knowledge
  graph to agent team with guardrails" engine: candidates include regulatory/compliance
  document QA, codebase onboarding, incident postmortem graphs, research-literature graphs, and
  maintenance records in other industries.
- A combination of the two projects, for example using Lattice to organize PDE literature and
  experiment logs, or exposing MeanFlow surrogates as Lattice tools.
- A new project idea grounded in a specific new model or dataset, with a one-paragraph plan and
  the first week of work.

Each long shot needs: who it is for, why now (what changed), the first experiment, and what
existing code it reuses.

## Limits

- Read-only on the owner's machine unless the owner approves a specific action.
- Never add dependencies, change shared config, or touch `main` directly; propose through PRs
  and issues as described in `HANDOFF.md`.
- Do not start paid compute (cloud GPUs, ALCF jobs) or sign up for services.
