"""Command line entry point for a persistent local Lattice knowledge base."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter

import click
import structlog
from rich.console import Console
from rich.table import Table

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.state import TaskStatus
from lattice.config import InferenceProvider, get_settings
from lattice.evaluation import DEMO_BENCHMARK_TASKS, EvaluationEngine, save_report
from lattice.graph.clustering import cluster_graph
from lattice.guardrails import GuardrailRunner
from lattice.inference.mock import MockProvider
from lattice.inference.provider import get_provider
from lattice.ingestion.cache import IngestionCache
from lattice.knowledge_base import KnowledgeBase

console = Console(markup=False)


class FriendlyGroup(click.Group):
    """Report operational failures without exposing a Python traceback."""

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, click.Abort):
            raise
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(cls=FriendlyGroup)
@click.option("--data-dir", type=click.Path(path_type=Path),
              default=lambda: get_settings().data_dir, show_default="settings.data_dir")
@click.option("-v", "--verbose", is_flag=True, help="Show INFO logs.")
@click.pass_context
def main(ctx: click.Context, data_dir: Path, verbose: bool) -> None:
    """Ingest documents, explore knowledge, and run an agent team."""
    level = logging.INFO if verbose else logging.WARNING
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


def _kb(ctx: click.Context, *, enable_cache: bool = True) -> KnowledgeBase:
    kb = KnowledgeBase(persist_dir=ctx.obj["data_dir"], enable_cache=enable_cache)
    ctx.call_on_close(kb.close)
    return kb


def _table(title: str, columns: tuple[str, ...], rows) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(value) for value in row))
    console.print(table)


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--no-cache", is_flag=True, help="Bypass the parsed document cache.")
@click.option("--reset", "reset_first", is_flag=True, help="Clear the knowledge base first.")
@click.pass_context
def ingest(ctx: click.Context, path: Path, no_cache: bool, reset_first: bool) -> None:
    """Ingest a file or directory into the knowledge base."""
    started = perf_counter()
    kb = _kb(ctx, enable_cache=not no_cache)
    if reset_first:
        kb.reset()
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    cache = IngestionCache(ctx.obj["data_dir"] / "ingestion_cache")
    hits = sum(cache.is_cached(p) for p in files) if not no_cache else 0
    chunks = kb.ingest_file(path) if path.is_file() else kb.ingest_directory(path)
    stats = kb.stats
    communities = {e.community for e in kb.graph.get_all_entities() if e.community is not None}
    _table("Ingestion summary", ("Metric", "Value"), [
        ("Files", len(files)), ("Cache hits", hits), ("Chunks", chunks),
        ("Entities", stats["kg_entities"]), ("Relations", stats["kg_relations"]),
        ("Communities", len(communities)), ("Elapsed", f"{perf_counter() - started:.2f}s"),
    ])


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Inspect stored chunks, graph counts, and entity/relation types."""
    kb = _kb(ctx)
    values = kb.stats
    _table("Knowledge base", ("Metric", "Value"), [
        ("Backend", type(kb.graph).__name__), ("Chunks stored", values["chunks_stored"]),
        ("Entities", values["kg_entities"]), ("Relations", values["kg_relations"]),
    ])
    _table("Entity types", ("Type", "Count"), sorted(values["kg_entity_types"].items()))
    _table("Relation types", ("Type", "Count"), sorted(values["kg_relation_types"].items()))


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, type=click.IntRange(min=1), show_default=True)
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
    """Search document chunks and matching graph entities."""
    kb = _kb(ctx)
    hits = kb.vector_search(query, top_k) if kb.vector_store.count else []
    _table("Vector hits", ("Chunk ID", "Score", "Source", "Snippet"), [
        (h.chunk_id, f"{h.similarity:.3f}", h.metadata.get("source", ""), h.content[:240])
        for h in hits
    ])
    _table("Matching graph entities", ("Name", "Type", "ID"), [
        (e.name, e.entity_type, e.id) for e in kb.graph_search(query)
    ])


@main.command()
@click.option("--communities", is_flag=True, help="Show communities and connecting nodes.")
@click.option("--entity", "entity_name", help="Show the neighborhood of a matching entity.")
@click.pass_context
def graph(ctx: click.Context, communities: bool, entity_name: str | None) -> None:
    """Explore graph communities or an entity's neighborhood."""
    if communities and entity_name:
        raise click.UsageError("Choose --communities or --entity.")
    kb = _kb(ctx)
    if entity_name:
        matches = kb.graph_search(entity_name)
        if not matches:
            console.print(f"No entity found: {entity_name}")
            return
        for entity in matches:
            neighborhood = kb.graph.get_neighbors(entity.id)
            console.print(f"Neighborhood: {entity.name} ({entity.entity_type})")
            _table("Neighbors", ("Name", "Type"), [
                (e.name, e.entity_type) for e in neighborhood.entities
            ])
            names = {e.id: e.name for e in neighborhood.entities}
            names[entity.id] = entity.name
            _table("Relations", ("Source", "Relation", "Target"), [
                (names.get(r.source_id, r.source_id), r.relation_type,
                 names.get(r.target_id, r.target_id)) for r in neighborhood.relations
            ])
        return
    clusters = cluster_graph(kb.graph)
    names = {e.id: e.name for e in kb.graph.get_all_entities()}
    _table("Communities", ("Community", "Entities", "Members"), [
        (cid, len(ids), ", ".join(names[i] for i in ids))
        for cid, ids in sorted(clusters.communities.items())
    ])
    console.print("God nodes: " + (", ".join(names[i] for i in clusters.god_nodes) or "None"))
    console.print("Surprising connections:")
    for _, _, explanation in clusters.surprising_connections:
        console.print(explanation)
    if not clusters.surprising_connections:
        console.print("None")


@main.command()
@click.argument("task")
@click.option("--provider", type=click.Choice(["ollama", "anthropic", "openai"]))
@click.option("--model", help="Override the provider's default model.")
@click.option("--max-iterations", default=6, type=click.IntRange(min=1), show_default=True)
@click.pass_context
def run(ctx: click.Context, task: str, provider: str | None, model: str | None,
        max_iterations: int) -> None:
    """Run the agent team on TASK using the stored knowledge base."""
    kb = _kb(ctx)
    try:
        llm = get_provider(
            InferenceProvider(provider) if provider else None,
            **({"model": model} if model else {}),
        )
        researcher = ResearcherAgent(
            provider=llm, vector_store=kb.vector_store, kg_backend=kb.graph,
        )
        team = OrchestratorAgent(provider=llm, researcher=researcher)
        result = asyncio.run(team.run(task, max_iterations=max_iterations))
    except Exception as exc:
        raise click.ClickException(
            f"Could not run the agent team. Check provider availability and configuration: {exc}"
        ) from exc
    console.print(f"Status: {result.status.value}")
    guardrails = GuardrailRunner.default(kg_backend=kb.graph)
    for artifact in result.artifacts:
        console.print(f"Artifact {artifact.id} ({artifact.artifact_type})")
        suffix = "\n[truncated]" if len(artifact.content) > 2000 else ""
        console.print(artifact.content[:2000] + suffix)
        report = guardrails.run(artifact)
        console.print(f"Guardrail report: {report.overall_verdict.value}")
        _table("Checks", ("Check", "Verdict", "Score", "Details"), [
            (r.guardrail_name, r.verdict.value, f"{r.score:.3f}", r.details)
            for r in report.results
        ])
    if result.status == TaskStatus.FAILED:
        ctx.exit(1)


@main.command("eval")
@click.option("--provider", type=click.Choice(["ollama", "anthropic", "openai", "mock"]),
              help="LLM provider; 'mock' runs offline to exercise the pipeline.")
@click.option("--model", help="Override the provider's default model.")
@click.option("--task", "task_ids", multiple=True, help="Run only these benchmark task ids.")
@click.pass_context
def eval_command(ctx: click.Context, provider: str | None, model: str | None,
                 task_ids: tuple[str, ...]) -> None:
    """Run the built-in benchmark and save results for the dashboard."""
    tasks = [t for t in DEMO_BENCHMARK_TASKS if not task_ids or t.id in task_ids]
    if not tasks:
        raise click.BadParameter(f"unknown task id(s): {', '.join(task_ids)}", param_hint="--task")

    if provider == "mock":
        llm = MockProvider()
    else:
        llm = get_provider(
            InferenceProvider(provider) if provider else None,
            **({"model": model} if model else {}),
        )
    report = asyncio.run(EvaluationEngine(llm).run_benchmark(tasks))

    path = Path(ctx.obj["data_dir"]) / "eval" / "latest.json"
    save_report(report, path, provider=provider or get_settings().default_provider.value,
                model=model or getattr(llm, "model", None))
    _table("Benchmark", ("Task", "Status", "Passed", "Quality", "Keyword recall", "Latency"), [
        (r.task_id, r.status, "yes" if r.passed else "no", f"{r.review_confidence:.2f}",
         f"{r.keyword_recall:.0%}", f"{r.elapsed_seconds:.1f}s")
        for r in report.results
    ])
    console.print(f"Pass rate {report.pass_rate:.0%}. Results saved to {path}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=click.IntRange(1, 65535), show_default=True)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the Lattice API server."""
    import os

    import uvicorn

    os.environ["LATTICE_DATA_DIR"] = str(ctx.obj["data_dir"])
    uvicorn.run("lattice_api.app:app", host=host, port=port)


@main.command()
@click.option("--yes", is_flag=True, required=True, help="Confirm deletion of stored knowledge.")
@click.pass_context
def reset(ctx: click.Context, yes: bool) -> None:
    """Clear the vector store, graph, and ingestion cache."""
    _kb(ctx).reset()
    console.print("Knowledge base reset.")


if __name__ == "__main__":
    main()
