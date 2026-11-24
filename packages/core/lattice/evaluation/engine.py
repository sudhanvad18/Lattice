"""
Evaluation engine.

Benchmarks the agent team's performance on ground-truth tasks.
Measures:
- Task completion rate
- Artifact quality (via reviewer scores)
- Citation accuracy
- Latency
- Iteration efficiency

Supports comparing:
- Full agent team vs single agent
- Different models (Ollama vs Claude)
- Different configurations
"""

from __future__ import annotations

import builtins
import inspect
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import TaskStatus
from lattice.agents.writer import WriterAgent
from lattice.inference.provider import LLMProvider
from lattice.knowledge_base import KnowledgeBase

logger = structlog.get_logger()


@dataclass
class EvalTask:
    """A ground-truth evaluation task."""

    id: str
    description: str
    expected_artifact_types: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    max_acceptable_iterations: int = 10
    category: str = "general"


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task_id: str
    task_description: str
    completed: bool = False
    status: str = "pending"
    artifact_count: int = 0
    review_approved: bool = False
    review_confidence: float = 0.0
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    iterations_used: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    groundedness: float = 0.0
    category: str | None = None

    @property
    def keyword_recall(self) -> float:
        total = len(self.keywords_found) + len(self.keywords_missing)
        if total == 0:
            return 1.0
        return len(self.keywords_found) / total

    @property
    def passed(self) -> bool:
        return self.completed and self.review_approved and self.keyword_recall >= 0.5

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-serializable view used by the CLI, API, and dashboard."""
        return {
            "groundedness": self.groundedness,
            "category": self.category,
            "task_id": self.task_id,
            "task_name": self.task_description,
            "status": self.status,
            "completed": self.completed,
            "passed": self.passed,
            "quality_score": self.review_confidence,
            "keyword_recall": self.keyword_recall,
            "keywords_missing": self.keywords_missing,
            "latency_seconds": self.elapsed_seconds,
            "iterations": self.iterations_used,
            "artifact_count": self.artifact_count,
            "errors": self.errors,
        }


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results."""

    results: list[EvalResult] = field(default_factory=list)
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_passed: int = 0
    avg_iterations: float = 0.0
    avg_latency_seconds: float = 0.0
    avg_keyword_recall: float = 0.0
    avg_groundedness: float = 0.0
    avg_review_confidence: float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.tasks_completed / self.total_tasks if self.total_tasks else 0.0

    @property
    def pass_rate(self) -> float:
        return self.tasks_passed / self.total_tasks if self.total_tasks else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "avg_groundedness": self.avg_groundedness,
            "total_tasks": self.total_tasks,
            "completion_rate": f"{self.completion_rate:.1%}",
            "pass_rate": f"{self.pass_rate:.1%}",
            "avg_iterations": f"{self.avg_iterations:.1f}",
            "avg_latency_seconds": f"{self.avg_latency_seconds:.2f}",
            "avg_keyword_recall": f"{self.avg_keyword_recall:.1%}",
            "avg_review_confidence": f"{self.avg_review_confidence:.2f}",
        }


def generate_kb_tasks(kb: KnowledgeBase, limit: int = 5) -> list[EvalTask]:
    """Select connected KB facts deterministically, preferring distinct communities."""
    if limit <= 0:
        return []
    generic = {name.casefold() for name in dir(builtins)} | {"self", "cls"}
    entities = {
        e.id: e
        for e in kb.graph.get_all_entities()
        if len(e.name.strip()) >= 3 and e.name.casefold() not in generic
    }
    neighbors: dict[str, set[str]] = {eid: set() for eid in entities}
    relations = kb.graph.get_all_relations()
    degree = dict.fromkeys(entities, 0)
    for rel in relations:
        for endpoint in (rel.source_id, rel.target_id):
            if endpoint in degree:
                degree[endpoint] += 1
        if (
            rel.source_id in entities
            and rel.target_id in entities
            and rel.source_id != rel.target_id
        ):
            neighbors[rel.source_id].add(rel.target_id)
            neighbors[rel.target_id].add(rel.source_id)

    def rank(eid):
        return (-degree[eid], entities[eid].name.casefold(), eid)

    def slug(eid):
        return re.sub(r"[^a-z0-9]+", "-", entities[eid].name.casefold()).strip("-") or "entity"

    ranked = sorted(entities, key=rank)
    selected, seen = [], set()
    for eid in ranked:
        community = entities[eid].community
        if community is not None and community in seen:
            continue
        selected.append(eid)
        if community is not None:
            seen.add(community)
    selected.extend(eid for eid in ranked if eid not in selected)
    tasks = []
    used_ids: set[str] = set()
    for eid in selected[:limit]:
        entity = entities[eid]
        names = [entities[n].name for n in sorted(neighbors[eid], key=rank)[:4]]
        task_id = f"kb-{slug(eid)}"
        if task_id in used_ids:
            task_id = f"{task_id}-{eid}"
        used_ids.add(task_id)
        related = ", ".join(f"`{name}`" for name in names)
        description = f"Explain what `{entity.name}` is in this codebase and how it works"
        if related:
            description += f", and how it relates to {related}"
        tasks.append(
            EvalTask(
                id=task_id,
                description=description + ".",
                expected_keywords=[entity.name, *names],
                category=entity.entity_type,
            )
        )
    cross = sorted(
        (
            r
            for r in relations
            if r.source_id in entities
            and r.target_id in entities
            and entities[r.source_id].community is not None
            and entities[r.target_id].community is not None
            and entities[r.source_id].community != entities[r.target_id].community
        ),
        key=lambda r: (
            degree[r.source_id] + degree[r.target_id],
            rank(r.source_id),
            rank(r.target_id),
            r.relation_type,
        ),
    )
    if cross and limit > 1:
        rel = cross[0]
        source, target = entities[rel.source_id], entities[rel.target_id]
        task = EvalTask(
            id=f"kb-cross-{slug(source.id)}-{slug(target.id)}",
            description=(
                f"Explain the cross-community connection between `{source.name}` and "
                f"`{target.name}` via `{rel.relation_type}` "
                "and why it matters in this codebase."
            ),
            expected_keywords=[source.name, target.name, rel.relation_type],
            category="cross-community",
        )
        tasks = tasks[: limit - 1] + [task]
    return tasks

class EvaluationEngine:
    """Runs evaluation benchmarks against the agent team."""

    def __init__(self, provider: LLMProvider, knowledge_base: KnowledgeBase | None = None) -> None:
        self._provider = provider
        self._kb = knowledge_base

    async def evaluate_task(self, eval_task: EvalTask) -> EvalResult:
        """Run a single evaluation task."""
        result = EvalResult(
            task_id=eval_task.id,
            task_description=eval_task.description,
            category=eval_task.category,
        )

        start = time.time()
        try:
            orchestrator = OrchestratorAgent(
                provider=self._provider,
                researcher=ResearcherAgent(
                    provider=self._provider,
                    vector_store=self._kb.vector_store if self._kb else None,
                    kg_backend=self._kb.graph if self._kb else None,
                ),
                writer=WriterAgent(provider=self._provider),
                reviewer=ReviewerAgent(provider=self._provider),
            )

            state = await orchestrator.run(
                eval_task.description,
                max_iterations=eval_task.max_acceptable_iterations,
            )

            result.elapsed_seconds = time.time() - start
            result.status = state.status.value
            result.completed = state.status == TaskStatus.COMPLETED
            result.artifact_count = len(state.artifacts)
            result.iterations_used = state.iteration_count
            result.errors.extend(state.errors)

            # Check review
            if state.reviews:
                latest = state.reviews[-1]
                result.review_approved = latest.approved
                result.review_confidence = latest.confidence

            final = state.get_latest_artifact()
            if self._kb is not None and final is not None and final.citations:
                valid_ids = set(self._kb.vector_store.chunk_ids())
                valid_ids.update(e.id for e in self._kb.graph.get_all_entities())
                valid_count = sum(c in valid_ids for c in final.citations)
                result.groundedness = valid_count / len(final.citations)

            # Check keywords in artifacts
            all_content = " ".join(a.content.lower() for a in state.artifacts)
            for kw in eval_task.expected_keywords:
                if kw.lower() in all_content:
                    result.keywords_found.append(kw)
                else:
                    result.keywords_missing.append(kw)

        except Exception as e:
            result.elapsed_seconds = time.time() - start
            result.errors.append(str(e))
            result.status = "error"

        return result

    async def run_benchmark(
        self, tasks: list[EvalTask], on_progress: Callable[[int, int], Any] | None = None,
    ) -> BenchmarkReport:
        """Run a full benchmark suite."""
        report = BenchmarkReport(total_tasks=len(tasks))

        for task in tasks:
            logger.info("eval_running", task_id=task.id, description=task.description[:50])
            result = await self.evaluate_task(task)
            report.results.append(result)

            if result.completed:
                report.tasks_completed += 1
            if result.passed:
                report.tasks_passed += 1

            if on_progress is not None:
                progress = on_progress(len(report.results), len(tasks))
                if inspect.isawaitable(progress):
                    await progress

        # Compute averages
        if report.results:
            report.avg_groundedness = (
                sum(r.groundedness for r in report.results) / len(report.results)
            )
            report.avg_iterations = sum(r.iterations_used for r in report.results) / len(report.results)
            report.avg_latency_seconds = sum(r.elapsed_seconds for r in report.results) / len(report.results)
            report.avg_keyword_recall = sum(r.keyword_recall for r in report.results) / len(report.results)
            confidences = [r.review_confidence for r in report.results if r.review_confidence > 0]
            report.avg_review_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info("benchmark_complete", **report.summary())
        return report


def save_report(
    report: BenchmarkReport, path: Path, kb_snapshot: dict | None = None, **run_info: Any,
) -> dict[str, Any]:
    """Write a benchmark run to ``path`` atomically and return the stored payload."""
    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "kb": kb_snapshot,
        **run_info,
        "summary": {
            "total_tasks": report.total_tasks,
            "completion_rate": report.completion_rate,
            "pass_rate": report.pass_rate,
            "avg_groundedness": report.avg_groundedness,
            "avg_iterations": report.avg_iterations,
            "avg_latency_seconds": report.avg_latency_seconds,
            "avg_keyword_recall": report.avg_keyword_recall,
            "avg_review_confidence": report.avg_review_confidence,
        },
        "results": [r.to_record() for r in report.results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    return payload


def load_report(path: Path) -> dict[str, Any] | None:
    """Return the stored benchmark run, or None when no valid run exists."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return None
    payload.setdefault("kb", None)
    if isinstance(payload.get("summary"), dict):
        payload["summary"].setdefault("avg_groundedness", 0.0)
    for row in payload["results"]:
        if isinstance(row, dict):
            row.setdefault("groundedness", 0.0)
            row.setdefault("category", None)
    return payload


# --- Built-in benchmark tasks ---

DEMO_BENCHMARK_TASKS = [
    EvalTask(
        id="doc-api",
        description="Write documentation for a REST API with user CRUD endpoints",
        expected_artifact_types=["research", "document"],
        expected_keywords=["GET", "POST", "user", "endpoint"],
        category="documentation",
    ),
    EvalTask(
        id="doc-architecture",
        description="Write an architecture document for a microservices-based e-commerce system",
        expected_artifact_types=["research", "document"],
        expected_keywords=["service", "database", "API", "gateway"],
        category="documentation",
    ),
    EvalTask(
        id="analysis-performance",
        description="Analyze common performance bottlenecks in Python web applications and recommend solutions",
        expected_artifact_types=["research", "document"],
        expected_keywords=["caching", "database", "async", "optimization"],
        category="analysis",
    ),
    EvalTask(
        id="doc-onboarding",
        description="Create a developer onboarding guide for a team using FastAPI, PostgreSQL, and Docker",
        expected_artifact_types=["research", "document"],
        expected_keywords=["FastAPI", "Docker", "setup", "environment"],
        category="documentation",
    ),
]
