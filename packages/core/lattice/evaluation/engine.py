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

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import AgentState, TaskStatus
from lattice.agents.writer import WriterAgent
from lattice.inference.provider import LLMProvider

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

    @property
    def keyword_recall(self) -> float:
        total = len(self.keywords_found) + len(self.keywords_missing)
        if total == 0:
            return 1.0
        return len(self.keywords_found) / total

    @property
    def passed(self) -> bool:
        return self.completed and self.review_approved and self.keyword_recall >= 0.5


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
    avg_review_confidence: float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.tasks_completed / self.total_tasks if self.total_tasks else 0.0

    @property
    def pass_rate(self) -> float:
        return self.tasks_passed / self.total_tasks if self.total_tasks else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completion_rate": f"{self.completion_rate:.1%}",
            "pass_rate": f"{self.pass_rate:.1%}",
            "avg_iterations": f"{self.avg_iterations:.1f}",
            "avg_latency_seconds": f"{self.avg_latency_seconds:.2f}",
            "avg_keyword_recall": f"{self.avg_keyword_recall:.1%}",
            "avg_review_confidence": f"{self.avg_review_confidence:.2f}",
        }


class EvaluationEngine:
    """Runs evaluation benchmarks against the agent team."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def evaluate_task(self, eval_task: EvalTask) -> EvalResult:
        """Run a single evaluation task."""
        result = EvalResult(
            task_id=eval_task.id,
            task_description=eval_task.description,
        )

        start = time.time()
        try:
            orchestrator = OrchestratorAgent(
                provider=self._provider,
                researcher=ResearcherAgent(provider=self._provider),
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

            # Check review
            if state.reviews:
                latest = state.reviews[-1]
                result.review_approved = latest.approved
                result.review_confidence = latest.confidence

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

    async def run_benchmark(self, tasks: list[EvalTask]) -> BenchmarkReport:
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

        # Compute averages
        if report.results:
            report.avg_iterations = sum(r.iterations_used for r in report.results) / len(report.results)
            report.avg_latency_seconds = sum(r.elapsed_seconds for r in report.results) / len(report.results)
            report.avg_keyword_recall = sum(r.keyword_recall for r in report.results) / len(report.results)
            confidences = [r.review_confidence for r in report.results if r.review_confidence > 0]
            report.avg_review_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info("benchmark_complete", **report.summary())
        return report


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
