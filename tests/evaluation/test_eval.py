"""
Tests for the evaluation engine.
"""

import json

import pytest

from lattice.evaluation import (
    BenchmarkReport,
    DEMO_BENCHMARK_TASKS,
    EvalResult,
    EvalTask,
    EvaluationEngine,
)
from lattice.inference.mock import MockProvider


@pytest.fixture
def provider():
    """Mock provider with responses for a full agent workflow."""
    def make_responses():
        return [
            # Plan
            json.dumps({"subtasks": [
                {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                {"description": "Write doc", "assigned_to": "writer", "depends_on": [0]},
            ]}),
            # Research
            json.dumps({
                "summary": "Found info about REST API endpoints including GET and POST methods for user management.",
                "key_findings": ["GET /users endpoint", "POST /users endpoint"],
                "confidence": 0.9,
            }),
            # Write
            "# API Documentation\n\n## Endpoints\n\n### GET /users\nReturns user list.\n\n### POST /users\nCreates a new user.\n\n## Architecture\nThis endpoint is part of the microservice gateway.",
            # Review (approve)
            json.dumps({
                "approved": True,
                "confidence": 0.88,
                "issues": [],
                "suggestions": [],
                "reasoning": "Comprehensive documentation.",
            }),
        ]
    return MockProvider(responses=make_responses())


class TestEvalTask:
    def test_eval_task_creation(self):
        task = EvalTask(
            id="test-1",
            description="Write docs",
            expected_keywords=["API", "endpoint"],
            category="documentation",
        )
        assert task.id == "test-1"
        assert task.category == "documentation"


class TestEvalResult:
    def test_keyword_recall_all_found(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            keywords_found=["a", "b", "c"],
            keywords_missing=[],
        )
        assert result.keyword_recall == 1.0

    def test_keyword_recall_partial(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            keywords_found=["a", "b"],
            keywords_missing=["c", "d"],
        )
        assert result.keyword_recall == 0.5

    def test_keyword_recall_none_found(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            keywords_found=[],
            keywords_missing=["a", "b"],
        )
        assert result.keyword_recall == 0.0

    def test_passed_requires_all_conditions(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            completed=True,
            review_approved=True,
            keywords_found=["a", "b"],
            keywords_missing=[],
        )
        assert result.passed

    def test_not_passed_if_not_completed(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            completed=False,
            review_approved=True,
            keywords_found=["a"],
            keywords_missing=[],
        )
        assert not result.passed

    def test_not_passed_if_review_rejected(self):
        result = EvalResult(
            task_id="t1",
            task_description="test",
            completed=True,
            review_approved=False,
            keywords_found=["a"],
            keywords_missing=[],
        )
        assert not result.passed


class TestEvaluationEngine:
    @pytest.mark.asyncio
    async def test_evaluate_single_task(self, provider):
        engine = EvaluationEngine(provider=provider)
        task = EvalTask(
            id="test-eval",
            description="Write documentation for a REST API with user CRUD endpoints",
            expected_keywords=["GET", "POST", "user"],
        )
        result = await engine.evaluate_task(task)

        assert result.completed
        assert result.status == "completed"
        assert result.artifact_count >= 2
        assert result.review_approved
        assert result.review_confidence > 0
        assert result.elapsed_seconds > 0
        assert "GET" in result.keywords_found
        assert "user" in result.keywords_found

    @pytest.mark.asyncio
    async def test_run_benchmark(self):
        # Give enough responses for 2 tasks
        responses = []
        for _ in range(2):
            responses.extend([
                json.dumps({"subtasks": [
                    {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                    {"description": "Write", "assigned_to": "writer", "depends_on": [0]},
                ]}),
                json.dumps({"summary": "Data about services and caching", "confidence": 0.9}),
                "# Report\n\nThe system uses caching and async patterns for database optimization.",
                json.dumps({"approved": True, "confidence": 0.85, "issues": [], "suggestions": [], "reasoning": "OK"}),
            ])

        provider = MockProvider(responses=responses)
        engine = EvaluationEngine(provider=provider)

        tasks = [
            EvalTask(
                id="bench-1",
                description="Analyze performance patterns",
                expected_keywords=["caching", "async"],
            ),
            EvalTask(
                id="bench-2",
                description="Write about database optimization",
                expected_keywords=["database", "optimization"],
            ),
        ]

        report = await engine.run_benchmark(tasks)

        assert report.total_tasks == 2
        assert report.tasks_completed == 2
        assert report.completion_rate == 1.0
        assert report.avg_iterations > 0
        assert report.avg_latency_seconds > 0

    @pytest.mark.asyncio
    async def test_benchmark_report_summary(self):
        responses = [
            json.dumps({"subtasks": [
                {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                {"description": "Write", "assigned_to": "writer", "depends_on": [0]},
            ]}),
            json.dumps({"summary": "FastAPI setup guide", "confidence": 0.9}),
            "# Setup Guide\n\nInstall FastAPI with Docker. Set up environment variables.",
            json.dumps({"approved": True, "confidence": 0.92, "issues": [], "suggestions": [], "reasoning": "Good."}),
        ]
        provider = MockProvider(responses=responses)
        engine = EvaluationEngine(provider=provider)

        tasks = [EvalTask(
            id="summary-test",
            description="Create a setup guide",
            expected_keywords=["FastAPI", "Docker", "environment"],
        )]

        report = await engine.run_benchmark(tasks)
        summary = report.summary()

        assert "total_tasks" in summary
        assert "completion_rate" in summary
        assert "pass_rate" in summary
        assert "avg_iterations" in summary


class TestDemoBenchmarkTasks:
    def test_demo_tasks_are_defined(self):
        assert len(DEMO_BENCHMARK_TASKS) >= 4

    def test_demo_tasks_have_keywords(self):
        for task in DEMO_BENCHMARK_TASKS:
            assert len(task.expected_keywords) > 0
            assert task.id
            assert task.description
