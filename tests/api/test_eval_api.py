"""GET /eval/results serves the latest `lattice eval` run."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from lattice.evaluation import BenchmarkReport, EvalResult, save_report
from lattice.inference.mock import MockProvider
from lattice_api.app import create_app
from lattice_api.eval_routes import eval_results_path
from lattice_api.task_manager import TaskManager


@pytest.fixture
def results_path(tmp_path):
    return tmp_path / "eval" / "latest.json"


@pytest.fixture
def client(results_path):
    app = create_app(task_manager=TaskManager(provider=MockProvider()), knowledge_base=MagicMock())
    app.dependency_overrides[eval_results_path] = lambda: results_path
    with TestClient(app) as test_client:
        yield test_client


def test_no_run_returns_empty_payload(client):
    response = client.get("/eval/results")
    assert response.status_code == 200
    assert response.json() == {
        "run_at": None, "provider": None, "model": None, "summary": None, "results": [],
    }


def test_corrupt_file_returns_empty_payload(client, results_path):
    results_path.parent.mkdir(parents=True)
    results_path.write_text("{not json")
    assert client.get("/eval/results").json()["results"] == []


def test_saved_run_is_served(client, results_path):
    result = EvalResult(
        task_id="doc-api", task_description="Write API docs", completed=True,
        status="completed", review_approved=True, review_confidence=0.9,
        keywords_found=["GET", "POST"], keywords_missing=["user"],
        iterations_used=2, elapsed_seconds=3.5,
    )
    report = BenchmarkReport(results=[result], total_tasks=1, tasks_completed=1, tasks_passed=1)
    save_report(report, results_path, provider="ollama", model="llama3.1")

    body = client.get("/eval/results").json()

    assert body["provider"] == "ollama"
    assert body["model"] == "llama3.1"
    assert body["run_at"] is not None
    assert body["summary"]["pass_rate"] == 1.0
    [row] = body["results"]
    assert row["task_name"] == "Write API docs"
    assert row["passed"] is True
    assert row["quality_score"] == 0.9
    assert row["keyword_recall"] == pytest.approx(2 / 3)
    assert row["keywords_missing"] == ["user"]
    assert row["iterations"] == 2
