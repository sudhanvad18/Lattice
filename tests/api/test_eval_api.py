"""GET /eval/results serves the latest `lattice eval` run."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from lattice.evaluation import BenchmarkReport, EvalResult, save_report
from lattice.graph.models import Entity
from lattice.inference.mock import MockProvider
from lattice_api import eval_routes
from lattice_api.app import create_app
from lattice_api.errors import api_error
from lattice_api.eval_routes import eval_results_path
from lattice_api.models import EvalStatus
from lattice_api.task_manager import TaskManager


@pytest.fixture
def results_path(tmp_path):
    return tmp_path / "eval" / "latest.json"


@pytest.fixture
def client(results_path):
    eval_routes._status = EvalStatus()
    kb = MagicMock()
    kb.is_empty = False
    kb.stats = {"chunks_stored": 1, "kg_entities": 1}
    kb.source_counts.return_value = {"test.py": 1}
    kb.graph.get_all_entities.return_value = [
        Entity(id="entity", name="Widget", entity_type="class"),
    ]
    kb.graph.get_all_relations.return_value = []
    kb.vector_store.chunk_ids.return_value = ["chunk"]
    app = create_app(task_manager=TaskManager(provider=MockProvider()), knowledge_base=kb)
    app.dependency_overrides[eval_results_path] = lambda: results_path
    with TestClient(app) as test_client:
        yield test_client


def test_no_run_returns_empty_payload(client):
    response = client.get("/eval/results")
    assert response.status_code == 200
    assert response.json() == {
        "run_at": None, "provider": None, "model": None, "summary": None, "results": [], "kb": None,
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


def test_run_rejects_empty_kb(client):
    client.app.state.knowledge_base.is_empty = True
    response = client.post("/eval/run")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "kb_empty"


def test_run_rejects_provider_down(client, monkeypatch):
    monkeypatch.setattr(eval_routes, "require_provider", AsyncMock(
        side_effect=api_error(503, "provider_unavailable", "Start Ollama"),
    ))
    response = client.post("/eval/run", json={"limit": 1})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "provider_unavailable"
    assert client.get("/eval/status").json()["running"] is False


def test_run_saves_results(client, results_path, monkeypatch):
    from lattice_api.app import get_task_manager

    events = MagicMock()
    monkeypatch.setattr(get_task_manager(), "emit", events, raising=False)
    monkeypatch.setattr(eval_routes, "require_provider", AsyncMock())
    provider = MockProvider(responses=[
        '{"subtasks":[{"description":"Research Widget","assigned_to":"researcher"},'
        '{"description":"Explain Widget","assigned_to":"writer","depends_on":[0]}]}',
        'Widget is a class in the knowledge base [chunk].',
        'Widget is a documented class [chunk].',
        '{"approved":true,"confidence":0.9,"issues":[],"suggestions":[],"reasoning":"Grounded"}',
    ])
    monkeypatch.setattr(eval_routes, "get_provider", lambda: provider)
    response = client.post("/eval/run", json={"limit": 1})
    assert response.status_code == 202
    assert response.json()["tasks"] == 1
    for _ in range(100):
        status = client.get("/eval/status").json()
        if not status["running"]:
            break
        time.sleep(0.01)
    assert status["running"] is False
    assert status["error"] is None
    assert status["completed"] == status["total"] == 1
    assert status["run_id"] == response.json()["run_id"]
    assert results_path.exists()
    body = client.get("/eval/results").json()
    assert body["kb"] == {"chunks": 1, "entities": 1, "sources": 1}
    assert body["results"][0]["task_id"] == "kb-widget"
    assert [call.args[1] for call in events.call_args_list] == [
        "eval_started", "eval_progress", "eval_complete",
    ]
    assert events.call_args_list[1].args[2] == {"completed": 1, "total": 1}


def test_run_busy_and_limit_validation(client):
    eval_routes._status = EvalStatus(running=True)
    response = client.post("/eval/run")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "busy"
    assert client.post("/eval/run", json={"limit": 0}).status_code == 422
    assert client.post("/eval/run", json={"limit": 11}).status_code == 422


def test_initial_status(client):
    assert client.get("/eval/status").json() == {
        "running": False, "run_id": None, "started_at": None,
        "completed": 0, "total": 0, "error": None,
    }


def test_failed_run_releases_holder(client, monkeypatch):
    monkeypatch.setattr(eval_routes, "require_provider", AsyncMock())
    monkeypatch.setattr(eval_routes, "get_provider", MockProvider)
    monkeypatch.setattr(eval_routes.EvaluationEngine, "run_benchmark", AsyncMock(
        side_effect=RuntimeError("Evaluation interrupted"),
    ))
    assert client.post("/eval/run").status_code == 202
    for _ in range(100):
        status = client.get("/eval/status").json()
        if not status["running"]:
            break
        time.sleep(0.01)
    assert status["running"] is False
    assert status["error"] == "Evaluation interrupted"
    assert client.post("/eval/run").status_code == 202
