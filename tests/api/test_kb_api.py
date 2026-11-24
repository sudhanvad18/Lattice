"""KB management, provider probes, and actionable task failures."""

import time
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from lattice.agents.state import AgentState, TaskStatus
from lattice.inference.health import ProviderHealth
from lattice.inference.mock import MockProvider
from lattice.knowledge_base import KnowledgeBase
from lattice_api.app import create_app
from lattice_api.task_manager import TaskManager


@pytest.fixture
def setup(tmp_path):
    kb = KnowledgeBase(tmp_path / "data")
    app = create_app(knowledge_base=kb)
    with TestClient(app) as client:
        yield kb, app, client
    kb.close()


def wait_for_ingestion(client):
    for _ in range(200):
        status = client.get("/kb/status").json()
        if not status["ingesting"]:
            return status
        time.sleep(0.01)
    pytest.fail("ingestion did not finish")


def test_empty_status(setup):
    kb, _, client = setup
    assert client.get("/kb/status").json() == {
        "ready": False, "chunks": 0, "entities": 0, "relations": 0,
        "communities": 0, "sources": [], "last_ingested_at": None,
        "ingesting": False, "data_dir": str(kb._persist_path),
    }


def test_directory_ingest_and_reset(setup, tmp_path):
    kb, app, client = setup
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("first.md", "second.md"):
        (folder / name).write_text(
            f"# {name}\n\n" + "Lattice manages knowledge graphs and documents. " * 30
        )
    response = client.post("/kb/ingest", json={"path": str(folder)})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = wait_for_ingestion(client)
    assert status["ready"]
    assert [source["source"] for source in status["sources"]] == [
        str(folder / "first.md"), str(folder / "second.md"),
    ]
    assert status["last_ingested_at"]
    reopened = KnowledgeBase(kb._persist_path)
    try:
        assert reopened.last_ingested_at == status["last_ingested_at"]
    finally:
        reopened.close()
    events = [e for e in app.state.task_manager.recent_events() if e["task_id"] == job_id]
    assert [e["event"] for e in events] == ["kb_ingest_started", "kb_ingest_complete"]
    assert events[-1]["data"]["chunks"] == status["chunks"]
    assert client.post("/kb/reset").json()["ready"] is False
    assert not (kb._persist_path / "kb_meta.json").exists()
    assert client.get("/kb/status").json()["last_ingested_at"] is None


def test_bad_path_and_busy(setup, tmp_path, monkeypatch):
    kb, _, client = setup
    response = client.post("/kb/ingest", json={"path": str(tmp_path / "missing")})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "bad_path"
    path = tmp_path / "doc.md"
    path.write_text("# Document\n\n" + "Knowledge graph stores documents and entities. " * 30)
    release = Event()
    original = kb.ingest_file

    def blocked(path):
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(kb, "ingest_file", blocked)
    try:
        assert client.post("/kb/ingest", json={"path": str(path)}).status_code == 202
        for route in ("/kb/ingest", "/kb/reset"):
            response = client.post(route, json={"path": str(path)})
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "busy"
    finally:
        release.set()
    assert wait_for_ingestion(client)["ready"]


def test_task_gating_and_health(setup, monkeypatch):
    kb, _, client = setup
    response = client.post("/tasks", json={"task": "Research"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "kb_empty"
    kb.ingest_text("Lattice manages knowledge graphs." * 30)

    async def unavailable():
        return ProviderHealth("ollama", "test", False, "Start with `ollama serve`.")

    monkeypatch.setattr("lattice_api.errors.check_provider", unavailable)
    monkeypatch.setattr("lattice_api.app.check_provider", unavailable)
    response = client.post("/tasks", json={"task": "Research"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "provider_unavailable"
    assert "ollama serve" in response.json()["detail"]["message"]
    health = client.get("/health").json()
    assert health["provider"] == {
        "name": "ollama", "model": "test", "reachable": False,
        "detail": "Start with `ollama serve`.",
    }
    assert health["providers_configured"] == ["ollama"]



async def test_provider_failure_is_actionable(monkeypatch):
    monkeypatch.setenv("LATTICE_DEFAULT_PROVIDER", "ollama")

    class BrokenProvider(MockProvider):
        async def generate(self, *args, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")

    manager = TaskManager(provider=BrokenProvider())
    state = await manager.submit_task("Research the service")
    await manager._running[state.task_id]
    assert_actionable(manager, state.task_id)


@pytest.mark.parametrize("message", ["All connection attempts failed", "Connection refused"])
async def test_failed_result_is_actionable(monkeypatch, message):
    async def failed(*args, **kwargs):
        return AgentState(original_task="Research", status=TaskStatus.FAILED, errors=[message])

    monkeypatch.setattr("lattice_api.task_manager.OrchestratorAgent.run", failed)
    manager = TaskManager(provider=MockProvider())
    state = await manager.submit_task("Research")
    await manager._running[state.task_id]
    assert_actionable(manager, state.task_id)


async def test_exception_is_actionable(monkeypatch):
    async def failed(*args, **kwargs):
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr("lattice_api.task_manager.OrchestratorAgent.run", failed)
    manager = TaskManager(provider=MockProvider())
    state = await manager.submit_task("Research")
    await manager._running[state.task_id]
    assert_actionable(manager, state.task_id)


def assert_actionable(manager, task_id):
    error = next(event for event in manager.recent_events() if event["event"] == "error")
    assert "ollama serve" in error["data"]["message"]
    assert "All connection attempts failed" not in error["data"]["message"]
    assert "ollama serve" in manager.get_task(task_id).errors[0]
    assert any("ollama serve" in record["summary"]
               for record in manager.audit_trail.to_dicts(task_id))
