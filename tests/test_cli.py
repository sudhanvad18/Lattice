"""Exercise the CLI against real local storage and a mocked LLM provider."""

import json
from pathlib import Path

import pytest
import structlog
from chromadb.api.models.CollectionCommon import CollectionCommon
from click.testing import CliRunner

from lattice.agents.state import AgentState, TaskStatus
from lattice.cli import main
from lattice.inference.mock import MockProvider
from lattice.knowledge_base import KnowledgeBase

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "aerospace"


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    previous = structlog.get_config()
    monkeypatch.setattr(CollectionCommon, "_embed", lambda self, input, **kw: [
        [1.0, float(len(text) % 17) / 17, 0.5] for text in input
    ])
    yield
    structlog.configure(**previous)


@pytest.fixture
def invoke(tmp_path):
    runner = CliRunner()

    def command(*args):
        return runner.invoke(main, ["--data-dir", str(tmp_path), *args])

    return command


def test_ingest_and_inspect_corpus(invoke, tmp_path):
    result = invoke("ingest", str(CORPUS))
    assert result.exit_code == 0, result.output
    assert "Ingestion summary" in result.output
    assert "Elapsed" in result.output
    kb = KnowledgeBase(tmp_path)
    before = kb.stats
    assert before["chunks_stored"] > 0
    assert before["kg_entities"] > 0
    assert before["kg_relations"] > 0
    kb.close()
    second = invoke("ingest", str(CORPUS))
    assert second.exit_code == 0, second.output
    assert "Cache hits" in second.output
    cache_row = next(line for line in second.output.splitlines() if "Cache hits" in line)
    assert "6" in cache_row
    kb = KnowledgeBase(tmp_path)
    assert kb.stats["chunks_stored"] == before["chunks_stored"]
    kb.close()
    for args, expected in [
        (("stats",), "NetworkXBackend"),
        (("search", "Fan Blade", "--top-k", "2"), "Matching graph entities"),
        (("graph", "--communities"), "God nodes"),
        (("graph", "--entity", "Fan Blade"), "Neighborhood"),
        (("graph", "--entity", "NoSuchEntity"), "No entity found"),
    ]:
        result = invoke(*args)
        assert result.exit_code == 0, result.output
        assert expected in result.output
    assert invoke("reset").exit_code != 0
    assert invoke("reset", "--yes").exit_code == 0
    kb = KnowledgeBase(tmp_path)
    assert kb.stats["chunks_stored"] == kb.stats["kg_entities"] == 0
    kb.close()


def test_ingest_file_no_cache_and_reset(invoke, tmp_path):
    file = CORPUS / "fan_blade_erosion.md"
    assert invoke("ingest", str(file)).exit_code == 0
    result = invoke("ingest", str(file), "--no-cache", "--reset")
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "Cache hits" in line)
    assert "0" in row
    kb = KnowledgeBase(tmp_path)
    assert not kb._pipeline._cache.is_cached(file)
    assert kb.stats["kg_entities"] > 0
    kb.close()


def test_run_mock_provider(invoke, monkeypatch):
    assert invoke("ingest", str(CORPUS)).exit_code == 0
    provider = MockProvider(responses=[
        json.dumps({"subtasks": [
            {"description": "Research Fan Blade", "assigned_to": "researcher", "depends_on": []},
            {"description": "Write findings", "assigned_to": "writer", "depends_on": [0]},
        ]}),
        json.dumps({"summary": "Inspect fan blades", "confidence": 0.9}),
        "# Findings\n\nFan Blade requires Inspection [chunk_1].",
        json.dumps({"approved": True, "confidence": 0.9, "issues": [], "suggestions": []}),
    ])
    calls = []

    def get_mock(*args, **kwargs):
        calls.append((args, kwargs))
        return provider

    monkeypatch.setattr("lattice.cli.get_provider", get_mock)
    result = invoke("run", "Summarize Fan Blade", "--provider", "openai", "--model", "test-model")
    assert result.exit_code == 0, result.output
    assert "Status: completed" in result.output
    assert "Findings" in result.output
    assert result.output.count("Guardrail report:") == 2
    assert "citation_enforcement" in result.output
    assert calls[0][1] == {"model": "test-model"}
    assert provider.call_count == 4


def test_unreachable_provider(invoke, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("lattice.cli.get_provider", unavailable)
    result = invoke("run", "Summarize maintenance")
    assert result.exit_code == 1
    assert "Check provider availability" in result.output
    assert "connection refused" in result.output
    assert "Traceback" not in result.output


def test_failed_task_exits_one(invoke, monkeypatch):
    monkeypatch.setattr("lattice.cli.get_provider", lambda *a, **kw: MockProvider())

    async def failed(self, task, **kwargs):
        assert kwargs["max_iterations"] == 2
        return AgentState(original_task=task, status=TaskStatus.FAILED)

    monkeypatch.setattr("lattice.cli.OrchestratorAgent.run", failed)
    result = invoke("run", "Fail", "--max-iterations", "2")
    assert result.exit_code == 1
    assert "Status: failed" in result.output


def test_serve(invoke, monkeypatch, tmp_path):
    import os

    calls = []
    # setenv first so monkeypatch restores the original value after serve overwrites it.
    monkeypatch.setenv("LATTICE_DATA_DIR", "unset")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(
        (args, kwargs, os.environ.get("LATTICE_DATA_DIR"))
    ))
    assert invoke("serve", "--host", "0.0.0.0", "--port", "9000").exit_code == 0
    assert calls == [
        (("lattice_api.app:app",), {"host": "0.0.0.0", "port": 9000}, str(tmp_path))
    ]


def test_eval_saves_results(invoke, tmp_path):
    result = invoke("eval", "--provider", "mock", "--task", "doc-api")
    assert result.exit_code == 0, result.output

    saved = json.loads((tmp_path / "eval" / "latest.json").read_text())
    assert saved["provider"] == "mock"
    assert [r["task_id"] for r in saved["results"]] == ["doc-api"]


def test_eval_rejects_unknown_task(invoke):
    result = invoke("eval", "--provider", "mock", "--task", "nope")
    assert result.exit_code == 2
    assert "unknown task id" in result.output


def test_subcommand_help_exits_cleanly(invoke):
    result = invoke("serve", "--help")
    assert result.exit_code == 0
    assert "Error" not in result.output


def test_help_and_empty_queries(invoke):
    assert invoke("--help").exit_code == 0
    assert invoke("stats").exit_code == 0
    assert invoke("search", "blade").exit_code == 0
    assert invoke("graph", "--communities").exit_code == 0
    assert invoke("search", "blade", "--top-k", "0").exit_code == 2
