"""KB facts determine benchmark tasks and citation groundedness."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.agents.state import AgentRole, AgentState, Artifact, TaskStatus
from lattice.evaluation import (
    EvalTask,
    EvaluationEngine,
    generate_kb_tasks,
    load_report,
    save_report,
)
from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity, Relation
from lattice.inference.mock import MockProvider


def make_kb():
    graph = NetworkXBackend()
    for name, community in [("Alpha", 1), ("Bravo", 1), ("Charlie", 2),
                            ("Delta", 2), ("len", 1), ("self", 1), ("xy", 1)]:
        graph.add_entity(Entity(id=name, name=name, entity_type="class", community=community))
    for source, target in [("Alpha", "Bravo"), ("Alpha", "Charlie"), ("Charlie", "Delta"),
                           ("Alpha", "len"), ("len", "self"), ("self", "xy")]:
        graph.add_relation(Relation(source_id=source, target_id=target, relation_type="calls"))
    store = MagicMock()
    store.chunk_ids.return_value = ["chunk-1"]
    return SimpleNamespace(graph=graph, vector_store=store)


def test_generation_deterministic_and_skips_generics():
    kb = make_kb()
    tasks = generate_kb_tasks(kb, limit=3)
    assert tasks == generate_kb_tasks(kb, limit=3)
    assert len(tasks) == 3
    assert tasks[0].id == "kb-alpha"
    assert tasks[1].id == "kb-charlie"
    assert tasks[-1].category == "cross-community"
    for task in tasks:
        assert not {"len", "self", "xy"}.intersection(task.expected_keywords)
    assert tasks[0].category == "class"
    assert generate_kb_tasks(kb, limit=0) == []


def test_generation_independent_of_storage_order():
    kb = make_kb()
    reversed_graph = MagicMock()
    reversed_graph.get_all_entities.return_value = list(reversed(kb.graph.get_all_entities()))
    reversed_graph.get_all_relations.return_value = list(reversed(kb.graph.get_all_relations()))
    assert generate_kb_tasks(kb) == generate_kb_tasks(SimpleNamespace(graph=reversed_graph))


def test_empty_graph_and_duplicate_slugs():
    graph = NetworkXBackend()
    kb = SimpleNamespace(graph=graph)
    assert generate_kb_tasks(kb) == []
    for eid, name in [("a", "Foo.Bar"), ("b", "Foo Bar"), ("c", "Foo Bar")]:
        graph.add_entity(Entity(id=eid, name=name, entity_type="class"))
    tasks = generate_kb_tasks(kb)
    assert len({task.id for task in tasks}) == len(tasks) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("citations,expected", [([], 0.0), (["missing"], 0.0),
                                                 (["chunk-1", "Alpha", "missing"], 2 / 3)])
async def test_groundedness_uses_final_artifact(monkeypatch, tmp_path, citations, expected):
    kb = make_kb()
    state = AgentState(original_task="explain Alpha", status=TaskStatus.COMPLETED, artifacts=[
        Artifact(artifact_type="research", source_agent=AgentRole.RESEARCHER,
                 content="Alpha", citations=["chunk-1"]),
        Artifact(artifact_type="document", source_agent=AgentRole.WRITER,
                 content="Alpha", citations=citations),
    ])
    orchestrator = MagicMock(return_value=SimpleNamespace(run=AsyncMock(return_value=state)))
    researcher = MagicMock()
    monkeypatch.setattr("lattice.evaluation.engine.OrchestratorAgent", orchestrator)
    monkeypatch.setattr("lattice.evaluation.engine.ResearcherAgent", researcher)
    progress = AsyncMock()
    report = await EvaluationEngine(MockProvider(), kb).run_benchmark(
        [EvalTask(id="kb-alpha", description="Alpha", category="class")], on_progress=progress,
    )
    assert report.results[0].groundedness == pytest.approx(expected)
    assert report.avg_groundedness == pytest.approx(expected)
    researcher.assert_called_once_with(provider=orchestrator.call_args.kwargs["provider"],
                                      vector_store=kb.vector_store, kg_backend=kb.graph)
    progress.assert_awaited_once_with(1, 1)
    saved = save_report(report, tmp_path / "report.json", kb_snapshot={"chunks": 1})
    assert saved["results"][0]["category"] == "class"
    assert saved["kb"] == {"chunks": 1}
    assert saved["summary"]["avg_groundedness"] == pytest.approx(expected)


def test_old_report_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"summary": {}, "results": [{"task_id": "old"}]}))
    report = load_report(path)
    assert report["kb"] is None
    assert report["summary"]["avg_groundedness"] == 0.0
    assert report["results"][0]["groundedness"] == 0.0
    assert report["results"][0]["category"] is None
