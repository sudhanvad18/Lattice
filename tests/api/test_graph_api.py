"""Knowledge graph wire contracts, KB lifecycle, and task retrieval wiring."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lattice.agents.state import AgentState
from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity, Relation
from lattice.inference.mock import MockProvider
from lattice.knowledge_base import KnowledgeBase
from lattice_api.app import create_app
from lattice_api.task_manager import TaskManager


@pytest.fixture
def kb(tmp_path, monkeypatch):
    # Replace only the external embedding model; retain the real persistent store.
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    monkeypatch.setattr(
        ONNXMiniLM_L6_V2, "__call__", lambda self, input: [[0.1] * 384 for _ in input]
    )
    knowledge_base = KnowledgeBase(persist_dir=tmp_path, graph_backend=NetworkXBackend())
    yield knowledge_base
    knowledge_base.close()


@pytest.fixture
def client(kb):
    manager = TaskManager(provider=MockProvider())
    with TestClient(create_app(task_manager=manager, knowledge_base=kb)) as test_client:
        yield test_client


@pytest.fixture
def seeded(kb):
    for node_id, name, entity_type in [
        ("a", "Alpha Service", "Service"),
        ("b", "Beta Service", "Service"),
        ("c", "Alpha Person", "Person"),
    ]:
        kb.graph.add_entity(Entity(id=node_id, name=name, entity_type=entity_type))
    kb.graph.add_relation(Relation(
        id="ab", source_id="a", target_id="b", relation_type="CALLS",
        provenance="inferred", confidence=0.8,
    ))
    kb.graph.add_relation(Relation(
        id="ca", source_id="c", target_id="a", relation_type="OWNS",
        provenance="ambiguous", confidence=0.4,
    ))
    return kb


def test_empty_graph(client):
    assert client.get("/kg/graph").json() == {"nodes": [], "edges": [], "truncated": False}
    assert client.get("/kg/entities").json() == []
    assert client.get("/kg/communities").json() == {
        "communities": [], "god_nodes": [], "surprising_connections": [],
    }


def test_graph_contract_and_limits(client, seeded):
    response = client.get("/kg/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["truncated"] is False
    assert data["nodes"][0] == {
        "id": "a", "name": "Alpha Service", "entity_type": "Service",
        "description": None, "community": None,
    }
    assert next(edge for edge in data["edges"] if edge["id"] == "ab") == {
        "id": "ab", "source": "a", "target": "b", "relation_type": "CALLS",
        "provenance": "inferred", "confidence": 0.8,
    }
    limited = client.get("/kg/graph?limit=2").json()
    assert limited["truncated"] is True
    assert len(limited["nodes"]) == 2
    assert [edge["id"] for edge in limited["edges"]] == ["ab"]
    assert client.get("/kg/graph?limit=1").json()["edges"] == []
    assert client.get("/kg/graph?limit=5000").status_code == 200
    for limit in (0, -1, 5001):
        assert client.get(f"/kg/graph?limit={limit}").status_code == 422


def test_entity_search(client, seeded):
    assert len(client.get("/kg/entities").json()) == 3
    assert len(client.get("/kg/entities?limit=1").json()) == 1
    assert len(client.get("/kg/entities?q=alpha").json()) == 2
    assert len(client.get("/kg/entities?entity_type=Service").json()) == 2
    assert [node["id"] for node in client.get(
        "/kg/entities?q=alpha&entity_type=Person"
    ).json()] == ["c"]
    assert client.get("/kg/entities?q=missing").json() == []
    assert client.get("/kg/entities?entity_type=Missing").json() == []
    assert client.get("/kg/entities?limit=0").status_code == 422


def test_entity_detail_preserves_edges(client, seeded):
    data = client.get("/kg/entities/a").json()
    assert data["entity"]["id"] == "a"
    assert {node["id"] for node in data["neighbors"]} == {"b", "c"}
    assert {edge["id"] for edge in data["relations"]} == {"ab", "ca"}
    assert {edge["provenance"] for edge in data["relations"]} == {"inferred", "ambiguous"}
    assert client.get("/kg/entities/missing").status_code == 404
    seeded.graph.add_entity(Entity(id="isolated", name="Isolated", entity_type="Service"))
    detail = client.get("/kg/entities/isolated").json()
    assert detail["neighbors"] == detail["relations"] == []


def test_stats_and_health(client, seeded):
    assert client.get("/kg/stats").json() == {
        **seeded.stats, "backend": "NetworkXBackend",
    }
    assert client.get("/health").json()["knowledge_base"] == {
        "backend": "NetworkXBackend", "entities": 3, "chunks": 0,
    }


def test_communities(client, seeded):
    data = client.get("/kg/communities").json()
    members = [member for community in data["communities"] for member in community["members"]]
    assert {member["id"] for member in members} == {"a", "b", "c"}
    for community in data["communities"]:
        assert isinstance(community["id"], int)
        assert community["size"] == len(community["members"])
        assert all(set(member) == {"id", "name"} for member in community["members"])
    assert data["god_nodes"][0] == {"id": "a", "name": "Alpha Service"}
    # Cover the cross-community serialization independently of the optional Leiden package.
    result = SimpleNamespace(
        communities={0: ["a", "b"], 1: ["c"]}, god_nodes=["a"],
        surprising_connections=[("c", "a", "Cross-community ownership")],
    )
    with patch("lattice.graph.clustering.cluster_graph", return_value=result) as cluster:
        response = client.get("/kg/communities").json()
    cluster.assert_called_once_with(seeded.graph)
    assert response["surprising_connections"] == [{
        "source": "c", "target": "a", "explanation": "Cross-community ownership",
    }]


def test_ingest(client, kb):
    before = kb.stats
    text = (
        "## Authentication\n\nOAuth 2.0 follows RFC 6749. "
        "Tokens follow RFC 7519 and security follows NIST 800-53.\n\n"
    ) * 5
    response = client.post("/kg/ingest", json={"text": text, "source": "test-doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] > 0
    assert data["stats"]["chunks_stored"] > before["chunks_stored"]
    assert data["stats"]["kg_entities"] > before["kg_entities"]
    assert client.get("/kg/graph").json()["nodes"]
    assert client.post("/kg/ingest", json={"text": ""}).status_code == 422
    assert client.post("/kg/ingest", json={}).status_code == 422


def test_ingest_runs_in_worker_thread(client, kb):
    thread_ids = []

    def ingest(text, source):
        thread_ids.append(threading.get_ident())
        assert text == "hello" and source == "api"
        return 1

    with patch.object(kb, "ingest_text", side_effect=ingest):
        assert client.post("/kg/ingest", json={"text": "hello"}).json()["chunks"] == 1
    assert thread_ids and thread_ids[0] != client.portal.call(threading.get_ident)


@pytest.mark.asyncio
async def test_task_manager_uses_kb_stores(kb):
    provider = MockProvider()
    manager = TaskManager(provider=provider, knowledge_base=kb)
    state = AgentState(original_task="Research")
    manager._tasks[state.task_id] = state
    with (
        patch("lattice_api.task_manager.ResearcherAgent") as researcher,
        patch("lattice_api.task_manager.OrchestratorAgent") as orchestrator,
    ):
        orchestrator.return_value.run = AsyncMock(return_value=state)
        await manager._execute_task(state.task_id, "never")
    researcher.assert_called_once_with(
        provider=provider, tool_registry=None,
        vector_store=kb.vector_store, kg_backend=kb.graph,
    )


def test_unavailable_kb():
    client = TestClient(create_app(task_manager=TaskManager(provider=MockProvider())))
    assert client.get("/kg/graph").status_code == 503
    assert client.get("/health").json()["knowledge_base"] is None


def test_owned_kb_lifecycle(tmp_path):
    kb = MagicMock()
    with (
        patch("lattice.knowledge_base.KnowledgeBase", return_value=kb) as constructor,
        patch("lattice.config.get_settings", return_value=SimpleNamespace(data_dir=tmp_path)),
        patch("lattice_api.app.TaskManager") as manager,
    ):
        app = create_app()
        constructor.assert_not_called()
        with TestClient(app):
            constructor.assert_called_once_with(persist_dir=tmp_path)
            manager.assert_called_once_with(knowledge_base=kb)
            assert app.state.knowledge_base is kb
            kb.close.assert_not_called()
        kb.close.assert_called_once_with()
        assert app.state.knowledge_base is None


def test_injected_kb_is_caller_owned(kb):
    with patch.object(kb, "close") as close:
        app = create_app(task_manager=TaskManager(provider=MockProvider()), knowledge_base=kb)
        with TestClient(app):
            pass
        close.assert_not_called()
