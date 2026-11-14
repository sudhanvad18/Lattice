"""Tests for the graph backend factory and KnowledgeBase backend injection."""

import sys
import types

from lattice.config import GraphBackend, Settings
from lattice.graph.engine import NetworkXBackend
from lattice.graph.factory import create_graph_backend
from lattice.knowledge_base import KnowledgeBase


def test_default_backend_is_networkx():
    backend = create_graph_backend(Settings(graph_backend=GraphBackend.NETWORKX))
    assert isinstance(backend, NetworkXBackend)


def test_neo4j_backend_receives_settings(monkeypatch):
    captured = {}

    class FakeNeo4jBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("lattice.graph.neo4j_backend")
    fake_module.Neo4jBackend = FakeNeo4jBackend
    monkeypatch.setitem(sys.modules, "lattice.graph.neo4j_backend", fake_module)

    settings = Settings(
        graph_backend=GraphBackend.NEO4J,
        neo4j_uri="bolt://example:7687",
        neo4j_user="u",
        neo4j_password="p",
        neo4j_database="lattice",
    )
    backend = create_graph_backend(settings)

    assert isinstance(backend, FakeNeo4jBackend)
    assert captured == {
        "uri": "bolt://example:7687",
        "user": "u",
        "password": "p",
        "database": "lattice",
    }


def test_knowledge_base_uses_injected_backend(tmp_path):
    backend = NetworkXBackend()
    kb = KnowledgeBase(persist_dir=tmp_path, graph_backend=backend)
    assert kb.graph is backend
    kb.close()
