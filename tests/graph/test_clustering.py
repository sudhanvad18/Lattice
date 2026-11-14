"""Tests for Leiden community detection and clustering."""

import pytest

from lattice.graph.clustering import (
    ClusterResult,
    cluster_graph,
    _build_nx_graph,
    _component_partition,
    _find_god_nodes,
    _find_surprising_connections,
)
from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity, Relation


def _build_test_graph() -> NetworkXBackend:
    """Build a small test graph with two clear communities."""
    backend = NetworkXBackend()

    # Community A: tightly connected
    a1 = Entity(name="A1", entity_type="Concept")
    a2 = Entity(name="A2", entity_type="Concept")
    a3 = Entity(name="A3", entity_type="Concept")

    # Community B: tightly connected
    b1 = Entity(name="B1", entity_type="Service")
    b2 = Entity(name="B2", entity_type="Service")
    b3 = Entity(name="B3", entity_type="Service")

    for e in [a1, a2, a3, b1, b2, b3]:
        backend.add_entity(e)

    # Intra-community A edges
    backend.add_relation(Relation(source_id=a1.id, target_id=a2.id, relation_type="RELATES", provenance="extracted"))
    backend.add_relation(Relation(source_id=a2.id, target_id=a3.id, relation_type="RELATES", provenance="extracted"))
    backend.add_relation(Relation(source_id=a1.id, target_id=a3.id, relation_type="RELATES", provenance="extracted"))

    # Intra-community B edges
    backend.add_relation(Relation(source_id=b1.id, target_id=b2.id, relation_type="DEPENDS", provenance="extracted"))
    backend.add_relation(Relation(source_id=b2.id, target_id=b3.id, relation_type="DEPENDS", provenance="extracted"))
    backend.add_relation(Relation(source_id=b1.id, target_id=b3.id, relation_type="DEPENDS", provenance="extracted"))

    # Single cross-community bridge
    backend.add_relation(Relation(source_id=a1.id, target_id=b1.id, relation_type="BRIDGE", provenance="inferred"))

    return backend


class TestClusterResult:
    def test_empty_result(self):
        r = ClusterResult()
        assert r.num_communities == 0
        assert r.god_nodes == []
        assert r.surprising_connections == []

    def test_num_communities(self):
        r = ClusterResult(communities={0: ["a", "b"], 1: ["c"]})
        assert r.num_communities == 2


class TestBuildNxGraph:
    def test_builds_undirected_graph(self):
        backend = _build_test_graph()
        G = _build_nx_graph(backend)

        assert G.number_of_nodes() == 6
        assert G.number_of_edges() >= 6

    def test_provenance_weights(self):
        backend = NetworkXBackend()
        e1 = Entity(name="X", entity_type="T")
        e2 = Entity(name="Y", entity_type="T")
        backend.add_entity(e1)
        backend.add_entity(e2)
        backend.add_relation(Relation(
            source_id=e1.id, target_id=e2.id,
            relation_type="R", provenance="inferred",
        ))

        G = _build_nx_graph(backend)
        weight = G[e1.id][e2.id]["weight"]
        assert weight == 0.5


class TestComponentPartition:
    def test_two_components(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        G.add_edges_from([("d", "e")])

        partition = _component_partition(G)
        assert partition["a"] == partition["b"] == partition["c"]
        assert partition["d"] == partition["e"]
        assert partition["a"] != partition["d"]


class TestClusterGraph:
    def test_assigns_communities(self):
        backend = _build_test_graph()
        result = cluster_graph(backend)

        assert result.num_communities >= 1
        total_assigned = sum(len(members) for members in result.communities.values())
        assert total_assigned == 6

        for e in backend.get_all_entities():
            assert e.community is not None

    def test_god_nodes(self):
        backend = _build_test_graph()
        result = cluster_graph(backend, top_k_gods=2)

        assert len(result.god_nodes) <= 2
        assert len(result.god_nodes) >= 1

    def test_surprising_connections(self):
        backend = _build_test_graph()
        result = cluster_graph(backend)

        # The bridge edge may or may not be surprising depending on partition.
        # But the result should be a list of tuples.
        assert isinstance(result.surprising_connections, list)
        for item in result.surprising_connections:
            assert len(item) == 3  # (source_id, target_id, explanation)

    def test_small_graph(self):
        backend = NetworkXBackend()
        e1 = Entity(name="Solo", entity_type="T")
        backend.add_entity(e1)

        result = cluster_graph(backend)
        assert result.num_communities == 0

    def test_empty_graph(self):
        backend = NetworkXBackend()
        result = cluster_graph(backend)
        assert result.num_communities == 0


class TestFindGodNodes:
    def test_god_node_ranking(self):
        backend = _build_test_graph()
        gods = _find_god_nodes(backend, top_k=3)
        assert len(gods) == 3

    def test_top_k_limit(self):
        backend = _build_test_graph()
        gods = _find_god_nodes(backend, top_k=1)
        assert len(gods) == 1


class TestFindSurprisingConnections:
    def test_cross_community_edges(self):
        backend = _build_test_graph()
        # Assign entities to two communities manually
        entities = backend.get_all_entities()
        for i, e in enumerate(entities):
            backend.set_entity_community(e.id, 0 if i < 3 else 1)

        partition = {e.id: e.community for e in backend.get_all_entities()}
        surprising = _find_surprising_connections(backend, partition)

        # The bridge edge should be detected
        assert len(surprising) >= 1
        assert any("BRIDGE" in s[2] for s in surprising)
