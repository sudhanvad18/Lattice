"""
Community detection for the knowledge graph using Leiden algorithm.

Groups entities into communities, identifies high-degree "god nodes",
and finds surprising cross-community connections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from lattice.graph.models import Entity, Relation

logger = structlog.get_logger()


@dataclass
class ClusterResult:
    """Result of community detection on the knowledge graph."""

    communities: dict[int, list[str]] = field(default_factory=dict)
    god_nodes: list[str] = field(default_factory=list)
    surprising_connections: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def num_communities(self) -> int:
        return len(self.communities)


def cluster_graph(backend, top_k_gods: int = 5) -> ClusterResult:
    """Run Leiden community detection on the graph.

    Uses graspologic's Leiden implementation when available,
    falls back to a simple connected-components approach otherwise.
    Sets community IDs on entities via the backend.
    """
    entities = backend.get_all_entities()
    if len(entities) < 2:
        return ClusterResult()

    try:
        import networkx as nx

        G = _build_nx_graph(backend)
        partition = _leiden_partition(G)
    except Exception as e:
        logger.debug("leiden_fallback_to_components", reason=str(e))
        import networkx as nx
        G = _build_nx_graph(backend)
        partition = _component_partition(G)

    communities: dict[int, list[str]] = {}
    for node_id, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node_id)
        backend.set_entity_community(node_id, comm_id)

    god_nodes = _find_god_nodes(backend, top_k_gods)

    surprising = _find_surprising_connections(backend, partition)

    result = ClusterResult(
        communities=communities,
        god_nodes=god_nodes,
        surprising_connections=surprising,
    )

    logger.info(
        "clustering_complete",
        communities=result.num_communities,
        god_nodes=len(god_nodes),
        surprising=len(surprising),
    )
    return result


def _build_nx_graph(backend) -> Any:
    """Build a plain NetworkX graph from the backend for clustering."""
    import networkx as nx

    G = nx.Graph()
    for e in backend.get_all_entities():
        G.add_node(e.id)

    for rel in backend.get_all_relations():
        weight = {"extracted": 1.0, "inferred": 0.5, "ambiguous": 0.25}.get(
            rel.provenance, 0.5
        )
        if G.has_edge(rel.source_id, rel.target_id):
            G[rel.source_id][rel.target_id]["weight"] += weight
        else:
            G.add_edge(rel.source_id, rel.target_id, weight=weight)

    return G


def _leiden_partition(G) -> dict[str, int]:
    """Run Leiden algorithm via graspologic."""
    from graspologic.partition import leiden

    node_list = list(G.nodes())
    if len(node_list) < 2:
        return {n: 0 for n in node_list}

    partition_map = leiden(G, resolution=1.0)
    return {str(k): int(v) for k, v in partition_map.items()}


def _component_partition(G) -> dict[str, int]:
    """Fallback: use connected components as communities."""
    import networkx as nx

    partition = {}
    for i, component in enumerate(nx.connected_components(G)):
        for node in component:
            partition[node] = i
    return partition


def _find_god_nodes(backend, top_k: int) -> list[str]:
    """Find the highest-degree entities (most connected)."""
    degree_map: dict[str, int] = {}
    for rel in backend.get_all_relations():
        degree_map[rel.source_id] = degree_map.get(rel.source_id, 0) + 1
        degree_map[rel.target_id] = degree_map.get(rel.target_id, 0) + 1

    sorted_nodes = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in sorted_nodes[:top_k]]


def _find_surprising_connections(backend, partition: dict[str, int]) -> list[tuple[str, str, str]]:
    """Find edges that cross community boundaries."""
    surprising: list[tuple[str, str, str]] = []
    entities_by_id = {e.id: e for e in backend.get_all_entities()}

    for rel in backend.get_all_relations():
        src_comm = partition.get(rel.source_id)
        tgt_comm = partition.get(rel.target_id)
        if src_comm is not None and tgt_comm is not None and src_comm != tgt_comm:
            src_name = entities_by_id.get(rel.source_id, Entity(name="?", entity_type="?")).name
            tgt_name = entities_by_id.get(rel.target_id, Entity(name="?", entity_type="?")).name
            explanation = (
                f"{src_name} (community {src_comm}) "
                f"→ {tgt_name} (community {tgt_comm}) "
                f"via {rel.relation_type}"
            )
            surprising.append((rel.source_id, rel.target_id, explanation))

    return surprising
