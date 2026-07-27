"""
Knowledge graph engine with pluggable backends.

Defines an abstract interface (KnowledgeGraphBackend) and provides a
NetworkX implementation for local development. A Neo4j implementation
can be added later without changing any code that uses the graph.

Design principles:
- All queries return KGQueryResult with provenance (traceability)
- The graph supports both typed queries (by entity/relation type) and
  structural queries (paths, neighborhoods, subgraphs)
- Thread-safe for read operations (writes should be serialized)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx
import structlog

from lattice.graph.models import Entity, KGQueryResult, KGStats, Relation

logger = structlog.get_logger()


class KnowledgeGraphBackend(ABC):
    """Abstract interface for knowledge graph backends."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def add_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def remove_entity(self, entity_id: str) -> None: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]: ...

    @abstractmethod
    def get_all_entities(self) -> list[Entity]: ...

    @abstractmethod
    def get_entities_by_type(self, entity_type: str) -> list[Entity]: ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both",
    ) -> KGQueryResult: ...

    @abstractmethod
    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> KGQueryResult: ...

    @abstractmethod
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> list[Entity]: ...

    @abstractmethod
    def get_stats(self) -> KGStats: ...

    @abstractmethod
    def get_subgraph(self, center_id: str, depth: int = 2) -> KGQueryResult: ...


class NetworkXBackend(KnowledgeGraphBackend):
    """In-memory knowledge graph using NetworkX.

    Zero external dependencies — works anywhere without Docker or servers.
    Suitable for development, testing, and demos with up to ~100K nodes.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._entities: dict[str, Entity] = {}

    @property
    def num_entities(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_relations(self) -> int:
        return self._graph.number_of_edges()

    def add_entity(self, entity: Entity) -> None:
        self._graph.add_node(
            entity.id,
            entity_type=entity.entity_type,
            name=entity.name,
            properties=entity.properties,
        )
        self._entities[entity.id] = entity

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self._graph:
            self._graph.remove_node(entity_id)
        self._entities.pop(entity_id, None)

    def get_all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self._graph:
            logger.warning("relation_source_missing", source_id=relation.source_id)
            return
        if relation.target_id not in self._graph:
            logger.warning("relation_target_missing", target_id=relation.target_id)
            return

        self._graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            relation_type=relation.relation_type,
            properties=relation.properties,
            confidence=relation.confidence,
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both",
    ) -> KGQueryResult:
        if entity_id not in self._graph:
            return KGQueryResult()

        result = KGQueryResult()
        seen_entities: set[str] = set()

        if direction in ("outgoing", "both"):
            for _, target, _, edge_data in self._graph.out_edges(entity_id, data=True, keys=True):
                if relation_type and edge_data.get("relation_type") != relation_type:
                    continue
                if target not in seen_entities:
                    entity = self._entities.get(target)
                    if entity:
                        result.entities.append(entity)
                        seen_entities.add(target)
                result.relations.append(
                    Relation(
                        source_id=entity_id,
                        target_id=target,
                        relation_type=edge_data.get("relation_type", ""),
                        properties=edge_data.get("properties", {}),
                        confidence=edge_data.get("confidence", 1.0),
                    )
                )

        if direction in ("incoming", "both"):
            for source, _, _, edge_data in self._graph.in_edges(entity_id, data=True, keys=True):
                if relation_type and edge_data.get("relation_type") != relation_type:
                    continue
                if source not in seen_entities:
                    entity = self._entities.get(source)
                    if entity:
                        result.entities.append(entity)
                        seen_entities.add(source)
                result.relations.append(
                    Relation(
                        source_id=source,
                        target_id=entity_id,
                        relation_type=edge_data.get("relation_type", ""),
                        properties=edge_data.get("properties", {}),
                        confidence=edge_data.get("confidence", 1.0),
                    )
                )

        return result

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> KGQueryResult:
        if source_id not in self._graph or target_id not in self._graph:
            return KGQueryResult()

        try:
            undirected = self._graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            if len(path) - 1 > max_depth:
                return KGQueryResult()

            result = KGQueryResult(paths=[path])
            for node_id in path:
                entity = self._entities.get(node_id)
                if entity:
                    result.entities.append(entity)
            return result
        except nx.NetworkXNoPath:
            return KGQueryResult()

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> list[Entity]:
        """Simple substring search over entity names and properties.

        For production, this would be replaced by a proper full-text index.
        """
        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            # Search in name
            if query_lower in entity.name.lower():
                results.append(entity)
                continue

            # Search in properties
            for value in entity.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(entity)
                    break

        return results

    def get_subgraph(self, center_id: str, depth: int = 2) -> KGQueryResult:
        """Extract a subgraph around a node up to N hops."""
        if center_id not in self._graph:
            return KGQueryResult()

        visited: set[str] = set()
        frontier = {center_id}
        result = KGQueryResult()

        for _ in range(depth + 1):
            if not frontier:
                break
            next_frontier: set[str] = set()

            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                entity = self._entities.get(node_id)
                if entity:
                    result.entities.append(entity)

                for _, target, _, edge_data in self._graph.out_edges(
                    node_id, data=True, keys=True
                ):
                    result.relations.append(
                        Relation(
                            source_id=node_id,
                            target_id=target,
                            relation_type=edge_data.get("relation_type", ""),
                            properties=edge_data.get("properties", {}),
                            confidence=edge_data.get("confidence", 1.0),
                        )
                    )
                    next_frontier.add(target)

                for source, _, _, edge_data in self._graph.in_edges(
                    node_id, data=True, keys=True
                ):
                    result.relations.append(
                        Relation(
                            source_id=source,
                            target_id=node_id,
                            relation_type=edge_data.get("relation_type", ""),
                            properties=edge_data.get("properties", {}),
                            confidence=edge_data.get("confidence", 1.0),
                        )
                    )
                    next_frontier.add(source)

            frontier = next_frontier - visited

        return result

    def get_stats(self) -> KGStats:
        entity_types: dict[str, int] = {}
        for entity in self._entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        relation_types: dict[str, int] = {}
        for _, _, edge_data in self._graph.edges(data=True):
            rt = edge_data.get("relation_type", "unknown")
            relation_types[rt] = relation_types.get(rt, 0) + 1

        return KGStats(
            total_entities=self.num_entities,
            total_relations=self.num_relations,
            entity_types=entity_types,
            relation_types=relation_types,
        )
