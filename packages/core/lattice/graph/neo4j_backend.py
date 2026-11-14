"""Neo4j storage for the knowledge graph; the driver is loaded only on use."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from lattice.graph.engine import KnowledgeGraphBackend
from lattice.graph.models import Entity, KGQueryResult, KGStats, Relation

logger = structlog.get_logger()


class Neo4jBackend(KnowledgeGraphBackend):
    """Persist entities and directed, typed relations using the synchronous driver."""

    def __init__(
        self, uri: str, user: str, password: str | None,
        database: str | None = None, driver=None,
    ) -> None:
        if driver is None:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver = driver
        self._database = database
        try:
            self._run_write(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )
            self._run_write("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            self._run_write(
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
            )
        except Exception:
            self.close()
            raise

    def _run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        options = {"database": self._database} if self._database is not None else {}
        with self._driver.session(**options) as session:
            # Materialize before the session closes, also consuming write results.
            return [dict(record) for record in session.run(cypher, **params)]

    def _run_write(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return self._run_read(cypher, **params)

    @staticmethod
    def _entity(data: Any) -> Entity:
        values = dict(data)
        values["properties"] = json.loads(values.pop("properties_json", "{}"))
        return Entity.model_validate(values)

    @staticmethod
    def _relation(record: dict[str, Any]) -> Relation:
        values = dict(record["r"])
        values["properties"] = json.loads(values.pop("properties_json", "{}"))
        return Relation.model_validate({
            **values, "source_id": record["source_id"], "target_id": record["target_id"],
        })

    @staticmethod
    def _relationship_type(value: str) -> str:
        normalized = value.upper()
        if not value.isascii() or not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
            raise ValueError("relation_type must start with a letter and contain only A-Z, 0-9, _")
        return normalized

    @staticmethod
    def _validate_depth(value: int, name: str, minimum: int = 1) -> None:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def add_entity(self, entity: Entity) -> None:
        values = entity.model_dump(mode="json", exclude={"properties"})
        values["properties_json"] = json.dumps(entity.properties)
        self._run_write(
            "MERGE (e:Entity {id: $entity_id}) SET e += $properties",
            entity_id=entity.id, properties=values,
        )

    def add_relation(self, relation: Relation) -> None:
        rel_type = self._relationship_type(relation.relation_type)
        endpoints = self._run_read(
            "OPTIONAL MATCH (s:Entity {id: $source_id}) "
            "OPTIONAL MATCH (t:Entity {id: $target_id}) "
            "RETURN s IS NOT NULL AS source_exists, t IS NOT NULL AS target_exists",
            source_id=relation.source_id, target_id=relation.target_id,
        )
        if not endpoints or not endpoints[0]["source_exists"]:
            logger.warning("relation_source_missing", source_id=relation.source_id)
            return
        if not endpoints[0]["target_exists"]:
            logger.warning("relation_target_missing", target_id=relation.target_id)
            return
        values = relation.model_dump(exclude={"properties", "source_id", "target_id"})
        values["properties_json"] = json.dumps(relation.properties)
        self._run_write(
            "MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id}) "
            f"MERGE (s)-[r:{rel_type} {{id: $relation_id}}]->(t) SET r += $properties",
            source_id=relation.source_id, target_id=relation.target_id,
            relation_id=relation.id, properties=values,
        )

    def remove_entity(self, entity_id: str) -> None:
        self._run_write("MATCH (e:Entity {id: $entity_id}) DETACH DELETE e", entity_id=entity_id)

    def get_entity(self, entity_id: str) -> Entity | None:
        records = self._run_read(
            "MATCH (e:Entity {id: $entity_id}) RETURN e", entity_id=entity_id,
        )
        return self._entity(records[0]["e"]) if records else None

    def get_all_entities(self) -> list[Entity]:
        return [self._entity(r["e"]) for r in self._run_read("MATCH (e:Entity) RETURN e")]

    def get_all_relations(self) -> list[Relation]:
        return [self._relation(r) for r in self._run_read(
            "MATCH (s:Entity)-[r]->(t:Entity) RETURN r, s.id AS source_id, t.id AS target_id"
        )]

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        return [self._entity(r["e"]) for r in self._run_read(
            "MATCH (e:Entity) WHERE e.entity_type = $entity_type RETURN e",
            entity_type=entity_type,
        )]

    def set_entity_community(self, entity_id: str, community_id: int) -> None:
        self._run_write(
            "MATCH (e:Entity {id: $entity_id}) SET e.community = $community_id",
            entity_id=entity_id, community_id=community_id,
        )

    def get_neighbors(
        self, entity_id: str, relation_type: str | None = None, direction: str = "both",
    ) -> KGQueryResult:
        patterns = {
            "outgoing": "(e)-[r]->(n:Entity)",
            "incoming": "(e)<-[r]-(n:Entity)",
            "both": "(e)-[r]-(n:Entity)",
        }
        if direction not in patterns:
            raise ValueError("direction must be outgoing, incoming, or both")
        records = self._run_read(
            "MATCH (e:Entity {id: $entity_id}) MATCH " + patterns[direction]
            + " WHERE ($relation_type IS NULL OR r.relation_type = $relation_type) "
            "RETURN n, r, startNode(r).id AS source_id, endNode(r).id AS target_id",
            entity_id=entity_id, relation_type=relation_type,
        )
        entities = {r["n"]["id"]: self._entity(r["n"]) for r in records}
        relations = {r["r"]["id"]: self._relation(r) for r in records}
        return KGQueryResult(entities=list(entities.values()), relations=list(relations.values()))

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> KGQueryResult:
        self._validate_depth(max_depth, "max_depth")
        if source_id == target_id:
            entity = self.get_entity(source_id)
            if entity is None:
                return KGQueryResult()
            return KGQueryResult(entities=[entity], paths=[[source_id]])
        records = self._run_read(
            "MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id}), "
            f"p = shortestPath((s)-[*..{max_depth}]-(t)) "
            "WHERE all(n IN nodes(p) WHERE n:Entity) RETURN nodes(p) AS entities",
            source_id=source_id, target_id=target_id,
        )
        if not records:
            return KGQueryResult()
        entities = [self._entity(e) for e in records[0]["entities"]]
        return KGQueryResult(entities=entities, paths=[[e.id for e in entities]])

    def search_entities(self, query: str, entity_type: str | None = None) -> list[Entity]:
        return [self._entity(r["e"]) for r in self._run_read(
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($search_query) "
            "AND ($entity_type IS NULL OR e.entity_type = $entity_type) RETURN e",
            search_query=query, entity_type=entity_type,
        )]

    def get_stats(self) -> KGStats:
        entity_types = {r["entity_type"]: r["count"] for r in self._run_read(
            "MATCH (e:Entity) RETURN e.entity_type AS entity_type, count(*) AS count"
        )}
        relation_types = {r["relation_type"]: r["count"] for r in self._run_read(
            "MATCH (:Entity)-[r]->(:Entity) "
            "RETURN r.relation_type AS relation_type, count(*) AS count"
        )}
        return KGStats(
            total_entities=sum(entity_types.values()), total_relations=sum(relation_types.values()),
            entity_types=entity_types, relation_types=relation_types,
        )

    def get_subgraph(self, center_id: str, depth: int = 2) -> KGQueryResult:
        self._validate_depth(depth, "depth", minimum=0)
        records = self._run_read(
            f"MATCH p = (e:Entity {{id: $center_id}})-[*0..{depth}]-(n:Entity) "
            "WHERE all(node IN nodes(p) WHERE node:Entity) "
            "WITH collect(DISTINCT n) AS entities "
            "UNWIND entities AS s OPTIONAL MATCH (s)-[r]->(t:Entity) WHERE t IN entities "
            "RETURN entities, collect(DISTINCT CASE WHEN r IS NOT NULL THEN "
            "{r: properties(r), source_id: s.id, target_id: t.id} END) AS relations",
            center_id=center_id,
        )
        if not records:
            return KGQueryResult()
        return KGQueryResult(
            entities=[self._entity(e) for e in records[0]["entities"]],
            relations=[self._relation(r) for r in records[0]["relations"]],
        )

    def clear(self) -> None:
        """Delete all Entity nodes and their incident relationships in this database."""
        self._run_write("MATCH (e:Entity) DETACH DELETE e")

    def close(self) -> None:
        self._driver.close()
