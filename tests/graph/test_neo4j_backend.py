"""Driver-free query and mapping tests, plus an explicitly enabled live smoke test."""

import json
import os
from unittest.mock import patch

import pytest

from lattice.graph.models import Entity, Relation
from lattice.graph.neo4j_backend import Neo4jBackend


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.sessions = []
        self.responses = []
        self.closed = False

    def session(self, **options):
        self.sessions.append(options)
        return FakeSession(self)

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query, **params):
        self.driver.calls.append((query, params))
        return self.driver.responses.pop(0) if self.driver.responses else []


@pytest.fixture
def backend():
    return Neo4jBackend("bolt://unused", "neo4j", None, driver=FakeDriver())


def entity_record(entity):
    values = entity.model_dump(mode="json", exclude={"properties"})
    values["properties_json"] = json.dumps(entity.properties)
    return values


def relation_record(relation):
    values = relation.model_dump(exclude={"source_id", "target_id", "properties"})
    values["properties_json"] = json.dumps(relation.properties)
    return {"r": values, "source_id": relation.source_id, "target_id": relation.target_id}


def test_schema_and_database():
    driver = FakeDriver()
    Neo4jBackend("unused", "user", "password", database="tenant", driver=driver)
    queries = [q for q, _ in driver.calls]
    assert queries == [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    ]
    assert driver.sessions == [{"database": "tenant"}] * 3


def test_lazy_driver_creation():
    driver = FakeDriver()
    with patch("neo4j.GraphDatabase.driver", return_value=driver) as create:
        backend = Neo4jBackend("bolt://host", "user", None)
    create.assert_called_once_with("bolt://host", auth=("user", None))
    backend.close()
    assert driver.closed


def test_entity_roundtrip_and_merge(backend):
    entity = Entity(
        name="O'Reilly", entity_type="Person", community=7, description="Author",
        properties={"nested": {"list": [1, True, None]}}, source_chunk_ids=["chunk"],
    )
    backend.add_entity(entity)
    first = backend._driver.calls[-1]
    backend.add_entity(entity)
    assert backend._driver.calls[-1] == first
    query, params = first
    assert "MERGE (e:Entity {id: $entity_id})" in query
    assert entity.name not in query
    assert json.loads(params["properties"]["properties_json"]) == entity.properties
    backend._driver.responses = [[{"e": params["properties"]}]]
    assert backend.get_entity(entity.id) == entity
    assert backend.get_entity("absent") is None
    assert all(options == {} for options in backend._driver.sessions)


@pytest.mark.parametrize("value", ["WORKS_WITH", "works_with", "A", "A123_"])
def test_relation_merge_and_type_normalization(backend, value):
    relation = Relation(source_id="s", target_id="t", relation_type=value)
    endpoint = [{"source_exists": True, "target_exists": True}]
    for _ in range(2):
        backend._driver.responses = [endpoint, []]
        backend.add_relation(relation)
    query, params = backend._driver.calls[-1]
    assert f"MERGE (s)-[r:{value.upper()} {{id: $relation_id}}]->(t)" in query
    assert params["relation_id"] == relation.id
    assert params["properties"]["relation_type"] == value
    assert backend._driver.calls[-1] == backend._driver.calls[-3]


@pytest.mark.parametrize("value", ["", "1ABC", "_ABC", "A B", "A-B", "A`]->()//", "é", "ß"])
def test_reject_relation_type(backend, value):
    with pytest.raises(ValueError):
        backend.add_relation(Relation(source_id="s", target_id="t", relation_type=value))
    assert len(backend._driver.calls) == 3


@pytest.mark.parametrize("source,target,event,key", [
    (False, False, "relation_source_missing", "source_id"),
    (True, False, "relation_target_missing", "target_id"),
])
def test_missing_endpoints(backend, source, target, event, key):
    backend._driver.responses = [[{"source_exists": source, "target_exists": target}]]
    with patch("lattice.graph.neo4j_backend.logger") as logger:
        backend.add_relation(Relation(source_id="s", target_id="t", relation_type="LINK"))
    logger.warning.assert_called_once_with(event, **{key: "s" if key == "source_id" else "t"})
    assert len(backend._driver.calls) == 4


def test_all_relations_preserves_metadata(backend):
    relation = Relation(
        source_id="s", target_id="t", relation_type="LINK", provenance="inferred",
        confidence=0.6, source_chunk_ids=["chunk"], properties={"nested": {"a": 1}},
    )
    backend._driver.responses = [[relation_record(relation)]]
    assert backend.get_all_relations() == [relation]
    assert "(s:Entity)-[r]->(t:Entity)" in backend._driver.calls[-1][0]


@pytest.mark.parametrize("direction,pattern", [
    ("outgoing", "(e)-[r]->(n:Entity)"),
    ("incoming", "(e)<-[r]-(n:Entity)"),
    ("both", "(e)-[r]-(n:Entity)"),
])
@pytest.mark.parametrize("relation_type", [None, "works_with"])
def test_neighbors(backend, direction, pattern, relation_type):
    neighbor = Entity(id="n", name="Neighbor", entity_type="Person")
    relation = Relation(
        source_id="n", target_id="e", relation_type="works_with", provenance="ambiguous",
    )
    record = {"n": entity_record(neighbor), **relation_record(relation)}
    backend._driver.responses = [[record, record]]
    result = backend.get_neighbors("e", relation_type, direction)
    assert result.entities == [neighbor]
    assert result.relations == [relation]
    query, params = backend._driver.calls[-1]
    assert pattern in query
    assert "r.relation_type = $relation_type" in query
    assert "startNode(r).id AS source_id" in query
    assert params == {"entity_id": "e", "relation_type": relation_type}


def test_invalid_direction(backend):
    with pytest.raises(ValueError):
        backend.get_neighbors("e", direction="sideways")


@pytest.mark.parametrize("depth", [0, -1, True, False, 1.5, "5", "1] DELETE e"])
def test_invalid_max_depth(backend, depth):
    with pytest.raises(ValueError):
        backend.find_path("s", "t", depth)
    assert len(backend._driver.calls) == 3


def test_path_mapping(backend):
    entities = [Entity(id=i, name=i, entity_type="Node") for i in ["s", "n", "t"]]
    backend._driver.responses = [[{"entities": [entity_record(e) for e in entities]}]]
    result = backend.find_path("s", "t", 3)
    assert result.entities == entities
    assert result.paths == [["s", "n", "t"]]
    query, params = backend._driver.calls[-1]
    assert "shortestPath((s)-[*..3]-(t))" in query
    assert params == {"source_id": "s", "target_id": "t"}
    assert backend.find_path("absent", "other").is_empty
    backend._driver.responses = [[{"e": entity_record(entities[0])}]]
    assert backend.find_path("s", "s").paths == [["s"]]
    assert backend.find_path("absent", "absent").is_empty


def test_stats(backend):
    backend._driver.responses = [
        [{"entity_type": "Person", "count": 2}, {"entity_type": "Company", "count": 1}],
        [{"relation_type": "WORKS_AT", "count": 2}],
    ]
    stats = backend.get_stats()
    assert stats.total_entities == 3
    assert stats.total_relations == 2
    assert stats.entity_types == {"Person": 2, "Company": 1}
    assert stats.relation_types == {"WORKS_AT": 2}
    assert backend.get_stats().total_entities == 0


def test_entity_queries_and_search(backend):
    entity = Entity(name="Alice", entity_type="Person")
    for method, args in [
        (backend.get_all_entities, ()),
        (backend.get_entities_by_type, ("Person",)),
        (backend.search_entities, ("ali'", "Person")),
    ]:
        backend._driver.responses = [[{"e": entity_record(entity)}]]
        assert method(*args) == [entity]
    query, params = backend._driver.calls[-1]
    assert "toLower(e.name) CONTAINS toLower($search_query)" in query
    assert params == {"search_query": "ali'", "entity_type": "Person"}
    assert "ali'" not in query


def test_subgraph(backend):
    entity = Entity(id="s", name="Start", entity_type="Node")
    relation = Relation(source_id="s", target_id="s", relation_type="SELF", provenance="inferred")
    backend._driver.responses = [[{
        "entities": [entity_record(entity)], "relations": [relation_record(relation)],
    }]]
    result = backend.get_subgraph("s", 0)
    assert result.entities == [entity]
    assert result.relations == [relation]
    query, params = backend._driver.calls[-1]
    assert "[*0..0]" in query
    assert "t IN entities" in query
    assert params == {"center_id": "s"}
    assert backend.get_subgraph("absent").is_empty
    for depth in [-1, True, "2", 1.5]:
        with pytest.raises(ValueError):
            backend.get_subgraph("s", depth)


def test_mutations_and_close(backend):
    backend.set_entity_community("e", 4)
    assert backend._driver.calls[-1] == (
        "MATCH (e:Entity {id: $entity_id}) SET e.community = $community_id",
        {"entity_id": "e", "community_id": 4},
    )
    backend.remove_entity("e")
    assert "DETACH DELETE e" in backend._driver.calls[-1][0]
    assert backend._driver.calls[-1][1] == {"entity_id": "e"}
    backend.clear()
    assert backend._driver.calls[-1] == ("MATCH (e:Entity) DETACH DELETE e", {})
    backend.close()
    assert backend._driver.closed


@pytest.mark.skipif(
    not os.getenv("LATTICE_TEST_NEO4J_URI"),
    reason="Requires a dedicated Neo4j test database",
)
class TestNeo4jIntegration:
    def test_roundtrip(self):
        from lattice.graph.clustering import cluster_graph

        backend = Neo4jBackend(
            os.environ["LATTICE_TEST_NEO4J_URI"],
            os.getenv("LATTICE_TEST_NEO4J_USER", "neo4j"),
            os.getenv("LATTICE_TEST_NEO4J_PASSWORD"),
        )
        try:
            backend.clear()
            source = Entity(name="Alice", entity_type="Person", properties={"nested": [1, 2]})
            target = Entity(name="Lattice", entity_type="Project")
            relation = Relation(
                source_id=source.id, target_id=target.id,
                relation_type="WORKS_ON", provenance="inferred",
            )
            for entity in [source, target]:
                backend.add_entity(entity)
                backend.add_entity(entity)
            backend.add_relation(relation)
            backend.add_relation(relation)
            assert backend.get_entity(source.id) == source
            assert backend.get_all_relations() == [relation]
            neighbors = backend.get_neighbors(source.id, "WORKS_ON", "outgoing")
            assert neighbors.entity_ids == [target.id]
            assert backend.get_neighbors(target.id, direction="incoming").relations == [relation]
            assert backend.find_path(target.id, source.id).paths == [[target.id, source.id]]
            assert backend.find_path(source.id, source.id).paths == [[source.id]]
            assert set(backend.get_subgraph(source.id, 1).entity_ids) == {source.id, target.id}
            assert backend.get_subgraph(source.id, 0).entity_ids == [source.id]
            assert backend.search_entities("ALI") == [source]
            assert backend.get_entities_by_type("Project") == [target]
            assert backend.get_stats().total_entities == 2
            assert backend.get_stats().total_relations == 1
            assert cluster_graph(backend).num_communities >= 1
            assert backend.get_entity(source.id).community is not None
            backend.remove_entity(target.id)
            assert backend.get_all_relations() == []
        finally:
            try:
                backend.clear()
                assert backend.get_stats().total_entities == 0
            finally:
                backend.close()
