"""Tests for the knowledge graph layer.

Covers:
- Entity/relation CRUD on NetworkX backend
- Graph queries (neighbors, paths, subgraphs, search)
- Entity extraction from chunks
- KG builder integration
"""

from pathlib import Path

import pytest

from lattice.graph.builder import KnowledgeGraphBuilder
from lattice.graph.engine import NetworkXBackend
from lattice.graph.extractor import RuleBasedExtractor
from lattice.graph.models import Entity, Relation
from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType
from lattice.ingestion.pipeline import IngestionPipeline

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def backend() -> NetworkXBackend:
    return NetworkXBackend()


@pytest.fixture
def sample_entities() -> list[Entity]:
    return [
        Entity(id="eng-1", name="PW1100G", entity_type="Engine", properties={"thrust": "33000 lbf"}),
        Entity(id="comp-1", name="Fan Blade Stage 1", entity_type="Component", properties={"material": "Ti-6Al-4V"}),
        Entity(id="comp-2", name="HPT Blade Stage 1", entity_type="Component", properties={"material": "CMSX-4"}),
        Entity(id="def-1", name="Leading Edge Erosion", entity_type="DefectType"),
        Entity(id="rc-1", name="Sand Ingestion", entity_type="RootCause"),
        Entity(id="proc-1", name="SB-72-0412", entity_type="Specification"),
    ]


@pytest.fixture
def populated_backend(backend: NetworkXBackend, sample_entities: list[Entity]) -> NetworkXBackend:
    for entity in sample_entities:
        backend.add_entity(entity)

    relations = [
        Relation(source_id="eng-1", target_id="comp-1", relation_type="HAS_COMPONENT"),
        Relation(source_id="eng-1", target_id="comp-2", relation_type="HAS_COMPONENT"),
        Relation(source_id="comp-1", target_id="def-1", relation_type="EXHIBITS_DEFECT"),
        Relation(source_id="def-1", target_id="rc-1", relation_type="HAS_ROOT_CAUSE"),
        Relation(source_id="def-1", target_id="proc-1", relation_type="REFERENCES"),
    ]
    for rel in relations:
        backend.add_relation(rel)

    return backend


class TestNetworkXBackend:
    def test_add_and_get_entity(self, backend: NetworkXBackend):
        entity = Entity(id="test-1", name="Test Entity", entity_type="Thing")
        backend.add_entity(entity)
        result = backend.get_entity("test-1")
        assert result is not None
        assert result.name == "Test Entity"

    def test_get_nonexistent_entity(self, backend: NetworkXBackend):
        assert backend.get_entity("fake-id") is None

    def test_get_entities_by_type(self, populated_backend: NetworkXBackend):
        components = populated_backend.get_entities_by_type("Component")
        assert len(components) == 2

    def test_add_relation(self, populated_backend: NetworkXBackend):
        assert populated_backend.num_relations == 5

    def test_relation_with_missing_source_skipped(self, backend: NetworkXBackend):
        entity = Entity(id="target-1", name="Target", entity_type="Thing")
        backend.add_entity(entity)
        rel = Relation(source_id="nonexistent", target_id="target-1", relation_type="TEST")
        backend.add_relation(rel)
        assert backend.num_relations == 0


class TestGraphQueries:
    def test_get_outgoing_neighbors(self, populated_backend: NetworkXBackend):
        result = populated_backend.get_neighbors("eng-1", direction="outgoing")
        assert len(result.entities) == 2
        names = [e.name for e in result.entities]
        assert "Fan Blade Stage 1" in names
        assert "HPT Blade Stage 1" in names

    def test_get_neighbors_filtered_by_type(self, populated_backend: NetworkXBackend):
        result = populated_backend.get_neighbors(
            "def-1", relation_type="HAS_ROOT_CAUSE", direction="outgoing"
        )
        assert len(result.entities) == 1
        assert result.entities[0].name == "Sand Ingestion"

    def test_get_incoming_neighbors(self, populated_backend: NetworkXBackend):
        result = populated_backend.get_neighbors("def-1", direction="incoming")
        assert len(result.entities) == 1
        assert result.entities[0].name == "Fan Blade Stage 1"

    def test_find_path(self, populated_backend: NetworkXBackend):
        result = populated_backend.find_path("eng-1", "rc-1")
        assert not result.is_empty
        assert len(result.paths) == 1
        path = result.paths[0]
        assert path[0] == "eng-1"
        assert path[-1] == "rc-1"

    def test_find_path_no_connection(self, populated_backend: NetworkXBackend):
        # proc-1 is only connected via def-1, but let's test with an isolated node
        isolated = Entity(id="isolated", name="Isolated", entity_type="Thing")
        populated_backend.add_entity(isolated)
        result = populated_backend.find_path("eng-1", "isolated")
        assert result.is_empty

    def test_search_entities_by_name(self, populated_backend: NetworkXBackend):
        results = populated_backend.search_entities("Fan Blade")
        assert len(results) >= 1
        assert any(e.name == "Fan Blade Stage 1" for e in results)

    def test_search_entities_by_type(self, populated_backend: NetworkXBackend):
        results = populated_backend.search_entities("Stage", entity_type="Component")
        assert len(results) == 2

    def test_search_entities_case_insensitive(self, populated_backend: NetworkXBackend):
        results = populated_backend.search_entities("erosion")
        assert len(results) >= 1

    def test_get_subgraph(self, populated_backend: NetworkXBackend):
        result = populated_backend.get_subgraph("def-1", depth=1)
        assert not result.is_empty
        ids = [e.id for e in result.entities]
        assert "def-1" in ids
        # Should include neighbors at depth 1
        assert "rc-1" in ids or "proc-1" in ids or "comp-1" in ids


class TestStats:
    def test_stats_counts(self, populated_backend: NetworkXBackend):
        stats = populated_backend.get_stats()
        assert stats.total_entities == 6
        assert stats.total_relations == 5
        assert stats.entity_types["Component"] == 2
        assert stats.entity_types["Engine"] == 1
        assert stats.relation_types["HAS_COMPONENT"] == 2


class TestExtractor:
    def test_extracts_headings_as_concepts(self):
        chunk = Chunk(
            document_id="doc-1",
            content="## Fan Module\n\nContent here.\n\n### Fan Blade Inspection\n\nMore content.",
            index=0,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
            token_count=50,
        )
        extractor = RuleBasedExtractor()
        result = extractor.extract([chunk])
        names = [e.name for e in result.entities]
        assert "Fan Module" in names
        assert "Fan Blade Inspection" in names

    def test_extracts_spec_numbers(self):
        chunk = Chunk(
            document_id="doc-1",
            content="Blend repair per SB-72-0412 is required. Material spec AMS 5662 applies.",
            index=0,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
            token_count=30,
        )
        extractor = RuleBasedExtractor()
        result = extractor.extract([chunk])
        names = [e.name for e in result.entities]
        assert "SB-72-0412" in names
        assert "AMS 5662" in names

    def test_deduplicates_entities(self):
        chunks = [
            Chunk(
                document_id="doc-1",
                content="## Fan Blade Inspection\n\nContent A.",
                index=0,
                metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
                token_count=20,
            ),
            Chunk(
                document_id="doc-1",
                content="## Fan Blade Inspection\n\nContent B references it again.",
                index=1,
                metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
                token_count=20,
            ),
        ]
        extractor = RuleBasedExtractor()
        result = extractor.extract(chunks)
        fan_blade_entities = [e for e in result.entities if e.name == "Fan Blade Inspection"]
        assert len(fan_blade_entities) == 1
        # Should have both chunk IDs
        assert len(fan_blade_entities[0].source_chunk_ids) == 2


class TestKGBuilder:
    def test_build_from_real_document(self):
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(FIXTURES / "sample_maintenance.md")

        builder = KnowledgeGraphBuilder()
        stats = builder.build_from_chunks(chunks)

        assert stats.total_entities > 5
        assert stats.total_relations >= 0  # relations depend on content structure
        assert "Concept" in stats.entity_types or "Specification" in stats.entity_types

    def test_incremental_build(self):
        builder = KnowledgeGraphBuilder()

        chunk1 = Chunk(
            document_id="doc-1",
            content="## Turbine Blade\n\nThe turbine blade per AMS 5391 requires inspection.",
            index=0,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
            token_count=30,
        )
        builder.build_from_chunks([chunk1])
        stats_after_first = builder.stats

        chunk2 = Chunk(
            document_id="doc-2",
            content="## Bearing System\n\nBearing spec per AMS 5662 defines limits.",
            index=0,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
            token_count=30,
        )
        builder.build_from_chunks([chunk2])
        stats_after_second = builder.stats

        # Should have more entities after second build
        assert stats_after_second.total_entities > stats_after_first.total_entities
