"""Tests for edge provenance labels on relations."""

import pytest

from lattice.graph.engine import NetworkXBackend
from lattice.graph.extractor import RuleBasedExtractor
from lattice.graph.models import Entity, Relation
from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType


class TestProvenanceField:
    """Test that Relation model supports provenance field."""

    def test_default_provenance(self):
        r = Relation(source_id="a", target_id="b", relation_type="TEST")
        assert r.provenance == "extracted"

    def test_set_provenance(self):
        r = Relation(source_id="a", target_id="b", relation_type="TEST", provenance="inferred")
        assert r.provenance == "inferred"

    def test_ambiguous_provenance(self):
        r = Relation(source_id="a", target_id="b", relation_type="TEST", provenance="ambiguous")
        assert r.provenance == "ambiguous"


class TestExtractorProvenance:
    """Test that the RuleBasedExtractor sets provenance correctly."""

    def _make_chunk(self, content: str) -> Chunk:
        return Chunk(
            document_id="doc1",
            content=content,
            index=0,
            metadata=DocumentMetadata(source="test.md", doc_type=DocumentType.MARKDOWN),
        )

    def test_pattern_match_provenance_is_extracted(self):
        content = "Turbine Blade requires regular Inspection per AMS-2000."
        chunk = self._make_chunk(content)
        extractor = RuleBasedExtractor()
        result = extractor.extract([chunk])

        for rel in result.relations:
            if rel.relation_type in ("REQUIRES", "REFERENCES", "IS_A", "CAUSES"):
                assert rel.provenance == "extracted", f"{rel.relation_type} should be 'extracted'"

    def test_heading_hierarchy_provenance_is_inferred(self):
        content = "# Engine Systems\n\n## Compressor Module\n\nDetails about the compressor."
        chunk = self._make_chunk(content)
        extractor = RuleBasedExtractor()
        result = extractor.extract([chunk])

        subtopic_rels = [r for r in result.relations if r.relation_type == "HAS_SUBTOPIC"]
        for rel in subtopic_rels:
            assert rel.provenance == "inferred"


class TestBackendProvenance:
    """Test that provenance is stored and retrievable via the backend."""

    def test_provenance_stored_on_edge(self):
        backend = NetworkXBackend()
        e1 = Entity(name="Alpha", entity_type="Concept")
        e2 = Entity(name="Beta", entity_type="Concept")
        backend.add_entity(e1)
        backend.add_entity(e2)

        rel = Relation(
            source_id=e1.id, target_id=e2.id,
            relation_type="RELATES_TO", provenance="inferred",
        )
        backend.add_relation(rel)

        relations = backend.get_all_relations()
        assert len(relations) == 1
        assert relations[0].provenance == "inferred"

    def test_extracted_provenance_default(self):
        backend = NetworkXBackend()
        e1 = Entity(name="A", entity_type="T")
        e2 = Entity(name="B", entity_type="T")
        backend.add_entity(e1)
        backend.add_entity(e2)

        rel = Relation(source_id=e1.id, target_id=e2.id, relation_type="R")
        backend.add_relation(rel)

        relations = backend.get_all_relations()
        assert relations[0].provenance == "extracted"

    def test_multiple_provenance_types(self):
        backend = NetworkXBackend()
        e1 = Entity(name="X", entity_type="T")
        e2 = Entity(name="Y", entity_type="T")
        e3 = Entity(name="Z", entity_type="T")
        for e in [e1, e2, e3]:
            backend.add_entity(e)

        backend.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="R1", provenance="extracted"))
        backend.add_relation(Relation(source_id=e2.id, target_id=e3.id, relation_type="R2", provenance="inferred"))
        backend.add_relation(Relation(source_id=e1.id, target_id=e3.id, relation_type="R3", provenance="ambiguous"))

        rels = backend.get_all_relations()
        provs = {r.provenance for r in rels}
        assert provs == {"extracted", "inferred", "ambiguous"}
