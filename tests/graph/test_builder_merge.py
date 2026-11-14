"""KnowledgeGraphBuilder merges repeated extractions and routes code chunks to the AST extractor."""

from lattice.graph.builder import KnowledgeGraphBuilder
from lattice.graph.engine import NetworkXBackend
from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType

DOC = """# Fan Blade Erosion

## Borescope Inspection

Fan Blade Erosion requires Borescope Inspection per AMS-4928.
"""


def _text_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        content=DOC,
        index=0,
        metadata=DocumentMetadata(source="erosion.md", doc_type=DocumentType.MARKDOWN),
    )


def _code_chunk() -> Chunk:
    ast_nodes = [
        {"type": "class_definition", "name": "BladeMetrics", "start_line": 1, "end_line": 9},
        {"type": "function_definition", "name": "is_critical", "start_line": 3, "end_line": 4},
        {"type": "import_statement", "name": "import math", "start_line": 1, "end_line": 1},
    ]
    return Chunk(
        id="code-1",
        document_id="code-doc",
        content="class BladeMetrics: ...",
        index=0,
        metadata=DocumentMetadata(
            source="examples/blade_metrics.py",
            doc_type=DocumentType.CODE,
            extra={"language": "python", "ast_nodes": ast_nodes},
        ),
    )


def test_reingesting_same_content_does_not_duplicate():
    builder = KnowledgeGraphBuilder(backend=NetworkXBackend())

    first = builder.build_from_chunks([_text_chunk("c1")])
    second = builder.build_from_chunks([_text_chunk("c1")])

    assert first.total_entities > 0
    assert second.total_entities == first.total_entities
    assert second.total_relations == first.total_relations


def test_repeated_entity_from_new_chunk_merges_source_ids():
    builder = KnowledgeGraphBuilder(backend=NetworkXBackend())
    builder.build_from_chunks([_text_chunk("c1")])
    builder.build_from_chunks([_text_chunk("c2")])

    [entity] = builder.backend.search_entities("Borescope Inspection", entity_type="Concept")
    assert entity.source_chunk_ids == ["c1", "c2"]


def test_merge_preserves_community_assignment():
    builder = KnowledgeGraphBuilder(backend=NetworkXBackend())
    builder.build_from_chunks([_text_chunk("c1")])
    [entity] = builder.backend.search_entities("Borescope Inspection", entity_type="Concept")
    builder.backend.set_entity_community(entity.id, 7)

    builder.build_from_chunks([_text_chunk("c2")])

    assert builder.backend.get_entity(entity.id).community == 7


def test_code_chunks_use_ast_extractor():
    builder = KnowledgeGraphBuilder(backend=NetworkXBackend())
    builder.build_from_chunks([_text_chunk("c1"), _code_chunk()])

    types = builder.backend.get_stats().entity_types
    assert types.get("Class") == 1
    assert types.get("Function") == 1
    assert types.get("Module", 0) >= 2
    relation_types = builder.backend.get_stats().relation_types
    assert relation_types.get("CONTAINS", 0) >= 1
    assert relation_types.get("IMPORTS") == 1
