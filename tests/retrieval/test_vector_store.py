"""Tests for the vector store."""

from pathlib import Path

import pytest

from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType
from lattice.ingestion.pipeline import IngestionPipeline
from lattice.retrieval.vector_store import VectorStore

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def vector_store(tmp_path) -> VectorStore:
    """Create a temporary vector store for testing."""
    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_chunks")


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    meta = DocumentMetadata(source="test.md", doc_type=DocumentType.MARKDOWN)
    return [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Fan blade erosion occurs when sand particles impact the leading edge at high velocity.",
            index=0,
            metadata=meta,
            token_count=20,
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            content="Thermal barrier coatings protect turbine blades from extreme temperatures above 2000 degrees.",
            index=1,
            metadata=meta,
            token_count=18,
        ),
        Chunk(
            id="chunk-3",
            document_id="doc-1",
            content="Bearing spalling is detected through spectrometric oil analysis monitoring iron content.",
            index=2,
            metadata=meta,
            token_count=16,
        ),
        Chunk(
            id="chunk-4",
            document_id="doc-2",
            content="Python is a programming language commonly used for machine learning and data science.",
            index=0,
            metadata=DocumentMetadata(source="other.md", doc_type=DocumentType.MARKDOWN),
            token_count=18,
        ),
    ]


class TestVectorStore:
    def test_add_chunks(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        count = vector_store.add_chunks(sample_chunks)
        assert count == 4
        assert vector_store.count == 4

    def test_add_empty_list(self, vector_store: VectorStore):
        count = vector_store.add_chunks([])
        assert count == 0

    def test_search_returns_relevant_results(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("What causes blade erosion?", top_k=2)
        assert len(results) == 2
        # The erosion chunk should rank highest
        assert "erosion" in results[0].content.lower() or "sand" in results[0].content.lower()

    def test_search_similarity_scores(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("fan blade erosion from sand", top_k=4)
        # Results should be ordered by similarity (highest first)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_search_with_document_filter(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("blade", top_k=10, filter_document_id="doc-1")
        for result in results:
            assert result.metadata["document_id"] == "doc-1"

    def test_search_top_k_limits_results(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("engine maintenance", top_k=2)
        assert len(results) <= 2

    def test_delete_document(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)
        assert vector_store.count == 4

        vector_store.delete_document("doc-1")
        assert vector_store.count == 1  # Only doc-2 chunk remains

    def test_reset(self, vector_store: VectorStore, sample_chunks: list[Chunk]):
        vector_store.add_chunks(sample_chunks)
        assert vector_store.count == 4

        vector_store.reset()
        assert vector_store.count == 0

    def test_integration_with_pipeline(self, vector_store: VectorStore):
        """Full pipeline: ingest file → store chunks → search."""
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(FIXTURES / "sample_maintenance.md")

        vector_store.add_chunks(chunks)
        assert vector_store.count == len(chunks)

        # Search for something in the document
        results = vector_store.search("What are the oil analysis limits for iron?", top_k=3)
        assert len(results) > 0
        # Should find the SOAP/oil analysis section
        assert any("iron" in r.content.lower() or "oil" in r.content.lower() for r in results)
