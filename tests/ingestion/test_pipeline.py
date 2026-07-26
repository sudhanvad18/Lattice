"""Tests for the document ingestion pipeline.

Covers:
- File parsing (markdown, text)
- Chunking (both strategies)
- Pipeline integration
- Edge cases (empty docs, very long docs)
"""

from pathlib import Path

import pytest

from lattice.ingestion.chunker import ChunkConfig, ChunkStrategy, ChunkingEngine
from lattice.ingestion.models import Document, DocumentMetadata, DocumentType
from lattice.ingestion.parser import parse_content, parse_file, parse_markdown
from lattice.ingestion.pipeline import IngestionPipeline

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestParsers:
    def test_parse_markdown_extracts_title(self):
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        assert doc.metadata.title == "Turbine Engine Maintenance Guide"
        assert doc.metadata.doc_type == DocumentType.MARKDOWN

    def test_parse_markdown_preserves_content(self):
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        assert "Fan Blade Inspection" in doc.content
        assert "HPT Blade Creep Assessment" in doc.content
        assert len(doc.content) > 1000

    def test_parse_file_routes_correctly(self):
        doc = parse_file(FIXTURES / "sample_maintenance.md")
        assert doc.metadata.doc_type == DocumentType.MARKDOWN

    def test_parse_file_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_file(Path("/nonexistent/file.md"))

    def test_parse_content_inline(self):
        doc = parse_content("Hello world", source="test")
        assert doc.content == "Hello world"
        assert doc.metadata.source == "test"


class TestFixedChunking:
    def test_fixed_chunks_respect_max_tokens(self):
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=100, overlap_tokens=20)
        engine = ChunkingEngine(config)
        doc = Document(
            content="word " * 500,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.TEXT),
        )
        chunks = engine.chunk_document(doc)
        for chunk in chunks:
            assert chunk.token_count <= 100

    def test_fixed_chunks_have_overlap(self):
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=100, overlap_tokens=20)
        engine = ChunkingEngine(config)
        doc = Document(
            content="word " * 500,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.TEXT),
        )
        chunks = engine.chunk_document(doc)
        # With overlap, consecutive chunks share some content
        assert len(chunks) > 5
        # Verify overlap exists by checking content sharing
        for i in range(len(chunks) - 1):
            words_current = set(chunks[i].content.split()[-10:])
            words_next = set(chunks[i + 1].content.split()[:10])
            # Some words should overlap
            assert len(words_current & words_next) > 0

    def test_fixed_chunks_sequential_indices(self):
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=100, overlap_tokens=10)
        engine = ChunkingEngine(config)
        doc = Document(
            content="text " * 300,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.TEXT),
        )
        chunks = engine.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestSemanticChunking:
    def test_semantic_splits_on_headings(self):
        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=200)
        engine = ChunkingEngine(config)
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        chunks = engine.chunk_document(doc)

        # Should produce multiple chunks from a structured document
        assert len(chunks) > 3

    def test_semantic_chunks_within_token_limit(self):
        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=300)
        engine = ChunkingEngine(config)
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        chunks = engine.chunk_document(doc)

        for chunk in chunks:
            assert chunk.token_count <= 300 + 10  # small tolerance for tokenizer edge cases

    def test_semantic_preserves_document_id(self):
        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=200)
        engine = ChunkingEngine(config)
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        chunks = engine.chunk_document(doc)

        for chunk in chunks:
            assert chunk.document_id == doc.id

    def test_semantic_no_empty_chunks(self):
        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=200)
        engine = ChunkingEngine(config)
        doc = parse_markdown(FIXTURES / "sample_maintenance.md")
        chunks = engine.chunk_document(doc)

        for chunk in chunks:
            assert len(chunk.content.strip()) > 0


class TestPipeline:
    def test_ingest_file(self):
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(FIXTURES / "sample_maintenance.md")
        assert len(chunks) > 0
        assert all(c.document_id for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_ingest_text(self):
        pipeline = IngestionPipeline()
        text = "This is a test document.\n\nIt has multiple paragraphs.\n\n" * 20
        chunks = pipeline.ingest_text(text, source="test-input")
        assert len(chunks) > 0
        assert chunks[0].metadata.source == "test-input"

    def test_ingest_directory(self, tmp_path):
        # Create temp files with enough content to exceed min_chunk_tokens
        (tmp_path / "doc1.md").write_text(
            "# Doc 1\n\n" + "This is paragraph content for testing. " * 20
        )
        (tmp_path / "doc2.txt").write_text(
            "Plain text document with enough content to form a chunk. " * 20
        )
        (tmp_path / "ignore.jpg").write_bytes(b"fake image")

        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_directory(tmp_path)
        # Should process .md and .txt but skip .jpg
        assert len(chunks) >= 2

    def test_ingest_with_custom_config(self):
        config = ChunkConfig(max_tokens=100, strategy=ChunkStrategy.FIXED)
        pipeline = IngestionPipeline(chunk_config=config)
        chunks = pipeline.ingest_file(FIXTURES / "sample_maintenance.md")
        for chunk in chunks:
            assert chunk.token_count <= 100

    def test_ingest_with_processor(self):
        call_count = {"n": 0}

        def counting_processor(chunks):
            call_count["n"] += 1
            return chunks

        pipeline = IngestionPipeline(processors=[counting_processor])
        pipeline.ingest_file(FIXTURES / "sample_maintenance.md")
        assert call_count["n"] == 1
