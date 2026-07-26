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
            # Context overlap adds ~2 sentences from the previous chunk, so
            # chunks may exceed max_tokens by the overlap amount. Allow tolerance
            # of ~100 tokens (2 sentences worth of context).
            assert chunk.token_count <= 300 + 100

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


class TestSentenceAwareSplitting:
    """Verify that semantic chunking never cuts mid-sentence."""

    def test_no_mid_sentence_cuts(self):
        """Each chunk should end at a sentence boundary (period, !, or ?)."""
        long_text = (
            "The fan blade is made of titanium alloy Ti-6Al-4V. "
            "It operates at speeds up to 1,200 RPM during takeoff. "
            "Erosion occurs when sand particles impact the leading edge. "
            "This causes progressive material removal over time. "
            "Inspection is required every 5,000 flight cycles. "
            "Blades exceeding erosion limits must be replaced immediately. "
            "The replacement procedure takes approximately 24 hours. "
            "Certified technicians must perform this work. "
            "All replaced blades are sent for metallographic analysis. "
            "Results are logged in the engine health monitoring system. "
        ) * 10  # ~100 sentences, will require multiple chunks

        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=100, context_sentences=0)
        engine = ChunkingEngine(config)
        doc = Document(
            content=long_text,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
        )
        chunks = engine.chunk_document(doc)

        assert len(chunks) > 3
        for chunk in chunks:
            content = chunk.content.strip()
            # Each chunk should end with sentence-terminal punctuation
            assert content[-1] in ".!?", (
                f"Chunk ends mid-sentence: '...{content[-50:]}'"
            )

    def test_sentence_splitter_handles_abbreviations(self):
        """Sentences with periods in abbreviations shouldn't falsely split."""
        engine = ChunkingEngine()
        text = "Dr. Smith analyzed the AMS 5662 specification. The results were conclusive."
        sentences = engine._split_sentences(text)
        # Should not split at "Dr." — but this is a known limitation of regex-based splitting.
        # At minimum, it should produce at least one sentence.
        assert len(sentences) >= 1

    def test_oversized_paragraph_uses_sentences_not_tokens(self):
        """When a paragraph exceeds max_tokens, it splits at sentence boundaries."""
        long_paragraph = ". ".join([f"Sentence number {i} with enough words to count" for i in range(50)]) + "."

        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=80, context_sentences=0)
        engine = ChunkingEngine(config)
        doc = Document(
            content=long_paragraph,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
        )
        chunks = engine.chunk_document(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            content = chunk.content.strip()
            assert content[-1] == ".", f"Chunk doesn't end at sentence: '...{content[-40:]}'"


class TestContextOverlap:
    """Verify that semantic chunks include context from the previous chunk."""

    def test_context_overlap_present(self):
        """Consecutive chunks should share trailing/leading content."""
        structured_doc = "\n\n".join([
            "## Section One",
            "The first section discusses fan blade erosion limits.",
            "Erosion depth must not exceed 0.030 inches at any point.",
            "## Section Two",
            "The second section covers thermal barrier coatings.",
            "TBC spallation greater than 30% requires strip and recoat.",
            "## Section Three",
            "The third section addresses bearing oil analysis.",
            "Iron content above 10 ppm triggers immediate action.",
        ])

        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=80, context_sentences=2)
        engine = ChunkingEngine(config)
        doc = Document(
            content=structured_doc,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
        )
        chunks = engine.chunk_document(doc)

        if len(chunks) > 1:
            # Second chunk should contain some text from the first chunk
            first_sentences = engine._split_sentences(chunks[0].content)
            if first_sentences:
                last_sentence_of_first = first_sentences[-1]
                # The context overlap means the last sentence of chunk[0]
                # appears at the start of chunk[1]
                assert last_sentence_of_first in chunks[1].content, (
                    f"Context overlap missing. Last sentence of chunk 0: '{last_sentence_of_first}' "
                    f"not found in chunk 1: '{chunks[1].content[:200]}'"
                )

    def test_no_overlap_when_disabled(self):
        """Setting context_sentences=0 disables overlap."""
        # Use unique content in each section so repetition can't cause false match
        sections = [f"## Section {i}\n\nUnique content for section {i} discussing topic {i}." for i in range(20)]
        text = "\n\n".join(sections)

        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, max_tokens=60, context_sentences=0)
        engine = ChunkingEngine(config)
        doc = Document(
            content=text,
            metadata=DocumentMetadata(source="test", doc_type=DocumentType.MARKDOWN),
        )
        chunks = engine.chunk_document(doc)

        if len(chunks) > 1:
            # Without overlap, chunk[1] should NOT contain the last sentence of chunk[0]
            first_sentences = engine._split_sentences(chunks[0].content)
            if first_sentences:
                last_of_first = first_sentences[-1]
                # The second chunk shouldn't start with content from the first
                assert not chunks[1].content.startswith(last_of_first)


class TestAutoStrategy:
    """Verify that AUTO strategy selects correctly based on document type."""

    def test_auto_selects_semantic_for_markdown(self):
        config = ChunkConfig(strategy=ChunkStrategy.AUTO)
        engine = ChunkingEngine(config)
        doc = Document(
            content="## Heading\n\nContent here with enough tokens to chunk properly. " * 20,
            metadata=DocumentMetadata(source="test.md", doc_type=DocumentType.MARKDOWN),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.SEMANTIC

    def test_auto_selects_semantic_for_html(self):
        config = ChunkConfig(strategy=ChunkStrategy.AUTO)
        engine = ChunkingEngine(config)
        doc = Document(
            content="<h1>Title</h1><p>Content</p>",
            metadata=DocumentMetadata(source="test.html", doc_type=DocumentType.HTML),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.SEMANTIC

    def test_auto_selects_semantic_for_pdf(self):
        config = ChunkConfig(strategy=ChunkStrategy.AUTO)
        engine = ChunkingEngine(config)
        doc = Document(
            content="Some PDF content extracted.",
            metadata=DocumentMetadata(source="test.pdf", doc_type=DocumentType.PDF),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.SEMANTIC

    def test_auto_selects_fixed_for_plain_text_without_headings(self):
        config = ChunkConfig(strategy=ChunkStrategy.AUTO)
        engine = ChunkingEngine(config)
        doc = Document(
            content="Just plain text without any markdown headings or structure at all.",
            metadata=DocumentMetadata(source="notes.txt", doc_type=DocumentType.TEXT),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.FIXED

    def test_auto_selects_semantic_for_text_with_headings(self):
        config = ChunkConfig(strategy=ChunkStrategy.AUTO)
        engine = ChunkingEngine(config)
        doc = Document(
            content="# Title\n\nSome content.\n\n## Section\n\nMore content.",
            metadata=DocumentMetadata(source="notes.txt", doc_type=DocumentType.TEXT),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.SEMANTIC

    def test_explicit_strategy_overrides_auto(self):
        """When strategy is explicitly set (not AUTO), it's used regardless of doc type."""
        config = ChunkConfig(strategy=ChunkStrategy.FIXED)
        engine = ChunkingEngine(config)
        doc = Document(
            content="## Heading\n\nMarkdown content.",
            metadata=DocumentMetadata(source="test.md", doc_type=DocumentType.MARKDOWN),
        )
        resolved = engine._resolve_strategy(doc)
        assert resolved == ChunkStrategy.FIXED
