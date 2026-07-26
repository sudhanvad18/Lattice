"""
Document ingestion pipeline.

Orchestrates: parse → chunk → (embed + extract entities)
Embedding and entity extraction are pluggable — they're injected as callbacks
so the pipeline doesn't depend on specific ML models at import time.

Usage:
    pipeline = IngestionPipeline()
    chunks = pipeline.ingest_file(Path("docs/manual.pdf"))
    # or
    chunks = pipeline.ingest_directory(Path("docs/"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import structlog

from lattice.ingestion.chunker import ChunkConfig, ChunkingEngine
from lattice.ingestion.models import Chunk, Document
from lattice.ingestion.parser import parse_content, parse_file

logger = structlog.get_logger()

# Type for optional post-processing hooks
ChunkProcessor = Callable[[list[Chunk]], list[Chunk]]


class IngestionPipeline:
    """Ingests documents through parse → chunk → process stages."""

    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        processors: list[ChunkProcessor] | None = None,
    ) -> None:
        self._chunker = ChunkingEngine(chunk_config)
        self._processors = processors or []

    def ingest_file(self, path: Path) -> list[Chunk]:
        """Ingest a single file: parse → chunk → process."""
        document = parse_file(path)
        return self._process_document(document)

    def ingest_text(self, content: str, source: str = "inline") -> list[Chunk]:
        """Ingest raw text content directly."""
        document = parse_content(content, source=source)
        return self._process_document(document)

    def ingest_directory(
        self,
        directory: Path,
        glob_pattern: str = "**/*",
        extensions: Optional[set[str]] = None,
    ) -> list[Chunk]:
        """Ingest all supported files in a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        valid_extensions = extensions or {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
        all_chunks: list[Chunk] = []
        files_processed = 0
        files_skipped = 0

        for path in sorted(directory.glob(glob_pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in valid_extensions:
                files_skipped += 1
                continue

            try:
                chunks = self.ingest_file(path)
                all_chunks.extend(chunks)
                files_processed += 1
            except Exception as e:
                logger.error("ingestion_failed", path=str(path), error=str(e))
                files_skipped += 1

        logger.info(
            "directory_ingested",
            directory=str(directory),
            files_processed=files_processed,
            files_skipped=files_skipped,
            total_chunks=len(all_chunks),
        )
        return all_chunks

    def _process_document(self, document: Document) -> list[Chunk]:
        """Chunk a document and run post-processors."""
        chunks = self._chunker.chunk_document(document)

        for processor in self._processors:
            chunks = processor(chunks)

        logger.info(
            "document_ingested",
            doc_id=document.id,
            source=document.metadata.source,
            chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )
        return chunks
