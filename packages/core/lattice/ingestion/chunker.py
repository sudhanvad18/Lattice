"""
Chunking engine for splitting documents into retrieval-ready pieces.

Two strategies implemented:
1. Fixed-size: split by token count with configurable overlap
2. Semantic: split at natural boundaries (headings, paragraphs, sentences)

The semantic strategy is preferred for most use cases because it produces
chunks that align with how humans organize information. Fixed-size is a
reliable fallback when documents lack structure.

Key design choice: chunk size is measured in TOKENS (via tiktoken) not
characters. This matters because LLM context windows and embedding models
operate in token-space. A 512-token chunk will consistently fit model limits
regardless of whether it's dense code or sparse prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import tiktoken

from lattice.ingestion.models import Chunk, Document


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


@dataclass
class ChunkConfig:
    """Configuration for the chunking engine."""

    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 50
    encoding_name: str = "cl100k_base"  # GPT-4 / Claude tokenizer


class ChunkingEngine:
    """Splits documents into chunks using the configured strategy."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._encoder = tiktoken.get_encoding(self.config.encoding_name)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using the configured strategy."""
        if self.config.strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(document)
        return self._fixed_chunk(document)

    def _token_count(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoder.encode(text))

    def _fixed_chunk(self, document: Document) -> list[Chunk]:
        """Split by token count with overlap.

        Simple and predictable. Each chunk is exactly max_tokens
        (except the last), with overlap_tokens repeated from the
        previous chunk to maintain context continuity.
        """
        tokens = self._encoder.encode(document.content)
        chunks: list[Chunk] = []
        step = self.config.max_tokens - self.config.overlap_tokens

        i = 0
        index = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.config.max_tokens]
            text = self._encoder.decode(chunk_tokens)

            if len(chunk_tokens) >= self.config.min_chunk_tokens:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=text.strip(),
                        index=index,
                        metadata=document.metadata,
                        token_count=len(chunk_tokens),
                    )
                )
                index += 1

            i += step

        return chunks

    def _semantic_chunk(self, document: Document) -> list[Chunk]:
        """Split at natural boundaries, then merge small pieces.

        Strategy:
        1. Split on heading boundaries (##, ###, etc.)
        2. Within each section, split on double-newlines (paragraphs)
        3. Merge consecutive small pieces until they hit max_tokens
        4. If a single piece exceeds max_tokens, fall back to fixed-split on it
        """
        sections = self._split_on_headings(document.content)
        raw_pieces: list[str] = []

        for section in sections:
            paragraphs = re.split(r"\n\n+", section)
            raw_pieces.extend(p.strip() for p in paragraphs if p.strip())

        # Merge small pieces, split large ones
        chunks: list[Chunk] = []
        buffer = ""
        buffer_tokens = 0
        index = 0

        for piece in raw_pieces:
            piece_tokens = self._token_count(piece)

            # If single piece is too large, fixed-split it
            if piece_tokens > self.config.max_tokens:
                # Flush buffer first
                if buffer.strip():
                    chunks.append(self._make_chunk(buffer, index, document))
                    index += 1
                    buffer = ""
                    buffer_tokens = 0

                # Fixed-split the oversized piece
                sub_doc = Document(
                    id=document.id, content=piece, metadata=document.metadata
                )
                sub_chunks = self._fixed_chunk(sub_doc)
                for sc in sub_chunks:
                    sc.index = index
                    index += 1
                chunks.extend(sub_chunks)
                continue

            # Would adding this piece exceed max?
            if buffer_tokens + piece_tokens > self.config.max_tokens:
                # Flush current buffer
                if buffer.strip():
                    chunks.append(self._make_chunk(buffer, index, document))
                    index += 1
                buffer = piece
                buffer_tokens = piece_tokens
            else:
                # Append to buffer
                buffer = f"{buffer}\n\n{piece}" if buffer else piece
                buffer_tokens += piece_tokens

        # Flush remaining buffer
        if buffer.strip() and self._token_count(buffer) >= self.config.min_chunk_tokens:
            chunks.append(self._make_chunk(buffer, index, document))

        return chunks

    def _make_chunk(self, text: str, index: int, document: Document) -> Chunk:
        """Create a Chunk from text."""
        return Chunk(
            document_id=document.id,
            content=text.strip(),
            index=index,
            metadata=document.metadata,
            token_count=self._token_count(text),
        )

    def _split_on_headings(self, text: str) -> list[str]:
        """Split text at markdown heading boundaries."""
        pattern = r"(?=^#{1,4}\s)", re.MULTILINE
        parts = re.split(r"(?m)(?=^#{1,4}\s)", text)
        return [p for p in parts if p.strip()]
