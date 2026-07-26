"""
Chunking engine for splitting documents into retrieval-ready pieces.

Three strategies:
1. Fixed-size: split by token count with configurable overlap
2. Semantic: split at natural boundaries (headings, paragraphs, sentences)
3. Auto: selects the best strategy based on document type

Key design choices:
- Chunk size is measured in TOKENS (via tiktoken) not characters
- Sentence boundaries are the atomic unit — never split mid-sentence
- Semantic chunks get context overlap (trailing sentences from previous chunk)
- Auto-detection routes structured docs to semantic, unstructured to fixed
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import tiktoken

from lattice.ingestion.models import Chunk, Document, DocumentType


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    AUTO = "auto"


@dataclass
class ChunkConfig:
    """Configuration for the chunking engine."""

    strategy: ChunkStrategy = ChunkStrategy.AUTO
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 50
    context_sentences: int = 2  # sentences from previous chunk prepended for context
    encoding_name: str = "cl100k_base"  # GPT-4 / Claude tokenizer


# Regex that splits at sentence boundaries while keeping the delimiter attached
# to the preceding sentence. Handles: ".", "!", "?" followed by whitespace or EOL.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])")


class ChunkingEngine:
    """Splits documents into chunks using the configured strategy."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._encoder = tiktoken.get_encoding(self.config.encoding_name)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using the configured strategy."""
        strategy = self._resolve_strategy(document)

        if strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(document)
        return self._fixed_chunk(document)

    def _resolve_strategy(self, document: Document) -> ChunkStrategy:
        """Determine the actual strategy to use (resolves AUTO)."""
        if self.config.strategy != ChunkStrategy.AUTO:
            return self.config.strategy

        doc_type = document.metadata.doc_type

        # Structured formats → semantic chunking
        if doc_type in (DocumentType.MARKDOWN, DocumentType.HTML, DocumentType.PDF):
            return ChunkStrategy.SEMANTIC

        # Plain text: check if it has heading-like structure
        if re.search(r"(?m)^#{1,4}\s", document.content):
            return ChunkStrategy.SEMANTIC

        # Unstructured → fixed with overlap
        return ChunkStrategy.FIXED

    def _token_count(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoder.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences at natural boundaries.

        This is the atomic unit — we never split within a sentence.
        Uses punctuation + capital letter heuristic which works well
        for technical prose. Not perfect for abbreviations (Dr., etc.)
        but good enough for our use case.
        """
        sentences = _SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

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
        4. If a single piece exceeds max_tokens, split at sentence
           boundaries (never mid-sentence)
        5. Post-process: add context overlap between chunks
        """
        sections = self._split_on_headings(document.content)
        raw_pieces: list[str] = []

        for section in sections:
            paragraphs = re.split(r"\n\n+", section)
            raw_pieces.extend(p.strip() for p in paragraphs if p.strip())

        # Merge small pieces, split large ones at sentence boundaries
        chunks: list[Chunk] = []
        buffer = ""
        buffer_tokens = 0
        index = 0

        for piece in raw_pieces:
            piece_tokens = self._token_count(piece)

            # If single piece is too large, split at sentence boundaries
            if piece_tokens > self.config.max_tokens:
                # Flush buffer first
                if buffer.strip():
                    chunks.append(self._make_chunk(buffer, index, document))
                    index += 1
                    buffer = ""
                    buffer_tokens = 0

                # Sentence-aware splitting (never cuts mid-sentence)
                sentence_chunks = self._split_by_sentences(piece, document)
                for sc in sentence_chunks:
                    sc.index = index
                    index += 1
                chunks.extend(sentence_chunks)
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

        # Add context overlap between consecutive chunks
        if self.config.context_sentences > 0 and len(chunks) > 1:
            chunks = self._add_context_overlap(chunks, document)

        return chunks

    def _split_by_sentences(self, text: str, document: Document) -> list[Chunk]:
        """Split oversized text at sentence boundaries.

        Groups sentences together until max_tokens is reached, then
        starts a new chunk. Never splits within a sentence.
        """
        sentences = self._split_sentences(text)

        # If sentence splitting didn't help (e.g., one giant sentence),
        # fall back to fixed chunking as last resort
        if len(sentences) <= 1:
            sub_doc = Document(id=document.id, content=text, metadata=document.metadata)
            return self._fixed_chunk(sub_doc)

        chunks: list[Chunk] = []
        buffer = ""
        buffer_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._token_count(sentence)

            # Single sentence exceeds max (rare but possible) — add it as-is
            if sentence_tokens > self.config.max_tokens:
                if buffer.strip():
                    chunks.append(self._make_chunk(buffer, 0, document))
                    buffer = ""
                    buffer_tokens = 0
                chunks.append(self._make_chunk(sentence, 0, document))
                continue

            if buffer_tokens + sentence_tokens > self.config.max_tokens:
                if buffer.strip():
                    chunks.append(self._make_chunk(buffer, 0, document))
                buffer = sentence
                buffer_tokens = sentence_tokens
            else:
                buffer = f"{buffer} {sentence}" if buffer else sentence
                buffer_tokens += sentence_tokens

        if buffer.strip() and self._token_count(buffer) >= self.config.min_chunk_tokens:
            chunks.append(self._make_chunk(buffer, 0, document))

        return chunks

    def _add_context_overlap(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Prepend trailing sentences from the previous chunk to the next.

        This ensures context continuity — references like "this component"
        or "the above procedure" still resolve correctly in each chunk.
        """
        n_context = self.config.context_sentences
        result = [chunks[0]]  # First chunk stays unchanged

        for i in range(1, len(chunks)):
            prev_sentences = self._split_sentences(chunks[i - 1].content)
            # Take the last N sentences from previous chunk
            context_sentences = prev_sentences[-n_context:] if len(prev_sentences) >= n_context else prev_sentences

            context_prefix = " ".join(context_sentences)
            new_content = f"{context_prefix}\n\n{chunks[i].content}"
            new_token_count = self._token_count(new_content)

            result.append(
                Chunk(
                    id=chunks[i].id,
                    document_id=chunks[i].document_id,
                    content=new_content,
                    index=chunks[i].index,
                    metadata=chunks[i].metadata,
                    token_count=new_token_count,
                )
            )

        return result

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
        parts = re.split(r"(?m)(?=^#{1,4}\s)", text)
        return [p for p in parts if p.strip()]
