"""
Document models used throughout the ingestion pipeline.

These models represent the lifecycle of a document:
  Raw Document → Parsed Document → Chunks → (Embeddings + KG Entities)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    """Metadata attached to every document and inherited by its chunks."""

    source: str = Field(..., description="Origin path, URL, or identifier")
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    doc_type: DocumentType = DocumentType.TEXT
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A raw document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Full text content")
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """A chunk produced by the chunking engine.

    Each chunk is a semantically coherent piece of a document,
    small enough for embedding/retrieval but retaining context.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text")
    index: int = Field(..., description="Position within parent document (0-based)")
    metadata: DocumentMetadata
    token_count: int = Field(0, description="Approximate token count")

    # Set after embedding
    embedding: Optional[list[float]] = Field(None, exclude=True)
