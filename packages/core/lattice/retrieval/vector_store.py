"""
Vector store for embedding-based retrieval.

Wraps ChromaDB to provide:
- Chunk storage with embeddings
- Similarity search with configurable top-k
- Metadata filtering (by document, type, etc.)
- Integration with the ingestion pipeline

Design: ChromaDB is used in embedded mode (no server needed).
It persists to disk so the vector store survives restarts.
For the local demo, we use ChromaDB's built-in embedding function.
For production, swap in sentence-transformers or a custom model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
import structlog

from lattice.ingestion.models import Chunk

logger = structlog.get_logger()


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "lattice_chunks",
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of chunks stored."""
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store.

        ChromaDB handles embedding automatically using its default model.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "source": chunk.metadata.source,
                "doc_type": chunk.metadata.doc_type.value,
                "index": chunk.index,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("chunks_stored", count=len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_document_id: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for chunks similar to the query.

        Returns ranked results with similarity scores.
        """
        where_filter = {}
        if filter_document_id:
            where_filter["document_id"] = filter_document_id
        if filter_doc_type:
            where_filter["doc_type"] = filter_doc_type

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - distance  # cosine distance → similarity

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        similarity=similarity,
                    )
                )

        return search_results

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self._collection.delete(where={"document_id": document_id})
        logger.info("document_deleted", document_id=document_id)

    def reset(self) -> None:
        """Clear all data from the vector store."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_reset")


class SearchResult:
    """A single result from vector similarity search."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        metadata: dict,
        similarity: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata
        self.similarity = similarity

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"SearchResult(sim={self.similarity:.3f}, text='{preview}')"
