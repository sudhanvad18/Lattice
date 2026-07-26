"""
End-to-end knowledge base builder.

This is the top-level orchestrator that combines:
  Ingestion Pipeline → Knowledge Graph Builder → Vector Store

Usage:
    from lattice.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.ingest_file(Path("docs/manual.pdf"))
    kb.ingest_directory(Path("docs/"))

    # Now you can query both:
    vector_results = kb.vector_search("What causes erosion?")
    graph_results = kb.graph_search("Fan Blade")
    path = kb.find_path("PW1100G", "Sand Ingestion")
"""

from __future__ import annotations

from pathlib import Path

import structlog

from lattice.graph.builder import KnowledgeGraphBuilder
from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity, KGQueryResult, KGStats
from lattice.ingestion.chunker import ChunkConfig
from lattice.ingestion.models import Chunk
from lattice.ingestion.pipeline import IngestionPipeline
from lattice.retrieval.vector_store import SearchResult, VectorStore

logger = structlog.get_logger()


class KnowledgeBase:
    """Unified interface to the knowledge base.

    Combines document ingestion, knowledge graph, and vector store
    into a single queryable system. This is what agents interact with.
    """

    def __init__(
        self,
        persist_dir: str | Path = "data",
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        persist_path = Path(persist_dir)

        self._pipeline = IngestionPipeline(chunk_config=chunk_config)
        self._vector_store = VectorStore(
            persist_dir=persist_path / "chroma",
            collection_name="lattice_chunks",
        )
        self._kg_builder = KnowledgeGraphBuilder(backend=NetworkXBackend())
        self._chunks: list[Chunk] = []

    @property
    def graph(self) -> NetworkXBackend:
        """Direct access to the graph backend for queries."""
        return self._kg_builder.backend  # type: ignore[return-value]

    @property
    def vector_store(self) -> VectorStore:
        """Direct access to the vector store."""
        return self._vector_store

    @property
    def stats(self) -> dict:
        """Summary statistics."""
        kg_stats = self._kg_builder.stats
        return {
            "chunks_stored": self._vector_store.count,
            "kg_entities": kg_stats.total_entities,
            "kg_relations": kg_stats.total_relations,
            "kg_entity_types": kg_stats.entity_types,
            "kg_relation_types": kg_stats.relation_types,
        }

    def ingest_file(self, path: Path) -> int:
        """Ingest a single file into both vector store and KG.

        Returns the number of chunks produced.
        """
        chunks = self._pipeline.ingest_file(path)
        return self._store_chunks(chunks)

    def ingest_text(self, content: str, source: str = "inline") -> int:
        """Ingest raw text into both vector store and KG."""
        chunks = self._pipeline.ingest_text(content, source=source)
        return self._store_chunks(chunks)

    def ingest_directory(self, directory: Path) -> int:
        """Ingest all supported files in a directory."""
        chunks = self._pipeline.ingest_directory(directory)
        return self._store_chunks(chunks)

    def vector_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic similarity search over document chunks."""
        return self._vector_store.search(query, top_k=top_k)

    def graph_search(self, query: str, entity_type: str | None = None) -> list[Entity]:
        """Search the knowledge graph for entities matching the query."""
        return self.graph.search_entities(query, entity_type=entity_type)

    def get_entity_context(self, entity_id: str, depth: int = 1) -> KGQueryResult:
        """Get the subgraph around an entity (for context enrichment)."""
        return self.graph.get_subgraph(entity_id, depth=depth)

    def find_path(self, source_name: str, target_name: str) -> KGQueryResult:
        """Find a path between two entities by name."""
        source_entities = self.graph.search_entities(source_name)
        target_entities = self.graph.search_entities(target_name)

        if not source_entities or not target_entities:
            return KGQueryResult()

        return self.graph.find_path(source_entities[0].id, target_entities[0].id)

    def _store_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks in both vector store and KG."""
        if not chunks:
            return 0

        # Add to vector store for similarity search
        self._vector_store.add_chunks(chunks)

        # Extract entities/relations and add to KG
        self._kg_builder.build_from_chunks(chunks)

        self._chunks.extend(chunks)

        logger.info(
            "knowledge_base_updated",
            new_chunks=len(chunks),
            total_chunks=self._vector_store.count,
            **self._kg_builder.stats.model_dump(),
        )
        return len(chunks)
