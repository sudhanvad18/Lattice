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

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import structlog

from lattice.graph.builder import KnowledgeGraphBuilder
from lattice.graph.engine import KnowledgeGraphBackend, NetworkXBackend
from lattice.graph.factory import create_graph_backend
from lattice.graph.models import Entity, KGQueryResult, KGStats
from lattice.ingestion.cache import IngestionCache
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
        enable_cache: bool = True,
        graph_backend: KnowledgeGraphBackend | None = None,
    ) -> None:
        persist_path = Path(persist_dir)
        self._persist_path = persist_path
        self._lock = threading.RLock()

        cache = IngestionCache(persist_path / "ingestion_cache") if enable_cache else None
        self._pipeline = IngestionPipeline(chunk_config=chunk_config, cache=cache)
        self._vector_store = VectorStore(
            persist_dir=persist_path / "chroma",
            collection_name="lattice_chunks",
        )
        backend = graph_backend if graph_backend is not None else create_graph_backend()
        graph_path = persist_path / "graph.json"
        if isinstance(backend, NetworkXBackend) and graph_path.exists():
            try:
                backend = NetworkXBackend.load(graph_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("graph_snapshot_corrupt", path=str(graph_path), error=str(exc))
                backend = NetworkXBackend()
        self._kg_builder = KnowledgeGraphBuilder(backend=backend)
        self._chunks: list[Chunk] = []

    @property
    def graph(self) -> KnowledgeGraphBackend:
        """Direct access to the graph backend for queries."""
        return self._kg_builder.backend

    def close(self) -> None:
        self.graph.close()

    def reset(self) -> None:
        """Clear persisted chunks, graph, and ingestion cache."""
        with self._lock:
            self._vector_store.reset()
            if isinstance(self.graph, NetworkXBackend):
                self._kg_builder.backend = NetworkXBackend()
            elif hasattr(self.graph, "clear"):
                self.graph.clear()
            IngestionCache(self._persist_path / "ingestion_cache").clear()
            if self._pipeline._cache is not None:
                self._pipeline._cache.clear()
            self._chunks.clear()
            (self._persist_path / "graph.json").unlink(missing_ok=True)
            (self._persist_path / "kb_meta.json").unlink(missing_ok=True)

    @property
    def vector_store(self) -> VectorStore:
        """Direct access to the vector store."""
        return self._vector_store

    @property
    def stats(self) -> dict:
        """Summary statistics."""
        with self._lock:
            kg_stats = self._kg_builder.stats
            return {
                "chunks_stored": self._vector_store.count,
                "kg_entities": kg_stats.total_entities,
                "kg_relations": kg_stats.total_relations,
                "kg_entity_types": kg_stats.entity_types,
                "kg_relation_types": kg_stats.relation_types,
            }

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._vector_store.count == 0

    def source_counts(self) -> dict[str, int]:
        """Chunks stored per source path, sorted by path."""
        with self._lock:
            return self._vector_store.source_counts()

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
        with self._lock:
            return self._vector_store.search(query, top_k=top_k)

    def graph_search(self, query: str, entity_type: str | None = None) -> list[Entity]:
        """Search the knowledge graph for entities matching the query."""
        with self._lock:
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
        with self._lock:
            if not chunks:
                return 0

            # Add to vector store for similarity search
            self._vector_store.add_chunks(chunks)

            # Extract entities/relations and add to KG
            self._kg_builder.build_from_chunks(chunks)

            # Update community clusters
            try:
                from lattice.graph.clustering import cluster_graph
                cluster_graph(self._kg_builder.backend)
            except ImportError:
                pass
            except Exception as e:
                logger.debug("clustering_skipped", reason=str(e))

            if isinstance(self.graph, NetworkXBackend):
                self.graph.save(self._persist_path / "graph.json")

            self._chunks.extend(chunks)

            logger.info(
                "knowledge_base_updated",
                new_chunks=len(chunks),
                total_chunks=self._vector_store.count,
                **self._kg_builder.stats.model_dump(),
            )
            meta_path = self._persist_path / "kb_meta.json"
            temporary = meta_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"last_ingested_at": datetime.now(UTC).isoformat()}))
            temporary.replace(meta_path)
            return len(chunks)

    @property
    def last_ingested_at(self) -> str | None:
        """Persisted timestamp of the most recent successful ingestion."""
        with self._lock:
            try:
                metadata = json.loads((self._persist_path / "kb_meta.json").read_text())
                return metadata["last_ingested_at"]
            except (OSError, ValueError, KeyError, TypeError):
                return None

    @property
    def community_count(self) -> int:
        with self._lock:
            return len({
                entity.community for entity in self.graph.get_all_entities()
                if entity.community is not None
            })
