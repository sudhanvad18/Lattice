"""
Knowledge graph builder.

Coordinates the full pipeline: chunks → extract → build graph.
This is the main entry point for populating the knowledge graph
from ingested documents.
"""

from __future__ import annotations

import structlog

from lattice.graph.engine import KnowledgeGraphBackend, NetworkXBackend
from lattice.graph.extractor import ExtractionResult, RuleBasedExtractor
from lattice.graph.models import KGStats
from lattice.ingestion.models import Chunk

logger = structlog.get_logger()


class KnowledgeGraphBuilder:
    """Builds and maintains the knowledge graph from document chunks."""

    def __init__(
        self,
        backend: KnowledgeGraphBackend | None = None,
        extractor: RuleBasedExtractor | None = None,
    ) -> None:
        self.backend = backend or NetworkXBackend()
        self._extractor = extractor or RuleBasedExtractor()

    def build_from_chunks(self, chunks: list[Chunk]) -> KGStats:
        """Extract entities/relations from chunks and add to graph.

        This is the primary method — call it after ingesting documents.
        Can be called multiple times to incrementally grow the graph.
        """
        result = self._extractor.extract(chunks)
        self._load_extraction(result)

        stats = self.backend.get_stats()
        logger.info(
            "kg_built",
            entities_added=len(result.entities),
            relations_added=len(result.relations),
            total_entities=stats.total_entities,
            total_relations=stats.total_relations,
        )
        return stats

    def add_extraction(self, result: ExtractionResult) -> None:
        """Directly add pre-computed extraction results to the graph."""
        self._load_extraction(result)

    def _load_extraction(self, result: ExtractionResult) -> None:
        """Load entities and relations into the backend."""
        for entity in result.entities:
            self.backend.add_entity(entity)

        for relation in result.relations:
            self.backend.add_relation(relation)

    @property
    def stats(self) -> KGStats:
        return self.backend.get_stats()
