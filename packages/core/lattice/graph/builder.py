"""
Knowledge graph builder.

Coordinates the full pipeline: chunks → extract → build graph.
This is the main entry point for populating the knowledge graph
from ingested documents.
"""

from __future__ import annotations

import structlog

from lattice.graph.engine import KnowledgeGraphBackend, NetworkXBackend
from lattice.graph.extractor import CodeExtractor, ExtractionResult, RuleBasedExtractor
from lattice.graph.models import KGStats
from lattice.ingestion.models import Chunk, DocumentType

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
        self._code_extractor = CodeExtractor()

    def build_from_chunks(self, chunks: list[Chunk]) -> KGStats:
        """Extract entities/relations from chunks and add to graph.

        This is the primary method — call it after ingesting documents.
        Can be called multiple times to incrementally grow the graph;
        re-ingesting the same content merges into existing entities.
        """
        code_chunks = [c for c in chunks if c.metadata.doc_type == DocumentType.CODE]
        text_chunks = [c for c in chunks if c.metadata.doc_type != DocumentType.CODE]

        result = self._extractor.extract(text_chunks)
        if code_chunks:
            code_result = self._code_extractor.extract(code_chunks)
            result = ExtractionResult(
                entities=result.entities + code_result.entities,
                relations=result.relations + code_result.relations,
            )
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
        """Merge entities and relations into the backend.

        Entities are identified by (entity_type, case-folded name) and relations by
        (source, target, relation_type), so extractor-assigned UUIDs never duplicate
        knowledge that is already in the graph.
        """
        known = {
            (e.entity_type, e.name.strip().casefold()): e
            for e in self.backend.get_all_entities()
        }
        id_map: dict[str, str] = {}

        for entity in result.entities:
            key = (entity.entity_type, entity.name.strip().casefold())
            existing = known.get(key)
            if existing is None:
                known[key] = entity
                id_map[entity.id] = entity.id
                self.backend.add_entity(entity)
                continue

            id_map[entity.id] = existing.id
            new_chunk_ids = [
                c for c in entity.source_chunk_ids if c not in existing.source_chunk_ids
            ]
            if new_chunk_ids or (entity.description and not existing.description):
                merged = existing.model_copy(update={
                    "source_chunk_ids": existing.source_chunk_ids + new_chunk_ids,
                    "description": existing.description or entity.description,
                })
                known[key] = merged
                self.backend.add_entity(merged)

        seen_edges = {
            (r.source_id, r.target_id, r.relation_type) for r in self.backend.get_all_relations()
        }
        for relation in result.relations:
            source_id = id_map.get(relation.source_id, relation.source_id)
            target_id = id_map.get(relation.target_id, relation.target_id)
            edge = (source_id, target_id, relation.relation_type)
            if source_id == target_id or edge in seen_edges:
                continue
            seen_edges.add(edge)
            self.backend.add_relation(
                relation.model_copy(update={"source_id": source_id, "target_id": target_id})
            )

    @property
    def stats(self) -> KGStats:
        return self.backend.get_stats()
