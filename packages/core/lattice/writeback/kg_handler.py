"""
Knowledge Graph write-back handler.

Expands the knowledge graph from agent discoveries:
- Adds new entities discovered during research
- Creates relations between existing and new entities
- Updates entity descriptions with richer context

This is what makes Lattice self-improving — the system grows
its own knowledge base from what agents discover.
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from lattice.graph.engine import KnowledgeGraphBackend
from lattice.graph.models import Entity, Relation
from lattice.writeback.base import (
    WriteBackHandler,
    WriteBackRequest,
    WriteBackResult,
    WriteBackStatus,
    WriteBackTarget,
)

logger = structlog.get_logger()


class KnowledgeGraphWriteBack(WriteBackHandler):
    """Write discoveries back to the knowledge graph."""

    target = WriteBackTarget.KNOWLEDGE_GRAPH

    def __init__(self, kg_backend: KnowledgeGraphBackend) -> None:
        self._kg = kg_backend

    async def validate(self, request: WriteBackRequest) -> tuple[bool, str]:
        """Validate that the content contains valid entities/relations."""
        try:
            data = json.loads(request.content)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            if not entities and not relations:
                return False, "No entities or relations to add"

            for e in entities:
                if not e.get("name") or not e.get("entity_type"):
                    return False, f"Entity missing name or type: {e}"

            return True, f"Valid: {len(entities)} entities, {len(relations)} relations"
        except json.JSONDecodeError:
            return False, "Content is not valid JSON"

    async def execute(self, request: WriteBackRequest) -> WriteBackResult:
        """Add entities and relations to the knowledge graph."""
        try:
            data = json.loads(request.content)
            entities_data = data.get("entities", [])
            relations_data = data.get("relations", [])

            added_entities = []
            added_relations = []

            for e_data in entities_data:
                entity = Entity(
                    name=e_data["name"],
                    entity_type=e_data["entity_type"],
                    description=e_data.get("description"),
                    properties=e_data.get("properties", {}),
                    source_chunk_ids=e_data.get("source_chunk_ids", []),
                )
                self._kg.add_entity(entity)
                added_entities.append(entity.id)

            for r_data in relations_data:
                relation = Relation(
                    source_id=r_data["source_id"],
                    target_id=r_data["target_id"],
                    relation_type=r_data["relation_type"],
                    properties=r_data.get("properties", {}),
                )
                self._kg.add_relation(relation)
                added_relations.append(relation.id)

            request.status = WriteBackStatus.COMPLETED
            request.executed_at = datetime.utcnow()
            request.metadata["_added_entity_ids"] = added_entities
            request.metadata["_added_relation_ids"] = added_relations

            logger.info(
                "kg_writeback_success",
                entities=len(added_entities),
                relations=len(added_relations),
            )
            return WriteBackResult(
                success=True,
                request_id=request.id,
                target=self.target,
                message=f"Added {len(added_entities)} entities, {len(added_relations)} relations",
                metadata={
                    "entity_ids": added_entities,
                    "relation_ids": added_relations,
                },
            )

        except Exception as e:
            request.status = WriteBackStatus.FAILED
            request.error = str(e)
            logger.error("kg_writeback_failed", error=str(e))
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=self.target,
                message=f"KG write-back failed: {e}",
            )

    async def rollback(self, request: WriteBackRequest) -> bool:
        """Remove entities and relations that were added."""
        try:
            entity_ids = request.metadata.get("_added_entity_ids", [])
            for entity_id in entity_ids:
                self._kg.remove_entity(entity_id)

            logger.info("kg_rollback_success", entities_removed=len(entity_ids))
            return True
        except Exception as e:
            logger.error("kg_rollback_failed", error=str(e))
            return False
