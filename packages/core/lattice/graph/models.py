"""
Knowledge graph data models.

These are domain-agnostic — any domain (aerospace, fintech, hardware validation)
can use the same entity/relationship structure. Domain-specific semantics come
from the entity types and relationship types, which are just strings.

This lets the same KG infrastructure support:
- Aerospace: Engine → Component → DefectType → RootCause
- Hardware validation: TestSuite → TestCase → Failure → RootCause
- Developer tools: Service → Endpoint → Error → Fix
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A node in the knowledge graph.

    Entities are typed (e.g., "Component", "Procedure", "Person") and
    carry properties as a flexible dict. The source_chunk_id links back
    to the document chunk this entity was extracted from (traceability).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Human-readable entity name")
    entity_type: str = Field(..., description="Type label (e.g., 'Component', 'Procedure')")
    properties: dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs this entity was extracted from (for traceability)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Relation(BaseModel):
    """A directed edge in the knowledge graph.

    Relations connect two entities with a typed relationship.
    Properties allow attaching metadata (confidence, source, etc.)
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Relationship type (e.g., 'HAS_COMPONENT')")
    properties: dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")


class KGQueryResult(BaseModel):
    """Result from a knowledge graph query with provenance."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def entity_ids(self) -> list[str]:
        return [e.id for e in self.entities]

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.relations) == 0


class KGStats(BaseModel):
    """Summary statistics for the knowledge graph."""

    total_entities: int = 0
    total_relations: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    relation_types: dict[str, int] = Field(default_factory=dict)
