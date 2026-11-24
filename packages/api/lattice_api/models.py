"""
API request/response models.

Defines the wire format for the REST API. These are separate from
the internal agent state to decouple the API contract from implementation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class TaskSubmitRequest(BaseModel):
    """Request to submit a new task to the agent team."""

    task: str = Field(..., min_length=1, max_length=5000, description="Task description in natural language")
    max_iterations: int = Field(10, ge=1, le=50)
    approval_mode: str = Field("never", description="Approval gate mode: always, on_low_confidence, never")
    write_back_targets: list[str] = Field(default_factory=list, description="Enabled write-back targets")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatusResponse(BaseModel):
    """Current status of a submitted task."""

    task_id: str
    status: str
    original_task: str
    current_agent: Optional[str] = None
    iteration_count: int = 0
    artifact_count: int = 0
    review_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    errors: list[str] = Field(default_factory=list)


class ArtifactResponse(BaseModel):
    """A single artifact produced by an agent."""

    id: str
    artifact_type: str
    content: str
    source_agent: str
    citations: list[str] = Field(default_factory=list)
    confidence: float
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResultResponse(BaseModel):
    """Full result of a completed task."""

    task_id: str
    status: str
    original_task: str
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    write_backs: list[dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = 0
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeBaseHealth(BaseModel):
    backend: str
    entities: int
    chunks: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    agents_available: list[str] = Field(default_factory=list)
    providers_configured: list[str] = Field(default_factory=list)
    knowledge_base: KnowledgeBaseHealth | None = None


class WebSocketMessage(BaseModel):
    """Message format for WebSocket streaming."""

    event: str  # "agent_started", "artifact_produced", "review_complete", "task_complete", "error"
    task_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Node(BaseModel):
    id: str
    name: str
    entity_type: str
    description: str | None = None
    community: int | None = None


class Edge(BaseModel):
    id: str
    source: str
    target: str
    relation_type: str
    provenance: str
    confidence: float


class GraphResponse(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    truncated: bool


class GraphStatsResponse(BaseModel):
    chunks_stored: int
    kg_entities: int
    kg_relations: int
    kg_entity_types: dict[str, int]
    kg_relation_types: dict[str, int]
    backend: str


class EntityDetailResponse(BaseModel):
    entity: Node
    neighbors: list[Node]
    relations: list[Edge]


class EntityReference(BaseModel):
    id: str
    name: str


class Community(BaseModel):
    id: int
    size: int
    members: list[EntityReference]


class SurprisingConnection(BaseModel):
    source: str
    target: str
    explanation: str


class CommunitiesResponse(BaseModel):
    communities: list[Community]
    god_nodes: list[EntityReference]
    surprising_connections: list[SurprisingConnection]


class ChatRequest(BaseModel):
    """One question from the dashboard Chat tab."""

    message: str = Field(..., min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    """Retrieval-backed answer. Citations are chunk ids and entity ids."""

    response: str
    citations: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "api"


class IngestResponse(BaseModel):
    chunks: int
    stats: dict[str, Any]


class EvalTaskResult(BaseModel):
    task_id: str
    task_name: str
    status: str
    completed: bool
    passed: bool
    quality_score: float
    keyword_recall: float
    keywords_missing: list[str] = Field(default_factory=list)
    latency_seconds: float
    iterations: int
    artifact_count: int = 0
    errors: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    total_tasks: int
    completion_rate: float
    pass_rate: float
    avg_iterations: float
    avg_latency_seconds: float
    avg_keyword_recall: float
    avg_review_confidence: float


class EvalRunResponse(BaseModel):
    """Latest ``lattice eval`` run; ``run_at`` is None when no run exists yet."""

    run_at: datetime | None = None
    provider: str | None = None
    model: str | None = None
    summary: EvalSummary | None = None
    results: list[EvalTaskResult] = Field(default_factory=list)
