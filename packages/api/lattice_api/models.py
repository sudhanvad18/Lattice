"""
API request/response models.

Defines the wire format for the REST API. These are separate from
the internal agent state to decouple the API contract from implementation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    agents_available: list[str] = Field(default_factory=list)
    providers_configured: list[str] = Field(default_factory=list)


class WebSocketMessage(BaseModel):
    """Message format for WebSocket streaming."""

    event: str  # "agent_started", "artifact_produced", "review_complete", "task_complete", "error"
    task_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
