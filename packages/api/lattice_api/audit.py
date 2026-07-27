"""
Task audit trail.

Records every significant event during task execution:
- Agent activations
- LLM calls (prompt/response summaries)
- Artifacts produced
- Reviews issued
- Write-back attempts
- Errors

Provides full traceability for debugging, compliance, and observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class AuditEventType(str, Enum):
    TASK_CREATED = "task_created"
    TASK_PLANNED = "task_planned"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    LLM_CALL = "llm_call"
    ARTIFACT_PRODUCED = "artifact_produced"
    REVIEW_ISSUED = "review_issued"
    WRITEBACK_REQUESTED = "writeback_requested"
    WRITEBACK_APPROVED = "writeback_approved"
    WRITEBACK_REJECTED = "writeback_rejected"
    WRITEBACK_EXECUTED = "writeback_executed"
    WRITEBACK_FAILED = "writeback_failed"
    ERROR = "error"
    TASK_COMPLETED = "task_completed"
    CHECKPOINT_SAVED = "checkpoint_saved"


@dataclass
class AuditEvent:
    """A single audit event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    event_type: AuditEventType = AuditEventType.TASK_CREATED
    agent: str = ""
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AuditTrail:
    """In-memory audit trail for a task execution.

    In production, events would be persisted to PostgreSQL
    and streamed to LangFuse for observability.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        task_id: str,
        event_type: AuditEventType,
        agent: str = "",
        summary: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            task_id=task_id,
            event_type=event_type,
            agent=agent,
            summary=summary,
            details=details or {},
        )
        self._events.append(event)
        return event

    def get_events(self, task_id: str | None = None) -> list[AuditEvent]:
        if task_id:
            return [e for e in self._events if e.task_id == task_id]
        return list(self._events)

    def get_events_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def to_dicts(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """Serialize events to dicts for API responses."""
        events = self.get_events(task_id)
        return [
            {
                "id": e.id,
                "event_type": e.event_type.value,
                "agent": e.agent,
                "summary": e.summary,
                "details": e.details,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]

    @property
    def count(self) -> int:
        return len(self._events)
