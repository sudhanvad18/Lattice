"""
Write-back abstraction layer.

All write-back targets (GitHub, KG, file system, webhooks) implement
the same interface so the orchestrator doesn't need to know the specifics.

Every write-back goes through:
1. Agent produces artifact → 2. Reviewer approves → 3. Approval gate → 4. Write-back executes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class WriteBackTarget(str, Enum):
    GITHUB = "github"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    FILE_SYSTEM = "file_system"
    WEBHOOK = "webhook"


class WriteBackStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WriteBackRequest:
    """A request to write an artifact to an external system."""

    id: str = field(default_factory=lambda: str(uuid4()))
    target: WriteBackTarget = WriteBackTarget.FILE_SYSTEM
    artifact_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: WriteBackStatus = WriteBackStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WriteBackResult:
    """Result of a write-back operation."""

    success: bool
    request_id: str
    target: WriteBackTarget
    message: str = ""
    url: Optional[str] = None  # e.g., PR URL, file path
    metadata: dict[str, Any] = field(default_factory=dict)


class WriteBackHandler(ABC):
    """Abstract interface for write-back targets."""

    target: WriteBackTarget

    @abstractmethod
    async def execute(self, request: WriteBackRequest) -> WriteBackResult:
        """Execute the write-back operation."""
        ...

    @abstractmethod
    async def validate(self, request: WriteBackRequest) -> tuple[bool, str]:
        """Validate that the request can be executed (pre-flight check).

        Returns (is_valid, reason).
        """
        ...

    @abstractmethod
    async def rollback(self, request: WriteBackRequest) -> bool:
        """Attempt to undo a write-back (best effort)."""
        ...
