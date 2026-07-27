"""
Agent state and communication models.

Defines the shared state that flows through the LangGraph state machine
and the structured artifacts agents produce and consume.

Key concepts:
- AgentState: the full state of a task execution (what LangGraph manages)
- Artifact: a structured output from one agent, consumable by others
- TaskPlan: the orchestrator's decomposition of a high-level task
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CODE = "code"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    INGESTION = "ingestion"


class Artifact(BaseModel):
    """A structured output produced by an agent.

    Artifacts are the communication currency between agents.
    The Researcher produces research artifacts, the Writer consumes
    them to produce document artifacts, the Reviewer validates them.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: str = Field(..., description="research, document, code, review, analysis")
    content: str = Field(..., description="The artifact's main content")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_agent: AgentRole = Field(..., description="Which agent produced this")
    citations: list[str] = Field(
        default_factory=list,
        description="IDs of KG entities or chunk IDs that support this artifact",
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReviewVerdict(BaseModel):
    """The Reviewer agent's judgment on an artifact."""

    artifact_id: str
    approved: bool
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class SubTask(BaseModel):
    """A single sub-task in the orchestrator's plan."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    assigned_to: AgentRole
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list, description="IDs of subtasks this depends on")
    result_artifact_id: Optional[str] = None


class TaskPlan(BaseModel):
    """The orchestrator's decomposition of a high-level task."""

    task_description: str
    subtasks: list[SubTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """The shared state flowing through the LangGraph state machine.

    This is what LangGraph manages. Every agent reads from and writes
    to this state. The orchestrator controls the flow.
    """

    # Task info
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    original_task: str = Field(..., description="The user's original task description")
    plan: Optional[TaskPlan] = None

    # Agent outputs
    artifacts: list[Artifact] = Field(default_factory=list)
    reviews: list[ReviewVerdict] = Field(default_factory=list)

    # Execution tracking
    current_agent: Optional[AgentRole] = None
    current_subtask_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    iteration_count: int = 0
    max_iterations: int = 10

    # Error handling
    errors: list[str] = Field(default_factory=list)

    # Conversation history for context
    messages: list[dict[str, str]] = Field(default_factory=list)

    def get_artifacts_by_type(self, artifact_type: str) -> list[Artifact]:
        return [a for a in self.artifacts if a.artifact_type == artifact_type]

    def get_artifacts_by_agent(self, agent: AgentRole) -> list[Artifact]:
        return [a for a in self.artifacts if a.source_agent == agent]

    def get_latest_artifact(self, artifact_type: str | None = None) -> Artifact | None:
        candidates = self.artifacts
        if artifact_type:
            candidates = [a for a in candidates if a.artifact_type == artifact_type]
        return candidates[-1] if candidates else None

    def get_pending_subtasks(self) -> list[SubTask]:
        if not self.plan:
            return []
        return [st for st in self.plan.subtasks if st.status == TaskStatus.PENDING]

    def get_next_subtask(self) -> SubTask | None:
        """Get the next subtask whose dependencies are all complete."""
        if not self.plan:
            return None
        completed_ids = {
            st.id for st in self.plan.subtasks if st.status == TaskStatus.COMPLETED
        }
        for st in self.plan.subtasks:
            if st.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in st.depends_on):
                    return st
        return None
