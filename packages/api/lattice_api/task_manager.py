"""
Task manager.

Manages the lifecycle of submitted tasks:
- Accepts new tasks via the API
- Dispatches to the orchestrator
- Tracks status, provides results
- Integrates with audit trail and write-back engine
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

import structlog

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import AgentState, TaskStatus
from lattice.agents.writer import WriterAgent
from lattice.comms.checkpoint import CheckpointStore, LocalCheckpointStore
from lattice.inference.provider import LLMProvider, get_provider
from lattice.writeback import ApprovalGate, ApprovalMode, WriteBackEngine

from lattice_api.audit import AuditEventType, AuditTrail

logger = structlog.get_logger()

EventCallback = Callable[[str, str, dict[str, Any]], None]


class TaskManager:
    """Manages task submission, execution, and retrieval."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        checkpoint_store: CheckpointStore | None = None,
        writeback_engine: WriteBackEngine | None = None,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        self._provider = provider
        self._checkpoint_store = checkpoint_store or LocalCheckpointStore()
        self._writeback_engine = writeback_engine
        self._audit = audit_trail or AuditTrail()
        self._tasks: dict[str, AgentState] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._event_callbacks: list[EventCallback] = []

    @property
    def audit_trail(self) -> AuditTrail:
        return self._audit

    def on_event(self, callback: EventCallback) -> None:
        """Register a callback for task events (used by WebSocket)."""
        self._event_callbacks.append(callback)

    def _emit_event(self, task_id: str, event: str, data: dict[str, Any]) -> None:
        for cb in self._event_callbacks:
            try:
                cb(task_id, event, data)
            except Exception:
                pass

    async def submit_task(
        self,
        task: str,
        max_iterations: int = 10,
        approval_mode: str = "never",
    ) -> AgentState:
        """Submit a new task for execution. Returns immediately with initial state."""
        state = AgentState(original_task=task, max_iterations=max_iterations)
        self._tasks[state.task_id] = state

        self._audit.record(
            task_id=state.task_id,
            event_type=AuditEventType.TASK_CREATED,
            summary=f"Task created: {task[:100]}",
            details={"max_iterations": max_iterations, "approval_mode": approval_mode},
        )

        self._emit_event(state.task_id, "task_created", {"task": task})

        # Launch execution in background
        async_task = asyncio.create_task(
            self._execute_task(state.task_id, approval_mode)
        )
        self._running[state.task_id] = async_task

        return state

    async def _execute_task(self, task_id: str, approval_mode: str) -> None:
        """Background task execution."""
        state = self._tasks[task_id]

        try:
            provider = self._provider
            if not provider:
                from lattice.inference.provider import get_provider
                provider = get_provider()

            orchestrator = OrchestratorAgent(
                provider=provider,
                researcher=ResearcherAgent(provider=provider),
                writer=WriterAgent(provider=provider),
                reviewer=ReviewerAgent(provider=provider),
            )

            self._audit.record(
                task_id=task_id,
                event_type=AuditEventType.AGENT_STARTED,
                agent="orchestrator",
                summary="Orchestrator started execution",
            )

            result = await orchestrator.run(
                state.original_task,
                max_iterations=state.max_iterations,
                checkpoint_store=self._checkpoint_store,
            )

            self._tasks[task_id] = result

            self._audit.record(
                task_id=task_id,
                event_type=AuditEventType.TASK_COMPLETED,
                summary=f"Task completed with status: {result.status.value}",
                details={
                    "artifacts": len(result.artifacts),
                    "reviews": len(result.reviews),
                    "iterations": result.iteration_count,
                },
            )

            self._emit_event(task_id, "task_complete", {
                "status": result.status.value,
                "artifacts": len(result.artifacts),
            })

        except Exception as e:
            logger.error("task_execution_failed", task_id=task_id, error=str(e))
            state.status = TaskStatus.FAILED
            state.errors.append(str(e))
            self._tasks[task_id] = state

            self._audit.record(
                task_id=task_id,
                event_type=AuditEventType.ERROR,
                summary=f"Task failed: {e}",
            )

            self._emit_event(task_id, "error", {"message": str(e)})

        finally:
            self._running.pop(task_id, None)

    def get_task(self, task_id: str) -> AgentState | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[AgentState]:
        return list(self._tasks.values())

    def is_running(self, task_id: str) -> bool:
        return task_id in self._running
