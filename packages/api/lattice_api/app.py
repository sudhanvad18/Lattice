"""
FastAPI application.

REST API + WebSocket streaming for the Lattice agent platform.

Endpoints:
- POST /tasks          Submit a new task
- GET  /tasks          List all tasks
- GET  /tasks/{id}     Get task status
- GET  /tasks/{id}/result   Get full task result with artifacts
- GET  /tasks/{id}/audit    Get audit trail
- POST /chat           Answer a question from the knowledge base
- WS   /ws/activity    Stream task events across every task
- WS   /ws/{task_id}   Stream real-time progress for one task
- GET  /health         Health check
- GET  /kg/graph, /kg/stats, /kg/entities, /kg/entities/{entity_id}
- GET  /kg/communities  Community detection
- POST /kg/ingest       Ingest text
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from lattice_api.audit import AuditTrail
from lattice_api.chat import build_chat_answer
from lattice_api.eval_routes import router as eval_router
from lattice_api.graph_routes import KnowledgeBaseDependency
from lattice_api.graph_routes import get_knowledge_base as get_knowledge_base
from lattice_api.graph_routes import router
from lattice_api.models import (
    ArtifactResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    TaskResultResponse,
    TaskStatusResponse,
    TaskSubmitRequest,
)
from lattice_api.task_manager import TaskManager

if TYPE_CHECKING:
    from lattice.knowledge_base import KnowledgeBase

logger = structlog.get_logger()

# --- Application State ---

_task_manager: TaskManager | None = None
_audit_trail: AuditTrail | None = None


def get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise RuntimeError("TaskManager not initialized")
    return _task_manager


_TERMINAL_EVENTS = frozenset({"task_complete", "error"})


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("task_id"), record.get("event"), record.get("timestamp"))


async def _stream_task_events(websocket: WebSocket, task_id: str | None) -> None:
    """Replay logged task events, then stream new ones.

    `task_id` None broadcasts every task (the activity feed). A specific id
    closes after that task completes or fails, matching the per-task socket.
    """
    await websocket.accept()
    tm = get_task_manager()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def matches(record: dict[str, Any]) -> bool:
        return task_id is None or record.get("task_id") == task_id

    def handler(record: dict[str, Any]) -> None:
        if matches(record):
            queue.put_nowait(record)

    tm.on_event(handler)
    try:
        seen: set[tuple[Any, ...]] = set()
        for record in tm.recent_events():
            if not matches(record):
                continue
            seen.add(_record_key(record))
            await websocket.send_json(record)
            if task_id is not None and record.get("event") in _TERMINAL_EVENTS:
                await websocket.close()
                return
        while True:
            try:
                record = await asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                await websocket.send_json({
                    "event": "ping",
                    "task_id": task_id or "",
                    "data": {},
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                continue
            key = _record_key(record)
            if key in seen:
                continue
            seen.add(key)
            await websocket.send_json(record)
            if task_id is not None and record.get("event") in _TERMINAL_EVENTS:
                await websocket.close()
                return
    except WebSocketDisconnect:
        pass
    finally:
        tm.off_event(handler)


# --- App Creation ---


def create_app(
    task_manager: TaskManager | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Accepts an optional TaskManager and caller-owned KnowledgeBase for injection.
    """
    global _task_manager, _audit_trail

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _task_manager, _audit_trail
        kb = knowledge_base
        if kb is None:
            from lattice.config import get_settings
            from lattice.knowledge_base import KnowledgeBase

            kb = KnowledgeBase(persist_dir=get_settings().data_dir)
        app.state.knowledge_base = kb
        manager = task_manager or TaskManager(knowledge_base=kb)
        _task_manager = manager
        _audit_trail = manager.audit_trail
        logger.info("lattice_api_started")
        try:
            yield
        finally:
            if knowledge_base is None:
                kb.close()
                app.state.knowledge_base = None
            logger.info("lattice_api_shutdown")

    app = FastAPI(
        title="Lattice Agent Platform",
        description="Autonomous multi-agent orchestration with write-back capabilities",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.knowledge_base = knowledge_base
    app.include_router(router)
    app.include_router(eval_router)

    if task_manager:
        _task_manager = task_manager
        _audit_trail = task_manager.audit_trail

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Auth Middleware ---

    async def verify_api_key(x_api_key: str = Header(default=None)):
        """Optional API key verification.

        In production, validates against stored keys.
        For demo, accepts any non-empty key or skips if not configured.
        """
        # TODO: implement real key validation in production
        return x_api_key

    # --- Health ---

    @app.get("/health", response_model=HealthResponse)
    async def health():
        kb = app.state.knowledge_base
        kb_health = None
        if kb is not None:
            stats = kb.stats
            kb_health = {
                "backend": type(kb.graph).__name__,
                "entities": stats["kg_entities"],
                "chunks": stats["chunks_stored"],
            }
        return HealthResponse(
            knowledge_base=kb_health,
            status="healthy",
            version="0.1.0",
            agents_available=["orchestrator", "researcher", "writer", "reviewer", "code", "ingestion"],
            providers_configured=["ollama"],
        )

    # --- Task Endpoints ---

    @app.post("/tasks", response_model=TaskStatusResponse)
    async def submit_task(request: TaskSubmitRequest):
        """Submit a new task to the agent team."""
        tm = get_task_manager()
        state = await tm.submit_task(
            task=request.task,
            max_iterations=request.max_iterations,
            approval_mode=request.approval_mode,
        )
        return TaskStatusResponse(
            task_id=state.task_id,
            status=state.status.value,
            original_task=state.original_task,
            current_agent=state.current_agent.value if state.current_agent else None,
            iteration_count=state.iteration_count,
            artifact_count=len(state.artifacts),
            review_count=len(state.reviews),
            created_at=datetime.utcnow(),
        )

    @app.get("/tasks", response_model=list[TaskStatusResponse])
    async def list_tasks():
        """List all submitted tasks."""
        tm = get_task_manager()
        tasks = tm.get_all_tasks()
        return [
            TaskStatusResponse(
                task_id=t.task_id,
                status=t.status.value,
                original_task=t.original_task,
                current_agent=t.current_agent.value if t.current_agent else None,
                iteration_count=t.iteration_count,
                artifact_count=len(t.artifacts),
                review_count=len(t.reviews),
                created_at=datetime.utcnow(),
            )
            for t in tasks
        ]

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(task_id: str):
        """Get current status of a task."""
        tm = get_task_manager()
        state = tm.get_task(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskStatusResponse(
            task_id=state.task_id,
            status=state.status.value,
            original_task=state.original_task,
            current_agent=state.current_agent.value if state.current_agent else None,
            iteration_count=state.iteration_count,
            artifact_count=len(state.artifacts),
            review_count=len(state.reviews),
            created_at=datetime.utcnow(),
            errors=state.errors,
        )

    @app.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
    async def get_task_result(task_id: str):
        """Get full task result with all artifacts."""
        tm = get_task_manager()
        state = tm.get_task(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="Task not found")

        artifacts = [
            ArtifactResponse(
                id=a.id,
                artifact_type=a.artifact_type,
                content=a.content,
                source_agent=a.source_agent.value,
                citations=a.citations,
                confidence=a.confidence,
                created_at=a.created_at,
                metadata=a.metadata,
            )
            for a in state.artifacts
        ]

        reviews = [
            {
                "artifact_id": r.artifact_id,
                "approved": r.approved,
                "confidence": r.confidence,
                "issues": r.issues,
                "suggestions": r.suggestions,
                "reasoning": r.reasoning,
            }
            for r in state.reviews
        ]

        audit_events = _audit_trail.to_dicts(task_id) if _audit_trail else []

        return TaskResultResponse(
            task_id=state.task_id,
            status=state.status.value,
            original_task=state.original_task,
            artifacts=artifacts,
            reviews=reviews,
            iteration_count=state.iteration_count,
            audit_trail=audit_events,
        )

    @app.get("/tasks/{task_id}/audit")
    async def get_task_audit(task_id: str):
        """Get the audit trail for a task."""
        tm = get_task_manager()
        state = tm.get_task(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="Task not found")

        if _audit_trail:
            return {"task_id": task_id, "events": _audit_trail.to_dicts(task_id)}
        return {"task_id": task_id, "events": []}

    # --- Chat ---

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, kb: KnowledgeBaseDependency):
        """Answer from retrieved passages and graph entities. Does not call an LLM."""
        try:
            response, citations = await run_in_threadpool(
                build_chat_answer, kb, request.message
            )
        except Exception as exc:
            logger.error("chat_retrieval_failed", error=str(exc))
            raise HTTPException(status_code=503, detail="Knowledge base retrieval failed") from exc
        return ChatResponse(response=response, citations=citations)

    # --- WebSocket ---
    # Register /ws/activity before /ws/{task_id}. Otherwise "activity" is captured
    # as a task id and the feed never sees real tasks.

    @app.websocket("/ws/activity")
    async def websocket_activity(websocket: WebSocket):
        """Stream events for every task, including events emitted before connect."""
        await _stream_task_events(websocket, None)

    @app.websocket("/ws/{task_id}")
    async def websocket_task_stream(websocket: WebSocket, task_id: str):
        """Stream real-time progress for one task."""
        await _stream_task_events(websocket, task_id)

    return app


app = create_app()
