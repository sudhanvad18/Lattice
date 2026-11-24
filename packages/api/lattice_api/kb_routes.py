"""Manage the API process's active knowledge base."""

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request

from lattice.inference.health import friendly_provider_error
from lattice_api.errors import api_error
from lattice_api.graph_routes import KnowledgeBaseDependency
from lattice_api.models import KBIngestAccepted, KBIngestRequest, KBSource, KBStatus

router = APIRouter(prefix="/kb", tags=["knowledge base"])


def _busy(request: Request) -> bool:
    job = getattr(request.app.state, "kb_ingest_job", None)
    return job is not None and not job.done()


def _require_idle(request: Request) -> None:
    if _busy(request):
        raise api_error(409, "busy", "An ingestion is already running.")


def _status(kb, ingesting: bool) -> KBStatus:
    with kb._lock:
        stats = kb.stats
        return KBStatus(
            ready=stats["chunks_stored"] > 0,
            chunks=stats["chunks_stored"],
            entities=stats["kg_entities"],
            relations=stats["kg_relations"],
            communities=kb.community_count,
            sources=[KBSource(source=source, chunks=count)
                     for source, count in sorted(kb.source_counts().items())],
            last_ingested_at=kb.last_ingested_at,
            ingesting=ingesting,
            data_dir=str(kb._persist_path),
        )


@router.get("/status", response_model=KBStatus)
async def status(request: Request, kb: KnowledgeBaseDependency):
    return await asyncio.to_thread(_status, kb, _busy(request))


@router.post("/ingest", status_code=202, response_model=KBIngestAccepted)
async def ingest(body: KBIngestRequest, request: Request, kb: KnowledgeBaseDependency):
    _require_idle(request)
    try:
        path = Path(body.path).expanduser().resolve()
        valid = path.is_file() or path.is_dir()
    except (OSError, ValueError, RuntimeError):
        valid = False
    if not valid:
        raise api_error(400, "bad_path", f"Path not found on the API host: {body.path}")

    manager = getattr(request.app.state, "task_manager", None)
    if manager is None:
        from lattice_api.app import get_task_manager

        manager = get_task_manager()
    job_id = str(uuid4())

    def build():
        if body.reset:
            kb.reset()
        return kb.ingest_directory(path) if path.is_dir() else kb.ingest_file(path)

    async def run():
        manager.emit(job_id, "kb_ingest_started", {"path": str(path)})
        try:
            chunks = await asyncio.to_thread(build)
            stats = await asyncio.to_thread(lambda: kb.stats)
            manager.emit(job_id, "kb_ingest_complete", {
                "path": str(path), "chunks": chunks,
                "entities": stats["kg_entities"], "relations": stats["kg_relations"],
            })
        except Exception as exc:
            manager.emit(job_id, "kb_ingest_failed", {
                "path": str(path), "message": friendly_provider_error(exc) or str(exc),
            })

    request.app.state.kb_ingest_job = asyncio.create_task(run())
    return KBIngestAccepted(job_id=job_id, path=str(path))


@router.post("/reset", response_model=KBStatus)
async def reset(request: Request, kb: KnowledgeBaseDependency):
    _require_idle(request)
    job = asyncio.create_task(asyncio.to_thread(kb.reset))
    request.app.state.kb_ingest_job = job
    await asyncio.shield(job)
    return await asyncio.to_thread(_status, kb, False)
