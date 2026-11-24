"""Benchmark results produced by ``lattice eval``."""

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends

from lattice.config import get_settings
from lattice.evaluation import EvaluationEngine, generate_kb_tasks, load_report, save_report
from lattice.inference.health import friendly_provider_error
from lattice.inference.provider import get_provider
from lattice_api.errors import api_error, require_kb, require_provider
from lattice_api.graph_routes import KnowledgeBaseDependency
from lattice_api.models import EvalRunAccepted, EvalRunRequest, EvalRunResponse, EvalStatus


def eval_results_path() -> Path:
    """Location of the latest benchmark run; overridable in tests."""
    return Path(get_settings().data_dir) / "eval" / "latest.json"


router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/results", response_model=EvalRunResponse)
def get_eval_results(path: Annotated[Path, Depends(eval_results_path)]):
    report = load_report(path)
    if report is None:
        return EvalRunResponse()
    return EvalRunResponse.model_validate(report)


# Keep the task referenced until completion; reserve the holder before probing.
_status = EvalStatus()
_run_task: asyncio.Task | None = None


@router.get("/status", response_model=EvalStatus)
def get_eval_status():
    return _status


@router.post("/run", response_model=EvalRunAccepted, status_code=202)
async def run_eval(
    kb: KnowledgeBaseDependency,
    path: Annotated[Path, Depends(eval_results_path)],
    request: EvalRunRequest | None = None,
):
    global _status, _run_task
    require_kb(kb)
    if _status.running:
        raise api_error(409, "busy", "An evaluation is already running.")
    previous = _status
    _status = EvalStatus(running=True)
    try:
        await require_provider()
        provider = get_provider()
        tasks = generate_kb_tasks(kb, limit=request.limit if request else 5)
        snapshot = {"chunks": kb.stats["chunks_stored"], "entities": kb.stats["kg_entities"],
                    "sources": len(kb.source_counts())}
    except BaseException:
        _status = previous
        raise
    run_id = str(uuid4())
    _status = EvalStatus(
        running=True, run_id=run_id, started_at=datetime.now(UTC), total=len(tasks),
    )
    status = _status

    async def emit(event, data):
        from lattice_api.app import get_task_manager

        try:
            tm = get_task_manager()
        except RuntimeError:
            return
        callback = getattr(tm, "emit", None)
        if callback is not None:
            value = callback(run_id, event, data)
            if inspect.isawaitable(value):
                await value

    async def progress(completed, total):
        status.completed = completed
        await emit("eval_progress", {"completed": completed, "total": total})

    async def execute():
        try:
            await emit("eval_started", {"total": len(tasks)})
            report = await EvaluationEngine(provider, knowledge_base=kb).run_benchmark(
                tasks, on_progress=progress,
            )
            save_report(report, path, kb_snapshot=snapshot,
                        provider=get_settings().default_provider.value,
                        model=getattr(provider, "model", None))
            # Evaluation captures per-task errors so other tasks can finish.
            errors = [error for result in report.results for error in result.errors]
            if errors:
                raise RuntimeError(errors[0])
            await emit("eval_complete", {"pass_rate": report.pass_rate})
        except Exception as exc:
            status.error = friendly_provider_error(exc) or str(exc)
            await emit("eval_failed", {"message": status.error})
        finally:
            status.running = False

    _run_task = asyncio.create_task(execute())
    return EvalRunAccepted(run_id=run_id, tasks=len(tasks))
