"""Structured API errors the dashboard can explain (see `{"detail": {"code", "message"}}`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException

from lattice.inference.health import check_provider

if TYPE_CHECKING:
    from lattice.knowledge_base import KnowledgeBase

KB_EMPTY_MESSAGE = (
    "The knowledge base is empty. Ingest a folder from the Knowledge Base view "
    "or run `lattice ingest <path>` first."
)


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def require_kb(kb: KnowledgeBase | None) -> KnowledgeBase:
    """409 `kb_empty` unless the knowledge base has at least one chunk."""
    if kb is None or kb.is_empty:
        raise api_error(409, "kb_empty", KB_EMPTY_MESSAGE)
    return kb


async def require_provider() -> None:
    """503 `provider_unavailable` unless the configured model answers a probe."""
    health = await check_provider()
    if not health.reachable:
        raise api_error(503, "provider_unavailable", health.detail or "The model is unavailable.")
