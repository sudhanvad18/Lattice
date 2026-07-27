"""
Checkpoint and crash recovery system.

Provides state persistence so long-running agent tasks can survive
process restarts, crashes, or network failures.

Two backends:
- LocalCheckpointStore: JSON file-based (zero dependencies, dev/demo)
- RedisCheckpointStore: Redis-backed (production, shared state)

Usage:
    store = LocalCheckpointStore("checkpoints/")
    await store.save("task-123", state.model_dump())
    # ... crash happens ...
    recovered = await store.load("task-123")
    state = AgentState(**recovered)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class CheckpointStore(ABC):
    """Abstract checkpoint storage interface."""

    @abstractmethod
    async def save(self, task_id: str, state: dict[str, Any]) -> None:
        """Save a checkpoint for a task."""
        ...

    @abstractmethod
    async def load(self, task_id: str) -> Optional[dict[str, Any]]:
        """Load the latest checkpoint for a task."""
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Delete all checkpoints for a task."""
        ...

    @abstractmethod
    async def list_tasks(self) -> list[str]:
        """List all tasks with saved checkpoints."""
        ...


class LocalCheckpointStore(CheckpointStore):
    """File-based checkpoint store for development.

    Each task gets a JSON file with its latest state.
    Simple, debuggable, zero dependencies.
    """

    def __init__(self, checkpoint_dir: str | Path = "data/checkpoints") -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, task_id: str) -> Path:
        safe_id = task_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_id}.json"

    async def save(self, task_id: str, state: dict[str, Any]) -> None:
        checkpoint = {
            "task_id": task_id,
            "state": state,
            "saved_at": datetime.utcnow().isoformat(),
        }
        path = self._path_for(task_id)
        path.write_text(json.dumps(checkpoint, default=str, indent=2))
        logger.debug("checkpoint_saved", task_id=task_id, path=str(path))

    async def load(self, task_id: str) -> Optional[dict[str, Any]]:
        path = self._path_for(task_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            logger.info("checkpoint_loaded", task_id=task_id)
            return data["state"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("checkpoint_load_failed", task_id=task_id, error=str(e))
            return None

    async def delete(self, task_id: str) -> None:
        path = self._path_for(task_id)
        if path.exists():
            path.unlink()
            logger.debug("checkpoint_deleted", task_id=task_id)

    async def list_tasks(self) -> list[str]:
        return [
            p.stem for p in self._dir.glob("*.json")
        ]


class RedisCheckpointStore(CheckpointStore):
    """Redis-backed checkpoint store for production.

    Uses Redis for shared state across multiple workers,
    with automatic expiry for completed tasks.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "lattice:checkpoint:",
        ttl_seconds: int = 86400 * 7,  # 7 days
    ) -> None:
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._redis_url = redis_url
        self._client: Any = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._redis_url)
        return self._client

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    async def save(self, task_id: str, state: dict[str, Any]) -> None:
        client = await self._get_client()
        checkpoint = {
            "task_id": task_id,
            "state": state,
            "saved_at": datetime.utcnow().isoformat(),
        }
        await client.setex(
            self._key(task_id),
            self._ttl,
            json.dumps(checkpoint, default=str),
        )
        logger.debug("checkpoint_saved_redis", task_id=task_id)

    async def load(self, task_id: str) -> Optional[dict[str, Any]]:
        client = await self._get_client()
        data = await client.get(self._key(task_id))
        if not data:
            return None
        try:
            checkpoint = json.loads(data)
            logger.info("checkpoint_loaded_redis", task_id=task_id)
            return checkpoint["state"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("checkpoint_load_failed_redis", task_id=task_id, error=str(e))
            return None

    async def delete(self, task_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(task_id))

    async def list_tasks(self) -> list[str]:
        client = await self._get_client()
        keys = await client.keys(f"{self._prefix}*")
        return [k.decode().removeprefix(self._prefix) for k in keys]
