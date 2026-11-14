"""
SHA256-based ingestion cache.

Avoids re-processing unchanged files when re-ingesting a directory.
Cache is stored on disk as a JSON index + per-hash chunk files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from lattice.ingestion.models import Chunk

logger = structlog.get_logger()


class IngestionCache:
    """Content-hash cache for the ingestion pipeline.

    Maps SHA256(file_content) -> cached chunk data so unchanged files
    are skipped on re-ingestion. Chunk content is stored in separate
    files to keep the index lightweight.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._index_path = self._cache_dir / "index.json"
        self._chunks_dir = self._cache_dir / "chunks"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Load cache index from disk, recovering from corruption."""
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("cache_index_corrupt", error=str(e))
        return {}

    def _save_index(self) -> None:
        """Persist cache index to disk."""
        self._index_path.write_text(
            json.dumps(self._index, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _hash_content(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def is_cached(self, path: Path) -> bool:
        """Check if a file's content hash exists in the cache."""
        try:
            content = Path(path).read_bytes()
        except OSError:
            return False
        h = self._hash_content(content)
        if h not in self._index:
            return False
        chunk_file = self._chunks_dir / f"{h}.json"
        return chunk_file.exists()

    def load(self, path: Path) -> Optional[list[Chunk]]:
        """Load cached chunks for a file, or None on miss/corruption."""
        try:
            content = Path(path).read_bytes()
        except OSError:
            return None
        h = self._hash_content(content)
        if h not in self._index:
            return None
        chunk_file = self._chunks_dir / f"{h}.json"
        if not chunk_file.exists():
            return None
        try:
            raw = json.loads(chunk_file.read_text(encoding="utf-8"))
            return [Chunk(**c) for c in raw]
        except (json.JSONDecodeError, OSError, Exception) as e:
            logger.warning("cache_chunk_load_failed", hash=h, error=str(e))
            self._index.pop(h, None)
            self._save_index()
            return None

    def save(self, path: Path, chunks: list[Chunk]) -> None:
        """Cache chunks for a file by its content hash."""
        try:
            content = Path(path).read_bytes()
        except OSError:
            return
        h = self._hash_content(content)

        chunk_data = [c.model_dump(mode="json", exclude={"embedding"}) for c in chunks]
        chunk_file = self._chunks_dir / f"{h}.json"
        chunk_file.write_text(json.dumps(chunk_data, default=str), encoding="utf-8")

        self._index[h] = {
            "source": str(path),
            "chunk_count": len(chunks),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_index()
        logger.debug("cache_saved", path=str(path), hash=h[:12], chunks=len(chunks))

    def invalidate(self, path: Path) -> bool:
        """Remove a file's cache entry. Returns True if it existed."""
        try:
            content = Path(path).read_bytes()
        except OSError:
            return False
        h = self._hash_content(content)
        if h in self._index:
            self._index.pop(h)
            chunk_file = self._chunks_dir / f"{h}.json"
            if chunk_file.exists():
                chunk_file.unlink()
            self._save_index()
            return True
        return False

    def clear(self) -> int:
        """Clear entire cache. Returns number of entries removed."""
        count = len(self._index)
        self._index = {}
        self._save_index()
        for f in self._chunks_dir.glob("*.json"):
            f.unlink()
        return count
