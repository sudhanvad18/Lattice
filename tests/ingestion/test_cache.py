"""Tests for SHA256 ingestion cache."""

import json
from pathlib import Path

import pytest

from lattice.ingestion.cache import IngestionCache
from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType


def _make_chunks(n: int = 2) -> list[Chunk]:
    return [
        Chunk(
            document_id="doc1",
            content=f"Chunk content number {i}",
            index=i,
            metadata=DocumentMetadata(source="test.txt", doc_type=DocumentType.TEXT),
        )
        for i in range(n)
    ]


class TestIngestionCache:

    def test_cache_miss_on_new_file(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "doc.txt"
        f.write_text("Hello world")

        assert cache.is_cached(f) is False
        assert cache.load(f) is None

    def test_cache_hit_after_save(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "doc.txt"
        f.write_text("Hello world")

        chunks = _make_chunks(3)
        cache.save(f, chunks)

        assert cache.is_cached(f) is True
        loaded = cache.load(f)
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0].content == "Chunk content number 0"
        assert loaded[2].content == "Chunk content number 2"

    def test_cache_invalidated_on_content_change(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "doc.txt"
        f.write_text("Version 1")

        chunks = _make_chunks()
        cache.save(f, chunks)
        assert cache.is_cached(f) is True

        f.write_text("Version 2 — content changed")
        assert cache.is_cached(f) is False
        assert cache.load(f) is None

    def test_cache_for_multiple_files(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")

        f1 = tmp_path / "a.txt"
        f1.write_text("File A")
        f2 = tmp_path / "b.txt"
        f2.write_text("File B")

        cache.save(f1, _make_chunks(1))
        cache.save(f2, _make_chunks(2))

        assert cache.is_cached(f1) is True
        assert cache.is_cached(f2) is True
        assert len(cache.load(f1)) == 1
        assert len(cache.load(f2)) == 2

    def test_invalidate(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "doc.txt"
        f.write_text("Content")

        cache.save(f, _make_chunks())
        assert cache.is_cached(f) is True

        result = cache.invalidate(f)
        assert result is True
        assert cache.is_cached(f) is False

    def test_invalidate_nonexistent(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "nope.txt"
        f.write_text("Content")
        assert cache.invalidate(f) is False

    def test_clear(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")

        for i in range(5):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"Content {i}")
            cache.save(f, _make_chunks(1))

        cleared = cache.clear()
        assert cleared == 5

        for i in range(5):
            f = tmp_path / f"file_{i}.txt"
            assert cache.is_cached(f) is False

    def test_corrupt_index_recovery(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        index_path = cache_dir / "index.json"
        index_path.write_text("THIS IS NOT JSON {{{{")

        cache = IngestionCache(cache_dir)
        f = tmp_path / "doc.txt"
        f.write_text("Hello")
        assert cache.is_cached(f) is False

    def test_corrupt_chunk_file_recovery(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        f = tmp_path / "doc.txt"
        f.write_text("Content")

        cache.save(f, _make_chunks())
        assert cache.is_cached(f) is True

        # Corrupt the chunk file
        import hashlib
        h = hashlib.sha256(b"Content").hexdigest()
        chunk_file = tmp_path / "cache" / "chunks" / f"{h}.json"
        chunk_file.write_text("CORRUPTED{{{")

        loaded = cache.load(f)
        assert loaded is None

    def test_nonexistent_file_path(self, tmp_path):
        cache = IngestionCache(tmp_path / "cache")
        missing = tmp_path / "does_not_exist.txt"

        assert cache.is_cached(missing) is False
        assert cache.load(missing) is None

    def test_index_persistence(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache1 = IngestionCache(cache_dir)
        f = tmp_path / "doc.txt"
        f.write_text("Persistent content")
        cache1.save(f, _make_chunks())

        cache2 = IngestionCache(cache_dir)
        assert cache2.is_cached(f) is True
        loaded = cache2.load(f)
        assert loaded is not None
        assert len(loaded) == 2
