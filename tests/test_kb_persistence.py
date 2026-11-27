"""Graph snapshots and knowledge base lifecycle, without embedding downloads."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from chromadb.api.models.CollectionCommon import CollectionCommon

from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity, Relation
from lattice.knowledge_base import KnowledgeBase

SAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "aerospace"
          / "fan_blade_erosion.md").read_text()


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.setattr(CollectionCommon, "_embed", lambda self, input, **kw: [
        [1.0, float(len(text) % 17) / 17, 0.5] for text in input
    ])


def test_graph_round_trip(tmp_path):
    graph = NetworkXBackend()
    source = Entity(name="Fan Blade", entity_type="Component", community=3,
                    properties={"alloy": "Ti-6Al-4V"}, source_chunk_ids=["chunk-1"])
    target = Entity(name="Inspection", entity_type="Procedure", community=4)
    graph.add_entity(source)
    graph.add_entity(target)
    relation = Relation(source_id=source.id, target_id=target.id, relation_type="REQUIRES",
                        provenance="inferred", confidence=0.7, properties={"reason": "wear"})
    graph.add_relation(relation)
    path = tmp_path / "nested" / "graph.json"
    graph.save(path)
    snapshot = json.loads(path.read_text())
    assert snapshot["version"] == 1
    restored = NetworkXBackend.load(path)
    assert restored.get_entity(source.id) == source
    assert restored.get_all_relations() == [relation]
    assert restored._graph.nodes[source.id]["community"] == 3
    assert list(path.parent.iterdir()) == [path]


def test_kb_reloads_graph_and_vectors(tmp_path):
    first = KnowledgeBase(tmp_path)
    assert first.ingest_text(SAMPLE) > 0
    expected = first.stats
    entities = [e.model_dump() for e in first.graph.get_all_entities()]
    first.close()
    second = KnowledgeBase(tmp_path)
    try:
        assert second.stats == expected
        assert [e.model_dump() for e in second.graph.get_all_entities()] == entities
        assert second.vector_search("blade")
    finally:
        second.close()


def test_identical_files_ingest_with_distinct_sources_and_stable_ids(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    paths = [sources / "first.md", sources / "second.md"]
    for path in paths:
        path.write_text(SAMPLE)
    kb = KnowledgeBase(tmp_path / "kb")
    try:
        assert kb.ingest_directory(sources) > 0
        original_ids = kb.vector_store.chunk_ids()
        counts = kb.source_counts()
        assert set(counts) == {str(path) for path in paths}
        assert counts[str(paths[0])] == counts[str(paths[1])] > 0
        assert kb.stats["chunks_stored"] == sum(counts.values())

        kb.ingest_directory(sources)
        assert kb.vector_store.chunk_ids() == original_ids
        assert kb.source_counts() == counts
    finally:
        kb.close()


@pytest.mark.parametrize("content", ["{broken", '{"version": 99}',
                                     '{"version": 1, "entities": [{}], "relations": []}'])
def test_corrupt_graph_recovers(tmp_path, content, monkeypatch):
    (tmp_path / "graph.json").write_text(content)
    warning = Mock()
    monkeypatch.setattr("lattice.knowledge_base.logger.warning", warning)
    kb = KnowledgeBase(tmp_path)
    try:
        assert kb.stats["kg_entities"] == 0
        warning.assert_called_once()
        kb.ingest_text(SAMPLE)
        assert NetworkXBackend.load(tmp_path / "graph.json").num_entities > 0
    finally:
        kb.close()


def test_reset_clears_disk_and_cache_even_when_disabled(tmp_path):
    source = tmp_path / "source.md"
    source.write_text(SAMPLE)
    kb = KnowledgeBase(tmp_path)
    kb.ingest_file(source)
    assert kb._pipeline._cache.is_cached(source)
    kb.reset()
    assert not kb._pipeline._cache.is_cached(source)
    kb.ingest_file(source)
    kb.close()
    uncached = KnowledgeBase(tmp_path, enable_cache=False)
    try:
        uncached.reset()
        assert uncached.stats["chunks_stored"] == 0
        assert uncached.stats["kg_entities"] == 0
        assert not (tmp_path / "graph.json").exists()
        assert json.loads((tmp_path / "ingestion_cache" / "index.json").read_text()) == {}
        uncached.reset()
    finally:
        uncached.close()
    reloaded = KnowledgeBase(tmp_path)
    try:
        assert reloaded.stats["kg_entities"] == 0
        assert reloaded.stats["chunks_stored"] == 0
    finally:
        reloaded.close()


def test_external_backend_is_not_snapshotted(tmp_path):
    backend = Mock(wraps=NetworkXBackend())
    backend.clear = Mock()
    kb = KnowledgeBase(tmp_path, graph_backend=backend)
    try:
        kb.ingest_text(SAMPLE)
        assert not (tmp_path / "graph.json").exists()
        backend.save.assert_not_called()
        kb.reset()
        backend.clear.assert_called_once()
    finally:
        kb.close()


def test_save_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "graph.json"
    graph = NetworkXBackend()
    graph.save(path)
    previous = path.read_bytes()
    graph.add_entity(Entity(name="Blade", entity_type="Component"))

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("lattice.graph.engine.os.replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        graph.save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
