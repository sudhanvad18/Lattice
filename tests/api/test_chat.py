"""Tests for the grounded-chat endpoint (POST /chat)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lattice.graph.engine import NetworkXBackend
from lattice.inference.mock import MockProvider
from lattice.inference.provider import GenerationConfig, GenerationResult, LLMProvider, Message
from lattice.ingestion.chunker import ChunkConfig
from lattice.knowledge_base import KnowledgeBase
from lattice_api.app import create_app

_SMALL_CHUNKS = ChunkConfig(min_chunk_tokens=1, max_tokens=512)


@pytest.fixture
def kb(tmp_path, monkeypatch):
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    monkeypatch.setattr(
        ONNXMiniLM_L6_V2, "__call__", lambda self, input: [[0.1] * 384 for _ in input]
    )
    knowledge_base = KnowledgeBase(
        persist_dir=tmp_path, graph_backend=NetworkXBackend(), chunk_config=_SMALL_CHUNKS
    )
    knowledge_base.ingest_text(
        "Fan blade erosion is caused by sand ingestion.", source="eng_doc.txt"
    )
    knowledge_base.ingest_text(
        "Turbine maintenance requires weekly inspections.", source="maint_guide.txt"
    )
    yield knowledge_base
    knowledge_base.close()


@pytest.fixture
def empty_kb(tmp_path, monkeypatch):
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    monkeypatch.setattr(
        ONNXMiniLM_L6_V2, "__call__", lambda self, input: [[0.1] * 384 for _ in input]
    )
    knowledge_base = KnowledgeBase(
        persist_dir=tmp_path, graph_backend=NetworkXBackend(), chunk_config=_SMALL_CHUNKS
    )
    yield knowledge_base
    knowledge_base.close()


def test_empty_kb_returns_409(empty_kb):
    client = TestClient(create_app(knowledge_base=empty_kb))
    response = client.post("/chat", json={"message": "What causes erosion?"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "kb_empty"


def test_answer_mode_cites_sources(kb, monkeypatch):
    scripted = "Fan blade erosion occurs due to sand [1]."
    provider = MockProvider(responses=[scripted])
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)

    client = TestClient(create_app(knowledge_base=kb))
    response = client.post("/chat", json={"message": "What causes fan blade erosion?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "answer"
    assert body["response"] == scripted
    # Only [1] was cited → one citation
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source"] == "eng_doc.txt"
    assert (citation["index"], citation["kind"]) == (1, "chunk")
    assert citation["id"]
    assert "sand ingestion" in citation["snippet"]


def test_citation_index_matches_marker_and_prompt_has_full_chunk(kb, monkeypatch):
    long_tail = "Inspection detail. " * 30
    kb.ingest_text("Fan blade erosion detail. " + long_tail, source="long.txt")
    provider = MockProvider(responses=["See [2]."])
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)

    body = TestClient(create_app(knowledge_base=kb)).post(
        "/chat", json={"message": "fan blade erosion inspection detail"}
    ).json()

    assert [c["index"] for c in body["citations"]] == [2]
    assert long_tail.strip() in provider.last_messages[-1].content

    # Prompt sent to provider contains the chunk text
    user_msg = provider.last_messages[-1].content
    assert "sand ingestion" in user_msg


def test_history_forwarded_last_6_turns(kb, monkeypatch):
    scripted = "Based on [1], erosion is caused by sand."
    provider = MockProvider(responses=[scripted])
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)

    client = TestClient(create_app(knowledge_base=kb))

    # 7 turns: answer_question must forward only the last 6
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
        for i in range(7)
    ]
    response = client.post(
        "/chat",
        json={"message": "What causes fan blade erosion?", "history": history},
    )
    assert response.status_code == 200

    messages = provider.last_messages
    assert messages is not None
    # system + 6 history turns + user (with Sources+question) = 8
    assert len(messages) == 8
    # First history turn (msg-0) must not appear as a standalone message
    history_msgs = [
        m for m in messages
        if m.role in ("user", "assistant") and m.content.startswith("msg-")
    ]
    assert len(history_msgs) == 6
    contents = [m.content for m in history_msgs]
    assert "msg-0" not in contents   # dropped (7th oldest)
    assert "msg-6" in contents       # kept (most recent)


def test_provider_connect_error_returns_retrieval_only(kb, monkeypatch):
    import httpx

    class _ConnectErrorProvider(LLMProvider):
        async def generate(
            self, messages: list[Message], config: GenerationConfig | None = None
        ) -> GenerationResult:
            raise httpx.ConnectError("Connection refused")

        def generate_sync(
            self, messages: list[Message], config: GenerationConfig | None = None
        ) -> GenerationResult:
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("lattice_api.app.get_provider", lambda: _ConnectErrorProvider())

    client = TestClient(create_app(knowledge_base=kb))
    response = client.post("/chat", json={"message": "What causes fan blade erosion?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "retrieval_only"
    assert body["notice"] is not None
    assert "ollama serve" in body["notice"].lower()
    # Response must contain retrieved passage text (not empty)
    assert len(body["response"]) > 0
    assert "eng_doc.txt" in body["response"] or "sand" in body["response"]


def test_no_relevant_results_skips_model(kb, monkeypatch):
    provider = MockProvider(responses=["Should not be called"])
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)

    from lattice.retrieval.vector_store import SearchResult

    # Monkeypatch instance to return only low-similarity results
    monkeypatch.setattr(
        kb,
        "vector_search",
        lambda query, top_k=5: [SearchResult("c1", "unrelated", {"source": "s"}, 0.05)],
    )

    client = TestClient(create_app(knowledge_base=kb))
    response = client.post("/chat", json={"message": "What is the boiling point of water?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert provider.call_count == 0
    assert "doesn't cover that yet" in body["response"]


def _stream_events(response):
    import json

    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.strip().split("\n\n"):
        name, payload = frame.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(payload.removeprefix("data: "))))
    return events


def test_stream_gates_empty_kb_before_sending_headers(empty_kb):
    response = TestClient(create_app(knowledge_base=empty_kb)).post(
        "/chat/stream", json={"message": "What causes erosion?"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "kb_empty"


def test_stream_emits_progress_deltas_and_final_citations(kb, monkeypatch):
    class StreamingProvider(MockProvider):
        async def stream(self, messages, config=None):
            self._call_history.append(messages)
            yield "Sand causes "
            yield "erosion [1]."

    provider = StreamingProvider()
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)
    response = TestClient(create_app(knowledge_base=kb)).post(
        "/chat/stream", json={"message": "What causes fan blade erosion?"},
    )
    events = _stream_events(response)
    assert [name for name, data in events] == [
        "status", "sources", "status", "delta", "delta", "done",
    ]
    assert "sand ingestion" in provider.last_messages[-1].content
    assert "".join(data["text"] for name, data in events if name == "delta") == (
        "Sand causes erosion [1]."
    )
    final = events[-1][1]
    assert final["response"] == "Sand causes erosion [1]."
    assert final["mode"] == "answer"
    assert [c["index"] for c in final["citations"]] == [1]


@pytest.mark.parametrize("partial", [False, True])
def test_stream_provider_failure_preserves_partial_or_uses_retrieval(kb, monkeypatch, partial):
    import httpx

    class BrokenStream(MockProvider):
        async def stream(self, messages, config=None):
            if partial:
                yield "Sand causes erosion [1]"
            raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr("lattice_api.app.get_provider", lambda: BrokenStream())
    events = _stream_events(TestClient(create_app(knowledge_base=kb)).post(
        "/chat/stream", json={"message": "What causes fan blade erosion?"},
    ))
    final = events[-1][1]
    assert events[-1][0] == "done"
    assert final["notice"]
    assert "All connection attempts failed" not in final["notice"]
    assert final["citations"]
    if partial:
        assert final["response"] == "Sand causes erosion [1]"
        assert "partial" in final["notice"]
    else:
        assert final["mode"] == "retrieval_only"
        assert "Retrieved passages" in final["response"]


def test_stream_no_coverage_skips_provider(kb, monkeypatch):
    provider = MockProvider()
    monkeypatch.setattr("lattice_api.app.get_provider", lambda: provider)
    monkeypatch.setattr(kb, "vector_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(kb, "graph_search", lambda *args, **kwargs: [])
    events = _stream_events(TestClient(create_app(knowledge_base=kb)).post(
        "/chat/stream", json={"message": "Uncovered topic"},
    ))
    assert [name for name, _ in events] == ["status", "done"]
    assert "doesn't cover" in events[-1][1]["response"]
    assert provider.call_count == 0


def test_stream_provider_configuration_failure_returns_cited_fallback(kb, monkeypatch):
    def unavailable():
        raise ValueError("missing secret credentials")

    monkeypatch.setattr("lattice_api.app.get_provider", unavailable)
    events = _stream_events(TestClient(create_app(knowledge_base=kb)).post(
        "/chat/stream", json={"message": "fan blade erosion"},
    ))
    assert events[-1][1]["mode"] == "retrieval_only"
    assert events[-1][1]["citations"]
    assert "secret credentials" not in events[-1][1]["notice"]


def test_stream_retrieval_error_is_an_explicit_error_event(kb, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private storage error")

    monkeypatch.setattr("lattice_api.app.get_provider", lambda: MockProvider())
    monkeypatch.setattr(kb, "vector_search", fail)
    events = _stream_events(TestClient(create_app(knowledge_base=kb)).post(
        "/chat/stream", json={"message": "erosion"},
    ))
    assert [name for name, _ in events] == ["status", "error"]
    assert "private storage" not in events[-1][1]["message"]
