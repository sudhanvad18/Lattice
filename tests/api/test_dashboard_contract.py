"""Dashboard-facing contracts: task body, retrieval chat, and activity events."""

import asyncio

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity
from lattice.inference.mock import MockProvider
from lattice.retrieval.vector_store import SearchResult
from lattice_api.app import _stream_task_events, create_app
from lattice_api.chat import build_chat_answer
from lattice_api.task_manager import TaskManager

PLAN = (
    '{"subtasks": ['
    '{"description": "Research", "assigned_to": "researcher", "depends_on": []},'
    '{"description": "Write", "assigned_to": "writer", "depends_on": [0]}'
    "]}"
)
RESEARCH = '{"summary": "Found data", "confidence": 0.9}'
DOCUMENT = "# Generated Document\n\nContent here."
APPROVAL = (
    '{"approved": true, "confidence": 0.9, "issues": [], "suggestions": [], "reasoning": "Good."}'
)


class _GraphKB:
    """Real graph search plus a fixed vector hit. No Chroma and no model."""

    def __init__(self) -> None:
        self.graph = NetworkXBackend()
        self.graph.add_entity(Entity(
            id="ent-blade",
            name="Fan Blade",
            entity_type="Component",
            description="Erodes when sand hits the leading edge.",
        ))
        self.chunks = [
            SearchResult(
                chunk_id="chunk-1",
                content="Fan blade erosion is caused by sand ingestion.",
                metadata={},
                similarity=0.9,
            )
        ]

    def vector_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.chunks[:top_k]

    def graph_search(self, query: str, entity_type: str | None = None) -> list[Entity]:
        return self.graph.search_entities(query, entity_type=entity_type)


class _EmptyKB:
    def vector_search(self, query: str, top_k: int = 5) -> list:
        return []

    def graph_search(self, query: str, entity_type: str | None = None) -> list:
        return []


class _FailingKB:
    def vector_search(self, query: str, top_k: int = 5) -> list:
        raise RuntimeError("vector store unavailable")

    def graph_search(self, query: str, entity_type: str | None = None) -> list:
        return []


def _manager() -> TaskManager:
    # Two tasks each consume a plan, research note, document, and approval.
    return TaskManager(provider=MockProvider(
        responses=[PLAN, RESEARCH, DOCUMENT, APPROVAL] * 4,
    ))


def _client(kb: object | None = None) -> TestClient:
    return TestClient(create_app(task_manager=_manager(), knowledge_base=kb))


class _FakeSocket:
    """Records what a task stream sends. Enough for `_stream_task_events`."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


async def test_completed_task_keeps_the_submitted_id():
    manager = _manager()
    state = await manager.submit_task("Document the API")
    submitted = state.task_id
    await manager._running[submitted]

    stored = manager.get_task(submitted)
    assert stored is not None
    assert stored.task_id == submitted
    assert stored.status == "completed"
    assert stored.original_task == "Document the API"
    assert any(artifact.artifact_type == "document" for artifact in stored.artifacts)


def test_submit_rejects_goal_field():
    client = _client(_EmptyKB())
    response = client.post("/tasks", json={"goal": "Write docs"})
    assert response.status_code == 422


def test_chat_cites_passages_and_entities_for_a_question():
    kb = _GraphKB()
    response, citations = build_chat_answer(kb, "What causes fan blade erosion?")
    assert "sand ingestion" in response
    assert "Fan Blade" in response
    assert citations == ["chunk-1", "ent-blade"]

    client = TestClient(create_app(task_manager=_manager(), knowledge_base=kb))
    http = client.post("/chat", json={"message": "What causes fan blade erosion?"})
    assert http.status_code == 200
    body = http.json()
    assert body["response"] == response
    assert body["citations"] == ["chunk-1", "ent-blade"]


def test_chat_says_when_nothing_matches():
    client = _client(_EmptyKB())
    response = client.post("/chat", json={"message": "unrelated question"})
    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert "no matching" in response.json()["response"].lower()


def test_chat_retrieval_failure_is_unavailable():
    client = _client(_FailingKB())
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503


def test_chat_rejects_blank_message():
    client = _client(_EmptyKB())
    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_activity_socket_replays_real_task_ids():
    manager = _manager()
    client = TestClient(create_app(task_manager=manager, knowledge_base=_EmptyKB()))
    first = client.post("/tasks", json={"task": "First task"}).json()["task_id"]
    second = client.post("/tasks", json={"task": "Second task"}).json()["task_id"]
    expected = [
        event for event in manager.recent_events()
        if event["event"] == "task_created"
    ]
    assert [event["task_id"] for event in expected] == [first, second]

    with client.websocket_connect("/ws/activity") as ws:
        messages = [ws.receive_json() for _ in expected]

    created = [message for message in messages if message["event"] == "task_created"]
    assert [message["task_id"] for message in created] == [first, second]
    assert created[0]["data"]["task"] == "First task"
    assert created[0]["timestamp"]


def test_task_socket_replays_only_that_task_then_closes():
    manager = _manager()
    client = TestClient(create_app(task_manager=manager, knowledge_base=_EmptyKB()))
    task_id = client.post("/tasks", json={"task": "Document the API"}).json()["task_id"]
    client.post("/tasks", json={"task": "Other task"})
    manager._emit_event(task_id, "task_complete", {"status": "completed", "artifacts": 1})

    messages = []
    with client.websocket_connect(f"/ws/{task_id}") as ws:
        while True:
            try:
                messages.append(ws.receive_json())
            except WebSocketDisconnect:
                break

    assert [message["event"] for message in messages] == ["task_created", "task_complete"]
    assert {message["task_id"] for message in messages} == {task_id}
    assert messages[-1]["data"]["status"] == "completed"


async def test_activity_stream_pushes_events_after_subscribe():
    manager = _manager()
    create_app(task_manager=manager, knowledge_base=_EmptyKB())
    socket = _FakeSocket()
    stream = asyncio.create_task(_stream_task_events(socket, None))
    await asyncio.sleep(0)
    assert socket.accepted
    assert socket.sent == []

    manager._emit_event("live-task", "task_created", {"task": "Live task"})
    await asyncio.sleep(0)

    stream.cancel()
    await asyncio.gather(stream, return_exceptions=True)
    assert socket.sent[0]["event"] == "task_created"
    assert socket.sent[0]["task_id"] == "live-task"
    assert socket.sent[0]["data"]["task"] == "Live task"
    assert socket.sent[0]["timestamp"]
