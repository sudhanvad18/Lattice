import json

import httpx
import pytest

from lattice.agents.openclaw import create_web_registry
from lattice.agents.tools import ToolCall

DOCS = "https://docs.example.com/integration"
FINAL = "https://docs.example.com/api"


@pytest.fixture
def gateway(monkeypatch):
    requests = []
    responses = []
    client = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    return requests, responses


def envelope(details):
    return httpx.Response(200, json={"ok": True, "result": {"details": details}})


@pytest.mark.parametrize("field", ["results", "citations"])
async def test_search_then_fetch_tracks_only_read_pages(gateway, field):
    requests, responses = gateway
    responses.extend([
        envelope({field: [{"title": "Reference", "url": DOCS}]}),
        httpx.Response(200, json={"ok": True, "result": {"content": [{
            "type": "text", "text": json.dumps({
                "url": DOCS, "finalUrl": FINAL, "text": "public passage", "status": 200,
            }),
        }]}}),
    ])
    registry = create_web_registry("http://localhost:18789/", "owner-token")
    search = await registry.invoke(ToolCall("web_search", {"query": "state machine guidance"}))
    assert search.success and DOCS in search.content and "chunk_ids" not in search.metadata
    unknown = await registry.invoke(ToolCall("web_fetch", {"url": FINAL}))
    assert not unknown.success
    read = await registry.invoke(ToolCall("web_fetch", {"url": DOCS}))
    assert read.success and read.metadata["chunk_ids"] == [FINAL]
    assert read.content == f"[{FINAL}]\npublic passage"
    assert all(str(req.url) == "http://localhost:18789/tools/invoke" for req in requests)
    assert all(req.headers["authorization"] == "Bearer owner-token" for req in requests)
    assert json.loads(requests[0].content) == {
        "tool": "web_search", "args": {"query": "state machine guidance", "count": 3},
    }
    assert json.loads(requests[1].content)["args"] == {
        "url": DOCS, "extractMode": "text", "maxChars": 4000,
    }
    other = create_web_registry("http://localhost:18789")
    assert not (await other.invoke(ToolCall("web_fetch", {"url": DOCS}))).success


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/test", "https://service.local/", "http://[::ffff:127.0.0.1]/",
    "file:///etc/passwd", "https://docs.example.com@localhost/", "https://a:b@docs.example.com/",
])
async def test_internal_or_credentialed_candidates_are_never_requested(gateway, url):
    requests, responses = gateway
    responses.append(envelope({"results": [{"url": url}]}))
    registry = create_web_registry("http://localhost:18789")
    search = await registry.invoke(ToolCall("web_search", {"query": "python library"}))
    assert url not in search.content
    assert not (await registry.invoke(ToolCall("web_fetch", {"url": url}))).success
    assert len(requests) == 1


async def test_remote_failures_and_budget(gateway):
    requests, responses = gateway
    responses.extend([httpx.Response(403), httpx.Response(200, content=b" " * 140_000)])
    registry = create_web_registry("http://localhost:18789")
    assert not (await registry.invoke(ToolCall("web_search", {"query": "/Users/me/app"}))).success
    query = ToolCall("web_search", {"query": "Python typing"})
    assert "403" in (await registry.invoke(query)).content
    assert "exceeded" in (await registry.invoke(query)).content
    assert "budget reached" in (await registry.invoke(query)).content
    assert len(requests) == 2


@pytest.mark.parametrize("override", [
    {"finalUrl": "http://127.0.0.1"}, {"text": ""}, {"status": 404}, {"url": FINAL},
])
async def test_unusable_pages_do_not_produce_citations(gateway, override):
    _, responses = gateway
    responses.extend([
        envelope({"results": [{"url": DOCS}]}),
        envelope({"url": DOCS, "status": 200, "text": "contents", **override}),
    ])
    registry = create_web_registry("http://localhost:18789")
    await registry.invoke(ToolCall("web_search", {"query": "documentation"}))
    result = await registry.invoke(ToolCall("web_fetch", {"url": DOCS}))
    assert not result.success and not result.metadata
