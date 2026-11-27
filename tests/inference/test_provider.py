"""
Tests for the inference provider abstraction.
"""

import pytest

from lattice.inference.mock import MockProvider
from lattice.inference.provider import (
    GenerationConfig,
    GenerationResult,
    Message,
    OllamaProvider,
    get_provider,
)
from lattice.config import InferenceProvider


class TestMockProvider:
    """Test the mock provider (used in all agent tests)."""

    def test_default_response(self):
        provider = MockProvider()
        result = provider.generate_sync([Message(role="user", content="Hello")])
        assert result.content == "Mock LLM response."
        assert result.provider == "mock"
        assert result.model == "mock-model"

    def test_custom_responses_consumed_in_order(self):
        provider = MockProvider(responses=["First", "Second", "Third"])
        r1 = provider.generate_sync([Message(role="user", content="1")])
        r2 = provider.generate_sync([Message(role="user", content="2")])
        r3 = provider.generate_sync([Message(role="user", content="3")])
        assert r1.content == "First"
        assert r2.content == "Second"
        assert r3.content == "Third"

    def test_falls_back_to_default_when_responses_exhausted(self):
        provider = MockProvider(responses=["Only one"], default_response="Fallback")
        r1 = provider.generate_sync([Message(role="user", content="1")])
        r2 = provider.generate_sync([Message(role="user", content="2")])
        assert r1.content == "Only one"
        assert r2.content == "Fallback"

    def test_tracks_call_count(self):
        provider = MockProvider()
        assert provider.call_count == 0
        provider.generate_sync([Message(role="user", content="1")])
        provider.generate_sync([Message(role="user", content="2")])
        assert provider.call_count == 2

    def test_tracks_last_messages(self):
        provider = MockProvider()
        msgs = [Message(role="user", content="Test message")]
        provider.generate_sync(msgs)
        assert provider.last_messages == msgs

    @pytest.mark.asyncio
    async def test_async_generate(self):
        provider = MockProvider(responses=["Async response"])
        result = await provider.generate([Message(role="user", content="Hi")])
        assert result.content == "Async response"
        assert provider.call_count == 1

    def test_usage_tracking(self):
        provider = MockProvider(responses=["A short response"])
        result = provider.generate_sync([Message(role="user", content="Hi")])
        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 3  # "A short response" = 3 words


class TestProviderFactory:
    """Test the get_provider factory function."""

    def test_creates_ollama_provider(self):
        provider = get_provider(InferenceProvider.OLLAMA)
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_requires_api_key(self):
        with pytest.raises(ValueError, match="LATTICE_ANTHROPIC_API_KEY"):
            get_provider(InferenceProvider.ANTHROPIC)

    def test_openai_requires_api_key(self):
        with pytest.raises(ValueError, match="LATTICE_OPENAI_API_KEY"):
            get_provider(InferenceProvider.OPENAI)


@pytest.mark.asyncio
async def test_custom_provider_stream_falls_back_to_generate():
    provider = MockProvider(responses=["Existing custom provider"])
    assert [text async for text in provider.stream([Message("user", "hi")])] == [
        "Existing custom provider",
    ]
    assert provider.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_ollama_stream_delivers_text_before_generation_finishes(monkeypatch, close_early):
    import asyncio
    import json

    import httpx

    release = asyncio.Event()
    requests = []

    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"message":{"content":"First "},"done":false}\n'
            await release.wait()
            # Split a multibyte character across transport chunks.
            line = '{"message":{"content":"caf\u00e9"},"done":false}\n'.encode()
            split = line.index(b'\xc3') + 1
            yield line[:split]
            yield line[split:]
            yield b'{"done":true}\n'

        async def aclose(self):
            self.closed = True

    body = Body()

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=body)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original_client(
        transport=httpx.MockTransport(handle), **kw,
    ))
    stream = OllamaProvider().stream(
        [Message("user", "hello")], GenerationConfig(max_tokens=20, stop_sequences=["END"]),
    )
    assert await asyncio.wait_for(anext(stream), timeout=1) == "First "
    assert not release.is_set()
    if close_early:
        await stream.aclose()
    else:
        release.set()
        assert [text async for text in stream] == ["caf\u00e9"]
    assert body.closed
    assert requests[0]["stream"] is True
    assert requests[0]["options"]["num_predict"] == 20
    assert requests[0]["options"]["stop"] == ["END"]


async def test_ollama_generation_can_request_native_json(monkeypatch):
    import json

    import httpx

    def handle(request):
        payload = json.loads(request.content)
        assert payload["format"] == "json"
        return httpx.Response(200, content=b'{"message":{"content":"{}"},"done":true}\n')

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original_client(
        transport=httpx.MockTransport(handle), **kw,
    ))
    result = await OllamaProvider().generate(
        [Message("user", "Return JSON")], GenerationConfig(json_output=True),
    )
    assert result.content == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b'', b'{"error":"model failed"}\n'])
async def test_ollama_stream_does_not_silently_accept_truncation(monkeypatch, ending):
    import httpx

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, content=b'{"message":{"content":"Partial"}}\n' + ending,
    ))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original_client(
        transport=transport, **kw,
    ))
    stream = OllamaProvider().stream([Message("user", "hello")])
    assert await anext(stream) == "Partial"
    with pytest.raises(RuntimeError):
        await anext(stream)


@pytest.mark.asyncio
async def test_cloud_provider_streams_and_closes_resources():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from lattice.inference.provider import AnthropicProvider, OpenAIProvider

    async def text_parts():
        yield "First "
        yield "second"

    class AnthropicStream:
        closed = False
        text_stream = text_parts()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    anthropic_stream = AnthropicStream()
    create = Mock(return_value=anthropic_stream)
    anthropic = AnthropicProvider.__new__(AnthropicProvider)
    anthropic._async_client = SimpleNamespace(messages=SimpleNamespace(stream=create))
    anthropic._model = "test-claude"
    messages = [Message("system", "grounded"), Message("user", "question")]
    assert [text async for text in anthropic.stream(messages)] == ["First ", "second"]
    assert anthropic_stream.closed
    assert create.call_args.kwargs["system"] == "grounded"
    assert create.call_args.kwargs["messages"] == [{"role": "user", "content": "question"}]

    class OpenAIStream:
        close = AsyncMock()

        async def __aiter__(self):
            for text in [None, "First ", "second"]:
                delta = SimpleNamespace(content=text)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    openai_stream = OpenAIStream()
    create = AsyncMock(return_value=openai_stream)
    openai = OpenAIProvider.__new__(OpenAIProvider)
    openai._async_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    openai._model = "test-openai"
    assert [text async for text in openai.stream(messages)] == ["First ", "second"]
    openai_stream.close.assert_awaited_once()
    assert create.call_args.kwargs["stream"] is True


@pytest.mark.parametrize("ending", ["complete", "disconnect", "cancel"])
async def test_ollama_generate_streams_counts_usage_and_closes(monkeypatch, ending):
    import asyncio
    import json

    import httpx

    started = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"message":{"content":"First "}}\n'
            started.set()
            if ending == "cancel":
                await asyncio.Event().wait()
            if ending == "complete":
                yield (b'{"message":{"content":"last"},"done":true,'
                       b'"eval_count":2,"prompt_eval_count":4}\n')

        async def aclose(self):
            self.closed = True

    body = Body()

    def handle(request):
        assert json.loads(request.content)["stream"] is True
        assert 0 < request.extensions["timeout"]["read"] <= 120
        return httpx.Response(200, stream=body)

    # Preserve the provider's configured timeout on the mock transport.
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original_client(
        transport=httpx.MockTransport(handle), **kw,
    ))
    task = asyncio.create_task(OllamaProvider(model="local").generate([Message("user", "hi")]))
    await asyncio.wait_for(started.wait(), 1)
    if ending == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif ending == "disconnect":
        with pytest.raises(RuntimeError, match="interrupted"):
            await task
    else:
        result = await task
        assert (result.content, result.provider, result.model) == ("First last", "ollama", "local")
        assert result.usage == {"prompt_tokens": 4, "completion_tokens": 2}
    assert body.closed
