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
