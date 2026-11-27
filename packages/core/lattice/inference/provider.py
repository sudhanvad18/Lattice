"""
Inference provider abstraction.

Agents interact with LLMs through a unified interface. The provider
handles model-specific details (API formats, streaming, token counting)
so agent code stays clean and portable.

Supported providers:
- Ollama (local, free, private — default for development)
- Anthropic Claude (cloud, high quality)
- OpenAI (cloud, broad model selection)

Usage:
    from lattice.inference.provider import get_provider

    provider = get_provider()  # Uses config defaults
    response = await provider.generate(messages=[
        {"role": "system", "content": "You are a researcher."},
        {"role": "user", "content": "What causes fan blade erosion?"},
    ])
    print(response.content)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

from lattice.config import InferenceProvider, Settings, get_settings

logger = structlog.get_logger()


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class GenerationConfig:
    """Parameters controlling LLM generation."""

    temperature: float = 0.7
    max_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)
    json_output: bool = False  # Request native JSON mode where supported.


@dataclass
class GenerationResult:
    """Result from an LLM generation call."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    raw_response: Optional[Any] = field(default=None, repr=False)


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult: ...

    @abstractmethod
    def generate_sync(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult: ...

    async def stream(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Yield generated text; existing custom providers can use this fallback."""
        result = await self.generate(messages, config)
        if result.content:
            yield result.content


class OllamaProvider(LLMProvider):
    """Local inference via Ollama.

    Requires Ollama to be running: `ollama serve`
    And a model pulled: `ollama pull llama3.2`
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self._base_url = base_url
        self._model = model

    async def _frames(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> AsyncIterator[dict]:
        import json

        import httpx

        config = config or GenerationConfig()
        options = {"temperature": config.temperature, "num_predict": config.max_tokens}
        if config.stop_sequences:
            options["stop"] = config.stop_sequences
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True, "options": options,
        }
        if config.json_output:
            payload["format"] = "json"
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client,
            client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError("Ollama could not finish generating the answer.")
                yield data
                if data.get("done") is True:
                    return
            raise RuntimeError("Ollama's response was interrupted.")

    async def stream(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        async with aclosing(self._frames(messages, config)) as frames:
            async for frame in frames:
                content = frame.get("message", {}).get("content", "")
                if content:
                    yield content

    async def generate(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> GenerationResult:
        parts = []
        async with aclosing(self._frames(messages, config)) as frames:
            async for data in frames:
                parts.append(data.get("message", {}).get("content", ""))
        content = "".join(parts)
        data["message"] = {**data.get("message", {}), "content": content}
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

        logger.debug("ollama_generation", model=self._model, tokens=usage)
        return GenerationResult(
            content=content,
            model=self._model,
            provider="ollama",
            usage=usage,
            raw_response=data,
        )

    def generate_sync(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        import httpx

        config = config or GenerationConfig()
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences

        if config.json_output:
            payload["format"] = "json"

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

        return GenerationResult(
            content=content,
            model=self._model,
            provider="ollama",
            usage=usage,
            raw_response=data,
        )


class AnthropicProvider(LLMProvider):
    """Cloud inference via Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def stream(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        config = config or GenerationConfig()
        kwargs: dict = {
            "model": self._model, "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages if m.role != "system"],
        }
        system = next((m.content for m in reversed(messages) if m.role == "system"), None)
        if system:
            kwargs["system"] = system
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        async with self._async_client.messages.stream(**kwargs) as response:
            async for text in response.text_stream:
                yield text

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        config = config or GenerationConfig()

        # Anthropic separates system message from the conversation
        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs: dict = {
            "model": self._model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": chat_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        response = await self._async_client.messages.create(**kwargs)

        content = response.content[0].text if response.content else ""
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        logger.debug("anthropic_generation", model=self._model, tokens=usage)
        return GenerationResult(
            content=content,
            model=self._model,
            provider="anthropic",
            usage=usage,
            raw_response=response,
        )

    def generate_sync(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        config = config or GenerationConfig()

        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs: dict = {
            "model": self._model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": chat_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        response = self._client.messages.create(**kwargs)

        content = response.content[0].text if response.content else ""
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        return GenerationResult(
            content=content,
            model=self._model,
            provider="anthropic",
            usage=usage,
            raw_response=response,
        )


class OpenAIProvider(LLMProvider):
    """Cloud inference via OpenAI API."""

    def __init__(
        self, api_key: str, model: str = "gpt-4o", *, base_url: str | None = None,
        provider_name: str = "openai",
    ) -> None:
        import openai

        self._provider_name = provider_name
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def stream(
        self, messages: list[Message], config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        config = config or GenerationConfig()
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=config.temperature, max_tokens=config.max_tokens,
            stop=config.stop_sequences or None, stream=True,
            **({"response_format": {"type": "json_object"}} if config.json_output else {}),
        )
        try:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await response.close()

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        config = config or GenerationConfig()

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences or None,
            **({"response_format": {"type": "json_object"}} if config.json_output else {}),
        )

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        logger.debug("openai_generation", model=self._model, tokens=usage)
        return GenerationResult(
            content=content,
            model=self._model,
            provider=self._provider_name,
            usage=usage,
            raw_response=response,
        )

    def generate_sync(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        config = config or GenerationConfig()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences or None,
            **({"response_format": {"type": "json_object"}} if config.json_output else {}),
        )

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return GenerationResult(
            content=content,
            model=self._model,
            provider=self._provider_name,
            usage=usage,
            raw_response=response,
        )


# --- Factory ---


def get_provider(
    provider_type: InferenceProvider | None = None,
    **kwargs,
) -> LLMProvider:
    """Create an LLM provider from configuration.

    Uses settings defaults if no provider_type is specified.
    """
    settings = get_settings()
    provider_type = provider_type or settings.default_provider

    if provider_type == InferenceProvider.OLLAMA:
        return OllamaProvider(
            base_url=kwargs.get("base_url", settings.ollama_base_url),
            model=kwargs.get("model", settings.ollama_model),
        )
    elif provider_type == InferenceProvider.ANTHROPIC:
        api_key = kwargs.get("api_key", settings.anthropic_api_key)
        if not api_key:
            raise ValueError("LATTICE_ANTHROPIC_API_KEY must be set to use Anthropic provider")
        return AnthropicProvider(
            api_key=api_key,
            model=kwargs.get("model", settings.anthropic_model),
        )
    elif provider_type == InferenceProvider.OPENAI:
        api_key = kwargs.get("api_key", settings.openai_api_key)
        if not api_key:
            raise ValueError("LATTICE_OPENAI_API_KEY must be set to use OpenAI provider")
        return OpenAIProvider(
            api_key=api_key,
            model=kwargs.get("model", settings.openai_model),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")

def get_writer_provider(default: LLMProvider, settings: Settings) -> LLMProvider:
    """Resolve the optional writer route for an Ask task."""
    if not settings.writer_model and not settings.writer_provider:
        return default
    if settings.writer_provider == "vllm":
        if not settings.writer_model:
            raise ValueError("Set LATTICE_WRITER_MODEL to the model served by vLLM.")
        return OpenAIProvider(
            api_key=settings.vllm_api_key or "local",
            model=settings.writer_model,
            base_url=settings.vllm_base_url,
            provider_name="vllm",
        )
    backend = InferenceProvider(settings.writer_provider or settings.default_provider)
    options = {"model": settings.writer_model} if settings.writer_model else {}
    return get_provider(backend, **options)
