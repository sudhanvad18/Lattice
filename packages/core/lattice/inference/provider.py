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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

from lattice.config import InferenceProvider, get_settings

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


class OllamaProvider(LLMProvider):
    """Local inference via Ollama.

    Requires Ollama to be running: `ollama serve`
    And a model pulled: `ollama pull llama3.2`
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self._base_url = base_url
        self._model = model

    async def generate(
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

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
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

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

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
            provider="openai",
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
        )

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return GenerationResult(
            content=content,
            model=self._model,
            provider="openai",
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
