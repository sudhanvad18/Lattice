"""
Mock inference provider for testing.

Returns configurable responses without making any network calls.
Mirrors the real provider interface exactly so agents can't tell
the difference during tests.
"""

from __future__ import annotations

from lattice.inference.provider import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    Message,
)


class MockProvider(LLMProvider):
    """Mock provider that returns pre-configured responses.

    Usage in tests:
        provider = MockProvider(responses=["Answer 1", "Answer 2"])
        result = provider.generate_sync(messages)
        assert result.content == "Answer 1"
        # Next call returns "Answer 2", etc.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "Mock LLM response.",
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default_response = default_response
        self._call_history: list[list[Message]] = []
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_messages(self) -> list[Message] | None:
        return self._call_history[-1] if self._call_history else None

    def _get_response(self) -> str:
        if self._responses:
            return self._responses.pop(0)
        return self._default_response

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return self._do_generate(messages)

    def generate_sync(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return self._do_generate(messages)

    def _do_generate(self, messages: list[Message]) -> GenerationResult:
        self._call_history.append(messages)
        self._call_count += 1
        content = self._get_response()

        return GenerationResult(
            content=content,
            model="mock-model",
            provider="mock",
            usage={"prompt_tokens": 10, "completion_tokens": len(content.split())},
        )
