"""
Base agent class.

All specialist agents inherit from BaseAgent. It provides:
- A consistent interface for the orchestrator to invoke agents
- Access to the inference provider
- Structured prompt management
- Artifact production helpers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from lattice.agents.state import AgentRole, AgentState, Artifact
from lattice.inference.provider import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    Message,
)

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Base class for all Lattice agents."""

    role: AgentRole

    def __init__(self, provider: LLMProvider, **kwargs) -> None:
        self._provider = provider
        self._config = GenerationConfig(**kwargs) if kwargs else GenerationConfig()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's system prompt defining its role and behavior."""
        ...

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's task given the current state.

        Returns a new state with the agent's outputs added.
        """
        ...

    async def _call_llm(
        self,
        user_message: str,
        context: str = "",
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Call the LLM with the agent's system prompt plus a user message."""
        messages = [Message(role="system", content=self.system_prompt)]

        if context:
            messages.append(Message(role="user", content=f"Context:\n{context}"))
            messages.append(Message(role="assistant", content="Understood. I'll use this context."))

        messages.append(Message(role="user", content=user_message))

        result = await self._provider.generate(messages, config or self._config)
        logger.info(
            "agent_llm_call",
            agent=self.role.value,
            tokens=result.usage,
        )
        return result

    def _create_artifact(
        self,
        artifact_type: str,
        content: str,
        citations: list[str] | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Helper to create a properly attributed artifact."""
        return Artifact(
            artifact_type=artifact_type,
            content=content,
            source_agent=self.role,
            citations=citations or [],
            confidence=confidence,
            metadata=metadata or {},
        )
