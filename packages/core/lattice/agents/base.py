"""
Base agent class.

All specialist agents inherit from BaseAgent. It provides:
- A consistent interface for the orchestrator to invoke agents
- Access to the inference provider
- Structured prompt management
- Artifact production helpers
- Tool-use loop: agents can call tools to gather richer data
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import structlog

from lattice.agents.state import AgentRole, AgentState, Artifact
from lattice.agents.tools import ToolCall, ToolRegistry, ToolResult
from lattice.inference.provider import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    Message,
)

logger = structlog.get_logger()

TOOL_USE_PROMPT_SUFFIX = """

## Tool Use

You have access to tools. To use a tool, include a JSON block in your response:
```tool_call
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}
```

You can make multiple tool calls. After receiving tool results, synthesize
them into your final output. Only call tools when you need additional
information — don't call tools for the sake of it.

{tool_descriptions}
"""


class BaseAgent(ABC):
    """Base class for all Lattice agents."""

    role: AgentRole

    def __init__(
        self,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        max_tool_rounds: int = 3,
        **kwargs,
    ) -> None:
        self._provider = provider
        self._config = GenerationConfig(**kwargs) if kwargs else GenerationConfig()
        self._tool_registry = tool_registry
        self._max_tool_rounds = max_tool_rounds

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

    def _build_system_prompt(self) -> str:
        """Build full system prompt, including tool descriptions if available."""
        base = self.system_prompt
        if self._tool_registry and self._tool_registry.available_tools:
            return base + TOOL_USE_PROMPT_SUFFIX.format(
                tool_descriptions=self._tool_registry.tool_descriptions()
            )
        return base

    async def _call_llm(
        self,
        user_message: str,
        context: str = "",
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Call the LLM with the agent's system prompt plus a user message."""
        messages = [Message(role="system", content=self._build_system_prompt())]

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

    async def _call_llm_with_tools(
        self,
        user_message: str,
        context: str = "",
        config: GenerationConfig | None = None,
    ) -> tuple[GenerationResult, list[ToolResult]]:
        """Call LLM with tool-use loop.

        The agent can request tool calls across multiple rounds. Each round:
        1. LLM generates a response (possibly with tool_call blocks)
        2. Tool calls are extracted and executed
        3. Results are fed back as context for the next round
        4. Repeat until no more tool calls or max rounds reached

        Returns the final LLM response and all tool results collected.
        """
        if not self._tool_registry or not self._tool_registry.available_tools:
            result = await self._call_llm(user_message, context, config)
            return result, []

        messages = [Message(role="system", content=self._build_system_prompt())]

        if context:
            messages.append(Message(role="user", content=f"Context:\n{context}"))
            messages.append(Message(role="assistant", content="Understood. I'll use this context."))

        messages.append(Message(role="user", content=user_message))

        all_tool_results: list[ToolResult] = []
        final_result = None

        for round_num in range(self._max_tool_rounds + 1):
            result = await self._provider.generate(messages, config or self._config)
            logger.info(
                "agent_llm_call",
                agent=self.role.value,
                round=round_num,
                tokens=result.usage,
            )

            tool_calls = self._extract_tool_calls(result.content)

            if not tool_calls or round_num == self._max_tool_rounds:
                final_result = result
                break

            # Execute tool calls
            tool_results_text = []
            for tc in tool_calls:
                tr = await self._tool_registry.invoke(tc)
                all_tool_results.append(tr)
                status = "✓" if tr.success else "✗"
                tool_results_text.append(
                    f"[{status} {tr.tool_name}]\n{tr.content}"
                )

            logger.info(
                "agent_tool_round",
                agent=self.role.value,
                round=round_num,
                tools_called=len(tool_calls),
                tools_succeeded=sum(1 for tr in all_tool_results[-len(tool_calls):] if tr.success),
            )

            # Feed tool results back to LLM
            messages.append(Message(role="assistant", content=result.content))
            messages.append(Message(
                role="user",
                content=f"Tool results:\n\n" + "\n\n".join(tool_results_text)
                + "\n\nNow synthesize these results into your final output. "
                "Do not make additional tool calls unless absolutely necessary.",
            ))

        return final_result or result, all_tool_results

    def _extract_tool_calls(self, content: str) -> list[ToolCall]:
        """Extract tool_call JSON blocks from LLM response."""
        calls = []
        marker = "```tool_call"
        pos = 0

        while True:
            start = content.find(marker, pos)
            if start == -1:
                break

            json_start = content.find("\n", start) + 1
            end = content.find("```", json_start)
            if end == -1:
                break

            try:
                data = json.loads(content[json_start:end].strip())
                calls.append(ToolCall(
                    tool_name=data["tool"],
                    arguments=data.get("arguments", {}),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("tool_call_parse_failed", error=str(e))

            pos = end + 3

        return calls

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
