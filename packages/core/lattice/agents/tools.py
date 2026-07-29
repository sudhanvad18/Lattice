"""
Agent tool registry and execution loop.

Gives agents the ability to call tools (MCP, local knowledge base, external
APIs) during their execution. Instead of a single LLM call, agents can:

1. Think about what information they need
2. Call one or more tools to gather it
3. Synthesize the tool results
4. Produce a richer artifact

This is what separates a "chat wrapper" from a real agent — the ability
to take autonomous actions to improve output quality.

The tool-use loop:
    think → decide_tools → call_tools → synthesize → (repeat if needed) → produce_artifact
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger()


@dataclass
class ToolCall:
    """A request to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    tool_name: str
    success: bool
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTool(ABC):
    """A tool that agents can invoke during execution."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """Registry of tools available to agents.

    Tools can be:
    - Local (knowledge base search, KG queries)
    - MCP-proxied (forwarded to an MCP server)
    - Custom (any async callable wrapped as a tool)
    """

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    @property
    def available_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def tool_descriptions(self) -> str:
        """Format tool descriptions for inclusion in agent prompts."""
        if not self._tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in self._tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in tool.parameters_schema.get("properties", {}).items()
            )
            lines.append(f"  - {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    async def invoke(self, call: ToolCall) -> ToolResult:
        """Invoke a tool by name."""
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Unknown tool: {call.tool_name}",
            )
        try:
            result = await tool.execute(call.arguments)
            logger.info("tool_invoked", tool=call.tool_name, success=result.success)
            return result
        except Exception as e:
            logger.error("tool_invocation_failed", tool=call.tool_name, error=str(e))
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Tool execution failed: {e}",
            )


# --- Built-in Tools ---


class KnowledgeSearchTool(AgentTool):
    """Search the vector store for relevant documents."""

    def __init__(self, vector_store) -> None:
        self._vs = vector_store

    @property
    def name(self) -> str:
        return "search_knowledge"

    @property
    def description(self) -> str:
        return "Search the knowledge base for documents relevant to a query. Returns chunks with relevance scores."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        results = self._vs.search(query, n_results=top_k)

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                content="No relevant documents found.",
                metadata={"result_count": 0},
            )

        formatted = []
        for r in results:
            formatted.append(f"[{r.get('id', 'unknown')}] (score: {1 - r.get('distance', 0):.2f})\n{r.get('content', '')}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            content="\n\n".join(formatted),
            metadata={"result_count": len(results), "chunk_ids": [r.get("id") for r in results]},
        )


class KGQueryTool(AgentTool):
    """Query the knowledge graph for entities and relations."""

    def __init__(self, kg_backend) -> None:
        self._kg = kg_backend

    @property
    def name(self) -> str:
        return "query_knowledge_graph"

    @property
    def description(self) -> str:
        return "Search the knowledge graph for entities matching a query. Returns entity names, types, and connections."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Entity name or keyword to search"},
                "entity_type": {"type": "string", "description": "Filter by entity type (optional)"},
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments["query"]
        entity_type = arguments.get("entity_type")

        entities = self._kg.search_entities(query, entity_type=entity_type)
        if not entities:
            return ToolResult(
                tool_name=self.name,
                success=True,
                content="No matching entities found in knowledge graph.",
                metadata={"result_count": 0},
            )

        lines = []
        for e in entities[:10]:
            lines.append(f"- {e.name} ({e.entity_type}): {e.description or 'no description'}")
            neighbors = self._kg.get_neighbors(e.id)
            for n in neighbors[:3]:
                lines.append(f"    → {n.name} ({n.entity_type})")

        return ToolResult(
            tool_name=self.name,
            success=True,
            content="\n".join(lines),
            metadata={"result_count": len(entities), "entity_ids": [e.id for e in entities]},
        )


class FunctionTool(AgentTool):
    """Wrap any async callable as a tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[[dict[str, Any]], Awaitable[str]],
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        content = await self._fn(arguments)
        return ToolResult(tool_name=self.name, success=True, content=content)
