"""
Tests for agent tool-use loop and tool registry.
"""

import json

import pytest

from lattice.agents.base import BaseAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.state import AgentRole, AgentState
from lattice.agents.tools import (
    AgentTool,
    FunctionTool,
    KnowledgeSearchTool,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from lattice.inference.mock import MockProvider
from lattice.inference.provider import Message


# --- Tool Registry Tests ---


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def mock_tool(self):
        async def search_fn(args):
            return f"Results for: {args['query']}"

        return FunctionTool(
            name="test_search",
            description="Search for things",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            fn=search_fn,
        )

    def test_register_and_get_tool(self, registry, mock_tool):
        registry.register(mock_tool)
        assert registry.get("test_search") is mock_tool
        assert registry.get("nonexistent") is None

    def test_available_tools(self, registry, mock_tool):
        assert len(registry.available_tools) == 0
        registry.register(mock_tool)
        assert len(registry.available_tools) == 1

    def test_tool_descriptions(self, registry, mock_tool):
        registry.register(mock_tool)
        desc = registry.tool_descriptions()
        assert "test_search" in desc
        assert "Search for things" in desc

    @pytest.mark.asyncio
    async def test_invoke_tool(self, registry, mock_tool):
        registry.register(mock_tool)
        result = await registry.invoke(ToolCall(tool_name="test_search", arguments={"query": "turbines"}))
        assert result.success
        assert "turbines" in result.content

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self, registry):
        result = await registry.invoke(ToolCall(tool_name="nonexistent", arguments={}))
        assert not result.success
        assert "Unknown tool" in result.content


# --- Tool-Use Loop Tests ---


class TestAgentToolUseLoop:
    """Test that agents can use tools in their execution loop."""

    @pytest.mark.asyncio
    async def test_researcher_with_tools_gathers_more_data(self):
        """When tools are available, researcher should call them and get richer citations."""
        # First response: researcher decides to call a tool
        first_response = """I need more information about turbine maintenance procedures.

```tool_call
{"tool": "test_search", "arguments": {"query": "turbine maintenance schedule"}}
```

Let me search for more specific data."""

        # Second response (after tool results): final research brief
        second_response = json.dumps({
            "summary": "Found detailed maintenance procedures from knowledge base.",
            "key_findings": [
                "Turbines require inspection every 500 hours",
                "Blade erosion is the primary failure mode",
            ],
            "sources": [{"id": "chunk_123", "relevance": "maintenance schedule data"}],
            "gaps": [],
            "confidence": 0.92,
        })

        provider = MockProvider(responses=[first_response, second_response])

        # Create a tool registry with a test tool
        registry = ToolRegistry()

        async def search_fn(args):
            return f"Found: Maintenance schedule requires inspection every 500 hours. Blade erosion detected in 30% of cases."

        registry.register(FunctionTool(
            name="test_search",
            description="Search knowledge base",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            fn=search_fn,
        ))

        agent = ResearcherAgent(provider=provider, tool_registry=registry)
        state = AgentState(original_task="Research turbine maintenance")
        state = await agent.execute(state)

        assert len(state.artifacts) == 1
        art = state.artifacts[0]
        assert art.artifact_type == "research"
        assert art.confidence == 0.92
        assert art.metadata.get("used_tools") is True
        # Provider was called twice (tool round + synthesis)
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_agent_without_tools_works_normally(self):
        """Without a tool registry, agents fall back to single-call behavior."""
        response = json.dumps({
            "summary": "Basic research without tools.",
            "confidence": 0.8,
        })
        provider = MockProvider(responses=[response])
        agent = ResearcherAgent(provider=provider, tool_registry=None)

        state = AgentState(original_task="Research something")
        state = await agent.execute(state)

        assert len(state.artifacts) == 1
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_extraction(self):
        """Test parsing of tool_call blocks from LLM responses."""
        from lattice.agents.base import BaseAgent

        provider = MockProvider()
        # Create a concrete agent subclass for testing
        agent = ResearcherAgent(provider=provider)

        content = """Let me search for this.

```tool_call
{"tool": "search_knowledge", "arguments": {"query": "turbine specs", "top_k": 3}}
```

And also check the knowledge graph:

```tool_call
{"tool": "query_knowledge_graph", "arguments": {"query": "compressor"}}
```

I'll synthesize after getting results."""

        calls = agent._extract_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].tool_name == "search_knowledge"
        assert calls[0].arguments == {"query": "turbine specs", "top_k": 3}
        assert calls[1].tool_name == "query_knowledge_graph"
        assert calls[1].arguments == {"query": "compressor"}

    @pytest.mark.asyncio
    async def test_tool_loop_respects_max_rounds(self):
        """Agent shouldn't call tools forever — max_tool_rounds is enforced."""
        # Every response contains a tool call
        infinite_tool_response = '```tool_call\n{"tool": "test_search", "arguments": {"query": "more"}}\n```'

        provider = MockProvider(
            responses=[infinite_tool_response] * 10,
            default_response=infinite_tool_response,
        )

        registry = ToolRegistry()

        async def always_search(args):
            return "Some result"

        registry.register(FunctionTool(
            name="test_search",
            description="Search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            fn=always_search,
        ))

        agent = ResearcherAgent(provider=provider, tool_registry=registry, max_tool_rounds=2)
        state = AgentState(original_task="Test max rounds")
        state = await agent.execute(state)

        # Should have been called at most max_tool_rounds + 1 times
        assert provider.call_count <= 4  # 2 tool rounds + final synthesis + initial

    @pytest.mark.asyncio
    async def test_multiple_tools_in_single_round(self):
        """Agent can call multiple tools in a single response."""
        response_with_two_tools = """I'll search two things:

```tool_call
{"tool": "tool_a", "arguments": {"q": "first"}}
```

```tool_call
{"tool": "tool_b", "arguments": {"q": "second"}}
```
"""
        final_response = json.dumps({"summary": "Combined results", "confidence": 0.9})
        provider = MockProvider(responses=[response_with_two_tools, final_response])

        registry = ToolRegistry()
        calls_log = []

        async def tool_a_fn(args):
            calls_log.append(("a", args))
            return "Result A"

        async def tool_b_fn(args):
            calls_log.append(("b", args))
            return "Result B"

        registry.register(FunctionTool("tool_a", "Tool A", {"type": "object", "properties": {"q": {"type": "string"}}}, tool_a_fn))
        registry.register(FunctionTool("tool_b", "Tool B", {"type": "object", "properties": {"q": {"type": "string"}}}, tool_b_fn))

        agent = ResearcherAgent(provider=provider, tool_registry=registry)
        state = AgentState(original_task="Test multi-tool")
        state = await agent.execute(state)

        assert len(calls_log) == 2
        assert ("a", {"q": "first"}) in calls_log
        assert ("b", {"q": "second"}) in calls_log
