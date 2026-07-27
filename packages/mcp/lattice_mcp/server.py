"""
Lattice MCP Server.

Exposes the agent team as Model Context Protocol tools so they can be
invoked directly from Cursor, Claude Code, or any MCP-compatible client.

Tools exposed:
- lattice_submit_task: Submit a task to the agent team
- lattice_task_status: Check task progress
- lattice_task_result: Get completed task artifacts
- lattice_search_knowledge: Query the knowledge base
- lattice_list_entities: Browse the knowledge graph

This makes the entire agent platform accessible as an IDE extension.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import AgentState, TaskStatus
from lattice.agents.writer import WriterAgent
from lattice.inference.provider import LLMProvider, get_provider

# In-memory task storage for the MCP server
_tasks: dict[str, AgentState] = {}
_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def create_mcp_server() -> Server:
    """Create the Lattice MCP server with all tools registered."""
    server = Server("lattice-agent-platform")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="lattice_submit_task",
                description=(
                    "Submit a task to the Lattice agent team. "
                    "A team of specialized agents (researcher, writer, reviewer, code) "
                    "will collaborate to complete it. Returns a task ID for tracking."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language task description",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum agent iterations (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="lattice_task_status",
                description="Check the current status of a submitted task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID returned from submit_task",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="lattice_task_result",
                description=(
                    "Get the full result of a completed task, including all "
                    "artifacts produced by the agent team."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to get results for",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="lattice_search_knowledge",
                description=(
                    "Search the Lattice knowledge base using semantic similarity. "
                    "Returns relevant document chunks with citations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in natural language",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="lattice_run_task_sync",
                description=(
                    "Submit a task and wait for completion. Returns the full result "
                    "including artifacts. Use for quick tasks that complete in seconds."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language task description",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum agent iterations (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["task"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "lattice_submit_task":
            return await _handle_submit_task(arguments)
        elif name == "lattice_task_status":
            return await _handle_task_status(arguments)
        elif name == "lattice_task_result":
            return await _handle_task_result(arguments)
        elif name == "lattice_search_knowledge":
            return await _handle_search_knowledge(arguments)
        elif name == "lattice_run_task_sync":
            return await _handle_run_task_sync(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _handle_submit_task(args: dict[str, Any]) -> list[TextContent]:
    """Submit a task for async execution."""
    task = args["task"]
    max_iterations = args.get("max_iterations", 10)

    state = AgentState(original_task=task, max_iterations=max_iterations)
    _tasks[state.task_id] = state

    # Launch in background
    asyncio.create_task(_run_task(state.task_id))

    return [TextContent(
        type="text",
        text=json.dumps({
            "task_id": state.task_id,
            "status": "submitted",
            "message": f"Task submitted. Use lattice_task_status with task_id '{state.task_id}' to check progress.",
        }),
    )]


async def _handle_task_status(args: dict[str, Any]) -> list[TextContent]:
    """Check task status."""
    task_id = args["task_id"]
    state = _tasks.get(task_id)
    if not state:
        return [TextContent(type="text", text=json.dumps({"error": "Task not found"}))]

    return [TextContent(
        type="text",
        text=json.dumps({
            "task_id": task_id,
            "status": state.status.value,
            "current_agent": state.current_agent.value if state.current_agent else None,
            "iterations": state.iteration_count,
            "artifacts": len(state.artifacts),
            "reviews": len(state.reviews),
        }),
    )]


async def _handle_task_result(args: dict[str, Any]) -> list[TextContent]:
    """Get full task result."""
    task_id = args["task_id"]
    state = _tasks.get(task_id)
    if not state:
        return [TextContent(type="text", text=json.dumps({"error": "Task not found"}))]

    artifacts = [
        {
            "type": a.artifact_type,
            "content": a.content[:2000],
            "agent": a.source_agent.value,
            "confidence": a.confidence,
        }
        for a in state.artifacts
    ]

    return [TextContent(
        type="text",
        text=json.dumps({
            "task_id": task_id,
            "status": state.status.value,
            "artifacts": artifacts,
            "reviews": [
                {"approved": r.approved, "confidence": r.confidence}
                for r in state.reviews
            ],
        }, indent=2),
    )]


async def _handle_search_knowledge(args: dict[str, Any]) -> list[TextContent]:
    """Search the knowledge base."""
    query = args["query"]
    top_k = args.get("top_k", 5)

    try:
        from lattice.retrieval.vector_store import VectorStore
        vs = VectorStore()
        results = vs.search(query, n_results=top_k)
        return [TextContent(
            type="text",
            text=json.dumps({"query": query, "results": results}, indent=2),
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Search failed: {e}"}),
        )]


async def _handle_run_task_sync(args: dict[str, Any]) -> list[TextContent]:
    """Run a task synchronously and return full result."""
    task = args["task"]
    max_iterations = args.get("max_iterations", 10)

    try:
        provider = _get_provider()
        orchestrator = OrchestratorAgent(
            provider=provider,
            researcher=ResearcherAgent(provider=provider),
            writer=WriterAgent(provider=provider),
            reviewer=ReviewerAgent(provider=provider),
        )

        result = await orchestrator.run(task, max_iterations=max_iterations)
        _tasks[result.task_id] = result

        artifacts = [
            {
                "type": a.artifact_type,
                "content": a.content[:2000],
                "agent": a.source_agent.value,
                "confidence": a.confidence,
            }
            for a in result.artifacts
        ]

        return [TextContent(
            type="text",
            text=json.dumps({
                "task_id": result.task_id,
                "status": result.status.value,
                "artifacts": artifacts,
                "iterations": result.iteration_count,
            }, indent=2),
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Task execution failed: {e}"}),
        )]


async def _run_task(task_id: str) -> None:
    """Background task runner for async submissions."""
    state = _tasks.get(task_id)
    if not state:
        return

    try:
        provider = _get_provider()
        orchestrator = OrchestratorAgent(
            provider=provider,
            researcher=ResearcherAgent(provider=provider),
            writer=WriterAgent(provider=provider),
            reviewer=ReviewerAgent(provider=provider),
        )
        result = await orchestrator.run(
            state.original_task,
            max_iterations=state.max_iterations,
        )
        _tasks[task_id] = result
    except Exception as e:
        state.status = TaskStatus.FAILED
        state.errors.append(str(e))


async def main():
    """Run the MCP server over stdio."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
