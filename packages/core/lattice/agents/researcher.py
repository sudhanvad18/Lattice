"""
Researcher Agent.

Queries the knowledge graph and vector store to gather information
relevant to the current task. Produces structured research artifacts
with citations back to source chunks and KG entities.

The Researcher does NOT generate new content — it synthesizes and
organizes existing knowledge from the knowledge base.

With tools enabled, the Researcher autonomously decides what to search
for, executes multiple queries, and synthesizes the combined results
into a richer research brief.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.state import AgentRole, AgentState, Artifact
from lattice.agents.tools import ToolRegistry, ToolResult
from lattice.graph.engine import KnowledgeGraphBackend
from lattice.inference.provider import LLMProvider
from lattice.retrieval.vector_store import VectorStore

logger = structlog.get_logger()


class ResearcherAgent(BaseAgent):
    """Queries knowledge base and produces research artifacts."""

    role = AgentRole.RESEARCHER

    def __init__(
        self,
        provider: LLMProvider,
        vector_store: VectorStore | None = None,
        kg_backend: KnowledgeGraphBackend | None = None,
        tool_registry: ToolRegistry | None = None,
        top_k: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(provider, tool_registry=tool_registry, **kwargs)
        self._vector_store = vector_store
        self._kg_backend = kg_backend
        self._top_k = top_k

    @property
    def system_prompt(self) -> str:
        return """You are a Research Specialist in an autonomous agent team. Your job is to:

1. Analyze the task and identify what information is needed
2. Use available tools to search for relevant data (call tools proactively!)
3. Synthesize retrieved documents and knowledge graph data into a structured research brief
4. Cite your sources precisely (chunk IDs and entity IDs)
5. Flag gaps where information is missing or uncertain

Your output MUST be a JSON object with this structure:
{
    "summary": "A concise summary of findings relevant to the task",
    "key_findings": ["finding 1", "finding 2", ...],
    "sources": [{"id": "chunk/entity id", "relevance": "why this is relevant"}],
    "gaps": ["information that is missing or uncertain"],
    "confidence": 0.0-1.0
}

IMPORTANT: Use tools to search for information BEFORE writing your research brief.
Multiple searches with different queries will give you better coverage.
Be thorough but concise. Focus on RELEVANCE to the task at hand."""

    async def execute(self, state: AgentState) -> AgentState:
        """Execute research by querying vector store and KG (with tool loop)."""
        task = state.original_task
        subtask = None
        if state.current_subtask_id and state.plan:
            for st in state.plan.subtasks:
                if st.id == state.current_subtask_id:
                    subtask = st
                    break

        query = subtask.description if subtask else task
        logger.info("researcher_executing", query=query[:100])

        # Gather initial context (direct queries for baseline)
        context_parts = []
        citations: list[str] = []

        if self._vector_store:
            vs_results = self._vector_store.search(query, top_k=self._top_k)
            if vs_results:
                context_parts.append("## Retrieved Documents\n")
                for i, result in enumerate(vs_results):
                    chunk_id = getattr(result, "chunk_id", f"chunk_{i}")
                    content = getattr(result, "content", "")
                    similarity = getattr(result, "similarity", 0.0)
                    context_parts.append(
                        f"[{chunk_id}] (relevance: {similarity:.2f})\n{content}\n"
                    )
                    citations.append(chunk_id)

        if self._kg_backend:
            kg_context = self._query_knowledge_graph(query)
            if kg_context:
                context_parts.append("## Knowledge Graph Context\n")
                context_parts.append(kg_context)

        context = "\n".join(context_parts) if context_parts else "No relevant documents found in knowledge base."

        user_msg = f"Task: {task}\n\nResearch query: {query}\n\nSynthesize the following retrieved information into a structured research brief."

        # Use tool loop if tools are available — agent autonomously gathers more data
        if self._tool_registry and self._tool_registry.available_tools:
            result, tool_results = await self._call_llm_with_tools(user_msg, context=context)
            # Collect additional citations from tool results
            for tr in tool_results:
                if tr.success and tr.metadata.get("chunk_ids"):
                    citations.extend(tr.metadata["chunk_ids"])
                if tr.success and tr.metadata.get("entity_ids"):
                    citations.extend(tr.metadata["entity_ids"])
        else:
            result = await self._call_llm(user_msg, context=context)

        # Parse response and create artifact
        confidence = self._extract_confidence(result.content)
        artifact = self._create_artifact(
            artifact_type="research",
            content=result.content,
            citations=list(set(citations)),
            confidence=confidence,
            metadata={
                "query": query,
                "num_sources": len(citations),
                "used_tools": bool(self._tool_registry and self._tool_registry.available_tools),
            },
        )

        state.artifacts.append(artifact)
        logger.info(
            "researcher_complete",
            citations=len(set(citations)),
            confidence=confidence,
        )
        return state

    def _query_knowledge_graph(self, query: str) -> str:
        """Extract relevant entities and relations from the KG."""
        if not self._kg_backend:
            return ""

        parts = []
        all_entities = self._kg_backend.get_all_entities()
        query_lower = query.lower()
        relevant = [
            e for e in all_entities
            if any(
                term in e.name.lower() or term in (e.description or "").lower()
                for term in query_lower.split()
                if len(term) > 3
            )
        ]

        if relevant:
            parts.append("Relevant entities:")
            for entity in relevant[:10]:
                parts.append(f"  - {entity.name} ({entity.entity_type}): {entity.description or 'no description'}")

                neighbors = self._kg_backend.get_neighbors(entity.id)
                for neighbor in neighbors.entities[:3]:
                    parts.append(f"    -> related to: {neighbor.name} ({neighbor.entity_type})")

        return "\n".join(parts)

    def _extract_confidence(self, content: str) -> float:
        """Try to extract the confidence score from the LLM's JSON response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return float(data.get("confidence", 0.8))
        except (json.JSONDecodeError, ValueError):
            pass
        return 0.8
