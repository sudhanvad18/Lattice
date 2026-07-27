"""
Ingestion Agent.

Monitors agent outputs and extracts new knowledge to feed back
into the knowledge graph. This is what makes the system self-improving.

The Ingestion Agent:
1. Reads artifacts produced by other agents
2. Identifies new entities and relations not yet in the KG
3. Produces structured KG expansion requests
4. Writes back via the KG write-back handler
"""

from __future__ import annotations

import json

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.state import AgentRole, AgentState, Artifact

logger = structlog.get_logger()


class IngestionAgent(BaseAgent):
    """Extracts new knowledge from agent outputs and expands the KG."""

    role = AgentRole.INGESTION

    @property
    def system_prompt(self) -> str:
        return """You are a Knowledge Ingestion Specialist in an autonomous agent team. Your job is to:

1. Analyze artifacts produced by other agents (research briefs, documents, code)
2. Extract NEW entities and relations that should be added to the knowledge graph
3. Avoid duplicating entities that likely already exist

Your output MUST be a JSON object with this structure:
{
    "entities": [
        {
            "name": "entity name",
            "entity_type": "concept|component|process|specification|person|organization",
            "description": "brief description",
            "properties": {"key": "value"}
        }
    ],
    "relations": [
        {
            "source_name": "source entity name",
            "target_name": "target entity name",
            "relation_type": "contains|depends_on|implements|relates_to|part_of"
        }
    ],
    "reasoning": "Why these entities/relations are worth adding"
}

Guidelines:
- Only extract entities that represent durable knowledge (not ephemeral task details)
- Prefer specific entity types over generic ones
- Include descriptions that provide context
- Relations should represent meaningful connections
- Be conservative — quality over quantity"""

    async def execute(self, state: AgentState) -> AgentState:
        """Extract new knowledge from recent artifacts."""
        # Gather non-research artifacts to learn from
        source_artifacts = [
            a for a in state.artifacts
            if a.source_agent != self.role and a.artifact_type != "research"
        ]

        if not source_artifacts:
            logger.info("ingestion_agent_no_artifacts")
            return state

        context_parts = [f"## Original Task: {state.original_task}\n"]
        for art in source_artifacts[-3:]:  # Last 3 artifacts
            context_parts.append(
                f"### {art.artifact_type} (by {art.source_agent.value})\n{art.content[:2000]}\n"
            )

        context = "\n".join(context_parts)
        user_msg = "Extract new entities and relations from these agent outputs that should be added to the knowledge graph."

        result = await self._call_llm(user_msg, context=context)

        artifact = self._create_artifact(
            artifact_type="kg_expansion",
            content=result.content,
            citations=[a.id for a in source_artifacts],
            confidence=0.75,
            metadata={"source_artifact_count": len(source_artifacts)},
        )

        state.artifacts.append(artifact)
        logger.info("ingestion_agent_complete", sources=len(source_artifacts))
        return state
