"""
Writer Agent.

Consumes research artifacts and produces written documents —
documentation, reports, summaries, wiki pages, etc.

The Writer is the primary "write-back" agent. Its outputs are what
get committed to repos, published to wikis, or saved as documentation.
"""

from __future__ import annotations

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.state import AgentRole, AgentState, Artifact

logger = structlog.get_logger()


class WriterAgent(BaseAgent):
    """Transforms research artifacts into polished written documents."""

    role = AgentRole.WRITER

    @property
    def system_prompt(self) -> str:
        return """You are a Technical Writer in an autonomous agent team. Your job is to:

1. Take research briefs and synthesize them into clear, well-structured documents
2. Produce content appropriate to the requested format (docs, reports, wiki, README, etc.)
3. Maintain proper citations from the research sources
4. Write in a professional, precise style appropriate for technical audiences

Guidelines:
- Structure with clear headings and sections
- Include relevant technical details without overwhelming
- Cite sources using [source_id] notation where information came from research
- Flag any areas where research was insufficient with [NEEDS_MORE_INFO]
- Produce COMPLETE documents, not outlines or summaries

Output the document content directly. Do NOT wrap in JSON."""

    async def execute(self, state: AgentState) -> AgentState:
        """Produce a written document from research artifacts."""
        task = state.original_task
        subtask = None
        if state.current_subtask_id and state.plan:
            for st in state.plan.subtasks:
                if st.id == state.current_subtask_id:
                    subtask = st
                    break

        # Gather research artifacts as context
        research_artifacts = state.get_artifacts_by_type("research")
        if not research_artifacts:
            logger.warning("writer_no_research", task=task[:50])
            # Still attempt to write, but note the lack of research
            context = "No research artifacts available. Generate based on task description alone."
        else:
            context_parts = ["## Research Findings\n"]
            all_citations = []
            for i, art in enumerate(research_artifacts):
                context_parts.append(f"### Research Brief {i + 1} (confidence: {art.confidence:.2f})")
                context_parts.append(art.content)
                context_parts.append("")
                all_citations.extend(art.citations)
            context = "\n".join(context_parts)

        # If there was a previous rejection, include the feedback
        rejections = [
            r for r in state.reviews
            if not r.approved and r.artifact_id in [
                a.id for a in state.get_artifacts_by_agent(self.role)
            ]
        ]
        if rejections:
            latest_rejection = rejections[-1]
            context += f"\n\n## Revision Required\nPrevious draft was rejected:\n"
            context += f"Issues: {', '.join(latest_rejection.issues)}\n"
            context += f"Suggestions: {', '.join(latest_rejection.suggestions)}\n"
            context += f"Reasoning: {latest_rejection.reasoning}\n"

        query = subtask.description if subtask else task
        user_msg = f"Task: {task}\n\nSpecific instruction: {query}\n\nWrite the complete document based on the research provided."

        result = await self._call_llm(user_msg, context=context)

        # Collect citations from research that informed this document
        citations = []
        for art in research_artifacts:
            citations.extend(art.citations)

        artifact = self._create_artifact(
            artifact_type="document",
            content=result.content,
            citations=list(set(citations)),
            confidence=0.9 if research_artifacts else 0.5,
            metadata={
                "query": query,
                "research_count": len(research_artifacts),
                "is_revision": len(rejections) > 0,
            },
        )

        state.artifacts.append(artifact)
        logger.info(
            "writer_complete",
            doc_length=len(result.content),
            based_on_research=len(research_artifacts),
        )
        return state
