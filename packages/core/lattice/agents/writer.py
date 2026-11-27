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
from lattice.agents.researcher import implementation_brief
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
- Distinguish research facts, proposed designs, and untested proof-of-concept code.
- Do not claim external web research or executed tests unless the supplied research records them.

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

        # Gather research artifacts as context. The newest brief goes first so
        # repair evidence from a prior rejection isn't buried below the stale
        # original research.
        research_artifacts = state.get_artifacts_by_type("research")
        citation_aliases = {}
        if not research_artifacts:
            logger.warning("writer_no_research", task=task[:50])
            # Still attempt to write, but note the lack of research
            context = "No research artifacts available. Generate based on task description alone."
        else:
            ordered = list(reversed(research_artifacts))
            context_parts = ["## Research Findings\n"]
            for i, art in enumerate(ordered):
                label = "Latest research (use these passages first)" if i == 0 else f"Research Brief {i}"
                context_parts.append(f"### {label} (confidence: {art.confidence:.2f})")
                context_parts.append(art.content)
                if implementation_brief(task):
                    context_parts.append("Retrieved source excerpts (cite the bracketed labels):")
                    for evidence in art.metadata.get("evidence", []):
                        key = f"S{len(citation_aliases) + 1}"
                        citation_aliases[key] = evidence["id"]
                        context_parts.append(f"[{key}] {evidence['source']}\n{evidence['excerpt']}")
            if implementation_brief(task) and citation_aliases:
                # Explicit cheat sheet: pair each retrieved file with its label
                # and the real source_id to write into the brief.
                evidence_by_id = {ev["id"]: ev for art in ordered
                                  for ev in art.metadata.get("evidence", [])}
                context_parts.append("\n## Files you may cite (use the exact bracket label)")
                for key, source_id in citation_aliases.items():
                    source = evidence_by_id.get(source_id, {}).get("source")
                    if source:
                        context_parts.append(f"- {source} → [{key}]")
            context = "\n".join(context_parts)

        # If there was a previous rejection, include the previous draft so the
        # writer revises it instead of starting from a new topic every time.
        own_artifacts = state.get_artifacts_by_agent(self.role)
        own_ids = {a.id for a in own_artifacts if subtask is None or a.id == subtask.result_artifact_id}
        rejections = [r for r in state.reviews if not r.approved and r.artifact_id in own_ids]
        latest_rejection = rejections[-1] if rejections else None
        previous_draft = next(
            (a for a in reversed(own_artifacts) if a.id == latest_rejection.artifact_id),
            None,
        ) if latest_rejection else None

        query = subtask.description if subtask else task
        if latest_rejection and previous_draft:
            context += "\n\n## Previous Draft (to revise)\n"
            context += previous_draft.content
            context += "\n\n## Reviewer Feedback on That Draft\n"
            context += f"Issues: {'; '.join(latest_rejection.issues)}\n"
            context += f"Suggestions: {'; '.join(latest_rejection.suggestions)}\n"
            context += f"Reasoning: {latest_rejection.reasoning}\n"
            user_msg = (
                f"Task: {task}\n\nSpecific instruction: {query}\n\n"
                "Revise the previous draft above to address the reviewer feedback. "
                "Keep the same topic and overall structure; address each issue listed. "
                "Do not switch to a different feature or subject."
            )
        else:
            user_msg = (
                f"Task: {task}\n\nSpecific instruction: {query}\n\n"
                "Write the complete document based on the research provided."
            )

        if implementation_brief(task):
            user_msg += (
                "\nDeliver a short implementation briefing. Name the target, limitation and chosen "
                "approach. Use an exact existing filename and its bracketed source label, e.g. [S1]. "
                "Describe the existing caller, proposed change, sample input and expected output. "
                "Include a complete fenced, language-labelled PoC in the target's language with "
                "representative values and a check for the expected result. Label verification "
                "status accurately. List unverified dependencies or upstream APIs as gaps."
            )
        result = await self._call_llm(user_msg, context=context)
        content = result.content
        for label, source_id in citation_aliases.items():
            content = content.replace(f"[{label}]", f"[{source_id}]")

        # Collect citations from research that informed this document
        citations = []
        for art in research_artifacts:
            citations.extend(art.citations)

        if implementation_brief(task):
            citations = [item for item in citations if f"[{item}]" in content]
        artifact = self._create_artifact(
            artifact_type="document",
            content=content,
            citations=list(dict.fromkeys(citations)),
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
