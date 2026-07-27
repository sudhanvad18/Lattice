"""
Reviewer Agent (LLM-as-Judge).

Validates artifacts produced by other agents before they are written back
to external systems. Acts as a quality gate.

Inspired by Moody's credit rating methodology — the Reviewer provides:
- Pass/Fail verdict
- Confidence score
- Specific issues identified
- Actionable suggestions for improvement

If an artifact is rejected, the producing agent gets another iteration
with the Reviewer's feedback.
"""

from __future__ import annotations

import json

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.state import AgentRole, AgentState, Artifact, ReviewVerdict

logger = structlog.get_logger()


class ReviewerAgent(BaseAgent):
    """LLM-as-judge that validates agent outputs."""

    role = AgentRole.REVIEWER

    def __init__(self, provider, approval_threshold: float = 0.7, **kwargs) -> None:
        super().__init__(provider, **kwargs)
        self._approval_threshold = approval_threshold

    @property
    def system_prompt(self) -> str:
        return """You are a Quality Reviewer in an autonomous agent team. Your job is to:

1. Evaluate artifacts produced by other agents for accuracy, completeness, and quality
2. Provide a clear APPROVE or REJECT verdict
3. If rejecting, provide specific, actionable feedback so the producing agent can improve

Your output MUST be a JSON object with this exact structure:
{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "reasoning": "Brief explanation of your verdict"
}

Evaluation criteria:
- ACCURACY: Is the content factually correct based on cited sources?
- COMPLETENESS: Does it fully address the task requirements?
- QUALITY: Is it well-structured, clear, and professional?
- CITATIONS: Are claims properly supported by source material?
- GAPS: Are there significant missing pieces flagged as [NEEDS_MORE_INFO]?

Be STRICT but FAIR. Only approve content that meets professional standards.
Reject if there are factual errors, missing critical information, or poor quality."""

    async def execute(self, state: AgentState) -> AgentState:
        """Review the most recent non-research artifact."""
        # Find the artifact to review (latest document/code, not research)
        artifact_to_review = None
        for art in reversed(state.artifacts):
            if art.artifact_type != "research" and art.source_agent != self.role:
                # Check it hasn't already been reviewed
                already_reviewed = any(r.artifact_id == art.id for r in state.reviews)
                if not already_reviewed:
                    artifact_to_review = art
                    break

        if not artifact_to_review:
            logger.warning("reviewer_nothing_to_review")
            return state

        # Build context: the artifact + original task + research it was based on
        context_parts = [
            f"## Original Task\n{state.original_task}\n",
            f"## Artifact to Review (type: {artifact_to_review.artifact_type})\n",
            artifact_to_review.content,
        ]

        # Include research for cross-referencing
        research = state.get_artifacts_by_type("research")
        if research:
            context_parts.append("\n## Source Research (for fact-checking)")
            for r in research:
                context_parts.append(r.content[:1000])

        context = "\n".join(context_parts)

        user_msg = (
            f"Review this {artifact_to_review.artifact_type} artifact.\n"
            f"It was produced by the {artifact_to_review.source_agent.value} agent.\n"
            f"Evaluate against the original task requirements and source research."
        )

        result = await self._call_llm(user_msg, context=context)

        # Parse the review verdict
        verdict = self._parse_verdict(result.content, artifact_to_review.id)
        state.reviews.append(verdict)

        logger.info(
            "reviewer_complete",
            artifact_id=artifact_to_review.id[:8],
            approved=verdict.approved,
            confidence=verdict.confidence,
            issues=len(verdict.issues),
        )
        return state

    def _parse_verdict(self, content: str, artifact_id: str) -> ReviewVerdict:
        """Parse the LLM's JSON verdict, with fallback for malformed responses."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return ReviewVerdict(
                    artifact_id=artifact_id,
                    approved=bool(data.get("approved", False)),
                    confidence=float(data.get("confidence", 0.5)),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    reasoning=data.get("reasoning", ""),
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("reviewer_parse_failed", error=str(e))

        # Fallback: try to infer from content
        content_lower = content.lower()
        approved = "approve" in content_lower and "reject" not in content_lower
        return ReviewVerdict(
            artifact_id=artifact_id,
            approved=approved,
            confidence=0.5,
            issues=["Could not parse structured review"],
            suggestions=["Review response was not in expected JSON format"],
            reasoning=content[:200],
        )
