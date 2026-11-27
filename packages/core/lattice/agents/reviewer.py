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

import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.researcher import evidence_context, implementation_brief
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
    "approved": false,
    "confidence": 0.8,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "reasoning": "Brief explanation of your verdict"
}

Set approved to true only when the artifact meets the criteria, otherwise false.
Confidence is a number from 0.0 to 1.0. Return only valid JSON, with concise feedback.

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
                if implementation_brief(state.original_task):
                    context_parts.append(evidence_context(r))

        context = "\n".join(context_parts)

        user_msg = (
            f"Review this {artifact_to_review.artifact_type} artifact.\n"
            f"It was produced by the {artifact_to_review.source_agent.value} agent.\n"
            f"Evaluate against the original task requirements and source research."
        )

        issues = []
        if implementation_brief(state.original_task):
            issues = self._check_brief(artifact_to_review, research)
            user_msg += (
                " Check the claimed integration against the actual repository excerpts. Verify "
                "the proposed input, caller, output and sample result match. Reject invented "
                "APIs or unresolved placeholders in the PoC. Treat the example as unexecuted."
            )
        if issues:
            verdict = ReviewVerdict(
                artifact_id=artifact_to_review.id, approved=False, confidence=1.0, issues=issues,
                reasoning="The draft failed the implementation briefing checks.",
            )
        else:
            result = await self._call_llm(
                user_msg, context=context,
                config=replace(self._config, temperature=0.1, json_output=True),
            )
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

    @staticmethod
    def _check_brief(draft: Artifact, research: list[Artifact]) -> list[str]:
        issues = []
        project_sources = [
            item for art in research for item in art.metadata.get("evidence", [])
            if item.get("kind") == "project"
        ]
        code_sources = [item for item in project_sources if item.get("doc_type") == "code"]
        if not any(f"[{item['id']}]" in draft.content and Path(item["source"]).name in draft.content
                   for item in (code_sources or project_sources)):
            issues.append("Name an existing project file and cite its matching retrieved [source_id].")
        blocks = re.findall(
            r"^ {0,3}```([\w+#.-]+)[^\S\n]*\n(.*?)^ {0,3}```[ \t]*$",
            draft.content, flags=re.MULTILINE | re.DOTALL,
        )
        if not any(code.strip() for _, code in blocks):
            issues.append("Supply a complete language-labelled fenced code example with sample input and output.")
        for language, code in blocks:
            if language.lower() in {"python", "py", "python3"}:
                try:
                    ast.parse(code)
                except SyntaxError as error:
                    issues.append(f"Fix Python syntax at example line {error.lineno}: {error.msg}.")
                except (ValueError, RecursionError):
                    issues.append("Simplify the Python example so that its syntax can be checked.")
        return issues

    def _parse_verdict(self, content: str, artifact_id: str) -> ReviewVerdict:
        """Parse the LLM's JSON verdict, with fallback for malformed responses."""
        try:
            payload = content.strip()
            if payload.startswith("```") and payload.endswith("```"):
                _, _, payload = payload[:-3].partition("\n")
            data = json.loads(payload)
            if not isinstance(data, dict) or type(data.get("approved")) is not bool:
                raise ValueError("Expected a boolean approved field")
            return ReviewVerdict(artifact_id=artifact_id, **data)
        except (ValueError, TypeError) as e:
            logger.warning("reviewer_parse_failed", error=str(e))

        return ReviewVerdict(
            artifact_id=artifact_id,
            approved=False,
            confidence=0.5,
            issues=["Could not parse structured review"],
            suggestions=["Review response was not in expected JSON format"],
            reasoning=content[:200],
        )
