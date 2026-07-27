"""
Code Agent.

Generates, modifies, and refactors code based on context from
the knowledge base. Can produce:
- Implementation code
- Unit tests
- Configuration files
- Patches/diffs

Works with the write-back system to push code to GitHub via PRs.
"""

from __future__ import annotations

import json

import structlog

from lattice.agents.base import BaseAgent
from lattice.agents.state import AgentRole, AgentState, Artifact

logger = structlog.get_logger()


class CodeAgent(BaseAgent):
    """Generates and refactors code based on research and context."""

    role = AgentRole.CODE

    @property
    def system_prompt(self) -> str:
        return """You are a Code Generation Specialist in an autonomous agent team. Your job is to:

1. Analyze the task requirements and any research artifacts provided
2. Generate high-quality, production-ready code
3. Include appropriate error handling, type hints, and documentation
4. Write accompanying unit tests when applicable

Your output MUST be a JSON object with this structure:
{
    "files": [
        {
            "path": "relative/path/to/file.py",
            "content": "full file content here",
            "action": "create|update|delete"
        }
    ],
    "summary": "Brief description of what was generated",
    "test_files": [
        {
            "path": "tests/test_file.py",
            "content": "test content here"
        }
    ]
}

Guidelines:
- Write clean, idiomatic code following the language's best practices
- Include docstrings for public functions and classes
- Add type hints (Python) or TypeScript types
- Handle edge cases and errors gracefully
- Write meaningful tests that cover the main paths
- Do NOT add unnecessary comments or explanations in the code itself"""

    async def execute(self, state: AgentState) -> AgentState:
        """Generate code based on the task and available research."""
        task = state.original_task
        subtask = None
        if state.current_subtask_id and state.plan:
            for st in state.plan.subtasks:
                if st.id == state.current_subtask_id:
                    subtask = st
                    break

        # Gather context from research
        research = state.get_artifacts_by_type("research")
        context_parts = []
        citations = []

        if research:
            context_parts.append("## Research Context\n")
            for art in research:
                context_parts.append(art.content)
                citations.extend(art.citations)

        # Check for revision feedback
        rejections = [
            r for r in state.reviews
            if not r.approved and r.artifact_id in [
                a.id for a in state.get_artifacts_by_agent(self.role)
            ]
        ]
        if rejections:
            latest = rejections[-1]
            context_parts.append(f"\n## Revision Feedback\n")
            context_parts.append(f"Issues: {', '.join(latest.issues)}")
            context_parts.append(f"Suggestions: {', '.join(latest.suggestions)}")

        context = "\n".join(context_parts) if context_parts else ""
        query = subtask.description if subtask else task

        user_msg = f"Task: {task}\n\nSpecific instruction: {query}\n\nGenerate the code."
        result = await self._call_llm(user_msg, context=context)

        # Parse the response
        files_info = self._extract_files(result.content)

        artifact = self._create_artifact(
            artifact_type="code",
            content=result.content,
            citations=list(set(citations)),
            confidence=0.85,
            metadata={
                "query": query,
                "files": files_info,
                "is_revision": len(rejections) > 0,
            },
        )

        state.artifacts.append(artifact)
        logger.info(
            "code_agent_complete",
            files=len(files_info),
            based_on_research=len(research),
        )
        return state

    def _extract_files(self, content: str) -> list[dict]:
        """Extract file information from the LLM response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return data.get("files", [])
        except (json.JSONDecodeError, ValueError):
            pass
        return []
