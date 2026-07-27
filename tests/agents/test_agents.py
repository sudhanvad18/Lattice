"""
Tests for individual agents and the full orchestrator workflow.
"""

import json

import pytest

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import (
    AgentRole,
    AgentState,
    Artifact,
    ReviewVerdict,
    SubTask,
    TaskPlan,
    TaskStatus,
)
from lattice.agents.writer import WriterAgent
from lattice.inference.mock import MockProvider
from lattice.inference.provider import Message


# --- State Model Tests ---


class TestAgentState:
    """Test the shared state models."""

    def test_state_creation(self):
        state = AgentState(original_task="Write docs about turbines")
        assert state.status == TaskStatus.PENDING
        assert state.artifacts == []
        assert state.iteration_count == 0

    def test_get_next_subtask_respects_dependencies(self):
        st1 = SubTask(description="Research", assigned_to=AgentRole.RESEARCHER)
        st2 = SubTask(
            description="Write",
            assigned_to=AgentRole.WRITER,
            depends_on=[st1.id],
        )
        state = AgentState(
            original_task="Test",
            plan=TaskPlan(task_description="Test", subtasks=[st1, st2]),
        )

        # st1 has no deps, should be next
        assert state.get_next_subtask() == st1

        # Mark st1 complete, now st2 should be next
        st1.status = TaskStatus.COMPLETED
        assert state.get_next_subtask() == st2

    def test_get_artifacts_by_type(self):
        state = AgentState(original_task="Test")
        state.artifacts = [
            Artifact(artifact_type="research", content="r1", source_agent=AgentRole.RESEARCHER),
            Artifact(artifact_type="document", content="d1", source_agent=AgentRole.WRITER),
            Artifact(artifact_type="research", content="r2", source_agent=AgentRole.RESEARCHER),
        ]
        research = state.get_artifacts_by_type("research")
        assert len(research) == 2
        docs = state.get_artifacts_by_type("document")
        assert len(docs) == 1

    def test_get_latest_artifact(self):
        state = AgentState(original_task="Test")
        state.artifacts = [
            Artifact(artifact_type="research", content="first", source_agent=AgentRole.RESEARCHER),
            Artifact(artifact_type="document", content="second", source_agent=AgentRole.WRITER),
        ]
        assert state.get_latest_artifact().content == "second"
        assert state.get_latest_artifact("research").content == "first"


# --- Researcher Agent Tests ---


class TestResearcherAgent:
    """Test the Researcher agent."""

    @pytest.mark.asyncio
    async def test_produces_research_artifact(self):
        mock_response = json.dumps({
            "summary": "Found relevant documentation on turbine maintenance.",
            "key_findings": ["Turbines require monthly inspection", "Blade erosion is common"],
            "sources": [{"id": "chunk_1", "relevance": "maintenance schedule"}],
            "gaps": ["No data on offshore turbines"],
            "confidence": 0.85,
        })
        provider = MockProvider(responses=[mock_response])
        agent = ResearcherAgent(provider=provider)

        state = AgentState(original_task="Research turbine maintenance procedures")
        state = await agent.execute(state)

        assert len(state.artifacts) == 1
        art = state.artifacts[0]
        assert art.artifact_type == "research"
        assert art.source_agent == AgentRole.RESEARCHER
        assert art.confidence == 0.85

    @pytest.mark.asyncio
    async def test_uses_subtask_description_as_query(self):
        provider = MockProvider(responses=['{"summary":"ok","confidence":0.9}'])
        agent = ResearcherAgent(provider=provider)

        st = SubTask(description="Find blade erosion data", assigned_to=AgentRole.RESEARCHER)
        state = AgentState(
            original_task="Write maintenance report",
            plan=TaskPlan(task_description="Write maintenance report", subtasks=[st]),
            current_subtask_id=st.id,
        )
        state = await agent.execute(state)

        # The agent should have used the subtask description
        last_msgs = provider.last_messages
        assert any("blade erosion" in m.content.lower() for m in last_msgs)


# --- Writer Agent Tests ---


class TestWriterAgent:
    """Test the Writer agent."""

    @pytest.mark.asyncio
    async def test_produces_document_artifact(self):
        provider = MockProvider(responses=[
            "# Turbine Maintenance Guide\n\nThis document covers..."
        ])
        agent = WriterAgent(provider=provider)

        state = AgentState(original_task="Write turbine maintenance docs")
        # Add research for the writer to consume
        state.artifacts.append(Artifact(
            artifact_type="research",
            content='{"summary": "Turbines need regular maintenance"}',
            source_agent=AgentRole.RESEARCHER,
            citations=["chunk_1"],
        ))

        state = await agent.execute(state)

        docs = state.get_artifacts_by_type("document")
        assert len(docs) == 1
        assert "Turbine Maintenance Guide" in docs[0].content
        assert docs[0].source_agent == AgentRole.WRITER
        assert "chunk_1" in docs[0].citations

    @pytest.mark.asyncio
    async def test_includes_rejection_feedback_in_revision(self):
        provider = MockProvider(responses=["Revised document with more detail..."])
        agent = WriterAgent(provider=provider)

        state = AgentState(original_task="Write docs")
        # Simulate a previous document that was rejected
        prev_doc = Artifact(
            artifact_type="document",
            content="Draft 1",
            source_agent=AgentRole.WRITER,
        )
        state.artifacts.append(prev_doc)
        state.reviews.append(ReviewVerdict(
            artifact_id=prev_doc.id,
            approved=False,
            issues=["Missing safety procedures section"],
            suggestions=["Add a dedicated safety section"],
            reasoning="Document is incomplete",
        ))

        state = await agent.execute(state)

        # The writer should have received the rejection feedback in its prompt
        last_msgs = provider.last_messages
        context_msg = next((m for m in last_msgs if "safety" in m.content.lower()), None)
        assert context_msg is not None


# --- Reviewer Agent Tests ---


class TestReviewerAgent:
    """Test the Reviewer agent (LLM-as-judge)."""

    @pytest.mark.asyncio
    async def test_approves_good_artifact(self):
        verdict_json = json.dumps({
            "approved": True,
            "confidence": 0.95,
            "issues": [],
            "suggestions": ["Consider adding a diagram"],
            "reasoning": "Document is comprehensive and well-cited.",
        })
        provider = MockProvider(responses=[verdict_json])
        agent = ReviewerAgent(provider=provider)

        state = AgentState(original_task="Write docs")
        state.artifacts.append(Artifact(
            artifact_type="document",
            content="A complete, well-written document.",
            source_agent=AgentRole.WRITER,
        ))

        state = await agent.execute(state)

        assert len(state.reviews) == 1
        assert state.reviews[0].approved is True
        assert state.reviews[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_rejects_poor_artifact(self):
        verdict_json = json.dumps({
            "approved": False,
            "confidence": 0.8,
            "issues": ["Missing citations", "Factual error in section 2"],
            "suggestions": ["Add source references", "Verify turbine RPM claims"],
            "reasoning": "Several accuracy issues need correction.",
        })
        provider = MockProvider(responses=[verdict_json])
        agent = ReviewerAgent(provider=provider)

        state = AgentState(original_task="Write docs")
        state.artifacts.append(Artifact(
            artifact_type="document",
            content="Some poorly written content.",
            source_agent=AgentRole.WRITER,
        ))

        state = await agent.execute(state)

        assert len(state.reviews) == 1
        review = state.reviews[0]
        assert review.approved is False
        assert len(review.issues) == 2
        assert "citations" in review.issues[0].lower()

    @pytest.mark.asyncio
    async def test_skips_already_reviewed_artifacts(self):
        provider = MockProvider()
        agent = ReviewerAgent(provider=provider)

        doc = Artifact(
            artifact_type="document",
            content="Already reviewed doc.",
            source_agent=AgentRole.WRITER,
        )
        state = AgentState(original_task="Test")
        state.artifacts.append(doc)
        state.reviews.append(ReviewVerdict(
            artifact_id=doc.id,
            approved=True,
        ))

        state = await agent.execute(state)

        # Should not have added a new review (nothing to review)
        assert len(state.reviews) == 1
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_response(self):
        provider = MockProvider(responses=["This is not JSON at all. I approve this."])
        agent = ReviewerAgent(provider=provider)

        state = AgentState(original_task="Test")
        state.artifacts.append(Artifact(
            artifact_type="document",
            content="Some content.",
            source_agent=AgentRole.WRITER,
        ))

        state = await agent.execute(state)

        # Should still produce a verdict using fallback parsing
        assert len(state.reviews) == 1
        assert state.reviews[0].confidence == 0.5  # Lower confidence for fallback


# --- Orchestrator Tests ---


class TestOrchestrator:
    """Test the orchestrator's planning and graph execution."""

    @pytest.mark.asyncio
    async def test_creates_task_plan(self):
        plan_response = json.dumps({
            "subtasks": [
                {"description": "Research turbine specs", "assigned_to": "researcher", "depends_on": []},
                {"description": "Write maintenance guide", "assigned_to": "writer", "depends_on": [0]},
            ]
        })
        provider = MockProvider(responses=[plan_response])
        orchestrator = OrchestratorAgent(provider=provider)

        state = AgentState(original_task="Create turbine maintenance documentation")
        state = await orchestrator.execute(state)

        assert state.plan is not None
        assert len(state.plan.subtasks) == 2
        assert state.plan.subtasks[0].assigned_to == AgentRole.RESEARCHER
        assert state.plan.subtasks[1].assigned_to == AgentRole.WRITER
        # Writer depends on researcher
        assert state.plan.subtasks[0].id in state.plan.subtasks[1].depends_on

    @pytest.mark.asyncio
    async def test_fallback_plan_on_bad_json(self):
        provider = MockProvider(responses=["I'll break down the task... (no JSON)"])
        orchestrator = OrchestratorAgent(provider=provider)

        state = AgentState(original_task="Some task")
        state = await orchestrator.execute(state)

        # Fallback: research → write
        assert state.plan is not None
        assert len(state.plan.subtasks) == 2
        assert state.plan.subtasks[0].assigned_to == AgentRole.RESEARCHER
        assert state.plan.subtasks[1].assigned_to == AgentRole.WRITER

    @pytest.mark.asyncio
    async def test_full_workflow_with_mock(self):
        """End-to-end: plan → research → write → review (approve)."""
        # Responses in order: plan, research, write, review
        responses = [
            # 1. Orchestrator plans
            json.dumps({
                "subtasks": [
                    {"description": "Research API endpoints", "assigned_to": "researcher", "depends_on": []},
                    {"description": "Write API documentation", "assigned_to": "writer", "depends_on": [0]},
                ]
            }),
            # 2. Researcher produces research
            json.dumps({
                "summary": "Found 5 API endpoints for turbine monitoring.",
                "key_findings": ["GET /turbines", "POST /maintenance"],
                "sources": [{"id": "chunk_api_1", "relevance": "API reference"}],
                "gaps": [],
                "confidence": 0.9,
            }),
            # 3. Writer produces document
            "# Turbine Monitoring API\n\n## Endpoints\n- GET /turbines\n- POST /maintenance\n\nDetailed documentation...",
            # 4. Reviewer approves
            json.dumps({
                "approved": True,
                "confidence": 0.92,
                "issues": [],
                "suggestions": [],
                "reasoning": "Comprehensive and accurate documentation.",
            }),
        ]
        provider = MockProvider(responses=responses)
        orchestrator = OrchestratorAgent(provider=provider)
        result = await orchestrator.run("Document the turbine monitoring API")

        # Verify the workflow completed
        assert result.status == TaskStatus.COMPLETED
        assert len(result.artifacts) >= 2  # research + document
        assert any(a.artifact_type == "research" for a in result.artifacts)
        assert any(a.artifact_type == "document" for a in result.artifacts)
        assert len(result.reviews) >= 1
        assert result.reviews[-1].approved is True

    @pytest.mark.asyncio
    async def test_revision_loop_on_rejection(self):
        """Writer gets rejected, revises, then gets approved."""
        responses = [
            # 1. Plan
            json.dumps({
                "subtasks": [
                    {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                    {"description": "Write report", "assigned_to": "writer", "depends_on": [0]},
                ]
            }),
            # 2. Research
            json.dumps({"summary": "Research data", "confidence": 0.9}),
            # 3. First write attempt
            "Draft 1: incomplete document",
            # 4. Reviewer REJECTS
            json.dumps({
                "approved": False,
                "confidence": 0.8,
                "issues": ["Too short", "Missing details"],
                "suggestions": ["Expand sections"],
                "reasoning": "Needs more content.",
            }),
            # 5. Writer revises
            "Draft 2: Complete and detailed document with all sections...",
            # 6. Reviewer APPROVES
            json.dumps({
                "approved": True,
                "confidence": 0.9,
                "issues": [],
                "suggestions": [],
                "reasoning": "Much improved.",
            }),
        ]
        provider = MockProvider(responses=responses)
        orchestrator = OrchestratorAgent(provider=provider)
        result = await orchestrator.run("Write a detailed report")

        assert result.status == TaskStatus.COMPLETED
        # Should have 2 reviews (reject + approve)
        assert len(result.reviews) == 2
        assert result.reviews[0].approved is False
        assert result.reviews[1].approved is True
        # Should have more than one document artifact (original + revision)
        docs = result.get_artifacts_by_type("document")
        assert len(docs) >= 2
