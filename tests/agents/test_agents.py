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

    @pytest.mark.asyncio
    async def test_revision_includes_previous_draft_and_locks_topic(self):
        provider = MockProvider(responses=["Revised briefing with the same topic..."])
        agent = WriterAgent(provider=provider)
        state = AgentState(original_task="Write a feature briefing")
        prev_doc = Artifact(
            artifact_type="document",
            content="## Active Session Notifications\nOriginal draft body.",
            source_agent=AgentRole.WRITER,
        )
        state.artifacts.append(prev_doc)
        state.reviews.append(ReviewVerdict(
            artifact_id=prev_doc.id, approved=False,
            issues=["No implementation plan"], suggestions=["Add integration steps"],
            reasoning="Plan missing",
        ))
        await agent.execute(state)
        joined = "\n".join(m.content for m in provider.last_messages)
        user_msg = provider.last_messages[-1].content
        assert "Active Session Notifications" in joined
        assert "Original draft body." in joined
        assert "Revise the previous draft" in user_msg
        assert "do not switch" in user_msg.lower()


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

    @pytest.mark.asyncio
    async def test_persistent_rejection_stops_at_max_iterations(self):
        """A reviewer that never approves must not loop the writer forever."""
        rejection = json.dumps({
            "approved": False, "confidence": 0.5, "issues": ["Still wrong"],
            "suggestions": [], "reasoning": "Rejected.",
        })
        responses = [
            json.dumps({"subtasks": [
                {"description": "Write report", "assigned_to": "writer", "depends_on": []},
            ]}),
        ] + ["Draft", rejection] * 50
        orchestrator = OrchestratorAgent(provider=MockProvider(responses=responses),
                                         max_revisions=50)

        result = await orchestrator.run("Write a report", max_iterations=4)

        assert result.iteration_count == 4
        assert result.status == TaskStatus.FAILED
        assert "Still wrong" in result.errors[-1]
        assert len(result.reviews) == 4
        assert all(not review.approved for review in result.reviews)

    @pytest.mark.asyncio
    async def test_persistent_rejection_breaks_out_at_max_revisions(self):
        """After max_revisions on one subtask, accept the draft to avoid a loop."""
        rejection = json.dumps({
            "approved": False, "confidence": 0.5, "issues": ["Needs more sources"],
            "suggestions": [], "reasoning": "Thin evidence.",
        })
        responses = [
            json.dumps({"subtasks": [
                {"description": "Write report", "assigned_to": "writer", "depends_on": []},
            ]}),
        ] + ["Draft", rejection] * 50
        orchestrator = OrchestratorAgent(provider=MockProvider(responses=responses),
                                         max_revisions=3)

        result = await orchestrator.run("Write a report", max_iterations=20)

        assert result.status == TaskStatus.COMPLETED
        assert result.iteration_count == 3
        assert len(result.reviews) == 3
        assert all(not review.approved for review in result.reviews)
        assert any("Accepted after 3 revisions" in e for e in result.errors)
        assert any("Needs more sources" in e for e in result.errors)


@pytest.mark.parametrize("items", [
    None, [], [None], [{}],
    [{"assigned_to": "code", "description": "Run code"}],
    [{"assigned_to": "writer", "description": " "}],
    *[[{"assigned_to": "writer", "description": "Draft", "depends_on": bad}]
      for bad in ([0], [1], [-1], [False], ["0"], None, "0")],
    [{"assigned_to": "writer", "description": "Draft", "depends_on": [1]},
     {"assigned_to": "researcher", "description": "Find", "depends_on": [0]}],
])
def test_invalid_plan_uses_safe_fallback(items):
    plan = OrchestratorAgent(MockProvider())._parse_plan(json.dumps({"subtasks": items}), "task")
    research, writer = plan.subtasks
    assert (research.assigned_to, writer.assigned_to) == (AgentRole.RESEARCHER, AgentRole.WRITER)
    assert writer.depends_on == [research.id]


@pytest.mark.parametrize("reply", ['{"approved": false', '{"approved":"false"}', "I approve"])
def test_unstructured_review_fails_closed(reply):
    verdict = ReviewerAgent(MockProvider())._parse_verdict(reply, "draft")
    assert not verdict.approved
    assert verdict.issues and verdict.suggestions


def test_fenced_review():
    verdict = ReviewerAgent(MockProvider())._parse_verdict(
        '```json\n{"approved":true,"reasoning":"Sources verified"}\n```', "draft",
    )
    assert verdict.approved and verdict.reasoning == "Sources verified"


async def test_structured_agent_steps_request_json():
    class RecordingProvider(MockProvider):
        def __init__(self):
            super().__init__(responses=[
                '{"subtasks":[{"description":"Find","assigned_to":"researcher"},'
                '{"description":"Draft","assigned_to":"writer","depends_on":[0]}]}',
                '{"approved":true,"confidence":0.9}',
            ])
            self.configs = []

        async def generate(self, messages, config=None):
            self.configs.append(config)
            return await super().generate(messages, config)

    provider = RecordingProvider()
    state = AgentState(original_task="Research and draft a briefing")
    await OrchestratorAgent(provider).execute(state)
    state.artifacts.append(Artifact(
        artifact_type="document", content="Draft", source_agent=AgentRole.WRITER,
    ))
    await ReviewerAgent(provider).execute(state)
    assert len(provider.configs) == 2
    assert all(config.json_output for config in provider.configs)


async def test_feature_research_anchors_project_purpose_in_readme():
    from lattice.retrieval.vector_store import SearchResult

    overview = SearchResult("overview", "This is an authentication automation tool.",
                            {"source": "/repo/README.md"}, 0.8)
    fragment = SearchResult("implementation", "export function login() {}",
                            {"source": "/repo/src/auth.ts"}, 0.7)

    class Store:
        def source_counts(self):
            return {"/repo/docs/nested/README.md": 1, "/repo/README.md": 1}

        def search(self, query, top_k=8, filter_source=None):
            if filter_source:
                assert filter_source == "/repo/README.md"
                return [overview]
            return [fragment, overview]

    class Provider(MockProvider):
        async def generate(self, messages, config=None):
            assert config.json_output
            return await super().generate(messages, config)

    provider = Provider()
    state = await ResearcherAgent(provider, vector_store=Store(), top_k=2).execute(
        AgentState(original_task="Propose a novelty feature for this tool"),
    )
    prompt = provider.last_messages[1].content
    assert prompt.index("authentication automation") < prompt.index("export function")
    assert "Source: /repo/README.md" in prompt
    assert prompt.count("authentication automation") == 1
    assert set(state.artifacts[0].citations) == {"overview", "implementation"}


async def test_long_plan_stops_at_dispatch_budget():
    plan = {"subtasks": [{"description": f"Section {i}", "assigned_to": "writer"}
                         for i in range(13)]}
    provider = MockProvider(responses=[json.dumps(plan)] + ["Draft", '{"approved":true}'] * 12)
    result = await OrchestratorAgent(provider).run("Write sections", max_iterations=12)
    assert result.status == TaskStatus.FAILED
    assert "Iteration limit" in result.errors[-1]
    assert result.iteration_count == len(result.artifacts) == len(result.reviews) == 12


@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.PENDING])
async def test_unschedulable_required_work_fails(status, monkeypatch):
    orchestrator = OrchestratorAgent(MockProvider())

    async def plan(state):
        state.plan = TaskPlan(task_description="task", subtasks=[
            SubTask(description="blocked", assigned_to=AgentRole.WRITER,
                    depends_on=["missing"], status=status),
        ])
        return state

    monkeypatch.setattr(orchestrator, "execute", plan)
    result = await orchestrator.run("task")
    assert result.status == TaskStatus.FAILED and result.errors
    assert result.iteration_count == 0
