"""
End-to-end integration test: full write-back workflow.

Tests the complete pipeline:
  Task → Plan → Research → Write → Review → Approve → Write-Back → Verify
"""

import json
import tempfile
from pathlib import Path

import pytest

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.state import AgentState, TaskStatus
from lattice.comms.checkpoint import LocalCheckpointStore
from lattice.inference.mock import MockProvider
from lattice.writeback import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalMode,
    FileSystemWriteBack,
    WriteBackEngine,
    WriteBackRequest,
    WriteBackTarget,
)


class TestEndToEndWriteBack:
    """Full workflow: agents produce → reviewer approves → write-back executes."""

    @pytest.mark.asyncio
    async def test_agent_to_file_writeback(self, tmp_path):
        """Agents produce a document, it gets written to file system."""
        responses = [
            # 1. Plan
            json.dumps({
                "subtasks": [
                    {"description": "Research the API", "assigned_to": "researcher", "depends_on": []},
                    {"description": "Write API docs", "assigned_to": "writer", "depends_on": [0]},
                ]
            }),
            # 2. Research
            json.dumps({
                "summary": "REST API with 3 endpoints.",
                "key_findings": ["GET /users", "POST /users", "DELETE /users/:id"],
                "sources": [],
                "gaps": [],
                "confidence": 0.9,
            }),
            # 3. Write
            "# Users API\n\n## Endpoints\n\n### GET /users\nReturns all users.\n\n### POST /users\nCreates a user.\n\n### DELETE /users/:id\nDeletes a user.",
            # 4. Review (approve)
            json.dumps({
                "approved": True,
                "confidence": 0.95,
                "issues": [],
                "suggestions": [],
                "reasoning": "Comprehensive API documentation.",
            }),
        ]

        provider = MockProvider(responses=responses)
        orchestrator = OrchestratorAgent(provider=provider)

        # Run the agent workflow
        result = await orchestrator.run("Document the Users API")
        assert result.status == TaskStatus.COMPLETED

        # Get the document artifact
        docs = result.get_artifacts_by_type("document")
        assert len(docs) >= 1
        doc_content = docs[-1].content

        # Now execute write-back
        engine = WriteBackEngine(approval_gate=ApprovalGate(mode=ApprovalMode.NEVER))
        engine.register_handler(FileSystemWriteBack(output_dir=tmp_path))

        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            artifact_id=docs[-1].id,
            content=doc_content,
            metadata={"file_path": "docs/users-api.md"},
        )
        wb_result = await engine.execute(request)

        assert wb_result.success
        written = (tmp_path / "docs" / "users-api.md").read_text()
        assert "GET /users" in written
        assert "DELETE /users/:id" in written

    @pytest.mark.asyncio
    async def test_approval_gate_blocks_low_confidence(self, tmp_path):
        """Low-confidence reviews trigger approval gate."""
        responses = [
            json.dumps({"subtasks": [
                {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                {"description": "Write", "assigned_to": "writer", "depends_on": [0]},
            ]}),
            json.dumps({"summary": "Minimal findings", "confidence": 0.4}),
            "A weakly-sourced document.",
            json.dumps({
                "approved": True,
                "confidence": 0.55,  # Low confidence
                "issues": ["Sparse sourcing"],
                "suggestions": [],
                "reasoning": "Acceptable but weak.",
            }),
        ]

        provider = MockProvider(responses=responses)
        orchestrator = OrchestratorAgent(provider=provider)
        result = await orchestrator.run("Write about advanced topic")

        # Set up approval gate with ON_LOW_CONFIDENCE mode
        approvals_requested = []

        def track_approval(req, review):
            approvals_requested.append(req)
            return ApprovalDecision(approved=False, reason="Human rejected: needs more research")

        gate = ApprovalGate(
            mode=ApprovalMode.ON_LOW_CONFIDENCE,
            confidence_threshold=0.8,
            callback=track_approval,
        )
        engine = WriteBackEngine(approval_gate=gate)
        engine.register_handler(FileSystemWriteBack(output_dir=tmp_path))

        docs = result.get_artifacts_by_type("document")
        review = result.reviews[-1] if result.reviews else None

        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content=docs[-1].content if docs else "content",
            metadata={"file_path": "blocked.md"},
        )
        wb_result = await engine.execute(request, review=review)

        assert not wb_result.success
        assert len(approvals_requested) == 1
        assert not (tmp_path / "blocked.md").exists()

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume(self, tmp_path):
        """Task state is saved and can be resumed after a crash."""
        store = LocalCheckpointStore(checkpoint_dir=tmp_path / "checkpoints")

        responses = [
            json.dumps({"subtasks": [
                {"description": "Research", "assigned_to": "researcher", "depends_on": []},
                {"description": "Write", "assigned_to": "writer", "depends_on": [0]},
            ]}),
            json.dumps({"summary": "Data collected", "confidence": 0.9}),
            "# Complete Document\n\nFull content here.",
            json.dumps({"approved": True, "confidence": 0.9, "issues": [], "suggestions": [], "reasoning": "Good."}),
        ]

        provider = MockProvider(responses=responses)
        orchestrator = OrchestratorAgent(provider=provider)
        result = await orchestrator.run("Write documentation", checkpoint_store=store)

        assert result.status == TaskStatus.COMPLETED

        # Verify checkpoint was saved
        tasks = await store.list_tasks()
        assert len(tasks) == 1

        # Load and verify
        loaded = await store.load(tasks[0])
        recovered = AgentState(**loaded)
        assert recovered.status == TaskStatus.COMPLETED
        assert len(recovered.artifacts) >= 2

    @pytest.mark.asyncio
    async def test_kg_writeback_from_ingestion(self, tmp_path):
        """Ingestion agent output gets written back to KG."""
        from lattice.agents.ingestion_agent import IngestionAgent
        from lattice.agents.state import AgentRole, Artifact
        from lattice.graph.engine import NetworkXBackend
        from lattice.writeback import KnowledgeGraphWriteBack

        kg = NetworkXBackend()
        handler = KnowledgeGraphWriteBack(kg_backend=kg)

        # Simulate ingestion agent producing KG expansion
        kg_expansion = json.dumps({
            "entities": [
                {"name": "UserService", "entity_type": "component", "description": "Manages user CRUD"},
                {"name": "AuthMiddleware", "entity_type": "component", "description": "JWT validation"},
            ],
            "relations": [],
        })

        provider = MockProvider(responses=[kg_expansion])
        ingestion_agent = IngestionAgent(provider=provider)

        state = AgentState(original_task="Document the system")
        state.artifacts.append(Artifact(
            artifact_type="document",
            content="The system has a UserService and AuthMiddleware.",
            source_agent=AgentRole.WRITER,
        ))

        state = await ingestion_agent.execute(state)

        # Get the KG expansion artifact
        kg_arts = state.get_artifacts_by_type("kg_expansion")
        assert len(kg_arts) == 1

        # Execute KG write-back
        engine = WriteBackEngine(approval_gate=ApprovalGate(mode=ApprovalMode.NEVER))
        engine.register_handler(handler)

        request = WriteBackRequest(
            target=WriteBackTarget.KNOWLEDGE_GRAPH,
            content=kg_arts[0].content,
            metadata={},
        )
        result = await engine.execute(request)

        assert result.success
        assert kg.num_entities == 2
        entities = kg.get_all_entities()
        names = {e.name for e in entities}
        assert "UserService" in names
        assert "AuthMiddleware" in names
