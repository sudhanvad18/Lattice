"""
Tests for the write-back system: file handler, KG handler, approval gate, and engine.
"""

import json
import tempfile
from pathlib import Path

import pytest

from lattice.agents.state import AgentRole, ReviewVerdict
from lattice.graph.engine import NetworkXBackend
from lattice.writeback import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalMode,
    FileSystemWriteBack,
    KnowledgeGraphWriteBack,
    WriteBackEngine,
    WriteBackRequest,
    WriteBackStatus,
    WriteBackTarget,
)


# --- File System Write-Back Tests ---


class TestFileSystemWriteBack:
    """Test writing artifacts to the file system."""

    @pytest.fixture
    def handler(self, tmp_path):
        return FileSystemWriteBack(output_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, handler):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="# Hello\nThis is documentation.",
            metadata={"file_path": "docs/README.md"},
        )
        is_valid, reason = await handler.validate(request)
        assert is_valid

    @pytest.mark.asyncio
    async def test_validate_no_file_path(self, handler):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="content",
            metadata={},
        )
        is_valid, reason = await handler.validate(request)
        assert not is_valid
        assert "file_path" in reason

    @pytest.mark.asyncio
    async def test_validate_path_traversal_blocked(self, handler):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="content",
            metadata={"file_path": "../../etc/passwd"},
        )
        is_valid, reason = await handler.validate(request)
        assert not is_valid
        assert "traversal" in reason.lower()

    @pytest.mark.asyncio
    async def test_execute_creates_file(self, handler, tmp_path):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="# API Documentation\n\nEndpoints listed below.",
            metadata={"file_path": "docs/api.md"},
        )
        result = await handler.execute(request)

        assert result.success
        assert (tmp_path / "docs" / "api.md").exists()
        assert "API Documentation" in (tmp_path / "docs" / "api.md").read_text()

    @pytest.mark.asyncio
    async def test_execute_creates_nested_directories(self, handler, tmp_path):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="deep content",
            metadata={"file_path": "a/b/c/deep.txt"},
        )
        result = await handler.execute(request)

        assert result.success
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()

    @pytest.mark.asyncio
    async def test_rollback_deletes_new_file(self, handler, tmp_path):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="temporary content",
            metadata={"file_path": "temp.md"},
        )
        await handler.execute(request)
        assert (tmp_path / "temp.md").exists()

        success = await handler.rollback(request)
        assert success
        assert not (tmp_path / "temp.md").exists()

    @pytest.mark.asyncio
    async def test_rollback_restores_original_content(self, handler, tmp_path):
        # Create original file
        (tmp_path / "existing.md").write_text("original content")

        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="overwritten content",
            metadata={"file_path": "existing.md"},
        )
        await handler.execute(request)
        assert (tmp_path / "existing.md").read_text() == "overwritten content"

        success = await handler.rollback(request)
        assert success
        assert (tmp_path / "existing.md").read_text() == "original content"


# --- Knowledge Graph Write-Back Tests ---


class TestKnowledgeGraphWriteBack:
    """Test writing discoveries back to the knowledge graph."""

    @pytest.fixture
    def kg(self):
        return NetworkXBackend()

    @pytest.fixture
    def handler(self, kg):
        return KnowledgeGraphWriteBack(kg_backend=kg)

    @pytest.mark.asyncio
    async def test_validate_valid_content(self, handler):
        content = json.dumps({
            "entities": [{"name": "TurboFan", "entity_type": "component"}],
            "relations": [],
        })
        request = WriteBackRequest(content=content)
        is_valid, reason = await handler.validate(request)
        assert is_valid

    @pytest.mark.asyncio
    async def test_validate_empty_content(self, handler):
        content = json.dumps({"entities": [], "relations": []})
        request = WriteBackRequest(content=content)
        is_valid, reason = await handler.validate(request)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_validate_invalid_json(self, handler):
        request = WriteBackRequest(content="not json")
        is_valid, reason = await handler.validate(request)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_execute_adds_entities(self, handler, kg):
        content = json.dumps({
            "entities": [
                {"name": "Compressor", "entity_type": "component", "description": "Air compression stage"},
                {"name": "Turbine", "entity_type": "component", "description": "Power extraction stage"},
            ],
            "relations": [],
        })
        request = WriteBackRequest(content=content)
        result = await handler.execute(request)

        assert result.success
        assert kg.num_entities == 2
        entities = kg.get_all_entities()
        names = {e.name for e in entities}
        assert "Compressor" in names
        assert "Turbine" in names

    @pytest.mark.asyncio
    async def test_execute_adds_relations(self, handler, kg):
        content = json.dumps({
            "entities": [
                {"name": "Engine", "entity_type": "system"},
                {"name": "Fan", "entity_type": "component"},
            ],
            "relations": [],
        })
        request = WriteBackRequest(content=content)
        await handler.execute(request)

        # Now add a relation between them
        entities = kg.get_all_entities()
        engine_id = next(e.id for e in entities if e.name == "Engine")
        fan_id = next(e.id for e in entities if e.name == "Fan")

        content2 = json.dumps({
            "entities": [],
            "relations": [
                {"source_id": engine_id, "target_id": fan_id, "relation_type": "contains"},
            ],
        })
        request2 = WriteBackRequest(content=content2)
        result = await handler.execute(request2)
        assert result.success
        assert kg.num_relations == 1

    @pytest.mark.asyncio
    async def test_rollback_removes_entities(self, handler, kg):
        content = json.dumps({
            "entities": [{"name": "Temporary", "entity_type": "test"}],
            "relations": [],
        })
        request = WriteBackRequest(content=content)
        await handler.execute(request)
        assert kg.num_entities == 1

        success = await handler.rollback(request)
        assert success
        assert kg.num_entities == 0


# --- Approval Gate Tests ---


class TestApprovalGate:
    """Test the human approval gate."""

    def test_never_mode_auto_approves(self):
        gate = ApprovalGate(mode=ApprovalMode.NEVER)
        review = ReviewVerdict(artifact_id="x", approved=True, confidence=0.5)
        assert not gate.should_require_approval(review)

    def test_always_mode_requires_approval(self):
        gate = ApprovalGate(mode=ApprovalMode.ALWAYS)
        review = ReviewVerdict(artifact_id="x", approved=True, confidence=0.99)
        assert gate.should_require_approval(review)

    def test_low_confidence_mode_with_high_confidence(self):
        gate = ApprovalGate(mode=ApprovalMode.ON_LOW_CONFIDENCE, confidence_threshold=0.8)
        review = ReviewVerdict(artifact_id="x", approved=True, confidence=0.9)
        assert not gate.should_require_approval(review)

    def test_low_confidence_mode_with_low_confidence(self):
        gate = ApprovalGate(mode=ApprovalMode.ON_LOW_CONFIDENCE, confidence_threshold=0.8)
        review = ReviewVerdict(artifact_id="x", approved=True, confidence=0.6)
        assert gate.should_require_approval(review)

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        def approve_all(req, review):
            return ApprovalDecision(approved=True, reason="Auto-approved by test")

        gate = ApprovalGate(mode=ApprovalMode.ALWAYS, callback=approve_all)
        request = WriteBackRequest(content="test")
        decision = await gate.request_approval(request)
        assert decision.approved

    @pytest.mark.asyncio
    async def test_callback_can_reject(self):
        def reject_all(req, review):
            return ApprovalDecision(approved=False, reason="Rejected by policy")

        gate = ApprovalGate(mode=ApprovalMode.ALWAYS, callback=reject_all)
        request = WriteBackRequest(content="test")
        decision = await gate.request_approval(request)
        assert not decision.approved
        assert request.status == WriteBackStatus.REJECTED

    @pytest.mark.asyncio
    async def test_callback_can_modify_content(self):
        def edit_and_approve(req, review):
            return ApprovalDecision(
                approved=True,
                reason="Approved with edits",
                modified_content="edited content",
            )

        gate = ApprovalGate(mode=ApprovalMode.ALWAYS, callback=edit_and_approve)
        request = WriteBackRequest(content="original")
        await gate.request_approval(request)
        assert request.content == "edited content"

    @pytest.mark.asyncio
    async def test_never_mode_skips_callback(self):
        gate = ApprovalGate(mode=ApprovalMode.NEVER)
        request = WriteBackRequest(content="test")
        review = ReviewVerdict(artifact_id="x", approved=True, confidence=0.95)
        decision = await gate.request_approval(request, review)
        assert decision.approved


# --- Write-Back Engine Tests ---


class TestWriteBackEngine:
    """Test the full write-back engine pipeline."""

    @pytest.fixture
    def engine(self, tmp_path):
        gate = ApprovalGate(mode=ApprovalMode.NEVER)
        engine = WriteBackEngine(approval_gate=gate)
        engine.register_handler(FileSystemWriteBack(output_dir=tmp_path))
        return engine

    @pytest.mark.asyncio
    async def test_execute_file_writeback(self, engine, tmp_path):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="Generated documentation content.",
            metadata={"file_path": "output.md"},
        )
        result = await engine.execute(request)

        assert result.success
        assert (tmp_path / "output.md").exists()

    @pytest.mark.asyncio
    async def test_execute_with_no_handler(self, tmp_path):
        engine = WriteBackEngine(approval_gate=ApprovalGate(mode=ApprovalMode.NEVER))
        request = WriteBackRequest(
            target=WriteBackTarget.GITHUB,
            content="test",
            metadata={"file_path": "test.md"},
        )
        result = await engine.execute(request)
        assert not result.success
        assert "No handler" in result.message

    @pytest.mark.asyncio
    async def test_execute_with_approval_rejection(self, tmp_path):
        def reject(req, review):
            return ApprovalDecision(approved=False, reason="Not safe")

        gate = ApprovalGate(mode=ApprovalMode.ALWAYS, callback=reject)
        engine = WriteBackEngine(approval_gate=gate)
        engine.register_handler(FileSystemWriteBack(output_dir=tmp_path))

        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="should not be written",
            metadata={"file_path": "rejected.md"},
        )
        result = await engine.execute(request)

        assert not result.success
        assert "Rejected" in result.message
        assert not (tmp_path / "rejected.md").exists()

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, engine):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="",  # Empty content fails validation
            metadata={"file_path": "test.md"},
        )
        result = await engine.execute(request)
        assert not result.success

    @pytest.mark.asyncio
    async def test_rollback(self, engine, tmp_path):
        request = WriteBackRequest(
            target=WriteBackTarget.FILE_SYSTEM,
            content="will be rolled back",
            metadata={"file_path": "rollback_test.md"},
        )
        await engine.execute(request)
        assert (tmp_path / "rollback_test.md").exists()

        success = await engine.rollback(request.id)
        assert success
        assert not (tmp_path / "rollback_test.md").exists()

    @pytest.mark.asyncio
    async def test_history_tracking(self, engine, tmp_path):
        for i in range(3):
            request = WriteBackRequest(
                target=WriteBackTarget.FILE_SYSTEM,
                content=f"doc {i}",
                metadata={"file_path": f"doc_{i}.md"},
            )
            await engine.execute(request)

        assert len(engine.history) == 3
