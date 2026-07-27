"""
Tests for the checkpoint/crash recovery system.
"""

import pytest

from lattice.agents.state import AgentState, TaskStatus
from lattice.comms.checkpoint import LocalCheckpointStore


class TestLocalCheckpointStore:
    """Test file-based checkpoint persistence."""

    @pytest.fixture
    def store(self, tmp_path):
        return LocalCheckpointStore(checkpoint_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        state = AgentState(original_task="Test task")
        state.status = TaskStatus.IN_PROGRESS

        await store.save("task-1", state.model_dump())
        loaded = await store.load("task-1")

        assert loaded is not None
        recovered = AgentState(**loaded)
        assert recovered.original_task == "Test task"
        assert recovered.status == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, store):
        result = await store.load("nonexistent-task")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        state = AgentState(original_task="To be deleted")
        await store.save("task-del", state.model_dump())
        assert await store.load("task-del") is not None

        await store.delete("task-del")
        assert await store.load("task-del") is None

    @pytest.mark.asyncio
    async def test_list_tasks(self, store):
        for i in range(3):
            state = AgentState(original_task=f"Task {i}")
            await store.save(f"task-{i}", state.model_dump())

        tasks = await store.list_tasks()
        assert len(tasks) == 3
        assert "task-0" in tasks
        assert "task-2" in tasks

    @pytest.mark.asyncio
    async def test_overwrite_checkpoint(self, store):
        state = AgentState(original_task="Test")
        state.iteration_count = 1
        await store.save("task-ow", state.model_dump())

        state.iteration_count = 5
        await store.save("task-ow", state.model_dump())

        loaded = await store.load("task-ow")
        recovered = AgentState(**loaded)
        assert recovered.iteration_count == 5

    @pytest.mark.asyncio
    async def test_preserves_artifacts(self, store):
        from lattice.agents.state import AgentRole, Artifact

        state = AgentState(original_task="Test with artifacts")
        state.artifacts.append(Artifact(
            artifact_type="research",
            content="Important findings about turbines",
            source_agent=AgentRole.RESEARCHER,
            citations=["chunk_1", "chunk_2"],
        ))

        await store.save("task-art", state.model_dump())
        loaded = await store.load("task-art")
        recovered = AgentState(**loaded)

        assert len(recovered.artifacts) == 1
        assert recovered.artifacts[0].content == "Important findings about turbines"
        assert recovered.artifacts[0].citations == ["chunk_1", "chunk_2"]
