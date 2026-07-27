"""
Tests for the FastAPI application.
"""

import json

import pytest
from fastapi.testclient import TestClient

from lattice.agents.state import AgentState, TaskStatus
from lattice.inference.mock import MockProvider
from lattice_api.app import create_app
from lattice_api.audit import AuditTrail
from lattice_api.task_manager import TaskManager


@pytest.fixture
def mock_provider():
    """Provider that returns valid agent responses."""
    return MockProvider(responses=[
        # Plan
        json.dumps({"subtasks": [
            {"description": "Research", "assigned_to": "researcher", "depends_on": []},
            {"description": "Write", "assigned_to": "writer", "depends_on": [0]},
        ]}),
        # Research
        json.dumps({"summary": "Found data", "confidence": 0.9}),
        # Write
        "# Generated Document\n\nContent here.",
        # Review
        json.dumps({"approved": True, "confidence": 0.9, "issues": [], "suggestions": [], "reasoning": "Good."}),
    ])


@pytest.fixture
def task_manager(mock_provider):
    audit = AuditTrail()
    return TaskManager(provider=mock_provider, audit_trail=audit)


@pytest.fixture
def client(task_manager):
    app = create_app(task_manager=task_manager)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "orchestrator" in data["agents_available"]
        assert "researcher" in data["agents_available"]

    def test_health_includes_version(self, client):
        resp = client.get("/health")
        assert resp.json()["version"] == "0.1.0"


class TestTaskSubmission:
    def test_submit_task_returns_task_id(self, client):
        resp = client.post("/tasks", json={"task": "Write API documentation"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert data["original_task"] == "Write API documentation"

    def test_submit_task_validates_empty_task(self, client):
        resp = client.post("/tasks", json={"task": ""})
        assert resp.status_code == 422  # Validation error

    def test_list_tasks(self, client):
        client.post("/tasks", json={"task": "Task 1"})
        client.post("/tasks", json={"task": "Task 2"})
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    def test_get_task_status(self, client):
        submit_resp = client.post("/tasks", json={"task": "Test task"})
        task_id = submit_resp.json()["task_id"]

        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == task_id

    def test_get_nonexistent_task_returns_404(self, client):
        resp = client.get("/tasks/nonexistent-id")
        assert resp.status_code == 404


class TestTaskResults:
    def test_get_result_of_submitted_task(self, client):
        submit_resp = client.post("/tasks", json={"task": "Document APIs"})
        task_id = submit_resp.json()["task_id"]

        resp = client.get(f"/tasks/{task_id}/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert "artifacts" in data
        assert "audit_trail" in data

    def test_get_audit_trail(self, client):
        submit_resp = client.post("/tasks", json={"task": "Audit test"})
        task_id = submit_resp.json()["task_id"]

        resp = client.get(f"/tasks/{task_id}/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert "events" in data
        # Should have at least the task_created event
        assert len(data["events"]) >= 1
        assert data["events"][0]["event_type"] == "task_created"
