"""Exercise Lattice's real orchestration with scripted, offline model responses.

Run from an installed checkout: python examples/mock_workflow.py
This demonstrates control flow, artifacts, and revision handling, not model quality.
"""

from __future__ import annotations

import asyncio
import json
import logging

import structlog

from lattice.agents.orchestrator import OrchestratorAgent
from lattice.agents.state import TaskStatus
from lattice.inference.mock import MockProvider


async def main() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    responses = [
        json.dumps({"subtasks": [
            {"description": "Read the fixture service notes", "assigned_to": "researcher", "depends_on": []},
            {"description": "Write the introduction", "assigned_to": "writer", "depends_on": [0]},
        ]}),
        json.dumps({
            "summary": "Fixture notes: the demo service exposes GET /health and POST /jobs.",
            "key_findings": ["GET /health reports readiness", "POST /jobs accepts a job"],
            "sources": [], "gaps": [], "confidence": 0.9,
        }),
        "# Demo service\n\nA small service for jobs.",
        json.dumps({
            "approved": False, "confidence": 0.8,
            "issues": ["The introduction omits the health endpoint."],
            "suggestions": ["Mention GET /health and POST /jobs."],
            "reasoning": "Both fixture endpoints should be described.",
        }),
        "# Demo service\n\nA small service with GET /health for readiness and POST /jobs for job submission.",
        json.dumps({
            "approved": True, "confidence": 0.95, "issues": [], "suggestions": [],
            "reasoning": "The revised introduction includes both fixture endpoints.",
        }),
    ]
    provider = MockProvider(responses=responses)
    result = await OrchestratorAgent(provider=provider).run(
        "Write a short README introduction for the fixture service."
    )
    documents = result.get_artifacts_by_type("document")
    if (
        result.status != TaskStatus.COMPLETED
        or len(documents) != 2
        or [review.approved for review in result.reviews] != [False, True]
        or provider.call_count != len(responses)
    ):
        raise RuntimeError("The fixture workflow did not follow its expected revision path.")

    print("LATTICE / OFFLINE WORKFLOW")
    print("Provider: MockProvider (scripted responses)")
    print("Network/model requests: none")
    print("Flow: plan -> research -> write -> reject -> revise -> approve")
    print(f"Status: {result.status.value}")
    print(f"Artifacts: {len(result.artifacts)} (1 research, 2 document versions)")
    print("Reviews: rejected -> approved")
    print(f"Provider calls: {provider.call_count}")
    print("\nFinal artifact:\n")
    print(documents[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
