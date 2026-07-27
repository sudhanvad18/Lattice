"""
Human approval gate.

Configurable safety layer that sits between the Reviewer's approval
and the actual write-back execution. Three modes:

- ALWAYS: every write-back requires human approval (safest)
- ON_LOW_CONFIDENCE: only when reviewer confidence < threshold
- NEVER: fully autonomous (for demo/testing)

In a real deployment, this integrates with the API/dashboard for
user interaction. For local demo, uses a callback pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from lattice.agents.state import ReviewVerdict
from lattice.writeback.base import WriteBackRequest, WriteBackStatus

logger = structlog.get_logger()


class ApprovalMode(str, Enum):
    ALWAYS = "always"
    ON_LOW_CONFIDENCE = "on_low_confidence"
    NEVER = "never"


@dataclass
class ApprovalDecision:
    """The human's decision on a write-back request."""

    approved: bool
    reason: str = ""
    modified_content: Optional[str] = None  # Human can edit before approving


ApprovalCallback = Callable[[WriteBackRequest, ReviewVerdict | None], ApprovalDecision]


class ApprovalGate:
    """Configurable human-in-the-loop gate for write-back safety."""

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.ALWAYS,
        confidence_threshold: float = 0.8,
        callback: ApprovalCallback | None = None,
    ) -> None:
        self._mode = mode
        self._confidence_threshold = confidence_threshold
        self._callback = callback
        self._pending_requests: list[WriteBackRequest] = []
        self._decisions: dict[str, ApprovalDecision] = {}

    @property
    def mode(self) -> ApprovalMode:
        return self._mode

    @property
    def pending_count(self) -> int:
        return len(self._pending_requests)

    def should_require_approval(self, review: ReviewVerdict | None) -> bool:
        """Determine if this write-back needs human approval."""
        if self._mode == ApprovalMode.ALWAYS:
            return True
        elif self._mode == ApprovalMode.NEVER:
            return False
        elif self._mode == ApprovalMode.ON_LOW_CONFIDENCE:
            if review is None:
                return True  # No review = uncertain
            return review.confidence < self._confidence_threshold
        return True

    async def request_approval(
        self,
        request: WriteBackRequest,
        review: ReviewVerdict | None = None,
    ) -> ApprovalDecision:
        """Request human approval for a write-back.

        If a callback is configured, calls it immediately.
        Otherwise, adds to pending queue for later resolution.
        """
        if not self.should_require_approval(review):
            logger.info("approval_auto_granted", request_id=request.id, mode=self._mode.value)
            decision = ApprovalDecision(approved=True, reason="Auto-approved (gate mode)")
            self._decisions[request.id] = decision
            request.status = WriteBackStatus.APPROVED
            return decision

        if self._callback:
            decision = self._callback(request, review)
            self._decisions[request.id] = decision
            if decision.approved:
                request.status = WriteBackStatus.APPROVED
                if decision.modified_content:
                    request.content = decision.modified_content
            else:
                request.status = WriteBackStatus.REJECTED
            logger.info(
                "approval_decision",
                request_id=request.id,
                approved=decision.approved,
                reason=decision.reason,
            )
            return decision

        # No callback — queue for manual resolution
        self._pending_requests.append(request)
        logger.info("approval_queued", request_id=request.id)
        return ApprovalDecision(approved=False, reason="Queued for manual approval")

    def resolve_pending(self, request_id: str, decision: ApprovalDecision) -> None:
        """Manually resolve a pending approval request."""
        self._decisions[request_id] = decision
        self._pending_requests = [
            r for r in self._pending_requests if r.id != request_id
        ]
        logger.info("approval_resolved", request_id=request_id, approved=decision.approved)

    def get_decision(self, request_id: str) -> ApprovalDecision | None:
        return self._decisions.get(request_id)
