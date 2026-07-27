"""
Write-back engine.

Orchestrates the full write-back flow:
1. Validate the request against the target handler
2. Pass through approval gate
3. Execute the write-back
4. Record the result for audit

Provides a registry pattern so handlers can be added dynamically.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from lattice.agents.state import ReviewVerdict
from lattice.writeback.approval_gate import ApprovalDecision, ApprovalGate, ApprovalMode
from lattice.writeback.base import (
    WriteBackHandler,
    WriteBackRequest,
    WriteBackResult,
    WriteBackStatus,
    WriteBackTarget,
)

logger = structlog.get_logger()


class WriteBackEngine:
    """Central engine for executing write-backs with safety gates."""

    def __init__(self, approval_gate: ApprovalGate | None = None) -> None:
        self._handlers: dict[WriteBackTarget, WriteBackHandler] = {}
        self._approval_gate = approval_gate or ApprovalGate(mode=ApprovalMode.NEVER)
        self._history: list[WriteBackRequest] = []

    def register_handler(self, handler: WriteBackHandler) -> None:
        """Register a write-back handler for its target type."""
        self._handlers[handler.target] = handler
        logger.info("writeback_handler_registered", target=handler.target.value)

    async def execute(
        self,
        request: WriteBackRequest,
        review: ReviewVerdict | None = None,
    ) -> WriteBackResult:
        """Execute a write-back request through the full safety pipeline."""
        self._history.append(request)

        # 1. Find handler
        handler = self._handlers.get(request.target)
        if not handler:
            request.status = WriteBackStatus.FAILED
            request.error = f"No handler registered for {request.target.value}"
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=request.target,
                message=request.error,
            )

        # 2. Validate
        is_valid, reason = await handler.validate(request)
        if not is_valid:
            request.status = WriteBackStatus.FAILED
            request.error = f"Validation failed: {reason}"
            logger.warning("writeback_validation_failed", reason=reason)
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=request.target,
                message=reason,
            )

        # 3. Approval gate
        decision = await self._approval_gate.request_approval(request, review)
        if not decision.approved:
            request.status = WriteBackStatus.REJECTED
            logger.info("writeback_rejected", request_id=request.id, reason=decision.reason)
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=request.target,
                message=f"Rejected: {decision.reason}",
            )

        # 4. Execute
        request.status = WriteBackStatus.EXECUTING
        result = await handler.execute(request)
        result.request_id = request.id

        logger.info(
            "writeback_executed",
            request_id=request.id,
            target=request.target.value,
            success=result.success,
        )
        return result

    async def rollback(self, request_id: str) -> bool:
        """Attempt to rollback a completed write-back."""
        request = next((r for r in self._history if r.id == request_id), None)
        if not request:
            return False

        handler = self._handlers.get(request.target)
        if not handler:
            return False

        return await handler.rollback(request)

    @property
    def history(self) -> list[WriteBackRequest]:
        return list(self._history)
