"""
File system write-back handler.

Writes agent outputs to the local file system:
- Generated documentation (markdown, HTML)
- Reports and analyses
- Code files
- Exported data

All writes go to a configurable output directory with full audit trail.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog

from lattice.writeback.base import (
    WriteBackHandler,
    WriteBackRequest,
    WriteBackResult,
    WriteBackStatus,
    WriteBackTarget,
)

logger = structlog.get_logger()


class FileSystemWriteBack(WriteBackHandler):
    """Write artifacts to the local file system."""

    target = WriteBackTarget.FILE_SYSTEM

    def __init__(self, output_dir: str | Path = "output") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def validate(self, request: WriteBackRequest) -> tuple[bool, str]:
        """Check that we can write to the target path."""
        file_path = request.metadata.get("file_path")
        if not file_path:
            return False, "No file_path specified in metadata"

        target = self._output_dir / file_path
        # Prevent path traversal
        try:
            target.resolve().relative_to(self._output_dir.resolve())
        except ValueError:
            return False, f"Path traversal detected: {file_path}"

        if not request.content:
            return False, "No content to write"

        return True, "Validation passed"

    async def execute(self, request: WriteBackRequest) -> WriteBackResult:
        """Write content to a file."""
        file_path = request.metadata.get("file_path", "output.md")
        target = self._output_dir / file_path

        try:
            target.parent.mkdir(parents=True, exist_ok=True)

            # Track if we're overwriting (for rollback)
            existing_content = None
            if target.exists():
                existing_content = target.read_text()

            target.write_text(request.content)

            request.status = WriteBackStatus.COMPLETED
            request.executed_at = datetime.utcnow()
            request.result = str(target)

            # Store original content for potential rollback
            if existing_content is not None:
                request.metadata["_original_content"] = existing_content
            request.metadata["_written_path"] = str(target)

            logger.info("file_writeback_success", path=str(target), size=len(request.content))
            return WriteBackResult(
                success=True,
                request_id=request.id,
                target=self.target,
                message=f"Written to {target}",
                url=str(target),
                metadata={"size_bytes": len(request.content.encode())},
            )

        except Exception as e:
            request.status = WriteBackStatus.FAILED
            request.error = str(e)
            logger.error("file_writeback_failed", path=str(target), error=str(e))
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=self.target,
                message=f"File write failed: {e}",
            )

    async def rollback(self, request: WriteBackRequest) -> bool:
        """Restore original file content or delete created file."""
        try:
            written_path = request.metadata.get("_written_path")
            if not written_path:
                return False

            target = Path(written_path)
            original = request.metadata.get("_original_content")

            if original is not None:
                target.write_text(original)
                logger.info("file_rollback_restored", path=written_path)
            elif target.exists():
                target.unlink()
                logger.info("file_rollback_deleted", path=written_path)

            return True
        except Exception as e:
            logger.error("file_rollback_failed", error=str(e))
            return False
