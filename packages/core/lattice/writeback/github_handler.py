"""
GitHub write-back handler.

Pushes agent outputs to GitHub repositories:
- Create/update files (documentation, code)
- Open Pull Requests with descriptive bodies
- Create branches for isolated changes

Uses PyGithub for API interactions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog

from lattice.writeback.base import (
    WriteBackHandler,
    WriteBackRequest,
    WriteBackResult,
    WriteBackStatus,
    WriteBackTarget,
)

logger = structlog.get_logger()


class GitHubWriteBack(WriteBackHandler):
    """Write artifacts to GitHub repositories."""

    target = WriteBackTarget.GITHUB

    def __init__(
        self,
        token: str,
        default_repo: str | None = None,
        default_branch: str = "main",
    ) -> None:
        from github import Github

        self._github = Github(token)
        self._default_repo = default_repo
        self._default_branch = default_branch

    async def validate(self, request: WriteBackRequest) -> tuple[bool, str]:
        """Check that we can write to the target repo/path."""
        repo_name = request.metadata.get("repo", self._default_repo)
        if not repo_name:
            return False, "No repository specified and no default configured"

        file_path = request.metadata.get("file_path")
        if not file_path:
            return False, "No file_path specified in metadata"

        try:
            repo = self._github.get_repo(repo_name)
            # Check we have push access
            permissions = repo.permissions
            if not permissions.push:
                return False, f"No push access to {repo_name}"
            return True, "Validation passed"
        except Exception as e:
            return False, f"GitHub validation failed: {e}"

    async def execute(self, request: WriteBackRequest) -> WriteBackResult:
        """Push content to GitHub (create/update file or open PR)."""
        repo_name = request.metadata.get("repo", self._default_repo)
        file_path = request.metadata.get("file_path", "")
        branch = request.metadata.get("branch", self._default_branch)
        commit_message = request.metadata.get(
            "commit_message", f"[Lattice] Update {file_path}"
        )
        create_pr = request.metadata.get("create_pr", False)
        pr_title = request.metadata.get("pr_title", f"[Lattice] {file_path}")
        pr_body = request.metadata.get("pr_body", "Automated update by Lattice agent team.")

        try:
            repo = self._github.get_repo(repo_name)

            if create_pr:
                result = await self._create_pr_with_file(
                    repo, file_path, request.content,
                    branch, commit_message, pr_title, pr_body,
                )
            else:
                result = await self._update_file(
                    repo, file_path, request.content, branch, commit_message
                )

            request.status = WriteBackStatus.COMPLETED
            request.executed_at = datetime.utcnow()
            logger.info("github_writeback_success", repo=repo_name, path=file_path)
            return result

        except Exception as e:
            request.status = WriteBackStatus.FAILED
            request.error = str(e)
            logger.error("github_writeback_failed", repo=repo_name, error=str(e))
            return WriteBackResult(
                success=False,
                request_id=request.id,
                target=self.target,
                message=f"GitHub write-back failed: {e}",
            )

    async def _update_file(
        self, repo, file_path: str, content: str,
        branch: str, commit_message: str,
    ) -> WriteBackResult:
        """Create or update a single file on a branch."""
        try:
            existing = repo.get_contents(file_path, ref=branch)
            result = repo.update_file(
                file_path, commit_message, content, existing.sha, branch=branch
            )
            action = "updated"
        except Exception:
            result = repo.create_file(
                file_path, commit_message, content, branch=branch
            )
            action = "created"

        return WriteBackResult(
            success=True,
            request_id="",
            target=self.target,
            message=f"File {action}: {file_path}",
            url=result["content"].html_url if isinstance(result, dict) else None,
            metadata={"sha": result["commit"].sha if isinstance(result, dict) else ""},
        )

    async def _create_pr_with_file(
        self, repo, file_path: str, content: str,
        base_branch: str, commit_message: str,
        pr_title: str, pr_body: str,
    ) -> WriteBackResult:
        """Create a new branch, commit file, and open a PR."""
        import time

        # Create feature branch
        branch_name = f"lattice/{file_path.replace('/', '-')}-{int(time.time())}"
        base_ref = repo.get_git_ref(f"heads/{base_branch}")
        repo.create_git_ref(f"refs/heads/{branch_name}", base_ref.object.sha)

        # Commit file to new branch
        try:
            existing = repo.get_contents(file_path, ref=branch_name)
            repo.update_file(
                file_path, commit_message, content, existing.sha, branch=branch_name
            )
        except Exception:
            repo.create_file(file_path, commit_message, content, branch=branch_name)

        # Open PR
        pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=base_branch,
        )

        return WriteBackResult(
            success=True,
            request_id="",
            target=self.target,
            message=f"PR created: {pr_title}",
            url=pr.html_url,
            metadata={"pr_number": pr.number, "branch": branch_name},
        )

    async def rollback(self, request: WriteBackRequest) -> bool:
        """Best-effort rollback: close PR or revert file."""
        try:
            repo_name = request.metadata.get("repo", self._default_repo)
            if not repo_name:
                return False

            repo = self._github.get_repo(repo_name)

            # If it was a PR, close it
            pr_number = request.metadata.get("pr_number")
            if pr_number:
                pr = repo.get_pull(pr_number)
                pr.edit(state="closed")
                logger.info("github_rollback_pr_closed", pr=pr_number)
                return True

            return False
        except Exception as e:
            logger.error("github_rollback_failed", error=str(e))
            return False
