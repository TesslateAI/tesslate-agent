"""Integration tests for the ``git_status`` tool."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tesslate_agent.agent.tools.git_ops.git_status_tool import git_status_tool

pytestmark = pytest.mark.asyncio


def _git(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True
    )


async def test_git_status_mixed_worktree_state(
    bound_orchestrator, temp_git_repo: Path, git_context
) -> None:
    """The fixture leaves a staged change, an unstaged change, and an untracked file."""
    result = await git_status_tool({}, git_context)

    assert result["success"] is True
    assert result["branch"]["name"] == "main"

    # Staged README.md + unstaged src/app.py.
    paths = [c["path"] for c in result["changes"]]
    assert "README.md" in paths
    assert "src/app.py" in paths

    # README was staged: index_status should NOT be a dot, worktree may be dot.
    readme_entry = next(c for c in result["changes"] if c["path"] == "README.md")
    assert readme_entry["index_status"] != "."

    # src/app.py was modified in the worktree: worktree_status should NOT be a dot.
    app_entry = next(c for c in result["changes"] if c["path"] == "src/app.py")
    assert app_entry["worktree_status"] != "."

    # Untracked file shows up.
    assert "untracked.txt" in result["untracked"]


async def test_git_status_clean_repo(
    bound_orchestrator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_env: dict[str, str],
    git_context,
) -> None:
    """Point the orchestrator at a brand new, clean repo with one commit."""
    clean_repo = tmp_path / "clean_repo"
    clean_repo.mkdir()
    _git(["init", "-q", "-b", "main"], clean_repo, git_env)
    (clean_repo / "only.txt").write_text("only\n", encoding="utf-8")
    _git(["add", "only.txt"], clean_repo, git_env)
    _git(["commit", "-q", "-m", "first"], clean_repo, git_env)

    monkeypatch.setenv("PROJECT_ROOT", str(clean_repo))

    result = await git_status_tool({}, git_context)

    assert result["success"] is True
    assert result["branch"]["name"] == "main"
    assert result["changes"] == []
    assert result["untracked"] == []
    assert "clean" in result["message"].lower()


async def test_git_status_exclude_untracked(
    bound_orchestrator, git_context
) -> None:
    """``include_untracked=False`` should hide the untracked list."""
    result = await git_status_tool({"include_untracked": False}, git_context)

    assert result["success"] is True
    assert result["untracked"] == []
    # Tracked changes are still reported.
    assert len(result["changes"]) >= 2


async def test_git_status_path_filter(
    bound_orchestrator, git_context
) -> None:
    result = await git_status_tool({"path": "src"}, git_context)

    assert result["success"] is True
    paths = [c["path"] for c in result["changes"]]
    # Filtering on src/ excludes README.md.
    assert "README.md" not in paths
    assert "src/app.py" in paths
