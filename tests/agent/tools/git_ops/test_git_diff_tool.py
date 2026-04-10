"""Integration tests for the ``git_diff`` tool."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.git_ops.git_diff_tool import git_diff_tool

pytestmark = pytest.mark.asyncio


async def test_git_diff_unstaged_worktree_changes(
    bound_orchestrator, git_context
) -> None:
    """The fixture left ``src/app.py`` modified in the worktree."""
    result = await git_diff_tool({}, git_context)

    assert result["success"] is True
    # At least one file should appear in the diff.
    paths = [f["path"] for f in result["files"]]
    assert "src/app.py" in paths

    stats = result["stats"]
    assert stats["files_changed"] >= 1
    assert stats["insertions"] >= 1


async def test_git_diff_staged(
    bound_orchestrator, git_context
) -> None:
    """README.md was staged by the fixture."""
    result = await git_diff_tool({"staged": True}, git_context)

    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert "README.md" in paths


async def test_git_diff_base_target_across_branches(
    bound_orchestrator, git_context
) -> None:
    """Diff between ``main`` and ``feature/extra`` should surface ``src/extra.py``."""
    result = await git_diff_tool(
        {"base": "main", "target": "feature/extra"}, git_context
    )

    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert "src/extra.py" in paths


async def test_git_diff_unified_context_lines(
    bound_orchestrator, git_context
) -> None:
    """Setting ``unified`` changes the amount of context in the unified diff."""
    zero = await git_diff_tool({"unified": 0}, git_context)
    three = await git_diff_tool({"unified": 3}, git_context)

    assert zero["success"] is True
    assert three["success"] is True
    # With fewer context lines the raw diff should be shorter or equal.
    assert len(zero["raw"]) <= len(three["raw"])


async def test_git_diff_rejects_negative_unified(
    bound_orchestrator, git_context
) -> None:
    result = await git_diff_tool({"unified": -1}, git_context)
    assert result["success"] is False
    assert "unified" in result["message"]


async def test_git_diff_path_filter(
    bound_orchestrator, git_context
) -> None:
    result = await git_diff_tool({"path": "src/app.py"}, git_context)

    assert result["success"] is True
    for f in result["files"]:
        assert f["path"] == "src/app.py"


async def test_git_diff_hunk_structure(
    bound_orchestrator, git_context
) -> None:
    """Every parsed hunk carries old/new line numbers and typed line entries."""
    result = await git_diff_tool({}, git_context)

    assert result["success"] is True
    assert result["files"], "expected at least one changed file"
    for f in result["files"]:
        for hunk in f["hunks"]:
            assert "old_start" in hunk
            assert "new_start" in hunk
            assert "lines" in hunk
            for line in hunk["lines"]:
                assert line["type"] in {"addition", "deletion", "context"}
