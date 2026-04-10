"""Integration tests for the ``git_log`` tool."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.git_ops.git_log_tool import git_log_tool

pytestmark = pytest.mark.asyncio


async def test_git_log_returns_all_commits(
    bound_orchestrator, git_context
) -> None:
    result = await git_log_tool({}, git_context)

    assert result["success"] is True
    commits = result["commits"]
    # The fixture creates exactly 3 commits on ``main``.
    assert len(commits) == 3
    subjects = [c["subject"] for c in commits]
    assert subjects[0] == "fix: say hello instead of hi"  # most recent first
    assert subjects[-1] == "initial: add README"

    for commit in commits:
        assert commit["hash"]
        assert len(commit["hash"]) == 40
        assert commit["abbrev"]
        assert commit["author"]["name"] == "Test Author"
        assert commit["author"]["email"] == "author@example.com"
        assert commit["date"]


async def test_git_log_max_count_limit(
    bound_orchestrator, git_context
) -> None:
    result = await git_log_tool({"max_count": 1}, git_context)

    assert result["success"] is True
    assert len(result["commits"]) == 1
    assert result["commits"][0]["subject"] == "fix: say hello instead of hi"


async def test_git_log_rejects_non_positive_max_count(
    bound_orchestrator, git_context
) -> None:
    result = await git_log_tool({"max_count": 0}, git_context)
    assert result["success"] is False
    assert "max_count" in result["message"]


async def test_git_log_author_filter(
    bound_orchestrator, git_context
) -> None:
    # All three fixture commits are by "Test Author".
    matched = await git_log_tool({"author": "Test"}, git_context)
    unmatched = await git_log_tool(
        {"author": "no-such-author-xyz"}, git_context
    )

    assert matched["success"] is True
    assert unmatched["success"] is True
    assert len(matched["commits"]) == 3
    assert unmatched["commits"] == []


async def test_git_log_path_filter(
    bound_orchestrator, git_context
) -> None:
    """Filtering on ``src/app.py`` should omit the README-only first commit."""
    result = await git_log_tool({"path": "src/app.py"}, git_context)

    assert result["success"] is True
    subjects = {c["subject"] for c in result["commits"]}
    assert "feat: add greet()" in subjects
    assert "fix: say hello instead of hi" in subjects
    assert "initial: add README" not in subjects
    assert len(result["commits"]) == 2


async def test_git_log_grep_filter(
    bound_orchestrator, git_context
) -> None:
    result = await git_log_tool({"grep": "initial"}, git_context)

    assert result["success"] is True
    assert len(result["commits"]) == 1
    assert result["commits"][0]["subject"] == "initial: add README"
