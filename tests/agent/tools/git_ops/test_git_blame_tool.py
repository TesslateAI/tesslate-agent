"""Integration tests for the ``git_blame`` tool."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.git_ops.git_blame_tool import git_blame_tool

pytestmark = pytest.mark.asyncio


async def test_git_blame_full_file(
    bound_orchestrator,
    tmp_path,
    monkeypatch,
    git_env,
    git_context,
) -> None:
    """
    Build a fresh clean repo with one committed file and verify the blame
    tool returns structurally complete, correctly attributed per-line
    records.
    """
    import subprocess

    clean = tmp_path / "blame_repo"
    clean.mkdir()

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=clean, env=git_env, check=True, capture_output=True
        )

    _git("init", "-q", "-b", "main")
    (clean / "lib.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n",
        encoding="utf-8",
    )
    _git("add", "lib.py")
    _git("commit", "-q", "-m", "add lib")

    monkeypatch.setenv("PROJECT_ROOT", str(clean))

    result = await git_blame_tool({"file_path": "lib.py"}, git_context)

    assert result["success"] is True
    assert result["file"] == "lib.py"
    lines = result["lines"]
    # 6 lines in the file.
    assert len(lines) == 6

    for entry in lines:
        assert entry["hash"]
        assert len(entry["hash"]) == 40
        assert entry["abbrev"] == entry["hash"][:7]
        assert entry["author"] == "Test Author"
        assert entry["author_mail"] == "author@example.com"
        assert "line_number" in entry
        assert entry["summary"] == "add lib"


async def test_git_blame_line_range(
    bound_orchestrator, git_context
) -> None:
    result = await git_blame_tool(
        {"file_path": "src/app.py", "line_start": 1, "line_end": 1},
        git_context,
    )

    assert result["success"] is True
    assert len(result["lines"]) == 1
    assert result["lines"][0]["line_number"] == 1


async def test_git_blame_requires_file_path(git_context) -> None:
    result = await git_blame_tool({}, git_context)
    assert result["success"] is False
    assert "file_path" in result["message"]


async def test_git_blame_missing_file_errors(
    bound_orchestrator, git_context
) -> None:
    result = await git_blame_tool(
        {"file_path": "does_not_exist.py"}, git_context
    )
    assert result["success"] is False


async def test_git_blame_rejects_partial_range(
    bound_orchestrator, git_context
) -> None:
    result = await git_blame_tool(
        {"file_path": "README.md", "line_start": 1},
        git_context,
    )
    assert result["success"] is False
    assert "line_start" in result["message"] and "line_end" in result["message"]


async def test_git_blame_rejects_invalid_range(
    bound_orchestrator, git_context
) -> None:
    result = await git_blame_tool(
        {"file_path": "README.md", "line_start": 5, "line_end": 1},
        git_context,
    )
    assert result["success"] is False
    assert "line" in result["message"].lower()
