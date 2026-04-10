"""Integration tests for the ``glob`` navigation tool."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tesslate_agent.agent.tools.nav_ops.glob_tool import glob_tool

pytestmark = pytest.mark.asyncio


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def test_glob_simple_pattern_match(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / "a.py", "print(1)\n")
    _touch(project_root / "b.py", "print(2)\n")
    _touch(project_root / "c.txt", "hi\n")

    result = await glob_tool({"pattern": "*.py"}, nav_context)

    assert result["success"] is True
    paths = {entry["path"] for entry in result["matches"]}
    assert paths == {"a.py", "b.py"}
    assert result["total_found"] == 2
    assert result["truncated"] is False


async def test_glob_recursive_double_star(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / "src" / "app" / "main.py", "x = 1\n")
    _touch(project_root / "src" / "util.py", "y = 2\n")
    _touch(project_root / "tests" / "test_x.py", "z = 3\n")
    _touch(project_root / "README.md", "# readme\n")

    result = await glob_tool({"pattern": "**/*.py"}, nav_context)

    assert result["success"] is True
    paths = {entry["path"] for entry in result["matches"]}
    assert "src/app/main.py" in paths
    assert "src/util.py" in paths
    assert "tests/test_x.py" in paths
    assert "README.md" not in paths


async def test_glob_sort_by_mtime_returns_newest_first(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    old = project_root / "old.py"
    new = project_root / "new.py"
    _touch(old, "old\n")
    # Set old's mtime firmly in the past so ordering is deterministic.
    past = time.time() - 10_000
    os.utime(old, (past, past))
    _touch(new, "new\n")
    os.utime(new, (time.time(), time.time()))

    result = await glob_tool({"pattern": "*.py", "sort": "mtime"}, nav_context)

    assert result["success"] is True
    assert [entry["path"] for entry in result["matches"]] == ["new.py", "old.py"]


async def test_glob_sort_by_name(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / "c.py")
    _touch(project_root / "a.py")
    _touch(project_root / "b.py")

    result = await glob_tool({"pattern": "*.py", "sort": "name"}, nav_context)

    assert result["success"] is True
    assert [entry["path"] for entry in result["matches"]] == ["a.py", "b.py", "c.py"]


async def test_glob_limit_honored_and_truncation_flag(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    for i in range(6):
        _touch(project_root / f"f{i}.py", str(i))

    result = await glob_tool({"pattern": "*.py", "limit": 3}, nav_context)

    assert result["success"] is True
    assert len(result["matches"]) == 3
    assert result["total_found"] == 6
    assert result["truncated"] is True


async def test_glob_gitignore_exclusion(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / ".gitignore", "ignored/\n*.log\n")
    _touch(project_root / "keep.py", "keep\n")
    _touch(project_root / "ignored" / "hidden.py", "secret\n")
    _touch(project_root / "trace.log", "logs\n")

    result = await glob_tool({"pattern": "**/*"}, nav_context)

    assert result["success"] is True
    paths = {entry["path"] for entry in result["matches"]}
    assert "keep.py" in paths
    assert "ignored/hidden.py" not in paths
    assert "trace.log" not in paths


async def test_glob_requires_pattern(
    bound_orchestrator, nav_context
) -> None:
    result = await glob_tool({}, nav_context)
    assert result["success"] is False
    assert "pattern" in result["message"]


async def test_glob_rejects_invalid_sort(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / "a.py")
    result = await glob_tool({"pattern": "*.py", "sort": "bogus"}, nav_context)
    assert result["success"] is False
    assert "sort" in result["message"].lower()


async def test_glob_no_matches_returns_empty_success(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _touch(project_root / "readme.md")

    result = await glob_tool({"pattern": "*.py"}, nav_context)

    assert result["success"] is True
    assert result["matches"] == []
    assert result["total_found"] == 0
    assert result["truncated"] is False
