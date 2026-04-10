"""Integration tests for the ``grep`` navigation tool."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tesslate_agent.agent.tools.nav_ops.grep_tool import grep_tool

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        shutil.which("rg") is None,
        reason="ripgrep (`rg`) binary is not installed on PATH",
    ),
]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def grep_fixture(bound_orchestrator, project_root: Path):
    _write(
        project_root / "src" / "hello.py",
        "def hello():\n    return 'hi'\n\nTODO: fix\n",
    )
    _write(
        project_root / "src" / "world.py",
        "def world():\n    return 'world'\n# TODO: wire up\n# TODO: add tests\n",
    )
    _write(
        project_root / "README.md",
        "# Project\n\nNo todos here.\n",
    )
    return project_root


async def test_grep_files_with_matches_default(
    grep_fixture, nav_context
) -> None:
    result = await grep_tool({"pattern": "TODO"}, nav_context)

    assert result["success"] is True
    files = {f for f in result["files"]}
    assert any(f.endswith("hello.py") for f in files)
    assert any(f.endswith("world.py") for f in files)
    # README has no TODO, so it should be absent.
    assert not any(f.endswith("README.md") for f in files)
    assert result["details"]["output_mode"] == "files_with_matches"


async def test_grep_content_mode_with_context(
    grep_fixture, nav_context
) -> None:
    result = await grep_tool(
        {"pattern": "TODO", "output_mode": "content", "-C": 1},
        nav_context,
    )

    assert result["success"] is True
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) >= 3  # 1 in hello.py, 2 in world.py

    # Every match record has a line_number and line_text.
    for entry in result["matches"]:
        assert "line_number" in entry
        assert "line_text" in entry
        assert "TODO" in entry["line_text"]

    # At least one match should have surrounding context because of ``-C 1``.
    has_context = any(
        entry.get("before_context") or entry.get("after_context")
        for entry in result["matches"]
    )
    assert has_context, "Expected at least one match to carry context lines"


async def test_grep_count_mode(
    grep_fixture, nav_context
) -> None:
    result = await grep_tool(
        {"pattern": "TODO", "output_mode": "count"},
        nav_context,
    )

    assert result["success"] is True
    counts = result["counts"]
    assert counts, "expected at least one file with matches"
    # world.py has 2 TODOs, hello.py has 1.
    total = sum(counts.values())
    assert total >= 3
    assert result["details"]["total_matches"] == total


async def test_grep_case_insensitive(
    grep_fixture, nav_context
) -> None:
    # "todo" lowercase shouldn't match TODO unless ``-i`` is passed.
    baseline = await grep_tool(
        {"pattern": "todo", "output_mode": "count"},
        nav_context,
    )
    assert baseline["success"] is True
    baseline_total = sum(baseline["counts"].values())

    insensitive = await grep_tool(
        {"pattern": "todo", "-i": True, "output_mode": "count"},
        nav_context,
    )
    assert insensitive["success"] is True
    insensitive_total = sum(insensitive["counts"].values())

    assert insensitive_total > baseline_total


async def test_grep_head_limit_applied(
    grep_fixture, nav_context
) -> None:
    result = await grep_tool(
        {"pattern": "TODO", "output_mode": "content", "head_limit": 1},
        nav_context,
    )

    assert result["success"] is True
    assert len(result["matches"]) == 1
    assert result["details"]["total"] >= 3
    assert result["details"]["returned"] == 1


async def test_grep_requires_pattern(nav_context) -> None:
    result = await grep_tool({}, nav_context)
    assert result["success"] is False
    assert "pattern" in result["message"]


async def test_grep_rejects_invalid_regex(nav_context) -> None:
    result = await grep_tool({"pattern": "[unclosed"}, nav_context)
    assert result["success"] is False
    assert "regular expression" in result["message"].lower() or "regex" in result["message"].lower()


async def test_grep_rejects_unknown_output_mode(nav_context) -> None:
    result = await grep_tool(
        {"pattern": "TODO", "output_mode": "bogus"}, nav_context
    )
    assert result["success"] is False
    assert "output_mode" in result["message"]


async def test_grep_no_matches_returns_clean_empty(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    (project_root / "only.md").write_text("nothing to see here\n", encoding="utf-8")

    result = await grep_tool(
        {"pattern": "THIS_DEFINITELY_DOES_NOT_EXIST_XYZ_123"},
        nav_context,
    )

    assert result["success"] is True
    assert result["files"] == []
