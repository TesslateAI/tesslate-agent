"""Integration tests for the ``list_dir`` navigation tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tesslate_agent.agent.tools.nav_ops.list_dir_tool import list_dir_tool

pytestmark = pytest.mark.asyncio


def _make_tree(root: Path) -> None:
    """
    Build a small tree under ``root``::

        root/
        ├── a.txt
        ├── b.txt
        ├── c.txt
        ├── d.txt
        └── sub/
            ├── one.py
            ├── two.py
            └── deeper/
                └── buried.py
    """
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("bb", encoding="utf-8")
    (root / "c.txt").write_text("ccc", encoding="utf-8")
    (root / "d.txt").write_text("dddd", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "one.py").write_text("1", encoding="utf-8")
    (root / "sub" / "two.py").write_text("2", encoding="utf-8")
    (root / "sub" / "deeper").mkdir()
    (root / "sub" / "deeper" / "buried.py").write_text("3", encoding="utf-8")


async def test_list_dir_basic_listing_at_root(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _make_tree(project_root)

    result = await list_dir_tool({"dir_path": "."}, nav_context)

    assert result["success"] is True
    names = {entry["name"] for entry in result["entries"]}
    # Default depth=2 traverses into sub/ and its direct children, but
    # NOT into sub/deeper/* (that would require depth >= 3).
    assert "a.txt" in names
    assert "sub" in names
    assert "one.py" in names
    assert "buried.py" not in names


async def test_list_dir_depth_limit_stops_descent(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _make_tree(project_root)

    shallow = await list_dir_tool({"dir_path": ".", "depth": 1}, nav_context)
    deep = await list_dir_tool({"dir_path": ".", "depth": 3, "limit": 100}, nav_context)

    assert shallow["success"] is True
    assert deep["success"] is True

    shallow_names = {entry["name"] for entry in shallow["entries"]}
    deep_names = {entry["name"] for entry in deep["entries"]}

    # depth=1 sees the top-level sub/ dir but none of its children.
    assert "sub" in shallow_names
    assert "one.py" not in shallow_names
    assert "buried.py" not in shallow_names

    # depth=3 reaches the deepest file.
    assert "buried.py" in deep_names


async def test_list_dir_offset_and_limit_pagination(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _make_tree(project_root)

    first = await list_dir_tool(
        {"dir_path": ".", "depth": 1, "offset": 1, "limit": 2}, nav_context
    )
    second = await list_dir_tool(
        {"dir_path": ".", "depth": 1, "offset": 3, "limit": 2}, nav_context
    )

    assert first["success"] is True
    assert second["success"] is True
    assert len(first["entries"]) == 2
    assert first["has_more"] is True
    # Second page should contain different entries than the first.
    first_names = {e["name"] for e in first["entries"]}
    second_names = {e["name"] for e in second["entries"]}
    assert first_names.isdisjoint(second_names)


async def test_list_dir_offset_beyond_total_errors(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    (project_root / "only.txt").write_text("x", encoding="utf-8")

    result = await list_dir_tool(
        {"dir_path": ".", "offset": 999, "limit": 10}, nav_context
    )

    assert result["success"] is False
    assert "offset" in result["message"].lower()


async def test_list_dir_hidden_files_are_skipped_by_default(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    """Dotfiles must not leak into the listing under the default settings."""
    (project_root / "visible.txt").write_text("x", encoding="utf-8")
    (project_root / ".secret").write_text("y", encoding="utf-8")

    result = await list_dir_tool({"dir_path": "."}, nav_context)

    assert result["success"] is True
    names = {entry["name"] for entry in result["entries"]}
    assert "visible.txt" in names
    assert ".secret" not in names


async def test_list_dir_nested_subdir(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    _make_tree(project_root)

    result = await list_dir_tool({"dir_path": "sub", "depth": 1}, nav_context)

    assert result["success"] is True
    names = {entry["name"] for entry in result["entries"]}
    assert "one.py" in names
    assert "two.py" in names
    assert "deeper" in names


async def test_list_dir_empty_directory(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    (project_root / "empty").mkdir()

    result = await list_dir_tool({"dir_path": "empty"}, nav_context)

    assert result["success"] is True
    assert result["entries"] == []
    assert result["total"] == 0


async def test_list_dir_requires_dir_path(nav_context) -> None:
    result = await list_dir_tool({}, nav_context)
    assert result["success"] is False
    assert "dir_path" in result["message"]


async def test_list_dir_rejects_zero_offset(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    (project_root / "a.txt").write_text("x", encoding="utf-8")

    result = await list_dir_tool(
        {"dir_path": ".", "offset": 0}, nav_context
    )

    assert result["success"] is False
    assert "offset" in result["message"].lower()


async def test_list_dir_rejects_zero_limit(
    bound_orchestrator, project_root: Path, nav_context
) -> None:
    (project_root / "a.txt").write_text("x", encoding="utf-8")

    result = await list_dir_tool(
        {"dir_path": ".", "limit": 0}, nav_context
    )

    assert result["success"] is False
    assert "limit" in result["message"].lower()
