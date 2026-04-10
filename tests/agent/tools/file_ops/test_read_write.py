"""Integration tests for the ``read_file`` / ``write_file`` tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from tesslate_agent.agent.tools.file_ops.read_write import (
    read_file_tool,
    write_file_tool,
)

pytestmark = pytest.mark.asyncio


async def test_read_file_returns_content(
    bound_orchestrator, project_root: Path, fops_context
) -> None:
    (project_root / "hello.txt").write_text("greetings\n", encoding="utf-8")

    result = await read_file_tool({"file_path": "hello.txt"}, fops_context)

    assert result["success"] is True
    assert result["content"] == "greetings\n"
    assert result["details"]["size_bytes"] == len("greetings\n")


async def test_read_file_missing(
    bound_orchestrator, fops_context
) -> None:
    result = await read_file_tool({"file_path": "nope.txt"}, fops_context)
    assert result["success"] is False
    assert "does not exist" in result["message"]


async def test_read_file_requires_path(fops_context) -> None:
    with pytest.raises(ValueError, match="file_path parameter is required"):
        await read_file_tool({}, fops_context)


async def test_write_file_creates_new_file(
    bound_orchestrator, project_root: Path, fops_context
) -> None:
    result = await write_file_tool(
        {"file_path": "new/hello.txt", "content": "payload"},
        fops_context,
    )
    assert result["success"] is True
    assert (project_root / "new" / "hello.txt").read_text(encoding="utf-8") == "payload"


async def test_write_file_overwrites(
    bound_orchestrator, project_root: Path, fops_context
) -> None:
    target = project_root / "file.txt"
    target.write_text("old", encoding="utf-8")

    result = await write_file_tool(
        {"file_path": "file.txt", "content": "new content\nline 2\n"},
        fops_context,
    )
    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == "new content\nline 2\n"
    assert result["details"]["line_count"] == 3


async def test_write_file_requires_content(fops_context) -> None:
    with pytest.raises(ValueError, match="content parameter is required"):
        await write_file_tool({"file_path": "foo.txt"}, fops_context)


async def test_write_file_round_trip_via_read(
    bound_orchestrator, project_root: Path, fops_context
) -> None:
    write = await write_file_tool(
        {"file_path": "rt.txt", "content": "roundtrip data"},
        fops_context,
    )
    assert write["success"] is True

    read = await read_file_tool({"file_path": "rt.txt"}, fops_context)
    assert read["success"] is True
    assert read["content"] == "roundtrip data"
