"""Integration tests for the persistent-memory tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tesslate_agent.agent.tools.memory_ops.memory_tool import (
    MemoryStore,
    load_memory_prefix,
    memory_read_tool,
    memory_write_tool,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def memory_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated project root for memory tests."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(project))
    return project


@pytest.fixture
def memory_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HOME so ``scope='global'`` does not touch the real filesystem."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # ``Path.home()`` respects HOME on POSIX.
    return home


@pytest.fixture
def memory_ctx() -> dict:
    return {"run_id": "memory-tests"}


async def test_read_nonexistent_returns_structured_error(
    memory_project, memory_home, memory_ctx
) -> None:
    result = await memory_read_tool({}, memory_ctx)

    assert result["success"] is False
    assert result["exists"] is False
    assert result["scope"] == "project"
    assert str(memory_project) in result["path"]


async def test_write_then_read_round_trip(
    memory_project, memory_home, memory_ctx
) -> None:
    write = await memory_write_tool(
        {"section": "Conventions", "body": "- Use pytest.\n- Keep it simple.\n"},
        memory_ctx,
    )
    assert write["success"] is True
    assert write["section"] == "Conventions"
    assert write["mode"] == "replace"

    read = await memory_read_tool({"section": "Conventions"}, memory_ctx)
    assert read["success"] is True
    assert "Use pytest" in read["content"]
    assert "Keep it simple" in read["content"]
    assert "Conventions" in read["sections"]


async def test_replace_existing_section_overwrites(
    memory_project, memory_home, memory_ctx
) -> None:
    await memory_write_tool(
        {"section": "Decisions", "body": "Old decision.\n"}, memory_ctx
    )
    await memory_write_tool(
        {
            "section": "Decisions",
            "body": "New decision.\n",
            "mode": "replace",
        },
        memory_ctx,
    )

    read = await memory_read_tool({"section": "Decisions"}, memory_ctx)
    assert read["success"] is True
    assert "New decision" in read["content"]
    assert "Old decision" not in read["content"]


async def test_append_to_existing_section_preserves_prior(
    memory_project, memory_home, memory_ctx
) -> None:
    await memory_write_tool(
        {"section": "Decisions", "body": "First point"}, memory_ctx
    )
    await memory_write_tool(
        {
            "section": "Decisions",
            "body": "Second point",
            "mode": "append",
        },
        memory_ctx,
    )

    read = await memory_read_tool({"section": "Decisions"}, memory_ctx)
    assert read["success"] is True
    assert "First point" in read["content"]
    assert "Second point" in read["content"]


async def test_list_sections_returns_h2_headings(
    memory_project, memory_home, memory_ctx
) -> None:
    await memory_write_tool(
        {"section": "Alpha", "body": "a"}, memory_ctx
    )
    await memory_write_tool(
        {"section": "Beta", "body": "b"}, memory_ctx
    )
    await memory_write_tool(
        {"section": "Gamma", "body": "g"}, memory_ctx
    )

    store = MemoryStore(context=memory_ctx)
    sections = await store.list_sections("project")
    assert sections == ["Alpha", "Beta", "Gamma"]


async def test_project_vs_global_scope_isolated(
    memory_project, memory_home, memory_ctx
) -> None:
    await memory_write_tool(
        {"section": "ProjectOnly", "body": "p"}, memory_ctx
    )
    await memory_write_tool(
        {
            "section": "GlobalOnly",
            "body": "g",
            "scope": "global",
        },
        memory_ctx,
    )

    project_read = await memory_read_tool(
        {"section": "ProjectOnly"}, memory_ctx
    )
    global_read = await memory_read_tool(
        {"section": "GlobalOnly", "scope": "global"}, memory_ctx
    )
    assert project_read["success"] is True
    assert global_read["success"] is True

    # Cross-scope lookups must not find the other scope's section.
    missing_project = await memory_read_tool(
        {"section": "GlobalOnly"}, memory_ctx
    )
    missing_global = await memory_read_tool(
        {"section": "ProjectOnly", "scope": "global"}, memory_ctx
    )
    assert missing_project["success"] is False
    assert missing_global["success"] is False


async def test_concurrent_writes_do_not_corrupt_file(
    memory_project, memory_home, memory_ctx
) -> None:
    """Fire off many parallel writes and verify the file stays parseable."""

    async def _write(n: int) -> None:
        await memory_write_tool(
            {"section": f"Section{n}", "body": f"value {n}"}, memory_ctx
        )

    await asyncio.gather(*[_write(i) for i in range(15)])

    store = MemoryStore(context=memory_ctx)
    sections = await store.list_sections("project")
    # Every section that "won" the race should be present — at minimum all
    # 15 eventually land because each writer takes the exclusive file lock.
    assert set(sections) == {f"Section{i}" for i in range(15)}

    # Each section should be readable and carry its own body.
    for i in range(15):
        body = await store.read_section("project", f"Section{i}")
        assert f"value {i}" in body


async def test_memory_write_rejects_empty_section(
    memory_project, memory_home, memory_ctx
) -> None:
    result = await memory_write_tool(
        {"section": "   ", "body": "x"}, memory_ctx
    )
    assert result["success"] is False
    assert "section" in result["message"].lower()


async def test_memory_write_rejects_invalid_mode(
    memory_project, memory_home, memory_ctx
) -> None:
    result = await memory_write_tool(
        {"section": "X", "body": "y", "mode": "wat"}, memory_ctx
    )
    assert result["success"] is False
    assert "mode" in result["message"].lower()


async def test_load_memory_prefix_returns_wrapped_content(
    memory_project, memory_home, memory_ctx
) -> None:
    await memory_write_tool(
        {"section": "Conventions", "body": "Use snake_case."}, memory_ctx
    )

    prefix = load_memory_prefix(memory_project)
    assert "## Persistent Memory" in prefix
    assert "Conventions" in prefix
    assert "Use snake_case" in prefix
    assert prefix.startswith("\n\n---\n")
    assert prefix.endswith("\n---\n")


async def test_load_memory_prefix_empty_when_missing(
    memory_project, memory_home
) -> None:
    assert load_memory_prefix(memory_project) == ""
