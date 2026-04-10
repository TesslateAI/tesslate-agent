"""Tests for the background-process inspection tools."""

from __future__ import annotations

import asyncio

import pytest

from tesslate_agent.agent.tools.shell_ops.background import (
    list_background_processes_tool,
    read_background_output_tool,
)
from tesslate_agent.agent.tools.shell_ops.bash import bash_exec_tool
from tesslate_agent.orchestration import PTY_SESSIONS


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    return {"run_id": "run-bg-tests"}


async def test_list_and_read_background(ctx) -> None:
    spawned = await bash_exec_tool(
        {
            "command": "for i in 1 2 3; do echo line-$i; sleep 0.05; done",
            "is_background": True,
        },
        ctx,
    )
    assert spawned["success"] is True
    session_id = spawned["session_id"]

    try:
        # Give the loop time to produce some output.
        await asyncio.sleep(0.5)

        listed = await list_background_processes_tool({}, ctx)
        assert listed["success"] is True
        listed_ids = [s["session_id"] for s in listed["sessions"]]
        assert session_id in listed_ids

        read = await read_background_output_tool(
            {"session_id": session_id, "lines": 10},
            ctx,
        )
        assert read["success"] is True
        # The tail output is a top-level field (see success_output usage).
        output_text = read["output"]
        assert "line-1" in output_text
        assert "line-3" in output_text
    finally:
        PTY_SESSIONS.close(session_id)


async def test_read_background_unknown_session(ctx) -> None:
    result = await read_background_output_tool(
        {"session_id": "bogus-session"},
        ctx,
    )
    assert result["success"] is False
    assert "Unknown" in result["message"]


async def test_run_id_scoping(ctx, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    run_a = {"run_id": "run-A"}
    run_b = {"run_id": "run-B"}

    spawned_a = await bash_exec_tool(
        {"command": "sleep 2", "is_background": True},
        run_a,
    )
    spawned_b = await bash_exec_tool(
        {"command": "sleep 2", "is_background": True},
        run_b,
    )

    try:
        listed_a = await list_background_processes_tool({}, run_a)
        ids_a = {s["session_id"] for s in listed_a["sessions"]}
        assert spawned_a["session_id"] in ids_a
        assert spawned_b["session_id"] not in ids_a

        listed_b = await list_background_processes_tool({}, run_b)
        ids_b = {s["session_id"] for s in listed_b["sessions"]}
        assert spawned_b["session_id"] in ids_b
        assert spawned_a["session_id"] not in ids_b

        # Cross-run read is rejected.
        cross = await read_background_output_tool(
            {"session_id": spawned_b["session_id"]},
            run_a,
        )
        assert cross["success"] is False
        assert "Access denied" in cross["message"]
    finally:
        PTY_SESSIONS.close(spawned_a["session_id"])
        PTY_SESSIONS.close(spawned_b["session_id"])


async def test_read_background_requires_session_id(ctx) -> None:
    with pytest.raises(ValueError):
        await read_background_output_tool({}, ctx)


async def test_lines_must_be_positive(ctx) -> None:
    with pytest.raises(ValueError):
        await read_background_output_tool(
            {"session_id": "anything", "lines": 0},
            ctx,
        )
