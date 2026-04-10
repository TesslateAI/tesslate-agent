"""Tests for the ``write_stdin`` tool."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.shell_ops.bash import bash_exec_tool
from tesslate_agent.agent.tools.shell_ops.write_stdin import write_stdin_tool
from tesslate_agent.orchestration import PTY_SESSIONS


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    return {"run_id": "run-write-stdin-tests"}


async def test_round_trip_with_cat(ctx) -> None:
    # Launch ``cat`` as a background PTY session, feed it a line, and
    # read the echo back.
    spawned = await bash_exec_tool(
        {"command": "cat", "is_background": True},
        ctx,
    )
    assert spawned["success"] is True
    session_id = spawned["session_id"]

    try:
        result = await write_stdin_tool(
            {
                "session_id": session_id,
                "chars": "hello\n",
                "yield_time_ms": 500,
            },
            ctx,
        )
        assert result["success"] is True
        assert "hello" in result["details"]["new_output"]
        assert result["details"]["status"] == "running"
    finally:
        PTY_SESSIONS.close(session_id)


async def test_unknown_session_returns_error(ctx) -> None:
    result = await write_stdin_tool(
        {"session_id": "not-a-real-session", "chars": "hi\n"},
        ctx,
    )
    assert result["success"] is False
    assert "Unknown PTY session" in result["message"]


async def test_missing_session_id_raises(ctx) -> None:
    with pytest.raises(ValueError):
        await write_stdin_tool({"chars": "hi\n"}, ctx)


async def test_missing_chars_raises(ctx) -> None:
    with pytest.raises(ValueError):
        await write_stdin_tool({"session_id": "x"}, ctx)


async def test_chars_must_be_string(ctx) -> None:
    with pytest.raises(ValueError):
        await write_stdin_tool({"session_id": "x", "chars": 42}, ctx)
