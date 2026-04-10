"""Tests for the ``bash_exec`` tool."""

from __future__ import annotations

import asyncio

import pytest

from tesslate_agent.agent.tools.shell_ops.bash import bash_exec_tool


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    return {"run_id": "run-bash-tests"}


async def test_echo_returns_output(ctx) -> None:
    result = await bash_exec_tool({"command": "echo hello"}, ctx)
    assert result["success"] is True
    assert "hello" in result["details"]["output"]
    assert result["details"]["exit_code"] == 0
    assert result["details"]["status"] == "exited"


async def test_nonzero_exit_is_reported(ctx) -> None:
    result = await bash_exec_tool({"command": "exit 7"}, ctx)
    assert result["success"] is False
    assert result["details"]["exit_code"] == 7


async def test_hard_timeout_kills_and_marks_timed_out(ctx) -> None:
    result = await bash_exec_tool(
        {"command": "sleep 10", "timeout": 1, "yield_time_ms": 0},
        ctx,
    )
    assert result["success"] is False
    assert result["details"]["exit_code"] == 124


async def test_ansi_output_is_stripped(ctx) -> None:
    # \033[31mRED\033[0m — ANSI red sequence, should be stripped.
    result = await bash_exec_tool(
        {"command": "printf '\\033[31mRED\\033[0m\\n'"},
        ctx,
    )
    assert result["success"] is True
    output = result["details"]["output"]
    assert "RED" in output
    assert "\x1b[" not in output


async def test_output_truncation(ctx) -> None:
    # Produce ~12KB of output but set a tiny budget (1 token = 4 bytes).
    result = await bash_exec_tool(
        {
            "command": "printf 'x%.0s' $(seq 1 12000)",
            "max_output_tokens": 4,  # 16-byte budget
            "yield_time_ms": 0,
        },
        ctx,
    )
    assert result["success"] is True
    assert result["details"]["truncated"] is True
    # Budget should keep the output well below the original size.
    assert len(result["details"]["output"]) < 200


async def test_background_mode_returns_immediately(ctx) -> None:
    result = await bash_exec_tool(
        {"command": "sleep 2", "is_background": True},
        ctx,
    )
    assert result["success"] is True
    assert result["details"]["status"] == "running"
    assert result["details"]["is_background"] is True
    assert "session_id" in result

    # Give the session time to finish so it doesn't leak across tests.
    await asyncio.sleep(0.1)
    from tesslate_agent.orchestration import PTY_SESSIONS

    PTY_SESSIONS.close(result["session_id"])


async def test_yield_returns_running_snapshot(ctx) -> None:
    result = await bash_exec_tool(
        {"command": "sleep 3", "timeout": 10, "yield_time_ms": 200},
        ctx,
    )
    assert result["success"] is True
    # Yielded before sleep completes — status should be running.
    assert result["details"]["status"] == "running"
    assert result["details"]["exit_code"] is None
    session_id = result["details"]["session_id"]

    # Clean up the still-running session.
    from tesslate_agent.orchestration import PTY_SESSIONS

    PTY_SESSIONS.close(session_id)


async def test_missing_command_raises(ctx) -> None:
    with pytest.raises(ValueError, match="command parameter is required"):
        await bash_exec_tool({}, ctx)
