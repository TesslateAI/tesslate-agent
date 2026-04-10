"""Tests for the ``shell_open`` / ``shell_exec`` / ``shell_close`` lifecycle."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.shell_ops.execute import shell_exec_executor
from tesslate_agent.agent.tools.shell_ops.session import (
    shell_close_executor,
    shell_open_executor,
)
from tesslate_agent.orchestration import PTY_SESSIONS


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> dict:
    """Per-test context with an isolated project root.

    Also guarantees that every PTY session created during the test is
    closed during teardown so a failing assertion cannot leak a blocking
    read() on a live PTY into asyncio loop shutdown.
    """
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    yield {"run_id": "run-session-tests"}
    for snapshot in PTY_SESSIONS.list():
        try:
            PTY_SESSIONS.close(snapshot["session_id"])
        except Exception:
            pass


async def test_open_exec_close_lifecycle(ctx) -> None:
    opened = await shell_open_executor({"command": "/bin/sh"}, ctx)
    assert opened["success"] is True
    session_id = opened["session_id"]
    assert session_id

    executed = await shell_exec_executor(
        {"session_id": session_id, "command": "echo session-hello", "wait_seconds": 0.5},
        ctx,
    )
    assert executed["success"] is True
    # ``output`` is a top-level field on the success payload, alongside
    # ``session_id`` and ``details`` — see ``output_formatter.success_output``.
    assert "session-hello" in executed["output"]
    assert executed["session_id"] == session_id
    assert executed["details"]["status"] in {"running", "exited"}

    closed = await shell_close_executor({"session_id": session_id}, ctx)
    assert closed["success"] is True

    # After closing the session must no longer be tracked.
    assert not PTY_SESSIONS.has(session_id)


async def test_exec_after_close_reports_unknown(ctx) -> None:
    opened = await shell_open_executor({"command": "/bin/sh"}, ctx)
    session_id = opened["session_id"]

    await shell_close_executor({"session_id": session_id}, ctx)

    result = await shell_exec_executor(
        {"session_id": session_id, "command": "echo late"},
        ctx,
    )
    assert result["success"] is False
    assert "Unknown shell session" in result["message"]


async def test_close_unknown_session_errors(ctx) -> None:
    result = await shell_close_executor({"session_id": "does-not-exist"}, ctx)
    assert result["success"] is False


async def test_exec_without_session_errors(ctx) -> None:
    result = await shell_exec_executor(
        {"session_id": "bogus", "command": "ls"},
        ctx,
    )
    assert result["success"] is False


async def test_shell_open_requires_non_empty_command(ctx) -> None:
    with pytest.raises(ValueError):
        await shell_open_executor({"command": ""}, ctx)


async def test_shell_exec_requires_session_id(ctx) -> None:
    with pytest.raises(ValueError):
        await shell_exec_executor({"command": "ls"}, ctx)


async def test_shell_exec_requires_command(ctx) -> None:
    with pytest.raises(ValueError):
        await shell_exec_executor({"session_id": "abc"}, ctx)


async def test_shell_close_requires_session_id(ctx) -> None:
    with pytest.raises(ValueError):
        await shell_close_executor({}, ctx)
