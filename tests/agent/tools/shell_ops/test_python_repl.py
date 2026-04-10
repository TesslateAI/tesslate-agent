"""Tests for the ``python_repl`` tool."""

from __future__ import annotations

import pytest

from tesslate_agent.agent.tools.shell_ops.python_repl import (
    PYTHON_REPL_SESSIONS,
    python_repl_tool,
)


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    return {"run_id": "run-py-repl-tests"}


async def test_expression_returns_value(ctx) -> None:
    result = await python_repl_tool({"code": "2 + 2"}, ctx)
    assert result["success"] is True
    assert result["value"] == "4"
    assert result["stderr"] == ""
    assert result["timed_out"] is False


async def test_persistent_state_across_calls(ctx) -> None:
    first = await python_repl_tool({"code": "x = 41\ny = x + 1"}, ctx)
    assert first["success"] is True
    session_id = first["session_id"]

    second = await python_repl_tool(
        {"code": "y", "session_id": session_id},
        ctx,
    )
    assert second["success"] is True
    assert second["value"] == "42"


async def test_print_is_captured(ctx) -> None:
    result = await python_repl_tool({"code": "print('hi-there')"}, ctx)
    assert result["success"] is True
    assert "hi-there" in result["stdout"]


async def test_reset_clears_state(ctx) -> None:
    first = await python_repl_tool({"code": "secret = 'original'"}, ctx)
    session_id = first["session_id"]

    assert PYTHON_REPL_SESSIONS.has(session_id)

    await python_repl_tool(
        {
            "code": "secret",
            "session_id": session_id,
            "reset": True,
        },
        ctx,
    )

    # After reset, ``secret`` should be gone — looking it up should raise.
    lookup = await python_repl_tool(
        {"code": "secret", "session_id": session_id},
        ctx,
    )
    assert lookup["success"] is True
    assert "NameError" in lookup["stderr"]


async def test_exception_is_captured_in_stderr(ctx) -> None:
    result = await python_repl_tool({"code": "raise ValueError('boom')"}, ctx)
    assert result["success"] is True
    assert "ValueError" in result["stderr"]
    assert "boom" in result["stderr"]


async def test_timeout_marks_session_bad(ctx) -> None:
    result = await python_repl_tool(
        {"code": "while True:\n    pass", "timeout_ms": 200},
        ctx,
    )
    assert result["success"] is True
    assert result["timed_out"] is True
    session_id = result["session_id"]

    # Subsequent call without reset should be rejected.
    rejected = await python_repl_tool(
        {"code": "1 + 1", "session_id": session_id},
        ctx,
    )
    assert rejected["success"] is False
    assert "bad" in rejected["message"].lower()

    # With reset=true the session is reusable.
    recovered = await python_repl_tool(
        {"code": "1 + 1", "session_id": session_id, "reset": True},
        ctx,
    )
    assert recovered["success"] is True
    assert recovered["value"] == "2"


async def test_missing_code_raises(ctx) -> None:
    with pytest.raises(ValueError):
        await python_repl_tool({}, ctx)


async def test_invalid_timeout_raises(ctx) -> None:
    with pytest.raises(ValueError):
        await python_repl_tool({"code": "1", "timeout_ms": 0}, ctx)
