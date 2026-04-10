"""
Shell Session Management Tools.

Open / close persistent shell sessions backed by the local PTY
registry. A session is a long-running PTY process (``/bin/sh`` by
default) that the agent can feed commands to via ``shell_exec`` and
keep alive across multiple tool calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tesslate_agent.agent.tools.output_formatter import error_output, success_output
from tesslate_agent.agent.tools.registry import Tool, ToolCategory
from tesslate_agent.orchestration import PTY_SESSIONS

logger = logging.getLogger(__name__)


def _resolve_run_id(context: dict[str, Any]) -> str | None:
    for key in ("run_id", "chat_id", "task_id", "message_id"):
        value = context.get(key)
        if value:
            return str(value)
    return None


def _resolve_cwd(context: dict[str, Any]) -> str:
    return context.get("cwd") or os.environ.get("PROJECT_ROOT") or os.getcwd()


async def shell_open_executor(
    params: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Open a new persistent shell session via the local PTY registry."""
    command = params.get("command", "/bin/sh")
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")

    run_id = _resolve_run_id(context)
    cwd = _resolve_cwd(context)

    # Use list form for bare binaries so the PTY runs the shell directly
    # rather than wrapping with ``/bin/sh -c``.
    argv: list[str] | str = command.split() if " " in command else [command]
    try:
        session_id = PTY_SESSIONS.create(argv, cwd=cwd, run_id=run_id)
    except FileNotFoundError as exc:
        return error_output(
            message=f"Failed to spawn shell session: {exc}",
            suggestion="Verify the shell binary exists (e.g. /bin/sh or /bin/bash)",
            details={"command": command, "tier": "local"},
        )
    except OSError as exc:
        return error_output(
            message=f"Failed to spawn shell session: {exc}",
            suggestion="Check that /dev/ptmx is available and writable",
            details={"command": command, "tier": "local"},
        )

    logger.info("[SHELL-OPEN] session=%s command=%r", session_id, command)
    return success_output(
        message=f"Opened shell session {session_id}",
        session_id=session_id,
        details={"command": command, "tier": "local"},
    )


async def shell_close_executor(
    params: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Close an open persistent shell session."""
    session_id = params.get("session_id")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id parameter is required")

    if not PTY_SESSIONS.has(session_id):
        return error_output(
            message=f"Unknown shell session: {session_id}",
            suggestion="The session may have already exited or been closed",
            details={"session_id": session_id, "tier": "local"},
        )

    PTY_SESSIONS.close(session_id)
    logger.info("[SHELL-CLOSE] session=%s", session_id)
    return success_output(
        message=f"Closed shell session {session_id}",
        session_id=session_id,
        details={"tier": "local"},
    )


def register_session_tools(registry) -> None:
    """Register the ``shell_open`` and ``shell_close`` tools."""
    registry.register(
        Tool(
            name="shell_open",
            description=(
                "Open an interactive shell session in the current project "
                "directory. Returns session_id for subsequent operations. MUST "
                "be called before shell_exec. The shell remains open until "
                "explicitly closed with shell_close."
            ),
            category=ToolCategory.SHELL,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to run (default: /bin/sh). The shell "
                            "starts in the project directory with all your source files."
                        ),
                    },
                },
                "required": [],
            },
            executor=shell_open_executor,
            examples=[
                '{"tool_name": "shell_open", "parameters": {}}',
                '{"tool_name": "shell_open", "parameters": {"command": "/bin/sh"}}',
            ],
        )
    )

    registry.register(
        Tool(
            name="shell_close",
            description=(
                "Close an active shell session. Always close sessions when "
                "done to free resources."
            ),
            category=ToolCategory.SHELL,
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Shell session ID to close",
                    },
                },
                "required": ["session_id"],
            },
            executor=shell_close_executor,
            examples=['{"tool_name": "shell_close", "parameters": {"session_id": "abc123"}}'],
        )
    )

    logger.info("Registered 2 shell session management tools")
