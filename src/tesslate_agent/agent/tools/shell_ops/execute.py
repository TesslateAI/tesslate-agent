"""
Shell Execution Tool.

Writes a command into a previously-opened persistent PTY session
(``shell_open``) and drains the response within a short window.

Retry Strategy:
- Automatically retries on transient failures (ConnectionError,
  TimeoutError, IOError).
- Exponential backoff: 1s -> 2s -> 4s (up to 3 attempts).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tesslate_agent.agent.tools.output_formatter import (
    error_output,
    strip_ansi_codes,
    success_output,
)
from tesslate_agent.agent.tools.registry import Tool, ToolCategory
from tesslate_agent.agent.tools.retry_config import tool_retry
from tesslate_agent.orchestration import PTY_SESSIONS

logger = logging.getLogger(__name__)

_BYTES_PER_TOKEN = 4

# Security Guardrail to prevent destructive shell commands
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",         # Root deletion
    r"chmod\s+.*777",        # Dangerous permission changes
    r":\(\){ :\|:& };:",     # Fork bombs
    r"mv\s+.*\s+/dev/null",  # Deleting data by moving to null
    r"> /dev/sda",           # Overwriting disk directly
]

def is_command_safe(command: str) -> tuple[bool, str | None]:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return False, f"Potentially dangerous command pattern detected: {pattern}"
    return True, None

@tool_retry
async def shell_exec_executor(
    params: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a command inside an open persistent shell session.

    Retry behavior:
    - Automatically retries on ConnectionError, TimeoutError, IOError.
    - Up to 3 attempts with exponential backoff (1s, 2s, 4s).
    """
    session_id = params.get("session_id")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id parameter is required")

    command = params.get("command")
    if command is None or not isinstance(command, str):
        raise ValueError("command parameter is required and must be a string")

    wait_seconds = float(params.get("wait_seconds", 2.0))
    max_output_tokens = int(params.get("max_output_tokens", 16384))

    if not PTY_SESSIONS.has(session_id):
        return error_output(
            message=f"Unknown shell session: {session_id}",
            suggestion=(
                "Call shell_open first to obtain a session_id, or list "
                "existing sessions with list_background_processes."
            ),
            details={"session_id": session_id, "tier": "local"},
        )

    # Add newline if not present so the shell actually runs the command.
    if not command.endswith("\n"):
        command += "\n"

    is_safe, error_msg = is_command_safe(command.strip())
    if not is_safe:
        logger.warning("[SECURITY-BLOCK] session=%s command=%s", session_id, command.strip())
        return error_output(
            message=error_msg,
            suggestion="Refine the command to target specific directories or avoid destructive flags.",
            details={"session_id": session_id, "tier": "local", "security_risk": "high"}
        )

    try:
        PTY_SESSIONS.write(session_id, command)
    except KeyError:
        return error_output(
            message=f"Unknown shell session: {session_id}",
            suggestion="Session may have exited between lookup and write",
            details={"session_id": session_id, "tier": "local"},
        )
    except (OSError, BrokenPipeError) as exc:
        return error_output(
            message=f"Failed to write to shell session {session_id}: {exc}",
            suggestion="The session may have exited — check its status",
            details={"session_id": session_id, "error": str(exc), "tier": "local"},
        )

    max_bytes_budget = max_output_tokens * _BYTES_PER_TOKEN if max_output_tokens > 0 else None

    try:
        raw = await PTY_SESSIONS.drain(
            session_id=session_id,
            max_duration_ms=max(1, int(wait_seconds * 1000)),
            idle_timeout_ms=0,
            max_bytes=max_bytes_budget,
            wait_for_exit=False,
        )
    except KeyError:
        return error_output(
            message=f"Unknown shell session: {session_id}",
            suggestion="Session was removed during drain",
            details={"session_id": session_id, "tier": "local"},
        )

    output_text = raw.decode("utf-8", errors="replace") if raw else ""
    output_text = strip_ansi_codes(output_text)

    status_snapshot = PTY_SESSIONS.status(session_id)

    logger.info(
        "[SHELL-EXEC] session=%s status=%s bytes=%d",
        session_id,
        status_snapshot["status"],
        len(output_text),
    )

    return success_output(
        message=f"Executed '{command.strip()}' in session {session_id}",
        output=output_text,
        session_id=session_id,
        details={
            "bytes": len(raw) if raw else 0,
            "status": status_snapshot["status"],
            "exit_code": status_snapshot.get("exit_code"),
            "tier": "local",
        },
    )


def register_execute_tools(registry) -> None:
    """Register the ``shell_exec`` tool on ``registry``."""
    registry.register(
        Tool(
            name="shell_exec",
            description=(
                "Execute a command in an open shell session and wait for "
                "output. REQUIRES session_id from shell_open first. DO NOT "
                "use 'exit' or close the shell - it stays open for multiple "
                "commands."
            ),
            category=ToolCategory.SHELL,
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Shell session ID obtained from shell_open",
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "Command to execute (automatically adds \\n). "
                            "DO NOT include 'exit' - the shell stays open."
                        ),
                    },
                    "wait_seconds": {
                        "type": "number",
                        "description": "Seconds to wait before reading output (default: 2)",
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "description": (
                            "Approximate output budget in model tokens "
                            "(4 bytes/token). Default: 16384."
                        ),
                        "default": 16384,
                    },
                },
                "required": ["session_id", "command"],
            },
            executor=shell_exec_executor,
            examples=[
                '{"tool_name": "shell_exec", "parameters": {"session_id": "abc123", "command": "npm install"}}',
                '{"tool_name": "shell_exec", "parameters": {"session_id": "abc123", "command": "echo \'Hello\'"}}',
            ],
        )
    )

    logger.info("Registered 1 shell execution tool")
