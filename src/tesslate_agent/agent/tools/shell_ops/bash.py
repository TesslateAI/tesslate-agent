"""
Bash Convenience Tool.

Executes shell commands under a dedicated PTY session spawned by the
shared :data:`tesslate_agent.orchestration.PTY_SESSIONS` registry. Supports:

- Soft yield via ``yield_time_ms`` — returns a partial snapshot with
  ``status=running`` and the ``session_id`` if the command has not exited
  by the yield deadline.
- Hard timeout via ``timeout`` — kills the process group with SIGTERM
  (and SIGKILL if it refuses to die).
- Idle-kill via ``idle_timeout_ms`` — yields when no new output has
  arrived for the configured window.
- Background mode via ``is_background=True`` — spawns the command and
  returns immediately with ``session_id``.
- Output truncation via ``max_output_tokens`` (roughly 4 bytes/token).
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any

from tesslate_agent.agent.tools.output_formatter import (
    error_output,
    strip_ansi_codes,
    success_output,
)
from tesslate_agent.agent.tools.registry import Tool, ToolCategory
from tesslate_agent.orchestration import PTY_SESSIONS

logger = logging.getLogger(__name__)

# One token is roughly four bytes of English text; the tool truncates the
# accumulated PTY output at ``max_output_tokens * _BYTES_PER_TOKEN`` bytes.
_BYTES_PER_TOKEN = 4
_TRUNCATION_MARKER = "\n[truncated]\n"


def _resolve_run_id(context: dict[str, Any]) -> str | None:
    """Extract the invocation identifier from the tool-call context."""
    for key in ("run_id", "chat_id", "task_id", "message_id"):
        value = context.get(key)
        if value:
            return str(value)
    return None


def _truncate_output(text: str, max_output_tokens: int) -> tuple[str, bool]:
    """Truncate ``text`` to at most ``max_output_tokens * 4`` bytes."""
    if max_output_tokens <= 0:
        return text, False
    budget = max_output_tokens * _BYTES_PER_TOKEN
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= budget:
        return text, False
    tail = encoded[-budget:]
    try:
        decoded = tail.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        decoded = tail.decode("latin-1", errors="replace")
    return _TRUNCATION_MARKER + decoded, True


def _resolve_cwd(context: dict[str, Any], params_cwd: str | None) -> str:
    """Resolve a working directory for the spawned process.

    ``params_cwd`` — when provided — is treated as relative to the
    project root (``context["cwd"]`` or ``$PROJECT_ROOT`` or the current
    process ``cwd``). Absolute paths are rejected silently by joining
    under the root.
    """
    base = context.get("cwd") or os.environ.get("PROJECT_ROOT") or os.getcwd()
    if not params_cwd:
        return base
    candidate = os.path.join(base, params_cwd) if not os.path.isabs(params_cwd) else params_cwd
    return candidate


async def _run_local_pty(
    context: dict[str, Any],
    command: str,
    timeout: int,
    yield_time_ms: int,
    max_output_tokens: int,
    is_background: bool,
    idle_timeout_ms: int,
    cwd: str | None,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    """Execute ``command`` in local mode under a dedicated PTY session."""
    run_id = _resolve_run_id(context)
    resolved_cwd = _resolve_cwd(context, cwd)

    try:
        session_id = PTY_SESSIONS.create(
            command,
            cwd=resolved_cwd,
            env=env,
            run_id=run_id,
        )
    except FileNotFoundError as exc:
        return error_output(
            message=f"Failed to spawn PTY session: {exc}",
            suggestion="Verify the shell is installed and accessible",
            details={"command": command, "tier": "local"},
        )
    except OSError as exc:
        return error_output(
            message=f"Failed to spawn PTY session: {exc}",
            suggestion="Check that /dev/ptmx is available and writable",
            details={"command": command, "tier": "local"},
        )

    snapshot = PTY_SESSIONS.status(session_id)

    if is_background:
        logger.info(
            "[BASH-LOCAL] Background PTY session %s spawned pid=%s cmd=%r",
            session_id,
            snapshot.get("pid"),
            command,
        )
        return success_output(
            message=f"Started background PTY session {session_id}",
            session_id=session_id,
            details={
                "command": command,
                "pid": snapshot.get("pid"),
                "status": "running",
                "tier": "local",
                "is_background": True,
            },
        )

    # Foreground: drain until exit / timeout / yield / idle.
    hard_deadline_ms = max(1, int(timeout)) * 1000
    yield_ms = max(0, int(yield_time_ms))
    max_duration_ms = hard_deadline_ms if yield_ms == 0 else min(hard_deadline_ms, yield_ms)

    output_bytes = bytearray()
    start = time.monotonic()
    truncated = False

    max_bytes_budget = max_output_tokens * _BYTES_PER_TOKEN if max_output_tokens > 0 else None

    try:
        while True:
            remaining_ms = hard_deadline_ms - int((time.monotonic() - start) * 1000)
            if remaining_ms <= 0:
                # Hard timeout — kill the process group.
                logger.warning(
                    "[BASH-LOCAL] Command timed out after %ss: %s",
                    timeout,
                    command[:100],
                )
                entry_pgid: int | None = None
                try:
                    entry_pgid = PTY_SESSIONS._sessions[session_id].get("pgid")  # noqa: SLF001
                except KeyError:
                    entry_pgid = None
                if entry_pgid:
                    try:
                        os.killpg(entry_pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    import asyncio as _asyncio

                    await _asyncio.sleep(2.0)
                    try:
                        if PTY_SESSIONS._sessions[session_id]["pty"].isalive():  # noqa: SLF001
                            os.killpg(entry_pgid, signal.SIGKILL)
                    except (KeyError, ProcessLookupError, PermissionError, OSError):
                        pass

                # Capture whatever is left, then close.
                tail = PTY_SESSIONS.read(session_id, max_bytes=65536)
                if tail:
                    output_bytes.extend(tail)
                PTY_SESSIONS.close(session_id)

                text = output_bytes.decode("utf-8", errors="replace")
                clean = strip_ansi_codes(text)
                clean, trunc_now = _truncate_output(clean, max_output_tokens)
                return error_output(
                    message=f"Command timed out after {timeout}s: {command}",
                    suggestion=(
                        "Increase the timeout, use is_background=True, or split the "
                        "command into smaller steps"
                    ),
                    details={
                        "command": command,
                        "timeout": timeout,
                        "output": clean,
                        "truncated": truncated or trunc_now,
                        "exit_code": 124,
                        "session_id": session_id,
                        "tier": "local",
                    },
                )

            drain_budget_ms = min(max_duration_ms, remaining_ms)
            chunk = await PTY_SESSIONS.drain(
                session_id=session_id,
                max_duration_ms=drain_budget_ms,
                idle_timeout_ms=idle_timeout_ms,
                max_bytes=max_bytes_budget,
                wait_for_exit=True,
            )
            if chunk:
                output_bytes.extend(chunk)

            status_snapshot = PTY_SESSIONS.status(session_id)
            if status_snapshot["status"] == "exited":
                # Final flush.
                tail = PTY_SESSIONS.read(session_id, max_bytes=65536)
                if tail:
                    output_bytes.extend(tail)
                exit_code = status_snapshot.get("exit_code")
                PTY_SESSIONS.close(session_id)

                text = output_bytes.decode("utf-8", errors="replace")
                clean = strip_ansi_codes(text)
                clean, trunc_now = _truncate_output(clean, max_output_tokens)
                truncated = truncated or trunc_now

                logger.info(
                    "[BASH-LOCAL] Command completed exit=%s output_length=%d",
                    exit_code,
                    len(clean),
                )

                details = {
                    "command": command,
                    "exit_code": exit_code if exit_code is not None else 0,
                    "output": clean,
                    "status": "exited",
                    "truncated": truncated,
                    "session_id": session_id,
                    "tier": "local",
                }

                if exit_code not in (None, 0):
                    return error_output(
                        message=f"Command failed (exit code {exit_code}): {command}",
                        suggestion="Check the output for errors",
                        details=details,
                    )
                return success_output(
                    message=f"Executed '{command}'",
                    output=clean,
                    details=details,
                )

            if max_bytes_budget is not None and len(output_bytes) >= max_bytes_budget:
                # Budget hit — mark truncated, kill the process, return early.
                truncated = True
                entry_pgid = None
                try:
                    entry_pgid = PTY_SESSIONS._sessions[session_id].get("pgid")  # noqa: SLF001
                except KeyError:
                    entry_pgid = None
                if entry_pgid:
                    try:
                        os.killpg(entry_pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                PTY_SESSIONS.close(session_id)

                text = output_bytes.decode("utf-8", errors="replace")
                clean = strip_ansi_codes(text)
                clean, _ = _truncate_output(clean, max_output_tokens)
                return success_output(
                    message=f"Executed '{command}' (output truncated at budget)",
                    output=clean,
                    details={
                        "command": command,
                        "exit_code": None,
                        "status": "truncated",
                        "truncated": True,
                        "session_id": session_id,
                        "tier": "local",
                    },
                )

            # If the yield window has elapsed without an exit, return a
            # partial snapshot so the agent can decide whether to continue.
            if yield_ms > 0 and int((time.monotonic() - start) * 1000) >= yield_ms:
                text = output_bytes.decode("utf-8", errors="replace")
                clean = strip_ansi_codes(text)
                clean, trunc_now = _truncate_output(clean, max_output_tokens)
                truncated = truncated or trunc_now
                logger.info(
                    "[BASH-LOCAL] Yield after %dms, session %s still running",
                    yield_ms,
                    session_id,
                )
                return success_output(
                    message=f"Yielded after {yield_ms}ms; session {session_id} still running",
                    output=clean,
                    details={
                        "command": command,
                        "exit_code": None,
                        "status": "running",
                        "truncated": truncated,
                        "session_id": session_id,
                        "tier": "local",
                    },
                )
    except Exception as exc:
        logger.error("[BASH-LOCAL] Execution error for %r: %s", command, exc, exc_info=True)
        try:
            PTY_SESSIONS.close(session_id)
        except Exception:
            pass
        return error_output(
            message=f"Command execution failed: {exc}",
            suggestion="Inspect the traceback and retry",
            details={"command": command, "error": str(exc), "tier": "local"},
        )


async def bash_exec_tool(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single shell command under a PTY session.

    Args:
        params: {
            command: str,
            cwd: str | None,
            timeout: int = 120,
            timeout_ms: int | None,
            yield_time_ms: int = 10000,
            max_output_tokens: int = 16384,
            env: dict[str, str] | None,
            is_background: bool = False,
            idle_timeout_ms: int = 0,
        }
        context: {run_id?, chat_id?, cwd?, ...}
    """
    command = params.get("command")
    if not command:
        raise ValueError("command parameter is required")

    # Accept either ``timeout`` (seconds) or ``timeout_ms`` (milliseconds).
    if "timeout_ms" in params and params["timeout_ms"] is not None:
        timeout = max(1, int(params["timeout_ms"]) // 1000)
    else:
        timeout = int(params.get("timeout", 120))

    yield_time_ms = int(params.get("yield_time_ms", 10000))
    max_output_tokens = int(params.get("max_output_tokens", 16384))
    is_background = bool(params.get("is_background", False))
    idle_timeout_ms = int(params.get("idle_timeout_ms", 0))
    cwd = params.get("cwd")
    env = params.get("env")
    if env is not None and not isinstance(env, dict):
        raise ValueError("env must be a mapping of str -> str")

    logger.info("[BASH] Executing: %s... (bg=%s)", command[:100], is_background)

    return await _run_local_pty(
        context=context,
        command=command,
        timeout=timeout,
        yield_time_ms=yield_time_ms,
        max_output_tokens=max_output_tokens,
        is_background=is_background,
        idle_timeout_ms=idle_timeout_ms,
        cwd=cwd,
        env=env,
    )


def register_bash_tools(registry) -> None:
    """Register the ``bash_exec`` tool on ``registry``."""
    registry.register(
        Tool(
            name="bash_exec",
            description=(
                "Execute a bash/sh command under a PTY session and return its "
                "output. Supports soft yielding via yield_time_ms, idle "
                "detection via idle_timeout_ms, background spawning via "
                "is_background=True, and output truncation via max_output_tokens."
            ),
            category=ToolCategory.SHELL,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Command to execute (e.g., 'npm install', 'ls -la', "
                            "'cat package.json')."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Working directory relative to the project root. "
                            "Defaults to the project root itself."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Hard timeout in seconds — the process group is killed "
                            "when it elapses (default: 120)."
                        ),
                        "default": 120,
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": (
                            "Alternative hard timeout expressed in milliseconds. "
                            "When provided, overrides ``timeout``."
                        ),
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "description": (
                            "Soft yield window in milliseconds. If the command is "
                            "still running after this window elapses, bash_exec "
                            "returns a partial snapshot with status=running and the "
                            "session_id so the agent can poll or send stdin. 0 "
                            "disables soft yield. Default: 10000."
                        ),
                        "default": 10000,
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "description": (
                            "Approximate output budget in model tokens (4 "
                            "bytes/token). Output beyond this is truncated with a "
                            "[truncated] marker. Default: 16384."
                        ),
                        "default": 16384,
                    },
                    "env": {
                        "type": "object",
                        "description": (
                            "Optional environment variable overrides applied on "
                            "top of the current process environment."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "is_background": {
                        "type": "boolean",
                        "description": (
                            "When true, spawn the command as a detached PTY "
                            "session and return immediately with the session_id. "
                            "Use list_background_processes and read_background_output "
                            "to inspect it later."
                        ),
                        "default": False,
                    },
                    "idle_timeout_ms": {
                        "type": "integer",
                        "description": (
                            "Idle output timeout in milliseconds. When >0 and no "
                            "new output arrives for this long, bash_exec yields a "
                            "partial snapshot. 0 disables the idle timeout. Default: 0."
                        ),
                        "default": 0,
                    },
                },
                "required": ["command"],
            },
            executor=bash_exec_tool,
            examples=[
                '{"tool_name": "bash_exec", "parameters": {"command": "npm install"}}',
                '{"tool_name": "bash_exec", "parameters": {"command": "ls -la", "timeout": 30}}',
                '{"tool_name": "bash_exec", "parameters": {"command": "npm run dev", "is_background": true}}',
                '{"tool_name": "bash_exec", "parameters": {"command": "pytest -x", "yield_time_ms": 5000, "idle_timeout_ms": 2000}}',
            ],
        )
    )

    logger.info("Registered 1 bash convenience tool")
