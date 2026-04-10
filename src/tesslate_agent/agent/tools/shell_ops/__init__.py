"""
Shell Operations Module.

Essential shell execution tools for AI agents backed by the local PTY
session registry. Supports one-off commands (``bash_exec``), persistent
shell sessions (``shell_open`` / ``shell_exec`` / ``shell_close``),
writing into a running session (``write_stdin``), background process
inspection, and a persistent Python REPL.
"""

from __future__ import annotations

from .background import register_background_tools
from .bash import register_bash_tools
from .execute import register_execute_tools
from .python_repl import register_python_repl_tool
from .session import register_session_tools
from .write_stdin import register_write_stdin_tool


def register_shell_ops_tools(registry) -> None:
    """Register every shell-ops tool on ``registry``."""
    register_bash_tools(registry)         # bash_exec
    register_session_tools(registry)      # shell_open, shell_close
    register_execute_tools(registry)      # shell_exec
    register_write_stdin_tool(registry)   # write_stdin
    register_background_tools(registry)   # list_background_processes, read_background_output
    register_python_repl_tool(registry)   # python_repl


__all__ = [
    "register_shell_ops_tools",
    "register_bash_tools",
    "register_session_tools",
    "register_execute_tools",
    "register_write_stdin_tool",
    "register_background_tools",
    "register_python_repl_tool",
]
