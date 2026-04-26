"""
Tool registry.

Manages the set of tools an agent can invoke and handles tool execution.
Each tool is defined with name, description, JSON-schema parameters, and
an async executor function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCategory(StrEnum):
    """Categories used to group tools in prompts and in the UI."""

    FILE_OPS = "file_operations"
    SHELL = "shell_commands"
    PROJECT = "project_management"
    BUILD = "build_operations"
    WEB = "web_operations"
    NAV_OPS = "navigation_operations"
    MEMORY_OPS = "memory_operations"
    GIT_OPS = "git_operations"
    DELEGATION_OPS = "delegation_operations"
    PLANNING = "planning_operations"
    VIEW_GRAPH = "graph_view_tools"


@dataclass
class Tool:
    """
    A tool the agent can call.

    Attributes:
        name: Unique tool identifier.
        description: What the tool does (shown to the LLM).
        parameters: JSON schema describing the tool's parameters.
        executor: Async function implementing the tool.
        category: Tool category for grouping.
        examples: Optional example invocations shown in the prompt.
        system_prompt: Optional extra system-prompt instructions.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable
    category: ToolCategory
    examples: list[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_prompt_format(self) -> str:
        """Render the tool as a block suitable for inclusion in a system prompt."""
        param_descriptions: list[str] = []
        for param_name, param_info in self.parameters.get("properties", {}).items():
            required = param_name in self.parameters.get("required", [])
            req_str = "required" if required else "optional"
            param_type = param_info.get("type", "string")
            desc = param_info.get("description", "")
            param_descriptions.append(
                f"  - {param_name} ({param_type}, {req_str}): {desc}"
            )

        params_text = (
            "\n".join(param_descriptions) if param_descriptions else "  No parameters"
        )

        examples_text = ""
        if self.examples:
            examples_text = "\n  Examples:\n    " + "\n    ".join(self.examples)

        system_prompt_text = ""
        if self.system_prompt:
            system_prompt_text = f"\n  Instructions: {self.system_prompt}"

        return (
            f"{self.name}: {self.description}\n"
            f"  Parameters:\n"
            f"{params_text}"
            f"{examples_text}"
            f"{system_prompt_text}"
        ).strip()


class ToolRegistry:
    """
    Registry of tools available to the agent.

    Manages tool registration, lookup, listing, and execution. Execution
    applies optional API-key scope checks and edit-mode gating (plan / ask
    / auto) before calling the tool's executor.
    """

    # Mapping from tool names to required permission scope values.
    # Tools not listed here are unrestricted (e.g. read_file, todo_write).
    TOOL_REQUIRED_SCOPES: dict[str, str] = {
        # File write operations
        "write_file": "file.write",
        "patch_file": "file.write",
        "multi_edit": "file.write",
        "apply_patch": "file.write",
        # File delete is separate
        "delete_file": "file.delete",
        # Shell / terminal operations
        "bash_exec": "terminal.access",
        "shell_exec": "terminal.access",
        "shell_open": "terminal.access",
        "shell_close": "terminal.access",
        # Web operations
        "web_fetch": "file.read",
        "web_search": "file.read",
        # Messaging
        "send_message": "channel.manage",
        # Container control
        "container_status": "container.view",
        "container_restart": "container.start_stop",
        "container_logs": "container.view",
        "container_health": "container.view",
        # Kanban
        "kanban_create": "kanban.edit",
        "kanban_move": "kanban.edit",
        "kanban_update": "kanban.edit",
        "kanban_comment": "kanban.edit",
    }

    # Tools that mutate state or reach out to the network — require
    # approval in ``ask`` mode and are blocked entirely in ``plan`` mode.
    DANGEROUS_TOOLS: frozenset[str] = frozenset(
        {
            "write_file",
            "patch_file",
            "multi_edit",
            "apply_patch",
            "bash_exec",
            "shell_exec",
            "shell_open",
            "web_fetch",
            "web_search",
            "send_message",
        }
    )

    # Tools that are allowed in ``plan`` mode (read-only context gathering).
    PLAN_MODE_ALLOWED: frozenset[str] = frozenset({"bash_exec"})

    def __init__(self, approval_handler: Any = None) -> None:
        self._tools: dict[str, Tool] = {}
        # Optional async callable: (tool_name, params, session_id) -> str.
        # When set, replaces the env-var ApprovalManager for ask-mode gating so
        # the orchestrator can inject its own interactive approval flow (Redis
        # pub/sub + frontend dialog) without the submodule knowing about it.
        self._approval_handler = approval_handler
        logger.info("ToolRegistry initialized")

    def register(self, tool: Tool) -> None:
        """Register ``tool`` in this registry, replacing any existing entry."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.info(
            "Registered tool: %s (category: %s)", tool.name, tool.category.value
        )

    def get(self, name: str) -> Tool | None:
        """Return the tool registered under ``name``, or ``None``."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return every registered tool."""
        return list(self._tools.values())

    def list_tools(self, category: ToolCategory | None = None) -> list[Tool]:
        """
        Return registered tools, optionally filtered by ``category``.
        """
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return the names of every registered tool."""
        return list(self._tools.keys())

    def get_system_prompt_section(self) -> str:
        """Render every registered tool as a grouped prompt section."""
        sections: list[str] = []

        for category in ToolCategory:
            tools = self.list_tools(category)
            if tools:
                sections.append(f"\n## {category.value.replace('_', ' ').title()}\n")
                for i, tool in enumerate(tools, 1):
                    sections.append(f"{i}. {tool.to_prompt_format()}\n")

        return "\n".join(sections)

    def _check_tool_scope(self, tool_name: str, scopes: list[str]) -> str | None:
        """
        Enforce API-key scopes for ``tool_name``.

        Returns an error message when the required scope is missing, or
        ``None`` when the scope check passes (or the tool is unrestricted).
        """
        required = self.TOOL_REQUIRED_SCOPES.get(tool_name)
        if required is None:
            return None
        if required in scopes:
            return None
        return (
            f"API key scope restriction: '{tool_name}' requires the '{required}' "
            f"permission, but this key only has: {scopes}"
        )

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute ``tool_name`` with ``parameters`` and ``context``.

        Context keys respected:
            - ``api_key_scopes``: optional list[str] for scope enforcement.
            - ``edit_mode``: one of ``"plan"`` / ``"ask"`` / ``"auto"``
              (default: ``"ask"``).
            - ``skip_approval_check``: when ``True``, skip the approval gate
              even in ``ask`` mode.
            - ``chat_id``: session identifier for approval memory.

        Returns:
            A dict with ``success`` (bool), ``tool`` (name), and either
            ``result`` (on success) or ``error`` (on failure). When the
            tool needs approval, returns ``{"approval_required": True, ...}``.
        """
        tool = self.get(tool_name)

        if not tool:
            logger.error("Unknown tool: %s", tool_name)
            return {
                "success": False,
                "error": (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools.keys())}"
                ),
            }

        # API-key scope enforcement.
        api_key_scopes = context.get("api_key_scopes")
        if api_key_scopes is not None:
            scope_result = self._check_tool_scope(tool_name, api_key_scopes)
            if scope_result is not None:
                logger.warning("[SCOPE] Blocked tool %s: %s", tool_name, scope_result)
                return {
                    "success": False,
                    "tool": tool_name,
                    "error": scope_result,
                }

        # Edit-mode control — applies to every agent.
        edit_mode = context.get("edit_mode", "ask")
        is_dangerous = tool_name in self.DANGEROUS_TOOLS

        if (
            edit_mode == "plan"
            and is_dangerous
            and tool_name not in self.PLAN_MODE_ALLOWED
        ):
            logger.warning("[PLAN MODE] Blocked tool execution: %s", tool_name)
            return {
                "success": False,
                "tool": tool_name,
                "error": (
                    f"Plan mode active - {tool_name} is disabled. You can only "
                    f"read files, run shell commands, and gather information. "
                    f"Explain what changes you would make instead."
                ),
            }

        skip_approval = context.get("skip_approval_check", False)
        if edit_mode == "ask" and is_dangerous and not skip_approval:
            session_id = context.get("chat_id", "default")

            if self._approval_handler is not None:
                # Delegate to the injected handler (e.g. orchestrator's
                # PendingUserInputManager which suspends until the user responds).
                logger.info(
                    "[ASK MODE] Approval required for %s in session %s",
                    tool_name,
                    session_id,
                )
                response = await self._approval_handler(tool_name, parameters, session_id)
                if response not in ("allow_once", "allow_all"):
                    return {
                        "approval_required": True,
                        "tool": tool_name,
                        "parameters": parameters,
                        "session_id": session_id,
                        "approval_id": None,
                        "response": response or "stop",
                    }
            else:
                # Fall back to the env-var-based manager (CLI / tests).
                from .approval_manager import get_approval_manager

                approval_mgr = get_approval_manager()

                if not approval_mgr.is_tool_approved(session_id, tool_name):
                    logger.info(
                        "[ASK MODE] Approval required for %s in session %s",
                        tool_name,
                        session_id,
                    )
                    approval_id, request = approval_mgr.request_approval(
                        tool_name, parameters, session_id
                    )
                    if request.response != "allow_once" and request.response != "allow_all":
                        return {
                            "approval_required": True,
                            "tool": tool_name,
                            "parameters": parameters,
                            "session_id": session_id,
                            "approval_id": approval_id,
                            "response": request.response,
                        }
                else:
                    logger.info(
                        "[ASK MODE] Tool %s already approved for session %s",
                        tool_name,
                        session_id,
                    )

        try:
            logger.info(
                "[TOOL-EXEC] Starting tool: %s with params: %s [edit_mode=%s]",
                tool_name,
                parameters,
                edit_mode,
            )

            result = await tool.executor(parameters, context)

            tool_succeeded = (
                result.get("success", True) if isinstance(result, dict) else True
            )

            if tool_succeeded:
                logger.info("[TOOL-EXEC] Completed tool: %s, success=True", tool_name)
            else:
                logger.warning(
                    "[TOOL-EXEC] Completed tool: %s, success=False, error: %s",
                    tool_name,
                    result.get("message", "Unknown error")
                    if isinstance(result, dict)
                    else "Unknown error",
                )

            return {"success": tool_succeeded, "tool": tool_name, "result": result}

        except Exception as exc:
            logger.error(
                "[TOOL-EXEC] Tool %s execution FAILED with exception: %s",
                tool_name,
                exc,
                exc_info=True,
            )
            return {"success": False, "tool": tool_name, "error": str(exc)}


# Global registry instance.
_registry: ToolRegistry | None = None


def register_all_tools(registry: ToolRegistry) -> None:
    """
    Populate ``registry`` with every built-in tool.

    Registration order mirrors the category layout used throughout the
    package: filesystem primitives first, then shell execution, then
    navigation / search, then git, memory, planning, web, and finally
    subagent delegation. Imports are kept inside the function to avoid
    circular-import risk at package import time.
    """
    from tesslate_agent.agent.tools.file_ops import register_file_ops_tools
    from tesslate_agent.agent.tools.shell_ops import register_shell_ops_tools
    from tesslate_agent.agent.tools.nav_ops import register_nav_ops_tools
    from tesslate_agent.agent.tools.git_ops import register_git_ops_tools
    from tesslate_agent.agent.tools.memory_ops import register_memory_ops_tools
    from tesslate_agent.agent.tools.planning_ops import register_planning_ops_tools
    from tesslate_agent.agent.tools.web_ops import register_web_ops_tools
    from tesslate_agent.agent.tools.delegation_ops import register_delegation_ops_tools

    register_file_ops_tools(registry)
    register_shell_ops_tools(registry)
    register_nav_ops_tools(registry)
    register_git_ops_tools(registry)
    register_memory_ops_tools(registry)
    register_planning_ops_tools(registry)
    register_web_ops_tools(registry)
    register_delegation_ops_tools(registry)


def get_tool_registry() -> ToolRegistry:
    """
    Return the process-wide :class:`ToolRegistry` singleton.

    On first call the registry is constructed and populated with every
    built-in tool via :func:`register_all_tools`. Subsequent calls
    return the cached instance.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        register_all_tools(_registry)
    return _registry


def reset_tool_registry() -> None:
    """
    Drop the cached tool registry singleton.

    Intended for tests that want a fresh registry after mutating or
    monkey-patching one of the ``register_*`` helpers.
    """
    global _registry
    _registry = None


def create_scoped_tool_registry(
    tool_names: list[str],
    tool_configs: dict[str, dict[str, Any]] | None = None,
) -> ToolRegistry:
    """
    Build a :class:`ToolRegistry` containing only ``tool_names``.

    Useful for giving agents restricted tool access with optional per-tool
    description / example overrides. Tools missing from the global registry
    are skipped with a warning.

    Args:
        tool_names: Names of tools to include.
        tool_configs: Optional mapping of tool name to override dict with
            any of ``description``, ``examples``, ``system_prompt``.

    Returns:
        A new :class:`ToolRegistry` populated with the requested tools.
    """
    scoped_registry = ToolRegistry()
    global_registry = get_tool_registry()
    tool_configs = tool_configs or {}

    missing_tools: list[str] = []
    for name in tool_names:
        tool = global_registry.get(name)
        if tool is None:
            missing_tools.append(name)
            logger.warning("Tool '%s' not found in global registry", name)
            continue

        if name in tool_configs:
            config = tool_configs[name]
            custom_tool = replace(
                tool,
                description=config.get("description", tool.description),
                examples=config.get("examples", tool.examples),
                system_prompt=config.get("system_prompt", tool.system_prompt),
            )
            scoped_registry.register(custom_tool)
            logger.info("Registered tool '%s' with custom configuration", name)
        else:
            scoped_registry.register(tool)

    if missing_tools:
        logger.warning(
            "Could not add %d tools to scoped registry: %s",
            len(missing_tools),
            missing_tools,
        )

    logger.info(
        "Created scoped tool registry with %d tools: %s",
        len(scoped_registry.list_names()),
        scoped_registry.list_names(),
    )

    return scoped_registry
