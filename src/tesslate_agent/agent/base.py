"""
Abstract Base Agent

Defines the core interface that every agent implementation must satisfy.
Agents are async generators that yield event dicts as they execute, so
callers can render progress live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from tesslate_agent.agent.tools.registry import ToolRegistry


class AbstractAgent(ABC):
    """
    The abstract base class every agent must implement.

    Subclasses must:

    1. Accept a system prompt and an optional :class:`ToolRegistry` in
       their constructor.
    2. Implement :meth:`run` as an async generator that yields event
       dicts describing the agent's progress.
    """

    def __init__(self, system_prompt: str, tools: ToolRegistry | None = None):
        """
        Initialise the agent.

        Args:
            system_prompt: Core instructions for the model.
            tools: Optional :class:`ToolRegistry` scoped to the tools this
                agent is allowed to call. ``None`` means the agent runs
                without tool access.
        """
        self.system_prompt = system_prompt
        self.tools = tools

    def get_processed_system_prompt(self, context: dict[str, Any]) -> str:
        """
        Return the system prompt with ``{marker}`` placeholders resolved
        from ``context``, with TESSLATE.md appended if available.

        Supported markers:
            - ``{mode}``            — edit mode (``"plan"`` / ``"ask"`` / ``"auto"``)
            - ``{project_name}``    — ``context["project_context"]["project_name"]``
            - ``{project_path}``    — standard container path (``/app``)
            - ``{timestamp}``       — current ISO timestamp
            - ``{user_name}``       — ``context["user_name"]``
            - ``{tool_list}``       — comma-separated list of tool names

        After placeholder substitution, if ``context["project_context"]["tesslate_context"]``
        is set, its content (the project's TESSLATE.md) is appended to the system prompt
        unconditionally so all agents receive project-specific documentation without
        needing an explicit ``{tesslate_context}`` placeholder in their template.
        """
        project_context = context.get("project_context") or {}
        tool_names = (
            list(self.tools._tools.keys()) if self.tools is not None else []
        )
        markers: dict[str, str] = {
            "mode": str(context.get("edit_mode", "auto")),
            "project_name": str(project_context.get("project_name", "")),
            "project_description": str(
                project_context.get("project_description", "")
            ),
            "project_path": "/app",
            "timestamp": datetime.now().isoformat(),
            "user_name": str(context.get("user_name", "")),
            "tool_list": ", ".join(tool_names),
        }
        result = self.system_prompt
        for marker, value in markers.items():
            placeholder = "{" + marker + "}"
            if placeholder in result:
                result = result.replace(placeholder, value)

        tesslate_ctx = project_context.get("tesslate_context")
        if tesslate_ctx:
            result = result + "\n\n" + tesslate_ctx

        return result

    @abstractmethod
    async def run(
        self, user_request: str, context: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Execute the agent loop for ``user_request``.

        Args:
            user_request: The user's message / prompt.
            context: Execution context. Agents pick whichever keys they
                understand and skip the rest. Common keys include
                ``user_id``, ``project_id``, ``project_slug``,
                ``container_name``, ``container_directory``,
                ``chat_history``, ``edit_mode``, and ``project_context``.

        Yields:
            Event dicts such as::

                {"type": "stream", "content": "..."}
                {"type": "agent_step", "data": {...}}
                {"type": "tool_result", "data": {...}}
                {"type": "complete", "data": {...}}
                {"type": "error", "content": "..."}
        """
        # Abstract: concrete subclasses implement the loop. The yield
        # below is only here so Python treats this as an async generator.
        yield {}
