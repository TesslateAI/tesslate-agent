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

    # ------------------------------------------------------------------
    # System-wide guardrails
    # ------------------------------------------------------------------
    # These blocks are prepended to EVERY agent's system prompt, before
    # the agent's own template. They define non-negotiable behaviour the
    # platform itself cares about — primarily, tool-result truthfulness.
    #
    # If you find yourself adding a per-agent system-prompt clause to fix
    # a fabrication / hallucination problem, add it here instead — that
    # way all agents benefit and the fix can't be lost when a marketplace
    # agent is forked.
    _TOOL_TRUTHFULNESS_CONTRACT = (
        "=== TOOL RESULT TRUTHFULNESS (non-negotiable) ===\n"
        "Tool results are the ONLY source of truth for facts about the\n"
        "user's data, environment, and code. Hold the following rules\n"
        "absolutely:\n"
        "\n"
        "1. NEVER invent record IDs, emails, names, file paths,\n"
        "   timestamps, counts, status values, URLs, or any other concrete\n"
        "   datum that did not appear in a tool result. If a tool returned\n"
        "   no records or no value for a field, say exactly that — do not\n"
        "   illustrate with a 'plausible example'.\n"
        "\n"
        "2. When reporting on tool output, quote primitives (IDs, counts,\n"
        "   values, fields) VERBATIM as the tool returned them. Do not\n"
        "   round, paraphrase, reorder, or 'tidy up' numbers.\n"
        "\n"
        "3. If you cannot find the data you need in a tool result, call\n"
        "   another tool to get it. Do not fall back to your own memory\n"
        "   of similar projects or pre-training data.\n"
        "\n"
        "4. If a tool returned a result you did not expect (or returned\n"
        "   nothing), report that honestly: \"the tool returned 0 records\",\n"
        "   \"the field is absent in every sampled record\", etc. Surprising\n"
        "   results are signal, not something to paper over.\n"
        "\n"
        "5. When summarising bounded analysis (e.g. ``aggregate`` with\n"
        "   ``sample_size``), state the bound: \"based on the most recent\n"
        "   N of total M records\". Never present a sampled result as exact.\n"
        "\n"
        "6. READ THE WHOLE TOOL RESULT. Tool results are JSON, and most\n"
        "   tools return a thin ``{success, message, ...}`` wrapper PLUS\n"
        "   the payload data alongside it (e.g. ``records: [...]``,\n"
        "   ``top_values: [...]``, ``fields: {...}``). Do NOT claim a\n"
        "   tool 'only returned a success message' when the same JSON\n"
        "   object also contains the data you asked for. If you can't\n"
        "   find a key, scan the entire result before deciding it's\n"
        "   absent — and certainly before escalating to a different tool\n"
        "   that bypasses the documented one.\n"
        "\n"
        "7. Do NOT bypass a structured tool with raw shell / filesystem\n"
        "   access when the structured tool already returned the answer.\n"
        "   ``workspace_data query``, ``list_collections``, ``summarize``,\n"
        "   etc. ARE the canonical way to read the data store — there is\n"
        "   no underlying file to ``grep`` in lieu of them.\n"
        "\n"
        "8. ONLY call tools that actually exist in your tool registry — the\n"
        "   tools provided to you THIS run are the complete set. NEVER invent\n"
        "   a tool name or call one you merely assume exists (e.g. there is\n"
        "   no ``print_tree`` / ``tree`` tool — use ``list_dir`` to inspect a\n"
        "   directory). If no tool offers a capability you need, say so\n"
        "   plainly rather than calling a non-existent tool.\n"
        "\n"
        "Violating any of these rules is a session-ending failure: the\n"
        "user can no longer trust any later claim you make. When in\n"
        "doubt, run the tool again rather than guess.\n"
    )

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

        The platform-wide :attr:`_TOOL_TRUTHFULNESS_CONTRACT` block is
        prepended to every agent's prompt so per-agent or per-skill drift
        cannot remove it.
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
        # Prepend the platform-wide tool-truthfulness contract. Goes FIRST
        # so it survives any per-agent prompt that re-defines tone or
        # output style afterward — model attention to early tokens is
        # higher, and the contract is the floor, not the ceiling.
        result = self._TOOL_TRUTHFULNESS_CONTRACT + "\n" + self.system_prompt
        for marker, value in markers.items():
            placeholder = "{" + marker + "}"
            if placeholder in result:
                result = result.replace(placeholder, value)

        # Append-keys: every entry in ``project_context`` that names a
        # pre-formatted text block ends up concatenated onto the system
        # prompt in the order listed. Each value is appended only if it's
        # a non-empty string — None / "" / non-strings are skipped so a
        # caller that stores something else under one of these keys never
        # corrupts the prompt. Keep this list deterministic (alpha-by-purpose,
        # not alpha-by-name) so a reader can trace what the agent saw.
        #
        # tesslate_context  — the project's TESSLATE.md (long-form docs)
        # data_overview     — passive workspace-data store discovery block
        # data_focus        — per-mention deep-dive (e.g. @data:subs)
        for key in ("tesslate_context", "data_overview", "data_focus"):
            block = project_context.get(key)
            if isinstance(block, str) and block.strip():
                result = result + "\n\n" + block

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
