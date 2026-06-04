"""Canonical, importable definitions of the built-in Tesslate Agent.

This package is the single source of truth for the Tesslate Agent's system
prompt and core runtime configuration. Downstream consumers import from
here instead of copying the prompt:

    from tesslate_agent.prompts import (
        TESSLATE_AGENT_SYSTEM_PROMPT,
        TESSLATE_AGENT_DEFINITION,
    )

OpenSail's code-resident "System Default Agent" is the Tesslate Agent baked
into the platform — same engine, same prompt, same tools. It builds its
catalog row from :data:`TESSLATE_AGENT_DEFINITION` so the two can never
drift, and so the prompt lives in this repo (which owns the agent runtime)
rather than being duplicated in every host.

The prompt body is a native Python string in :mod:`._tesslate_agent_prompt`
(no data files to bundle, so it imports identically from source or wheel).

The ``{tool_list}`` marker in the prompt is resolved per-run by
``AbstractAgent.get_processed_system_prompt`` from the live tool registry,
so the model always sees exactly which tools exist.
"""

from __future__ import annotations

from typing import Any

from ._tesslate_agent_prompt import TESSLATE_AGENT_SYSTEM_PROMPT

#: Core runtime configuration of the built-in Tesslate Agent. These are the
#: fields a host needs to construct an agent row that behaves identically to
#: the Tesslate Agent. Host-specific presentation (name, slug, description,
#: icon, features, tags, forkability) is intentionally NOT included here —
#: that belongs to whoever surfaces the agent.
#:
#: - ``model=None``  => resolve to the host's site-level default at runtime.
#: - ``tools=None``  => use the TesslateAgent default tool registry.
TESSLATE_AGENT_DEFINITION: dict[str, Any] = {
    "system_prompt": TESSLATE_AGENT_SYSTEM_PROMPT,
    "agent_type": "TesslateAgent",
    "tools": None,
    "model": None,
}

__all__ = ["TESSLATE_AGENT_SYSTEM_PROMPT", "TESSLATE_AGENT_DEFINITION"]
