"""
Context builders for the standalone Tesslate Agent CLI.

The agent expects a context dict with a handful of fields (user,
project identifiers, orchestrator-style container metadata, a model
adapter, a run id). The CLI runs outside any database or web
framework, so this module synthesises those fields from the local
working directory and a simple stub user.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

__all__ = ["StubUser", "make_standalone_context"]


@dataclass
class StubUser:
    """
    Lightweight stand-in for an ORM ``User`` row.

    The agent's tools only read ``id``, ``email``, and ``display_name``
    — none of the database-only fields — so a plain dataclass is
    sufficient for the CLI.
    """

    id: UUID = field(default_factory=uuid4)
    email: str = "benchmark@local"
    display_name: str = "benchmark"


def _project_identifier(project_root: Path) -> str:
    """
    Derive a stable short identifier for a project root path.

    Uses the first 12 characters of a SHA-256 digest of the absolute
    path so the same working directory always maps to the same id
    across invocations.
    """
    digest = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


def make_standalone_context(
    user: StubUser,
    project_root: Path,
    model_adapter: Any,
    run_id: str,
) -> dict[str, Any]:
    """
    Build the context dict passed to :meth:`TesslateAgent.run`.

    Args:
        user: Stub user identity (CLI has no real auth layer).
        project_root: Absolute path to the working directory the agent
            should operate in.
        model_adapter: Instantiated :class:`ModelAdapter` used for the
            LLM calls. Stored under ``model_adapter`` so tools that
            want to spawn a sub-model can reuse it.
        run_id: Unique identifier for this agent invocation. Used by
            the trajectory recorder and downstream logs.

    Returns:
        A context dict with every field the agent and tool registry
        require in standalone (no-database) mode.
    """
    resolved_root = project_root.resolve()
    identifier = _project_identifier(resolved_root)

    return {
        "user": user,
        "user_id": str(user.id),
        "project_id": identifier,
        "project_slug": identifier,
        "container_name": "local",
        "container_directory": None,
        "db": None,
        "model_adapter": model_adapter,
        "run_id": run_id,
        "project_root": resolved_root,
        "subagent_depth": 0,
    }
