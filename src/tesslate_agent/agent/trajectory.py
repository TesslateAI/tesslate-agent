"""
ATIF v1.4 trajectory recording.

Records agent execution as timestamped entries and converts them to the
Agent Trajectory Interchange Format (ATIF) v1.4 for observability.

This module is fully self-contained and has no database or Redis
dependencies — records live in memory and can be serialised by the caller.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

AGENT_NAME = "tesslate-agent"
AGENT_VERSION = "1.2.0"
SCHEMA_VERSION = "ATIF-v1.4"


class TrajectoryRecorder:
    """
    Records agent messages for ATIF conversion.

    Each ``record_*`` method appends a timestamped entry. After the run
    completes, call :meth:`to_atif` to get the full ATIF v1.4 dict.
    """

    def __init__(self, session_id: str, model_name: str):
        self.session_id = session_id
        self.model_name = model_name
        self.entries: list[dict[str, Any]] = []

    def record_system(self, content: str) -> None:
        self.entries.append(
            {
                "role": "system",
                "content": content,
                "timestamp": _now(),
            }
        )

    def record_user(self, content: str) -> None:
        self.entries.append(
            {
                "role": "user",
                "content": content,
                "timestamp": _now(),
            }
        )

    def record_assistant(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
            "timestamp": _now(),
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if usage:
            entry["usage"] = usage
        self.entries.append(entry)

    def record_tool_result(self, tool_call_id: str, content: str) -> None:
        self.entries.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "timestamp": _now(),
            }
        )

    def to_atif(self) -> dict[str, Any]:
        """Convert recorded entries to an ATIF v1.4 dict."""
        return convert_to_atif(
            trajectory=self.entries,
            session_id=self.session_id,
            model_name=self.model_name,
            agent_name=AGENT_NAME,
        )


def convert_to_atif(
    trajectory: list[dict[str, Any]],
    session_id: str,
    model_name: str,
    agent_name: str = AGENT_NAME,
    agent_version: str = AGENT_VERSION,
    extra_agent_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert a raw trajectory list to ATIF v1.4 format.

    Tool results are matched by ``tool_call_id`` to the preceding agent
    step and attached as ``observation.results``, not as separate steps.
    """
    steps: list[dict[str, Any]] = []
    step_id = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    agent_step_count = 0

    pending_tool_results: list[dict[str, Any]] = []

    for entry in trajectory:
        role = entry.get("role", "")

        if role == "tool":
            pending_tool_results.append(entry)
            continue

        if pending_tool_results and steps:
            _attach_tool_results(steps[-1], pending_tool_results)
            pending_tool_results = []

        step_id += 1

        if role == "system":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": entry.get("timestamp", ""),
                    "source": "system",
                    "message": entry.get("content", ""),
                }
            )

        elif role == "user":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": entry.get("timestamp", ""),
                    "source": "user",
                    "message": entry.get("content", ""),
                }
            )

        elif role == "assistant":
            agent_step_count += 1
            step: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": entry.get("timestamp", ""),
                "source": "agent",
                "model_name": model_name,
                "message": entry.get("content", ""),
            }

            tool_calls = entry.get("tool_calls")
            if tool_calls:
                step["tool_calls"] = [
                    {
                        "tool_call_id": tc.get("id", ""),
                        "function_name": tc.get("function", {}).get("name", ""),
                        "arguments": _safe_parse_arguments(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                    }
                    for tc in tool_calls
                ]

            usage = entry.get("usage") or {}
            prompt = usage.get("prompt_tokens", 0) or 0
            completion = usage.get("completion_tokens", 0) or 0
            cached = usage.get("cached_tokens", 0) or 0
            if prompt or completion:
                metrics: dict[str, int] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                }
                if cached:
                    metrics["cached_tokens"] = cached
                step["metrics"] = metrics

            total_prompt_tokens += prompt
            total_completion_tokens += completion
            total_cached_tokens += cached

            steps.append(step)

    if pending_tool_results and steps:
        _attach_tool_results(steps[-1], pending_tool_results)

    agent_info: dict[str, Any] = {
        "name": agent_name,
        "version": agent_version,
        "model_name": model_name,
    }
    if extra_agent_fields:
        agent_info["extra"] = extra_agent_fields

    final_metrics: dict[str, Any] = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_steps": agent_step_count,
    }
    if total_cached_tokens:
        final_metrics["total_cached_tokens"] = total_cached_tokens

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": agent_info,
        "steps": steps,
        "final_metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_parse_arguments(args: Any) -> Any:
    """Return parsed JSON if ``args`` is a string, otherwise return as-is."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return args
    return args


def _attach_tool_results(
    step: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> None:
    """Attach tool results as ``observation.results`` on ``step``."""
    results: list[dict[str, Any]] = []
    for tr in tool_results:
        results.append(
            {
                "source_call_id": tr.get("tool_call_id", ""),
                "content": tr.get("content", ""),
            }
        )
    if results:
        step.setdefault("observation", {})["results"] = results
