"""
Streaming event adapter for :class:`TrajectoryRecorder`.

The :class:`TesslateAgent` emits a small set of event dicts while it
runs. The CLI consumes those events and feeds them into a
:class:`TrajectoryRecorder` so the final ATIF document faithfully
reflects what the agent actually did — assistant turns, tool calls,
tool results, errors, and the final answer.

This module does not own the recorder; the caller builds the recorder
(so it can seed the system and user turns directly), passes it to the
bridge, and later calls :meth:`TrajectoryBridge.finalize` to obtain
the ATIF dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tesslate_agent.agent.trajectory import TrajectoryRecorder

logger = logging.getLogger(__name__)

__all__ = ["TrajectoryBridge"]


def _normalize_tool_calls(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise agent tool-call dicts into the OpenAI function-call shape.

    The :class:`TesslateAgent` emits two different tool-call shapes
    depending on the event:

    * ``agent_step`` events use a compact shape: ``{"name", "parameters"}``.
    * The underlying ``chat_with_tools`` response uses the OpenAI shape:
      ``{"id", "type": "function", "function": {"name", "arguments"}}``.

    :class:`TrajectoryRecorder.record_assistant` expects the OpenAI
    shape (it reads ``tc["id"]`` and ``tc["function"]["arguments"]``),
    so this helper coerces either input into that shape.
    """
    normalized: list[dict[str, Any]] = []
    for idx, tc in enumerate(raw or []):
        if not isinstance(tc, dict):
            continue

        # Already in OpenAI shape.
        if "function" in tc and isinstance(tc["function"], dict):
            fn = tc["function"]
            arguments = fn.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, default=str)
            normalized.append(
                {
                    "id": tc.get("id") or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )
            continue

        # Compact agent-step shape: {"name", "parameters"}.
        name = tc.get("name", "")
        parameters = tc.get("parameters", {})
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters, default=str)
        normalized.append(
            {
                "id": tc.get("id") or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": parameters,
                },
            }
        )

    return normalized


class TrajectoryBridge:
    """
    Forward :class:`TesslateAgent` events into a :class:`TrajectoryRecorder`.

    The bridge buffers streamed assistant text between ``agent_step``
    events so the recorded assistant turns match what the model
    actually produced on each iteration. It also tracks whether the
    run finished successfully, allowing the CLI to pick an exit code
    without needing to re-inspect the events.
    """

    def __init__(self, recorder: TrajectoryRecorder, project_root: Path):
        self._recorder = recorder
        self._project_root = project_root
        self._stream_buffer: list[str] = []
        self._finalized: bool = False
        self.has_error: bool = False
        self.final_response: str = ""

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def handle_event(self, event: dict[str, Any]) -> None:
        """
        Dispatch a single agent event into the recorder.

        Unknown event types are logged and ignored so new event kinds
        added to the agent do not break existing trajectories.
        """
        if not isinstance(event, dict):
            return

        event_type = event.get("type", "")

        if event_type == "stream":
            content = event.get("content")
            if isinstance(content, str) and content:
                self._stream_buffer.append(content)
            return

        if event_type == "agent_step":
            self._handle_agent_step(event)
            return

        if event_type == "tool_result":
            # Per-call streaming events for live UIs. Tool results are
            # already recorded by `_handle_agent_step` (matched to their
            # originating call IDs), so skip to avoid double-recording
            # entries with empty source_call_ids.
            return

        if event_type == "complete":
            data = event.get("data") or {}
            final = str(data.get("final_response") or "")
            # Flush any buffered stream text into a trailing assistant turn
            # if the agent produced text but never emitted an agent_step
            # (which happens when the final response has no tool calls).
            buffered = self._drain_stream_buffer()
            if final and final != buffered:
                self._recorder.record_assistant(content=final)
            elif buffered and not final:
                self._recorder.record_assistant(content=buffered)
            self.final_response = final
            if not data.get("success", True):
                self.has_error = True
            self._finalized = True
            return

        if event_type == "error":
            message = str(event.get("content") or "")
            self._recorder.record_assistant(content=f"[error] {message}")
            self.has_error = True
            return

        if event_type == "context_pressure":
            logger.debug("context_pressure event ignored by bridge")
            return

        logger.debug("unhandled agent event type: %s", event_type)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def mark_errored(self, message: str) -> None:
        """
        Record an assistant turn describing an external error.

        Used by the runner when the agent loop raises (timeout,
        unexpected exception, keyboard interrupt) before emitting a
        ``complete`` event itself.
        """
        self._recorder.record_assistant(content=f"[error] {message}")
        self.has_error = True
        self._finalized = True

    def finalize(self) -> dict[str, Any]:
        """
        Return the ATIF dict for the full recorded trajectory.

        Safe to call multiple times — the recorder state is not
        mutated.
        """
        return self._recorder.to_atif()

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_agent_step(self, event: dict[str, Any]) -> None:
        data = event.get("data") or {}
        response_text = str(data.get("response_text") or "")

        # Prefer streamed text if the agent_step response_text is empty
        # (which happens when the model only emitted tool calls without
        # any prose on this iteration).
        buffered = self._drain_stream_buffer()
        if not response_text and buffered:
            response_text = buffered

        raw_tool_calls = data.get("tool_calls") or []
        normalized_calls = _normalize_tool_calls(raw_tool_calls)

        usage = data.get("usage")
        self._recorder.record_assistant(
            content=response_text,
            tool_calls=normalized_calls,
            usage=usage if isinstance(usage, dict) else None,
        )

        tool_results = data.get("tool_results") or []
        for idx, result in enumerate(tool_results):
            if idx < len(normalized_calls):
                call_id = normalized_calls[idx]["id"]
            else:
                call_id = f"call_{idx}"
            self._recorder.record_tool_result(call_id, content=str(result))

    def _drain_stream_buffer(self) -> str:
        if not self._stream_buffer:
            return ""
        text = "".join(self._stream_buffer)
        self._stream_buffer.clear()
        return text
