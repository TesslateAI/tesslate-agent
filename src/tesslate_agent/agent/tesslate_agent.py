"""
TesslateAgent — concrete agent implementation.

Function-calling agent that drives a :class:`ModelAdapter` using the
OpenAI tools contract. Runs a loop of:

1. Build messages (system prompt + history + user request).
2. Call ``model.chat_with_tools`` with the current message list and the
   registered tools converted to OpenAI format.
3. If the model emits tool calls, execute them (parallel for read-only
   tools, sequential for mutating tools), feed results back, continue.
4. If the model returns plain text, yield a ``complete`` event and exit.
5. Compact the message history when it crosses ``compaction_threshold``
   of ``context_window_tokens`` (only when ``compaction_adapter`` is set).

No database, Redis, billing, locking, or orchestrator coupling — the
agent only yields events and reads from the caller-supplied ``context``
dict. Callers are responsible for persistence, lock acquisition, and
credit enforcement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid as uuid_mod
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from tesslate_agent.agent.base import AbstractAgent
from tesslate_agent.agent.models import ModelAdapter
from tesslate_agent.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_DELAY_MS = 200
MAX_RETRIES = 2
RETRYABLE_KEYWORDS: frozenset[str] = frozenset(
    {
        "timeout",
        "connection",
        "temporary",
        "transient",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "stream",
        "502",
        "503",
        "504",
        "429",
    }
)

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_COMPACTION_THRESHOLD = 0.8
DEFAULT_MAX_ITERATIONS = 0  # 0 = no cap; the loop runs until the model
# stops emitting tool calls or the host kills it. Set a positive integer
# to bound the loop explicitly.

# Tool output truncation (per tool result, when fed back to the model)
MAX_TOOL_OUTPUT = 10_000

# Read-only tool names that are safe to execute in parallel within a
# single model turn. Anything not listed here is executed sequentially
# to preserve filesystem / state ordering.
PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_files",
        "glob_search",
        "grep",
        "grep_search",
        "web_fetch",
        "web_search",
        "metadata",
        "container_status",
        "container_logs",
        "container_health",
    }
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter. Returns delay in seconds."""
    exp = 2.0 ** max(0, attempt)
    base_ms = INITIAL_DELAY_MS * exp
    jitter = random.uniform(0.9, 1.1)
    return (base_ms * jitter) / 1000.0


def _is_retryable_error(error: Exception) -> bool:
    """Classify an exception as retryable based on keyword heuristics."""
    error_str = str(error).lower()
    return any(kw in error_str for kw in RETRYABLE_KEYWORDS)


def _safe_json_loads(s: str) -> dict[str, Any]:
    """Parse a JSON string, returning ``{}`` on failure."""
    try:
        value = json.loads(s or "{}")
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _convert_uuids(obj: Any) -> Any:
    """Recursively convert :class:`UUID` instances to strings."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _convert_uuids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_uuids(item) for item in obj]
    return obj


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough 4-chars-per-token estimate across every message payload."""
    total_chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", "")))
        tool_calls = m.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            total_chars += len(str(fn.get("name", "")))
            total_chars += len(str(fn.get("arguments", "")))
    return max(1, total_chars // 4)


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    """Convert a :class:`Tool` to the OpenAI function-tool schema shape."""
    parameters = tool.parameters or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def registry_to_openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Render every tool in ``registry`` in OpenAI function-tool format."""
    return [_tool_to_openai(t) for t in registry.all_tools()]


def serialize_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Serialise an assistant turn for the conversation history.

    Follows the OpenAI contract: ``content`` is ``None`` (not omitted)
    when ``tool_calls`` are present, and each call carries an explicit
    ``"type": "function"``.
    """
    if not tool_calls:
        return {"role": "assistant", "content": content or ""}

    serialized: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        serialized.append(
            {
                "id": tc.get("id", "") if isinstance(tc, dict) else "",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                },
            }
        )

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": serialized,
    }


def format_tool_result(result: dict[str, Any]) -> str:
    """
    Render a tool execution result as the ``content`` of a ``role: tool``
    message. Truncates oversize outputs in the middle.
    """
    if result.get("approval_required"):
        return f"Awaiting approval for {result.get('tool', 'unknown')}"

    if result.get("success"):
        tool_result = result.get("result", {})
        if isinstance(tool_result, dict):
            parts: list[str] = []
            if "message" in tool_result:
                parts.append(str(tool_result["message"]))

            for field_name in ("content", "stdout", "output", "preview"):
                if field_name in tool_result:
                    output = str(tool_result[field_name])
                    if len(output) > MAX_TOOL_OUTPUT:
                        half = MAX_TOOL_OUTPUT // 2
                        elided = len(output) - MAX_TOOL_OUTPUT
                        output = (
                            f"{output[:half]}\n"
                            f"... ({elided} chars truncated) ...\n"
                            f"{output[-half:]}"
                        )
                    parts.append(output)

            files = tool_result.get("files")
            if isinstance(files, list):
                parts.append(f"Files ({len(files)} items): {files[:20]}")

            if tool_result.get("stderr"):
                parts.append(f"stderr: {tool_result['stderr']}")

            return "\n".join(parts) if parts else json.dumps(tool_result)
        return str(tool_result)

    error = result.get("error", "Unknown error")
    suggestion = ""
    inner = result.get("result")
    if isinstance(inner, dict):
        suggestion = str(inner.get("suggestion", ""))
    return f"Error: {error}" + (f"\nSuggestion: {suggestion}" if suggestion else "")


# ---------------------------------------------------------------------------
# TesslateAgent
# ---------------------------------------------------------------------------


class TesslateAgent(AbstractAgent):
    """
    Function-calling agent implementation.

    The agent loop:

    1. Build the system prompt (with marker substitution) and the initial
       message list (system + optional chat history + user request).
    2. Call :meth:`ModelAdapter.chat_with_tools` with the current message
       list and the OpenAI-format tool schemas.
    3. If the model response contains ``tool_calls``, execute them via
       the :class:`ToolRegistry` (parallel where safe, sequential
       otherwise), append results to the history, loop.
    4. If the response is plain text (``finish_reason == "stop"`` and no
       tool calls), yield a ``complete`` event and exit.
    5. If the history crosses the compaction threshold AND a
       ``compaction_adapter`` was supplied, compact the history via the
       cheap adapter before the next call.
    6. Cap iterations at ``max_iterations`` — terminate with a
       ``max_iterations`` completion if reached.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: ToolRegistry | None = None,
        model: ModelAdapter | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        compaction_adapter: ModelAdapter | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW,
        compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    ):
        super().__init__(system_prompt, tools)
        self.model = model
        # max_iterations <= 0 means "no cap" — the loop runs until the
        # model emits a turn with no tool calls, the orchestrator times
        # out, or an exception terminates the run.
        max_iter_int = int(max_iterations)
        self.max_iterations = max_iter_int if max_iter_int > 0 else 0
        self.compaction_adapter = compaction_adapter
        self.context_window_tokens = max(1000, int(context_window_tokens))
        self.compaction_threshold = max(0.1, min(0.99, float(compaction_threshold)))

        logger.info(
            "TesslateAgent initialised - tools=%d, max_iterations=%s, "
            "context_window=%d, compaction=%s",
            len(self.tools._tools) if self.tools else 0,
            self.max_iterations if self.max_iterations > 0 else "unlimited",
            self.context_window_tokens,
            "on" if self.compaction_adapter is not None else "off",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_request: str,
        context: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run the agent to completion for ``user_request``.

        Yields streaming event dicts. See :class:`AbstractAgent.run` for
        the event shapes.
        """
        if self.model is None:
            yield {"type": "error", "content": "Model adapter not set."}
            yield self._complete_event(
                success=False,
                iterations=0,
                final_response="",
                error="missing_model_adapter",
                tool_calls_made=0,
                reason="missing_model_adapter",
                session_id=None,
            )
            return

        session_id = str(uuid_mod.uuid4())
        iteration = 0
        tool_calls_count = 0

        messages = self._build_initial_messages(user_request, context)

        logger.info(
            "[TesslateAgent] run start - session=%s, request=%s",
            session_id,
            (user_request or "")[:100],
        )

        while True:
            iteration += 1
            logger.info("[TesslateAgent] iteration %d", iteration)

            if self.max_iterations > 0 and iteration > self.max_iterations:
                logger.warning(
                    "[TesslateAgent] max_iterations=%d reached",
                    self.max_iterations,
                )
                yield self._complete_event(
                    success=False,
                    iterations=iteration - 1,
                    final_response="",
                    error=f"Maximum iterations ({self.max_iterations}) reached.",
                    tool_calls_made=tool_calls_count,
                    reason="max_iterations",
                    session_id=session_id,
                )
                return

            # Pre-flight compaction
            if (
                self.compaction_adapter is not None
                and _estimate_tokens(messages)
                >= self.context_window_tokens * self.compaction_threshold
            ):
                messages = await self._compact_messages(messages)
                yield {
                    "type": "context_pressure",
                    "data": {"compacted": True, "iteration": iteration},
                }

            # Call the model
            try:
                response = await self._call_model_with_retry(messages)
            except Exception as exc:
                logger.error("[TesslateAgent] LLM call failed: %s", exc)
                yield {"type": "error", "content": f"Agent error: {exc}"}
                yield self._complete_event(
                    success=False,
                    iterations=iteration,
                    final_response="",
                    error=str(exc),
                    tool_calls_made=tool_calls_count,
                    reason="model_error",
                    session_id=session_id,
                )
                return

            content = str(response.get("content") or "")
            tool_calls = response.get("tool_calls") or []
            usage = response.get("usage") or {}
            finish_reason = response.get("finish_reason")

            # Stream the assistant text (if any) as a single chunk so
            # callers can render it without needing a separate streaming
            # path.
            if content:
                yield {"type": "stream", "content": content}

            # Terminal: no more tool calls
            if not tool_calls:
                yield self._agent_step_event(
                    iteration=iteration,
                    tool_calls=[],
                    tool_results=[],
                    response_text=content,
                    is_complete=True,
                )
                yield self._complete_event(
                    success=True,
                    iterations=iteration,
                    final_response=content or "Task completed.",
                    error=None,
                    tool_calls_made=tool_calls_count,
                    reason="stop" if finish_reason == "stop" else "no_more_actions",
                    session_id=session_id,
                    usage=usage,
                )
                return

            # Add the assistant turn to history
            messages.append(serialize_assistant_message(content, tool_calls))

            # Execute tool calls
            tool_results = await self._execute_tool_calls(tool_calls, context)
            tool_calls_count += len(tool_calls)

            for idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name = fn.get("name", "")
                tool_params = _convert_uuids(_safe_json_loads(fn.get("arguments", "{}")))
                result = tool_results[idx]

                yield {
                    "type": "tool_result",
                    "data": {
                        "iteration": iteration,
                        "index": idx,
                        "total": len(tool_calls),
                        "name": tool_name,
                        "parameters": tool_params,
                        "result": _convert_uuids(result),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                }

                tc_id = tc.get("id", f"call_{idx}") if isinstance(tc, dict) else f"call_{idx}"
                result_text = format_tool_result(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_text,
                    }
                )

            yield self._agent_step_event(
                iteration=iteration,
                tool_calls=tool_calls,
                tool_results=tool_results,
                response_text=content,
                is_complete=False,
            )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self,
        user_request: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the initial message list (system + history + user)."""
        system_prompt = self.get_processed_system_prompt(context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        chat_history = context.get("chat_history") or []
        if chat_history:
            logger.info(
                "[TesslateAgent] including %d history messages", len(chat_history)
            )
            for entry in chat_history:
                if isinstance(entry, dict) and entry.get("role"):
                    messages.append(entry)

        messages.append(
            self._build_user_turn(user_request, context.get("attachments") or [])
        )
        return messages

    # Defense-in-depth caps mirroring the API-schema limits. If a payload
    # somehow slips past the schema (legacy client, internal enqueue path),
    # we still clip rather than blow the context window.
    _MAX_PASTED_TEXT_CHARS = 100_000
    _MAX_IMAGE_BASE64_CHARS = 20_000_000

    @staticmethod
    def _build_user_turn(
        user_request: str,
        attachments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compose the user turn, folding chat attachments into the prompt.

        ``pasted_text`` and ``file_reference`` attachments are inlined as
        labeled text blocks. ``image`` attachments become OpenAI vision
        content-parts. Without attachments the payload is unchanged so
        non-attachment flows keep their existing shape. Oversized payloads
        are truncated with a visible marker instead of dropped silently.
        """
        if not attachments:
            return {"role": "user", "content": user_request}

        text_blocks: list[str] = []
        if user_request:
            text_blocks.append(user_request)

        image_parts: list[dict[str, Any]] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            att_type = att.get("type")
            label = att.get("label") or att_type or "attachment"

            if att_type == "pasted_text":
                body = att.get("content") or ""
                if len(body) > TesslateAgent._MAX_PASTED_TEXT_CHARS:
                    dropped = len(body) - TesslateAgent._MAX_PASTED_TEXT_CHARS
                    body = (
                        body[: TesslateAgent._MAX_PASTED_TEXT_CHARS]
                        + f"\n… [truncated: {dropped} more chars]"
                    )
                text_blocks.append(f"\n[{label}]\n{body}")
            elif att_type == "file_reference":
                fp = att.get("file_path") or "unknown"
                text_blocks.append(f"\n[attached file: {fp}]")
            elif att_type == "image":
                b64 = att.get("content") or ""
                if not b64:
                    continue
                if len(b64) > TesslateAgent._MAX_IMAGE_BASE64_CHARS:
                    logger.warning(
                        "[TesslateAgent] dropping oversized image attachment "
                        "(%d base64 chars, cap %d)",
                        len(b64),
                        TesslateAgent._MAX_IMAGE_BASE64_CHARS,
                    )
                    text_blocks.append(
                        f"\n[{label}: image exceeded size cap, not attached]"
                    )
                    continue
                mime = att.get("mime_type") or "image/png"
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )

        combined_text = "\n".join(text_blocks).strip() or "(attachments only)"

        if image_parts:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text},
                    *image_parts,
                ],
            }
        return {"role": "user", "content": combined_text}

    def _get_openai_tools(self) -> list[dict[str, Any]]:
        """Return the tool set in OpenAI function-tool format, or ``[]``."""
        if self.tools is None:
            return []
        return registry_to_openai_tools(self.tools)

    # ------------------------------------------------------------------
    # Model calling (with retry)
    # ------------------------------------------------------------------

    async def _call_model_with_retry(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Call ``model.chat_with_tools`` with exponential-backoff retry for
        transient failures.

        Always uses ``stream=False`` — the adapter returns a ready dict.
        """
        assert self.model is not None  # narrowed by run()
        tools = self._get_openai_tools()

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.model.chat_with_tools(
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else "none",
                    stream=False,
                )
                if isinstance(response, dict):
                    return response
                # If an adapter mis-implements the contract and returns an
                # async iterator for stream=False, collect the final state.
                return await _collect_stream(response)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt == MAX_RETRIES:
                    raise
                delay = _backoff(attempt)
                logger.warning(
                    "[TesslateAgent] retryable model error "
                    "(attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        # Unreachable — the final attempt re-raises above — but keeps
        # type checkers happy.
        raise last_error if last_error else RuntimeError("model call failed")

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute ``tool_calls`` against the registry.

        Parallel-safe tools (read_file, list_files, grep, web_fetch, ...)
        run concurrently via :func:`asyncio.gather`. Everything else runs
        sequentially to preserve state ordering.
        """
        if self.tools is None:
            return [
                {
                    "success": False,
                    "tool": (tc.get("function") or {}).get("name", ""),
                    "error": "No tool registry configured.",
                }
                for tc in tool_calls
            ]

        parallel: list[tuple[int, str, dict[str, Any]]] = []
        sequential: list[tuple[int, str, dict[str, Any]]] = []

        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            params = _safe_json_loads(fn.get("arguments", "{}"))
            if name in PARALLEL_SAFE_TOOLS:
                parallel.append((idx, name, params))
            else:
                sequential.append((idx, name, params))

        results: list[dict[str, Any] | None] = [None] * len(tool_calls)

        if parallel:
            coros = [
                self.tools.execute(name, params, context)
                for _idx, name, params in parallel
            ]
            outcomes = await asyncio.gather(*coros, return_exceptions=True)
            for (idx, name, _params), outcome in zip(parallel, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    logger.error(
                        "[TesslateAgent] parallel tool '%s' raised: %s", name, outcome
                    )
                    results[idx] = {
                        "success": False,
                        "tool": name,
                        "error": str(outcome),
                    }
                else:
                    results[idx] = outcome

        for idx, name, params in sequential:
            try:
                results[idx] = await self.tools.execute(name, params, context)
            except Exception as exc:
                logger.error(
                    "[TesslateAgent] sequential tool '%s' raised: %s", name, exc
                )
                results[idx] = {
                    "success": False,
                    "tool": name,
                    "error": str(exc),
                }

        # Guard against any remaining ``None`` slots.
        final: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            if r is None:
                final.append(
                    {
                        "success": False,
                        "tool": "",
                        "error": f"Tool execution produced no result (index={i}).",
                    }
                )
            else:
                final.append(r)
        return final

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _compact_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Summarise the middle of ``messages`` using ``compaction_adapter``.

        The system message and the most recent ~6 messages are kept
        verbatim; everything in between is replaced by a single summary
        system message generated by the cheap adapter.
        """
        if self.compaction_adapter is None or len(messages) <= 8:
            return messages

        keep_head = messages[:1]  # system
        keep_tail = messages[-6:]  # recent turns
        middle = messages[1:-6]
        if not middle:
            return messages

        summarise_prompt = [
            {
                "role": "system",
                "content": (
                    "You compact agent conversations. Summarise the "
                    "following messages into a concise bullet list of "
                    "decisions, file edits, tool results, and open "
                    "questions. Preserve critical identifiers verbatim."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(middle, default=str)[:60_000],
            },
        ]

        try:
            resp = await self.compaction_adapter.chat_with_tools(
                messages=summarise_prompt,
                tools=None,
                tool_choice="none",
                stream=False,
            )
        except Exception as exc:
            logger.warning("[TesslateAgent] compaction failed: %s", exc)
            return messages

        if not isinstance(resp, dict):
            resp = await _collect_stream(resp)

        summary = str(resp.get("content") or "").strip()
        if not summary:
            return messages

        summary_msg = {
            "role": "system",
            "content": f"[Conversation summary]\n{summary}",
        }
        return keep_head + [summary_msg] + keep_tail

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_step_event(
        iteration: int,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        response_text: str,
        is_complete: bool,
    ) -> dict[str, Any]:
        return {
            "type": "agent_step",
            "data": {
                "iteration": iteration,
                "tool_calls": [
                    {
                        "name": (tc.get("function") or {}).get("name", ""),
                        "parameters": _convert_uuids(
                            _safe_json_loads(
                                (tc.get("function") or {}).get("arguments", "{}")
                            )
                        ),
                    }
                    for tc in tool_calls
                ],
                "tool_results": _convert_uuids(tool_results),
                "response_text": response_text or "",
                "timestamp": datetime.now(UTC).isoformat(),
                "is_complete": is_complete,
            },
        }

    @staticmethod
    def _complete_event(
        success: bool,
        iterations: int,
        final_response: str,
        error: str | None,
        tool_calls_made: int,
        reason: str,
        session_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "success": success,
            "iterations": iterations,
            "final_response": final_response,
            "tool_calls_made": tool_calls_made,
            "completion_reason": reason,
        }
        if error:
            data["error"] = error
        if session_id:
            data["session_id"] = session_id
        if usage:
            data["usage"] = usage
        return {"type": "complete", "data": data}


# ---------------------------------------------------------------------------
# Streaming collector fallback
# ---------------------------------------------------------------------------


async def _collect_stream(stream: Any) -> dict[str, Any]:
    """
    Collapse an adapter streaming iterator into a dict matching the
    non-streaming ``chat_with_tools`` contract.

    Used as a safety net when an adapter returns an iterator even though
    ``stream=False`` was requested.
    """
    content_parts: list[str] = []
    tool_call_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}

    async for chunk in stream:
        if not isinstance(chunk, dict):
            continue
        delta = chunk.get("delta_content") or chunk.get("content") or ""
        if delta:
            content_parts.append(str(delta))
        for tc in chunk.get("delta_tool_calls") or chunk.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", len(tool_call_acc))
            bucket = tool_call_acc.setdefault(
                idx,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc.get("id"):
                bucket["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                bucket["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                bucket["function"]["arguments"] += fn["arguments"]
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]

    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_call_acc[k] for k in sorted(tool_call_acc.keys())],
        "usage": usage,
        "finish_reason": finish_reason,
    }
