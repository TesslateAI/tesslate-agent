"""
Tests for :class:`TesslateAgent`, :class:`AbstractAgent`, and
:class:`TrajectoryRecorder`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tesslate_agent.agent.base import AbstractAgent
from tesslate_agent.agent.models import ModelAdapter
from tesslate_agent.agent.tesslate_agent import TesslateAgent, format_tool_result
from tesslate_agent.agent.tools.registry import Tool, ToolCategory, ToolRegistry
from tesslate_agent.agent.trajectory import TrajectoryRecorder

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(ModelAdapter):
    """Model adapter that replays a canned sequence of responses."""

    def __init__(
        self,
        responses: list[dict[str, Any]],
        *,
        model_name: str = "fake/test-model",
    ) -> None:
        self._responses = list(responses)
        self._model_name = model_name
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools else [],
                "tool_choice": tool_choice,
            }
        )
        if not self._responses:
            return {
                "content": "",
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            }
        # Clone via dict(...) so callers don't mutate our canned data.
        return dict(self._responses.pop(0))


async def _fake_read_file(
    params: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    return {
        "success": True,
        "message": f"read {params.get('file_path')}",
        "content": f"contents of {params.get('file_path')}",
    }


def _make_read_file_tool() -> Tool:
    return Tool(
        name="read_file",
        description="Read a file from disk.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to read."},
            },
            "required": ["file_path"],
        },
        executor=_fake_read_file,
        category=ToolCategory.FILE_OPS,
    )


async def _collect_events(agent: TesslateAgent, request: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for event in agent.run(request, context={}):
        events.append(event)
    return events


async def _collect_events_with_context(
    agent: TesslateAgent,
    request: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for event in agent.run(request, context=context):
        events.append(event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiation_does_not_crash() -> None:
    agent = TesslateAgent(
        system_prompt="test",
        tools=ToolRegistry(),
        model=FakeAdapter(responses=[]),
    )
    assert isinstance(agent, AbstractAgent)
    assert agent.system_prompt == "test"
    assert agent.tools is not None


@pytest.mark.asyncio
async def test_happy_path_tool_call_then_stop() -> None:
    registry = ToolRegistry()
    registry.register(_make_read_file_tool())

    responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "hello.txt"}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "finish_reason": "tool_calls",
        },
        {
            "content": "Done.",
            "tool_calls": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            "finish_reason": "stop",
        },
    ]

    agent = TesslateAgent(
        system_prompt="You are a helpful assistant.",
        tools=registry,
        model=FakeAdapter(responses=responses),
    )

    events = await _collect_events(agent, "please read hello.txt")

    event_types = [e.get("type") for e in events]
    assert "tool_result" in event_types
    assert "agent_step" in event_types
    assert "complete" in event_types

    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["data"]["name"] == "read_file"
    assert tool_results[0]["data"]["parameters"] == {"file_path": "hello.txt"}

    complete = next(e for e in events if e.get("type") == "complete")
    assert complete["data"]["success"] is True
    assert complete["data"]["iterations"] == 2
    assert complete["data"]["tool_calls_made"] == 1
    assert complete["data"]["final_response"] == "Done."


@pytest.mark.asyncio
async def test_no_tools_path_plain_text() -> None:
    responses = [
        {
            "content": "Hello there!",
            "tool_calls": [],
            "usage": {},
            "finish_reason": "stop",
        }
    ]
    agent = TesslateAgent(
        system_prompt="test",
        tools=ToolRegistry(),
        model=FakeAdapter(responses=responses),
    )

    events = await _collect_events(agent, "say hi")
    complete = next(e for e in events if e.get("type") == "complete")
    assert complete["data"]["success"] is True
    assert complete["data"]["final_response"] == "Hello there!"
    assert complete["data"]["tool_calls_made"] == 0

    stream_events = [e for e in events if e.get("type") == "stream"]
    assert any(se.get("content") == "Hello there!" for se in stream_events)


@pytest.mark.asyncio
async def test_max_iterations_cap() -> None:
    """Adapter that always asks for a tool call → loop terminates at cap."""
    registry = ToolRegistry()
    registry.register(_make_read_file_tool())

    looping_response = {
        "content": "",
        "tool_calls": [
            {
                "id": "loop",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "loop.txt"}',
                },
            }
        ],
        "usage": {},
        "finish_reason": "tool_calls",
    }

    class LoopingAdapter(FakeAdapter):
        async def chat_with_tools(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] = "auto",
            temperature: float | None = None,
            max_tokens: int | None = None,
            stream: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.calls.append({"messages": list(messages)})
            return dict(looping_response)

    agent = TesslateAgent(
        system_prompt="test",
        tools=registry,
        model=LoopingAdapter(responses=[]),
        max_iterations=3,
    )

    events = await _collect_events(agent, "loop please")
    complete = next(e for e in events if e.get("type") == "complete")
    assert complete["data"]["success"] is False
    assert complete["data"]["completion_reason"] == "max_iterations"
    assert "Maximum iterations" in (complete["data"].get("error") or "")
    assert complete["data"]["iterations"] == 3


@pytest.mark.asyncio
async def test_max_iterations_zero_disables_cap() -> None:
    """max_iterations=0 means no cap; loop only stops when the model
    emits a turn with no tool calls."""
    registry = ToolRegistry()
    registry.register(_make_read_file_tool())

    tool_call = {
        "id": "loop",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"file_path": "loop.txt"}',
        },
    }

    class CountingAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__(responses=[])
            self.turn = 0

        async def chat_with_tools(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] = "auto",
            temperature: float | None = None,
            max_tokens: int | None = None,
            stream: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.calls.append({"messages": list(messages)})
            self.turn += 1
            # Default cap is 30; iterate well past it before terminating
            # to prove no cap is in force.
            if self.turn >= 50:
                return {
                    "content": "done",
                    "tool_calls": [],
                    "usage": {},
                    "finish_reason": "stop",
                }
            return {
                "content": "",
                "tool_calls": [tool_call],
                "usage": {},
                "finish_reason": "tool_calls",
            }

    agent = TesslateAgent(
        system_prompt="test",
        tools=registry,
        model=CountingAdapter(),
        max_iterations=0,
    )
    assert agent.max_iterations == 0

    events = await _collect_events(agent, "loop please")
    complete = next(e for e in events if e.get("type") == "complete")
    assert complete["data"]["success"] is True
    assert complete["data"]["iterations"] == 50
    assert complete["data"]["completion_reason"] == "stop"


@pytest.mark.asyncio
async def test_missing_model_adapter_yields_error() -> None:
    agent = TesslateAgent(system_prompt="test", tools=ToolRegistry(), model=None)
    events = await _collect_events(agent, "hi")
    assert any(e.get("type") == "error" for e in events)
    complete = next(e for e in events if e.get("type") == "complete")
    assert complete["data"]["success"] is False
    assert complete["data"]["completion_reason"] == "missing_model_adapter"


@pytest.mark.asyncio
async def test_trajectory_recorder_integration() -> None:
    """Drive the agent, feed its events into a TrajectoryRecorder, verify ATIF."""
    registry = ToolRegistry()
    registry.register(_make_read_file_tool())

    responses = [
        {
            "content": "thinking...",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "hello.txt"}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "finish_reason": "tool_calls",
        },
        {
            "content": "All done.",
            "tool_calls": [],
            "usage": {"prompt_tokens": 15, "completion_tokens": 3},
            "finish_reason": "stop",
        },
    ]

    agent = TesslateAgent(
        system_prompt="system-prompt-for-test",
        tools=registry,
        model=FakeAdapter(responses=responses),
    )

    recorder = TrajectoryRecorder(session_id="sess-1", model_name="fake/test-model")
    recorder.record_system("system-prompt-for-test")
    recorder.record_user("please read hello.txt")

    async for event in agent.run("please read hello.txt", context={}):
        t = event.get("type")
        if t == "agent_step":
            data = event.get("data", {})
            tool_calls_raw: list[dict[str, Any]] = []
            for idx, tc in enumerate(data.get("tool_calls", [])):
                tool_calls_raw.append(
                    {
                        "id": f"c{idx + 1}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": "{}",
                        },
                    }
                )
            recorder.record_assistant(
                content=data.get("response_text", ""),
                tool_calls=tool_calls_raw or None,
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            )
        elif t == "tool_result":
            data = event.get("data", {})
            recorder.record_tool_result(
                tool_call_id=f"c{int(data.get('index', 0)) + 1}",
                content=str(data.get("result", "")),
            )

    atif = recorder.to_atif()
    assert atif["schema_version"] == "ATIF-v1.4"
    assert atif["session_id"] == "sess-1"
    assert "final_metrics" in atif
    assert atif["final_metrics"]["total_steps"] >= 1
    assert atif["agent"]["model_name"] == "fake/test-model"
    assert any(step["source"] == "agent" for step in atif["steps"])


# ---------------------------------------------------------------------------
# format_tool_result tests — Bug #198
# ---------------------------------------------------------------------------


def test_format_tool_result_plan_mode_block_shows_message() -> None:
    """Plan-mode blocks return 'error' key directly; message must be visible."""
    result = {
        "success": False,
        "tool": "write_file",
        "error": "Plan mode active - write_file is disabled.",
    }
    text = format_tool_result(result)
    assert "Plan mode active" in text
    assert "write_file" in text


def test_format_tool_result_error_output_format_propagates_message() -> None:
    """Tools using error_output() embed the message in result.message, not
    at the top-level error key. format_tool_result must fall back to that."""
    result = {
        "success": False,
        "tool": "write_file",
        "result": {
            "success": False,
            "message": "Disk quota exceeded — cannot write 4.2 MB",
            "suggestion": "Remove large build artifacts before retrying",
        },
    }
    text = format_tool_result(result)
    assert "Disk quota exceeded" in text
    assert "Unknown error" not in text


def test_format_tool_result_suggestion_from_nested_result() -> None:
    """Suggestion from error_output() payload surfaces in the formatted text."""
    result = {
        "success": False,
        "tool": "bash_exec",
        "result": {
            "success": False,
            "message": "Command timed out",
            "suggestion": "Break into smaller steps",
        },
    }
    text = format_tool_result(result)
    assert "Command timed out" in text
    assert "Break into smaller steps" in text


def test_format_tool_result_unknown_error_fallback() -> None:
    """When neither 'error' nor 'result.message' is present, fall back
    to 'Unknown error' so the model always gets actionable text."""
    result = {"success": False, "tool": "mystery_tool"}
    text = format_tool_result(result)
    assert text == "Error: Unknown error"


def test_format_tool_result_success_unchanged() -> None:
    """Success results are not affected by the error-path fix."""
    result = {
        "success": True,
        "tool": "read_file",
        "result": {
            "message": "read /app/README.md",
            "content": "Hello world",
        },
    }
    text = format_tool_result(result)
    assert "read /app/README.md" in text
    assert "Hello world" in text


@pytest.mark.asyncio
async def test_plan_mode_block_message_visible_to_model() -> None:
    """End-to-end: when a tool is blocked in plan mode, the model sees the
    actual 'Plan mode active' message (not 'Unknown error')."""

    async def _blocked_write(
        params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "success": False,
            "tool": "write_file",
            "error": "Plan mode active - write_file is disabled.",
        }

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="write_file",
            description="Write a file.",
            parameters={
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"],
            },
            executor=_blocked_write,
            category=ToolCategory.FILE_OPS,
        )
    )

    responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"file_path": "out.txt", "content": "hi"}',
                    },
                }
            ],
            "usage": {},
            "finish_reason": "tool_calls",
        },
        {
            "content": "I cannot write in plan mode.",
            "tool_calls": [],
            "usage": {},
            "finish_reason": "stop",
        },
    ]

    adapter = FakeAdapter(responses=responses)
    agent = TesslateAgent(system_prompt="test", tools=registry, model=adapter)

    events = await _collect_events(agent, "write out.txt")
    tool_result_events = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_result_events) == 1

    # When the executor returns {success: False, error: "..."}, the registry
    # wraps it as {success: False, result: {error: "..."}}. The inner dict
    # holds the actual error message.
    outer = tool_result_events[0]["data"]["result"]
    inner = outer.get("result", {})
    assert inner.get("error") == "Plan mode active - write_file is disabled."

    # The model (second adapter call) must have seen the correct error text in
    # the tool-role message content — not "Unknown error".
    second_call_messages = adapter.calls[1]["messages"]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "Plan mode active" in tool_msg["content"]
    assert "Unknown error" not in tool_msg["content"]


# ---------------------------------------------------------------------------
# Bug #203: compact_messages must not inject a second system message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_single_system_message() -> None:
    """After compaction the result must have exactly one system message.

    Regression for bug #203 where _compact_messages() inserted a second
    role='system' entry as the conversation summary, causing dual system
    messages that violate Anthropic's API contract.
    """
    fake_summary = "Step 1: read file. Step 2: wrote output."

    class FakeCompactionAdapter(ModelAdapter):
        @property
        def model_name(self) -> str:
            return "fake/compactor"

        async def chat_with_tools(self, messages, tools=None, **kwargs):
            return {"content": fake_summary, "tool_calls": [], "usage": {}, "finish_reason": "stop"}

    agent = TesslateAgent(
        system_prompt="You are a coding agent.",
        tools=None,
        model=None,
        compaction_adapter=FakeCompactionAdapter(),
    )

    # Build a messages list that is long enough to trigger compaction.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent."},
    ]
    for i in range(10):
        messages.append({"role": "user", "content": f"user turn {i}"})
        messages.append({"role": "assistant", "content": f"assistant turn {i}"})

    compacted = await agent._compact_messages(messages)

    system_msgs = [m for m in compacted if m.get("role") == "system"]
    assert len(system_msgs) == 1, (
        f"Expected exactly 1 system message after compaction, got {len(system_msgs)}: "
        f"{system_msgs}"
    )
    # Summary text is merged into the original system message, not a new one.
    assert fake_summary in system_msgs[0]["content"]
    assert "You are a coding agent." in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_compact_messages_no_op_when_few_messages() -> None:
    """_compact_messages is a no-op when the history is short."""
    agent = TesslateAgent(system_prompt="sys", tools=None, model=None)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    result = await agent._compact_messages(messages)
    assert result is messages  # same object, not copied


@pytest.mark.asyncio
async def test_compact_messages_preserves_tail_verbatim() -> None:
    """The most recent 6 messages are always kept verbatim after compaction."""
    fake_summary = "compressed history"

    class FakeCompactionAdapter(ModelAdapter):
        @property
        def model_name(self) -> str:
            return "fake/compactor"

        async def chat_with_tools(self, messages, tools=None, **kwargs):
            return {
                "content": fake_summary,
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            }

    agent = TesslateAgent(
        system_prompt="sys",
        tools=None,
        model=None,
        compaction_adapter=FakeCompactionAdapter(),
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    compacted = await agent._compact_messages(messages)
    tail = messages[-6:]
    assert compacted[-6:] == tail

@pytest.mark.asyncio
async def test_pasted_text_attachment_reaches_model() -> None:
    """Empty typed message + pasted_text attachment must still surface to the LLM."""
    adapter = FakeAdapter(
        responses=[
            {
                "content": "Saw the paste.",
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            }
        ]
    )
    agent = TesslateAgent(system_prompt="sys", tools=ToolRegistry(), model=adapter)

    context = {
        "attachments": [
            {
                "type": "pasted_text",
                "content": "Build Error: Unexpected token at line 113",
                "label": "Pasted text (49 lines)",
            }
        ]
    }

    await _collect_events_with_context(agent, "", context)

    user_msg = adapter.calls[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert "Pasted text (49 lines)" in user_msg["content"]
    assert "Unexpected token at line 113" in user_msg["content"]


@pytest.mark.asyncio
async def test_image_attachment_uses_vision_parts() -> None:
    """Image attachments become OpenAI vision content-parts alongside text."""
    adapter = FakeAdapter(
        responses=[
            {
                "content": "ok",
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            }
        ]
    )
    agent = TesslateAgent(system_prompt="sys", tools=ToolRegistry(), model=adapter)

    context = {
        "attachments": [
            {"type": "image", "content": "AAAA", "mime_type": "image/jpeg"},
        ]
    }

    await _collect_events_with_context(agent, "what's this?", context)

    user_msg = adapter.calls[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    parts_by_type = {p.get("type"): p for p in user_msg["content"]}
    assert parts_by_type["text"]["text"] == "what's this?"
    assert parts_by_type["image_url"]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,AAAA"
    )


@pytest.mark.asyncio
async def test_oversized_pasted_text_is_truncated_not_dropped() -> None:
    """Pastes exceeding the cap are truncated with a marker, not silently dropped."""
    cap = TesslateAgent._MAX_PASTED_TEXT_CHARS
    oversize = "x" * (cap + 500)

    adapter = FakeAdapter(
        responses=[
            {"content": "ok", "tool_calls": [], "usage": {}, "finish_reason": "stop"}
        ]
    )
    agent = TesslateAgent(system_prompt="sys", tools=ToolRegistry(), model=adapter)

    context = {
        "attachments": [
            {"type": "pasted_text", "content": oversize, "label": "huge paste"}
        ]
    }

    await _collect_events_with_context(agent, "", context)

    user_msg = adapter.calls[0]["messages"][-1]
    content = user_msg["content"]
    assert "500 more chars" in content
    # truncated body + marker must be smaller than the raw oversize
    assert len(content) < len(oversize)


@pytest.mark.asyncio
async def test_no_attachments_preserves_string_content() -> None:
    """Without attachments the user turn is the bare string — no shape drift."""
    adapter = FakeAdapter(
        responses=[
            {
                "content": "hi",
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            }
        ]
    )
    agent = TesslateAgent(system_prompt="sys", tools=ToolRegistry(), model=adapter)

    await _collect_events(agent, "hello")

    user_msg = adapter.calls[0]["messages"][-1]
    assert user_msg == {"role": "user", "content": "hello"}
