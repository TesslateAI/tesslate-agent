"""
Tests for the delegation_ops subagent tools.

These tests exercise the standalone subagent-orchestration primitives
without requiring network access or real litellm calls. A lightweight
``FakeAgent`` stand-in for :class:`TesslateAgent` is monkeypatched into
``tesslate_agent.agent.tesslate_agent`` so ``task_executor`` can spawn
and drive it deterministically.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

from tesslate_agent.agent.tools.delegation_ops import (
    MAX_SUBAGENT_DEPTH,
    SUBAGENT_REGISTRY,
    SubagentRecord,
    SubagentRegistry,
    register_delegation_ops_tools,
)
from tesslate_agent.agent.tools.delegation_ops import task_tool as delegation_task_tool
from tesslate_agent.agent.tools.delegation_ops.agent_registry import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from tesslate_agent.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Minimal stand-in for the LiteLLM adapter interface."""

    def __init__(self, model_name: str = "fake/test-model") -> None:
        self.model_name = model_name
        self.thinking_effort: str | None = None


class FakeAgent:
    """
    Deterministic FakeAgent used in place of ``TesslateAgent``.

    Behaviour is driven by the ``prompt`` string contents to keep the
    tests self-contained:

    - "emit-events" → yields three agent_step events then completes
    - "fail-mid"    → yields one event then raises ``RuntimeError``
    - "slow:<ms>"   → sleeps ``ms`` milliseconds then completes
    - otherwise     → single complete event
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: ToolRegistry,
        model: Any,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model

    async def run(self, prompt: str, context: dict[str, Any]):
        if "fail-mid" in prompt:
            yield {
                "type": "agent_step",
                "data": {
                    "iteration": 1,
                    "response_text": "about to fail",
                    "tool_calls": [],
                },
            }
            raise RuntimeError("boom from FakeAgent")

        if "emit-events" in prompt:
            yield {
                "type": "agent_step",
                "data": {
                    "iteration": 1,
                    "response_text": "step-1",
                    "tool_calls": [
                        {"name": "pretend_tool", "parameters": {"a": 1}},
                    ],
                },
            }
            yield {
                "type": "tool_call",
                "data": {
                    "iteration": 1,
                    "index": 0,
                    "result": {"ok": True, "value": 42},
                },
            }
            yield {
                "type": "agent_step",
                "data": {
                    "iteration": 2,
                    "response_text": "step-2",
                    "tool_calls": [],
                },
            }
            yield {
                "type": "complete",
                "data": {
                    "iteration": 3,
                    "final_response": "done-after-events",
                },
            }
            return

        if prompt.startswith("slow:"):
            try:
                ms = int(prompt.split(":", 1)[1])
            except ValueError:
                ms = 1000
            await asyncio.sleep(ms / 1000.0)
            yield {
                "type": "complete",
                "data": {
                    "iteration": 1,
                    "final_response": f"slept-{ms}ms",
                },
            }
            return

        # Default path: single complete event.
        yield {
            "type": "complete",
            "data": {
                "iteration": 1,
                "final_response": f"ok:{prompt}",
            },
        }


@pytest.fixture(autouse=True)
def _reset_registry():
    """Wipe the module-global registry between tests."""
    SUBAGENT_REGISTRY.clear()
    yield
    SUBAGENT_REGISTRY.clear()


@pytest.fixture
def fake_tesslate_agent(monkeypatch):
    """
    Install ``FakeAgent`` as :class:`TesslateAgent` so the delegation
    runner's late ``from tesslate_agent.agent.tesslate_agent import
    TesslateAgent`` resolves to our stand-in regardless of whether the
    real class has been shipped yet.
    """
    mod_name = "tesslate_agent.agent.tesslate_agent"
    fake_mod = types.ModuleType(mod_name)
    fake_mod.TesslateAgent = FakeAgent
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    return FakeAgent


@pytest.fixture
def parent_context() -> dict[str, Any]:
    """A minimal parent-agent run context."""
    return {
        "agent_id": "parent-1",
        "model_adapter": FakeAdapter(),
        "subagent_depth": 0,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_delegation_ops_tools_exposes_five_tools():
    registry = ToolRegistry()
    register_delegation_ops_tools(registry)
    names = set(registry.list_names())
    assert {
        "task",
        "wait_agent",
        "send_message_to_agent",
        "close_agent",
        "list_agents",
    } <= names


# ---------------------------------------------------------------------------
# task executor
# ---------------------------------------------------------------------------


async def test_task_wait_false_returns_agent_id_immediately(
    fake_tesslate_agent, parent_context
):
    result = await delegation_task_tool.task_executor(
        {
            "role": "explore",
            "prompt": "slow:150",
            "wait": False,
        },
        parent_context,
    )
    assert result["success"] is True
    assert result["status"] == STATUS_RUNNING
    assert "agent_id" in result
    assert result["depth"] == 1

    # The background task is still in flight — wait for it so our cleanup
    # teardown does not yell about unawaited coroutines.
    rec = SUBAGENT_REGISTRY.get(result["agent_id"])
    assert rec is not None
    if rec.task is not None:
        await rec.task
    assert rec.status == STATUS_COMPLETED
    assert rec.final_response == "slept-150ms"


async def test_task_wait_true_returns_final_response_and_trajectory(
    fake_tesslate_agent, parent_context
):
    result = await delegation_task_tool.task_executor(
        {
            "role": "summariser",
            "prompt": "emit-events",
            "wait": True,
            "timeout_ms": 5_000,
        },
        parent_context,
    )
    assert result["success"] is True
    assert result["status"] == STATUS_COMPLETED
    assert result["final_response"] == "done-after-events"

    trajectory = result["trajectory"]
    assert isinstance(trajectory, dict)
    assert trajectory.get("schema_version")
    # Ensure trajectory is JSON-serializable.
    json.dumps(trajectory)


async def test_task_depth_cap_rejects_beyond_max(
    fake_tesslate_agent, parent_context
):
    parent_context["subagent_depth"] = MAX_SUBAGENT_DEPTH
    result = await delegation_task_tool.task_executor(
        {
            "role": "deep",
            "prompt": "hello",
            "wait": False,
        },
        parent_context,
    )
    assert result["success"] is False
    assert "max" in result["message"].lower() or "depth" in result["message"].lower()
    details = result.get("details") or {}
    assert details.get("max_depth") == MAX_SUBAGENT_DEPTH
    assert details.get("parent_depth") == MAX_SUBAGENT_DEPTH


async def test_task_recursion_guard_excludes_delegation_tools_from_child_registry(
    fake_tesslate_agent, monkeypatch, parent_context
):
    # Pre-populate the GLOBAL registry with the delegation tools so the
    # guard filter has something to strip out.
    from tesslate_agent.agent.tools.registry import get_tool_registry

    global_registry = get_tool_registry()
    # Stash and restore original state around the test.
    original_tools = dict(global_registry._tools)
    try:
        global_registry._tools.clear()
        register_delegation_ops_tools(global_registry)

        scoped = delegation_task_tool._resolve_tool_registry(None)
        child_names = set(scoped.list_names())
        assert not ({
            "task",
            "wait_agent",
            "send_message_to_agent",
            "close_agent",
            "list_agents",
        } & child_names), (
            f"child registry leaked delegation tools: {child_names}"
        )

        # Explicit allowlist containing delegation tools must also be stripped.
        scoped2 = delegation_task_tool._resolve_tool_registry(
            ["task", "wait_agent", "list_agents"]
        )
        assert set(scoped2.list_names()) == set()
    finally:
        global_registry._tools.clear()
        global_registry._tools.update(original_tools)


# ---------------------------------------------------------------------------
# wait_agent executor
# ---------------------------------------------------------------------------


async def test_wait_agent_unknown_agent_id_returns_error(parent_context):
    result = await delegation_task_tool.wait_agent_executor(
        {"agent_id": "does-not-exist"}, parent_context
    )
    assert result["success"] is False
    assert "unknown" in result["message"].lower()


async def test_wait_agent_blocks_until_complete(
    fake_tesslate_agent, parent_context
):
    spawn_result = await delegation_task_tool.task_executor(
        {"role": "wait-me", "prompt": "slow:200", "wait": False},
        parent_context,
    )
    agent_id = spawn_result["agent_id"]

    wait_result = await delegation_task_tool.wait_agent_executor(
        {"agent_id": agent_id, "timeout_ms": 5_000},
        parent_context,
    )
    assert wait_result["success"] is True
    assert wait_result["status"] == STATUS_COMPLETED
    assert wait_result["final_response"] == "slept-200ms"


async def test_wait_agent_short_timeout_returns_still_running(
    fake_tesslate_agent, parent_context
):
    spawn_result = await delegation_task_tool.task_executor(
        {"role": "slowpoke", "prompt": "slow:800", "wait": False},
        parent_context,
    )
    agent_id = spawn_result["agent_id"]

    wait_result = await delegation_task_tool.wait_agent_executor(
        {"agent_id": agent_id, "timeout_ms": 50},
        parent_context,
    )
    assert wait_result["success"] is True
    assert wait_result["status"] == "still_running"

    # The subagent must still be alive and then allowed to finish cleanly.
    rec = SUBAGENT_REGISTRY.get(agent_id)
    assert rec is not None
    assert rec.task is not None
    await rec.task
    assert rec.status == STATUS_COMPLETED


# ---------------------------------------------------------------------------
# send_message_to_agent executor
# ---------------------------------------------------------------------------


async def test_send_message_enqueues_into_pending_messages(
    fake_tesslate_agent, parent_context
):
    spawn_result = await delegation_task_tool.task_executor(
        {"role": "listener", "prompt": "slow:400", "wait": False},
        parent_context,
    )
    agent_id = spawn_result["agent_id"]

    send_result = await delegation_task_tool.send_message_to_agent_executor(
        {"agent_id": agent_id, "message": "hello from parent"},
        parent_context,
    )
    assert send_result["success"] is True
    assert send_result["queued"] is True
    assert send_result["queue_depth"] >= 1

    # The runner may drain the queue between ticks, but there must be at
    # least one event where the message sat in the pending queue. Ensure
    # the task runs to completion without error.
    rec = SUBAGENT_REGISTRY.get(agent_id)
    assert rec is not None
    if rec.task is not None:
        await rec.task
    assert rec.status == STATUS_COMPLETED


async def test_send_message_unknown_agent_returns_error(parent_context):
    result = await delegation_task_tool.send_message_to_agent_executor(
        {"agent_id": "nope", "message": "hi"}, parent_context
    )
    assert result["success"] is False
    assert "unknown" in result["message"].lower()


# ---------------------------------------------------------------------------
# close_agent executor
# ---------------------------------------------------------------------------


async def test_close_agent_cancels_running_subagent(
    fake_tesslate_agent, parent_context
):
    spawn_result = await delegation_task_tool.task_executor(
        {"role": "to-cancel", "prompt": "slow:5000", "wait": False},
        parent_context,
    )
    agent_id = spawn_result["agent_id"]

    close_result = await delegation_task_tool.close_agent_executor(
        {"agent_id": agent_id}, parent_context
    )
    assert close_result["success"] is True
    assert close_result["status"] == STATUS_CANCELLED

    rec = SUBAGENT_REGISTRY.get(agent_id)
    assert rec is not None
    assert rec.status == STATUS_CANCELLED


async def test_close_agent_is_idempotent(
    fake_tesslate_agent, parent_context
):
    spawn_result = await delegation_task_tool.task_executor(
        {"role": "to-cancel", "prompt": "slow:5000", "wait": False},
        parent_context,
    )
    agent_id = spawn_result["agent_id"]

    first = await delegation_task_tool.close_agent_executor(
        {"agent_id": agent_id}, parent_context
    )
    assert first["success"] is True

    second = await delegation_task_tool.close_agent_executor(
        {"agent_id": agent_id}, parent_context
    )
    assert second["success"] is True
    # Second call sees a terminal record — message should reflect that.
    assert "terminal" in second["message"].lower()


# ---------------------------------------------------------------------------
# list_agents executor
# ---------------------------------------------------------------------------


async def test_list_agents_filters_by_status_and_parent(
    fake_tesslate_agent, parent_context
):
    # Spawn one quick, one slow, both parented to parent-1.
    quick = await delegation_task_tool.task_executor(
        {"role": "quick", "prompt": "hello", "wait": True, "timeout_ms": 5000},
        parent_context,
    )
    slow = await delegation_task_tool.task_executor(
        {"role": "slow", "prompt": "slow:5000", "wait": False},
        parent_context,
    )
    assert quick["success"] is True
    assert slow["success"] is True

    # Spawn a third parented to a DIFFERENT parent so the filter has work.
    other_context = {
        "agent_id": "parent-2",
        "model_adapter": FakeAdapter(),
        "subagent_depth": 0,
    }
    other = await delegation_task_tool.task_executor(
        {"role": "other", "prompt": "hello", "wait": True, "timeout_ms": 5000},
        other_context,
    )
    assert other["success"] is True

    # Filter by parent_agent_id.
    res_parent1 = await delegation_task_tool.list_agents_executor(
        {"parent_agent_id": "parent-1"}, parent_context
    )
    assert res_parent1["success"] is True
    parent1_ids = {a["agent_id"] for a in res_parent1["agents"]}
    assert quick["agent_id"] in parent1_ids
    assert slow["agent_id"] in parent1_ids
    assert other["agent_id"] not in parent1_ids

    # Filter by status=running — only the slow one should remain.
    res_running = await delegation_task_tool.list_agents_executor(
        {"status": STATUS_RUNNING}, parent_context
    )
    running_ids = {a["agent_id"] for a in res_running["agents"]}
    assert slow["agent_id"] in running_ids
    assert quick["agent_id"] not in running_ids

    # Clean up the still-running slow agent so teardown is clean.
    await delegation_task_tool.close_agent_executor(
        {"agent_id": slow["agent_id"]}, parent_context
    )


# ---------------------------------------------------------------------------
# Error propagation + event buffering
# ---------------------------------------------------------------------------


async def test_child_exception_marks_subagent_failed(
    fake_tesslate_agent, parent_context
):
    result = await delegation_task_tool.task_executor(
        {
            "role": "exploder",
            "prompt": "fail-mid",
            "wait": True,
            "timeout_ms": 5_000,
        },
        parent_context,
    )
    assert result["success"] is True
    assert result["status"] == STATUS_FAILED
    assert result["error"]
    assert "boom" in result["error"]


async def test_events_buffered_in_order_on_registry(
    fake_tesslate_agent, parent_context
):
    spawn = await delegation_task_tool.task_executor(
        {
            "role": "emitter",
            "prompt": "emit-events",
            "wait": True,
            "timeout_ms": 5_000,
        },
        parent_context,
    )
    rec = SUBAGENT_REGISTRY.get(spawn["agent_id"])
    assert rec is not None
    types_seen = [e.get("type") for e in rec.events]
    # Must preserve the order the FakeAgent yielded.
    assert types_seen == ["agent_step", "tool_call", "agent_step", "complete"]


async def test_trajectory_is_json_serializable(
    fake_tesslate_agent, parent_context
):
    result = await delegation_task_tool.task_executor(
        {
            "role": "jsonable",
            "prompt": "emit-events",
            "wait": True,
            "timeout_ms": 5_000,
        },
        parent_context,
    )
    trajectory = result["trajectory"]
    # Will raise if there's any non-serializable content.
    encoded = json.dumps(trajectory)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


# ---------------------------------------------------------------------------
# SubagentRegistry unit-level sanity
# ---------------------------------------------------------------------------


async def test_subagent_registry_snapshot_excludes_task_handle():
    reg = SubagentRegistry()
    from datetime import UTC, datetime

    record = SubagentRecord(
        agent_id="a1",
        role="r",
        status="pending",
        spawned_at=datetime.now(UTC),
        task_text="t",
        model_name="m",
        depth=1,
    )
    await reg.register(record)
    snap = reg.snapshot_for_listing()
    assert len(snap) == 1
    assert "task" not in snap[0]
    # Snapshot must be JSON-serializable.
    json.dumps(snap)
