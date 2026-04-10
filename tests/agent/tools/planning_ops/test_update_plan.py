"""Integration tests for the ``update_plan`` planning tool."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tesslate_agent.agent.tools.planning_ops.update_plan import (
    PLAN_MIRROR_PATH,
    PLAN_STORE,
    update_plan_tool,
)
from tesslate_agent.orchestration import (
    DeploymentMode,
    LocalOrchestrator,
    OrchestratorFactory,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    return tmp_path


@pytest.fixture
def bound_orchestrator(project_root: Path) -> LocalOrchestrator:
    OrchestratorFactory.clear_cache()
    orchestrator = LocalOrchestrator()
    for mode in DeploymentMode:
        OrchestratorFactory._instances[mode] = orchestrator
    yield orchestrator
    OrchestratorFactory.clear_cache()


@pytest.fixture
async def fresh_plan_store():
    """Ensure PLAN_STORE starts empty and is cleared between tests."""
    PLAN_STORE._states.clear()
    yield
    PLAN_STORE._states.clear()


def _ctx(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "user_id": uuid4(),
        "project_id": uuid4(),
        "project_slug": "test-project",
        "container_name": None,
        "container_directory": None,
    }


async def test_empty_plan_rejected(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx = _ctx("run-empty")
    result = await update_plan_tool({"plan": []}, ctx)
    assert result["success"] is False
    assert "empty" in result["message"].lower()


async def test_missing_plan_rejected(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx = _ctx("run-missing")
    result = await update_plan_tool({}, ctx)
    assert result["success"] is False
    assert "plan" in result["message"].lower()


async def test_non_list_plan_rejected(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx = _ctx("run-wrong-type")
    result = await update_plan_tool({"plan": "just a string"}, ctx)
    assert result["success"] is False
    assert "array" in result["message"].lower() or "list" in result["message"].lower()


async def test_invalid_status_rejected(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx = _ctx("run-bad-status")
    result = await update_plan_tool(
        {
            "plan": [
                {"step": "Do thing", "status": "not-a-real-status"},
            ]
        },
        ctx,
    )
    assert result["success"] is False
    assert "status" in result["message"].lower()


async def test_missing_step_text_rejected(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx = _ctx("run-missing-text")
    result = await update_plan_tool(
        {"plan": [{"status": "pending"}]}, ctx
    )
    assert result["success"] is False
    assert "step" in result["message"].lower()


async def test_happy_path_populates_store_and_writes_mirror(
    bound_orchestrator,
    project_root: Path,
    fresh_plan_store,
) -> None:
    ctx = _ctx("run-happy")

    result = await update_plan_tool(
        {
            "plan": [
                {"step": "Inspect failing test", "status": "in_progress"},
                {"step": "Patch regression", "status": "pending"},
                {"step": "Re-run suite", "status": "pending"},
            ],
            "reasoning": "Break the fix into verifiable steps",
        },
        ctx,
    )

    assert result["success"] is True
    assert result["run_id"] == "run-happy"
    assert len(result["plan"]) == 3
    assert result["plan"][0]["index"] == 0
    assert result["plan"][0]["status"] == "in_progress"
    assert result["mirror_path"] == PLAN_MIRROR_PATH
    assert result["details"]["step_count"] == 3
    assert result["details"]["status_counts"]["in_progress"] == 1
    assert result["details"]["status_counts"]["pending"] == 2
    assert result["details"]["mirror_written"] is True

    # Store now has the plan.
    state = await PLAN_STORE.get("run-happy")
    assert state is not None
    assert len(state.plan) == 3
    assert state.reasoning == "Break the fix into verifiable steps"

    # Mirror file written to <project_root>/.tesslate/plan.json.
    mirror = project_root / PLAN_MIRROR_PATH
    assert mirror.exists()
    payload = json.loads(mirror.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-happy"
    assert len(payload["plan"]) == 3
    assert payload["reasoning"] == "Break the fix into verifiable steps"


async def test_second_call_replaces_previous_plan(
    bound_orchestrator,
    project_root: Path,
    fresh_plan_store,
) -> None:
    ctx = _ctx("run-replace")

    await update_plan_tool(
        {
            "plan": [
                {"step": "Original step 1", "status": "pending"},
                {"step": "Original step 2", "status": "pending"},
            ],
        },
        ctx,
    )

    result = await update_plan_tool(
        {
            "plan": [
                {"step": "Only step", "status": "completed"},
            ],
        },
        ctx,
    )

    assert result["success"] is True
    assert len(result["plan"]) == 1
    assert result["plan"][0]["step"] == "Only step"

    state = await PLAN_STORE.get("run-replace")
    assert state is not None
    assert len(state.plan) == 1

    # Mirror reflects the replacement.
    mirror = project_root / PLAN_MIRROR_PATH
    payload = json.loads(mirror.read_text(encoding="utf-8"))
    assert len(payload["plan"]) == 1
    assert payload["plan"][0]["step"] == "Only step"


async def test_event_sink_async_callable_receives_update(
    bound_orchestrator, fresh_plan_store
) -> None:
    received: list[dict] = []

    async def sink(event: dict) -> None:
        received.append(event)

    ctx = _ctx("run-event")
    ctx["event_sink"] = sink

    await update_plan_tool(
        {
            "plan": [
                {"step": "Step 1", "status": "pending"},
            ],
        },
        ctx,
    )

    assert len(received) == 1
    assert received[0]["type"] == "plan_update"
    assert received[0]["data"]["run_id"] == "run-event"
    assert len(received[0]["data"]["plan"]) == 1


async def test_concurrent_run_ids_are_isolated(
    bound_orchestrator, fresh_plan_store
) -> None:
    ctx_a = _ctx("run-a")
    ctx_b = _ctx("run-b")

    await asyncio.gather(
        update_plan_tool(
            {
                "plan": [
                    {"step": "A step", "status": "in_progress"},
                ],
            },
            ctx_a,
        ),
        update_plan_tool(
            {
                "plan": [
                    {"step": "B step 1", "status": "completed"},
                    {"step": "B step 2", "status": "pending"},
                ],
            },
            ctx_b,
        ),
    )

    state_a = await PLAN_STORE.get("run-a")
    state_b = await PLAN_STORE.get("run-b")

    assert state_a is not None
    assert state_b is not None
    assert len(state_a.plan) == 1
    assert len(state_b.plan) == 2
    assert state_a.plan[0].step == "A step"
    assert state_b.plan[0].step == "B step 1"
