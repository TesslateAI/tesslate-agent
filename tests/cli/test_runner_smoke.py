"""
Integration smoke tests for the CLI runner.

These tests monkey-patch ``TesslateAgent`` with a scripted fake so the
runner can exercise its full event dispatch, trajectory recording, and
atomic-write paths without touching a real LLM.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest


# Environment variables that need to be unset to force a genuine
# "no credentials configured" state for the missing-key test case.
_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "FIREWORKS_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "PERPLEXITYAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "LITELLM_API_BASE",
    "LITELLM_MASTER_KEY",
)


class _FakeAgent:
    """
    Scripted replacement for :class:`TesslateAgent` used in smoke tests.

    The fake agent ignores the model adapter entirely. On each call to
    :meth:`run` it emits a single ``agent_step`` event carrying one
    ``write_file`` tool call, actually executes that tool against the
    live tool registry (so the file lands on disk under
    ``PROJECT_ROOT``), then yields a ``complete`` event.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: Any = None,
        model: Any = None,
        **_: Any,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model

    async def run(self, task: str, context: dict[str, Any]):
        del task  # unused
        tool_call = {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": "output.txt", "content": "hello"}
                ),
            },
        }

        # Execute the tool call against the real registry so the
        # write actually lands on disk. This mirrors what TesslateAgent
        # would do during a normal iteration.
        exec_context = dict(context)
        exec_context.setdefault("edit_mode", "auto")
        result = await self.tools.execute(
            "write_file",
            {"file_path": "output.txt", "content": "hello"},
            exec_context,
        )

        # Streaming-style per-call event (consumed by live UIs, ignored
        # by the trajectory bridge).
        yield {
            "type": "tool_result",
            "data": {
                "iteration": 1,
                "index": 0,
                "total": 1,
                "name": "write_file",
                "parameters": {"file_path": "output.txt", "content": "hello"},
                "result": result,
            },
        }

        # Authoritative agent_step with the executed tool_results — this
        # is what the bridge records into the trajectory.
        yield {
            "type": "agent_step",
            "data": {
                "iteration": 1,
                "response_text": "I'll write a file",
                "tool_calls": [tool_call],
                "tool_results": [result],
            },
        }

        yield {
            "type": "complete",
            "data": {
                "success": True,
                "iterations": 1,
                "final_response": "Done.",
                "tool_calls_made": 1,
                "completion_reason": "stop",
            },
        }


@pytest.mark.asyncio
async def test_run_agent_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tesslate_agent.agent import tesslate_agent as agent_module
    from tesslate_agent.cli.runner import run_agent
    from tesslate_agent.orchestration.factory import OrchestratorFactory

    # Credential check for openai/* models requires OPENAI_API_KEY.
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    # Drop any cached orchestrator from previous tests so LocalOrchestrator
    # picks up the new PROJECT_ROOT.
    OrchestratorFactory.clear_cache()

    monkeypatch.setattr(agent_module, "TesslateAgent", _FakeAgent)

    output = tmp_path / "traj.json"

    code = await run_agent(
        task="write hello.txt",
        model="openai/gpt-4o-mini",
        workdir=tmp_path,
        output=output,
    )

    assert code == 0
    produced = tmp_path / "output.txt"
    assert produced.exists(), "fake agent should have written output.txt via write_file"
    assert produced.read_text() == "hello"

    assert output.exists(), "trajectory file should always be written"
    loaded = json.loads(output.read_text())
    assert loaded["schema_version"] == "ATIF-v1.4"
    assert "final_metrics" in loaded
    assert loaded["steps"], "trajectory should contain at least one step"

    # Regression: every observation result must reference a real tool
    # call in its parent step. Empty source_call_ids broke ATIF strict
    # validation in earlier versions because the trajectory bridge
    # double-recorded streaming `tool_result` events.
    for step in loaded["steps"]:
        observation = step.get("observation") or {}
        results = observation.get("results", [])
        if not results:
            continue
        call_ids = {tc.get("tool_call_id") for tc in step.get("tool_calls", [])}
        for result in results:
            source = result.get("source_call_id")
            assert source, (
                f"step {step['step_id']} has observation result with empty "
                f"source_call_id: {result}"
            )
            assert source in call_ids, (
                f"step {step['step_id']} observation references "
                f"source_call_id={source!r} not present in tool_calls={call_ids}"
            )

    # Restore cache state for downstream tests.
    OrchestratorFactory.clear_cache()


@pytest.mark.asyncio
async def test_run_agent_missing_key_writes_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tesslate_agent.cli.runner import run_agent
    from tesslate_agent.orchestration.factory import OrchestratorFactory

    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    OrchestratorFactory.clear_cache()

    output = tmp_path / "traj.json"

    code = await run_agent(
        task="unused",
        model="openai/gpt-4o-mini",
        workdir=tmp_path,
        output=output,
    )

    assert code == 2, f"expected exit code 2 for missing key, got {code}"
    assert output.exists(), "trajectory should still be written when credentials are missing"

    loaded = json.loads(output.read_text())
    assert loaded["schema_version"] == "ATIF-v1.4"
    # The error assistant turn should be present as a step.
    step_messages = [
        str(step.get("message", ""))
        for step in loaded.get("steps", [])
        if step.get("source") == "agent"
    ]
    assert any("[error]" in msg for msg in step_messages), (
        f"expected an agent error step, got: {step_messages}"
    )

    OrchestratorFactory.clear_cache()


def test_cli_tools_list_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tesslate_agent.cli.__main__ import main

    monkeypatch.setattr(sys, "argv", ["tesslate-agent", "tools", "list"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) >= 30, f"expected at least 30 tools, got {len(lines)}"
    names = {line.split("\t", 1)[0] for line in lines}
    assert "read_file" in names, f"read_file missing from tools list: {sorted(names)[:10]}..."


def test_cli_version_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tesslate_agent.cli.__main__ import main

    monkeypatch.setattr(sys, "argv", ["tesslate-agent", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "0.1.0" in captured.out
