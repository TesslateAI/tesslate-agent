"""
Standalone agent runner used by the Tesslate Agent CLI.

Exposes a single :func:`run_agent` coroutine that builds a local
orchestration context, instantiates a :class:`TesslateAgent`, drives
the agent loop until completion (or timeout / error), and always
writes an ATIF v1.4 trajectory to disk before returning.

The runner performs no I/O on behalf of the agent's tools; tools reach
the filesystem through the ``local`` orchestrator, which is why the
function pins ``DEPLOYMENT_MODE=local`` and ``PROJECT_ROOT=<workdir>``
before importing anything that might cache an orchestrator instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = ["run_agent"]


EXIT_SUCCESS = 0
EXIT_AGENT_ERROR = 1
EXIT_MISSING_KEY = 2


async def run_agent(
    task: str,
    model: str,
    workdir: Path,
    output: Path,
    *,
    max_iterations: int = 0,
    effort: str = "medium",
    tool_names: list[str] | None = None,
    timeout_ms: int = 900_000,
    system_prompt: str | None = None,
    event_printer: Callable[[dict[str, Any]], None] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> int:
    """
    Drive :class:`TesslateAgent` against a local working directory.

    Args:
        task: The natural-language task for the agent to execute.
        model: LiteLLM-style model identifier (e.g. ``"openai/gpt-4o-mini"``).
        workdir: Directory the agent should treat as ``PROJECT_ROOT``.
        output: Path to write the ATIF v1.4 trajectory JSON to.
        max_iterations: Hard cap on the agent loop. Default 0 = no cap;
            the agent runs until it terminates on its own. Pass any
            positive integer to bound the loop.
        effort: Extended-thinking effort tier for supporting models
            (``"low"`` / ``"medium"`` / ``"high"``).
        tool_names: Optional list of tool names to scope the registry
            to. When ``None``, the full built-in tool set is used.
        timeout_ms: Wall-clock timeout for the agent loop in milliseconds.
        system_prompt: Optional override for the base system prompt.
            The project memory preamble is still appended when present.
        event_printer: Optional sink invoked for every agent event.
            Used by the CLI to render live progress.
        api_base: Optional LiteLLM proxy base URL
            (overrides ``LITELLM_API_BASE``).
        api_key: Optional LiteLLM proxy master key
            (overrides ``LITELLM_MASTER_KEY``).

    Returns:
        Process exit code: ``0`` on success, ``1`` on agent error /
        timeout, ``2`` when the requested model has no credentials.
    """
    resolved_workdir = workdir.resolve()
    resolved_output = output.resolve()

    # Pin environment BEFORE importing anything that caches an
    # orchestrator. The local orchestrator reads PROJECT_ROOT at
    # instantiation time.
    os.environ["DEPLOYMENT_MODE"] = "local"
    os.environ["PROJECT_ROOT"] = str(resolved_workdir)

    # Lazy imports so the environment variables above take effect
    # before module-level orchestrator caches are built.
    from tesslate_agent.agent import tesslate_agent as _agent_module
    from tesslate_agent.agent.models import MissingApiKeyError, create_model_adapter
    from tesslate_agent.agent.tools.memory_ops import load_memory_prefix
    from tesslate_agent.agent.tools.registry import (
        create_scoped_tool_registry,
        get_tool_registry,
    )
    from tesslate_agent.agent.trajectory import TrajectoryRecorder
    from tesslate_agent.cli.context import StubUser, make_standalone_context
    from tesslate_agent.cli.prompts import DEFAULT_BENCHMARK_SYSTEM_PROMPT
    from tesslate_agent.cli.trajectory_bridge import TrajectoryBridge
    from tesslate_agent.orchestration.factory import OrchestratorFactory

    # Drop any cached orchestrator so the new PROJECT_ROOT is picked up.
    OrchestratorFactory.clear_cache()

    run_id = uuid4().hex
    recorder = TrajectoryRecorder(session_id=run_id, model_name=model)

    # Resolve model adapter up-front so we can fail fast on missing
    # credentials without writing half-initialised state.
    try:
        adapter = await create_model_adapter(
            model,
            api_base=api_base,
            api_key=api_key,
            thinking_effort=effort,
        )
    except MissingApiKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Set the indicated environment variable, or pass "
            "--api-base and --api-key to route through a LiteLLM proxy.",
            file=sys.stderr,
        )
        # Still emit a minimal trajectory so callers always get a file.
        recorder.record_system(system_prompt or DEFAULT_BENCHMARK_SYSTEM_PROMPT)
        recorder.record_user(task)
        recorder.record_assistant(content=f"[error] {exc}")
        _write_trajectory(resolved_output, recorder.to_atif())
        return EXIT_MISSING_KEY

    # Resolve the tool registry.
    if tool_names is None:
        tool_registry = get_tool_registry()
    else:
        # Trigger lazy population of the global registry so scoped
        # lookups find the built-in tools.
        get_tool_registry()
        tool_registry = create_scoped_tool_registry(tool_names)

    # Build the system prompt.
    base_prompt = system_prompt or DEFAULT_BENCHMARK_SYSTEM_PROMPT
    memory_prefix = load_memory_prefix(resolved_workdir)
    if memory_prefix:
        sys_prompt = f"{base_prompt}\n\n{memory_prefix}"
    else:
        sys_prompt = base_prompt

    # Construct the agent. We read TesslateAgent off the module so
    # monkeypatches applied to ``tesslate_agent.agent.tesslate_agent.TesslateAgent``
    # are honoured by the runner.
    TesslateAgentCls = _agent_module.TesslateAgent
    agent = TesslateAgentCls(
        system_prompt=sys_prompt,
        tools=tool_registry,
        model=adapter,
        max_iterations=max_iterations,
    )

    context = make_standalone_context(
        StubUser(),
        resolved_workdir,
        adapter,
        run_id=run_id,
    )
    # Auto-approve all tool calls in standalone mode — there is no
    # human to approve each destructive action.
    context["edit_mode"] = "auto"

    bridge = TrajectoryBridge(recorder, resolved_workdir)

    # Seed the recorder with the system prompt + user task so the
    # trajectory is self-describing even if the agent errors out on
    # the very first iteration.
    recorder.record_system(sys_prompt)
    recorder.record_user(task)

    timeout_seconds = max(1.0, timeout_ms / 1000.0)
    exit_code = EXIT_SUCCESS

    # Fix: Explicit signal handling for graceful shutdown in containers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_exit_signal():
        logger.warning("Received exit signal, stopping agent gracefully...")
        stop_event.set()
        # Trigger an exception in the running _drive_agent task
        for task in asyncio.all_tasks(loop):
            if task.get_coro().__name__ == "_drive_agent":
                task.cancel()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_exit_signal)

    try:
        # Combined wait_for and cancellation event
        agent_task = asyncio.create_task(
            _drive_agent(agent, task, context, bridge, event_printer)
        )
        
        # Wait for either completion, timeout, or signal
        done, pending = await asyncio.wait(
            [agent_task], 
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED
        )

        if agent_task in done:
            await agent_task
        else:
            # Handle Timeout or Signal
            agent_task.cancel()
            if stop_event.is_set():
                logger.error("agent run interrupted by signal")
                bridge.mark_errored("interrupted by system signal (SIGINT/SIGTERM)")
                exit_code = EXIT_AGENT_ERROR
            else:
                logger.error("agent run timed out after %.1fs", timeout_seconds)
                bridge.mark_errored(f"timeout after {timeout_seconds:.1f}s")
                exit_code = EXIT_AGENT_ERROR

    except (asyncio.CancelledError, asyncio.TimeoutError):
        # Handled by signal logic above
        pass

    except KeyboardInterrupt: # This handles Ctrl+C on Windows
        logger.error("agent run interrupted by user")
        bridge.mark_errored("interrupted by user")
        exit_code = EXIT_AGENT_ERROR

    except Exception as exc:
        logger.exception("agent run failed with unexpected exception")
        bridge.mark_errored(f"unexpected exception: {exc}")
        exit_code = EXIT_AGENT_ERROR
    finally:
        # Remove signal handlers to clean up the loop
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
            _write_trajectory(resolved_output, bridge.finalize())

    return exit_code


async def _drive_agent(
    agent: Any,
    task: str,
    context: dict[str, Any],
    bridge: Any,
    event_printer: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Iterate the agent's async event stream and feed the bridge."""
    async for event in agent.run(task, context):
        await bridge.handle_event(event)
        if event_printer is not None:
            try:
                event_printer(event)
            except Exception:  # pragma: no cover - printer is best-effort
                logger.exception("event printer raised; continuing")


def _write_trajectory(output: Path, atif: dict[str, Any]) -> None:
    """
    Write ``atif`` to ``output`` atomically.

    Uses a sibling ``tempfile.NamedTemporaryFile`` followed by
    :func:`os.replace` so concurrent readers never observe a
    half-written file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(atif, indent=2, default=str)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(output.parent),
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with tmp as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp.name, output)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
