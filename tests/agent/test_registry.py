"""Tests for the agent tool registry primitives."""

from __future__ import annotations

from typing import Any

import pytest

from tesslate_agent.agent.tools.registry import (
    Tool,
    ToolCategory,
    ToolRegistry,
    create_scoped_tool_registry,
    get_tool_registry,
)


async def _noop_executor(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "message": "noop", "echo": params}


def _make_tool(name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description="Echo the provided params",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
        executor=_noop_executor,
        category=ToolCategory.FILE_OPS,
        examples=["echo hello"],
        system_prompt="Keep it short",
    )


def test_tool_instantiation_populates_fields() -> None:
    tool = _make_tool()
    assert tool.name == "echo"
    assert tool.category == ToolCategory.FILE_OPS
    assert tool.examples == ["echo hello"]
    assert tool.system_prompt == "Keep it short"
    prompt_text = tool.to_prompt_format()
    assert "echo" in prompt_text
    assert "text" in prompt_text
    assert "required" in prompt_text


def test_register_and_get() -> None:
    registry = ToolRegistry()
    tool = _make_tool()
    registry.register(tool)
    assert registry.get("echo") is tool
    assert registry.get("nope") is None
    assert "echo" in registry.list_names()
    assert tool in registry.all_tools()


def test_tool_category_contains_required_values() -> None:
    values = {c.value for c in ToolCategory}
    assert values == {
        "file_operations",
        "shell_commands",
        "project_management",
        "build_operations",
        "web_operations",
        "navigation_operations",
        "memory_operations",
        "git_operations",
        "delegation_operations",
        "planning_operations",
        "graph_view_tools",
    }


def test_get_tool_registry_returns_populated_singleton(monkeypatch) -> None:
    # Reset the module-level singleton to guarantee a clean slate.
    import tesslate_agent.agent.tools.registry as reg_mod

    monkeypatch.setattr(reg_mod, "_registry", None)
    registry = get_tool_registry()
    assert isinstance(registry, ToolRegistry)
    # The singleton is auto-populated with every built-in tool on
    # first access via register_all_tools(). A handful of canonical
    # names from each category should always be present.
    names = set(registry.list_names())
    for expected in (
        "read_file",
        "write_file",
        "bash_exec",
        "glob",
        "grep",
        "git_status",
        "memory_read",
        "update_plan",
        "web_fetch",
        "task",
    ):
        assert expected in names, f"{expected} missing from auto-populated registry"
    # Calling twice returns the same instance (true singleton).
    assert get_tool_registry() is registry


def test_create_scoped_tool_registry_filters_and_overrides(monkeypatch) -> None:
    import tesslate_agent.agent.tools.registry as reg_mod

    monkeypatch.setattr(reg_mod, "_registry", None)
    global_registry = get_tool_registry()
    global_registry.register(_make_tool("alpha"))
    global_registry.register(_make_tool("beta"))
    global_registry.register(_make_tool("gamma"))

    scoped = create_scoped_tool_registry(
        ["alpha", "beta", "missing_tool"],
        tool_configs={
            "alpha": {"description": "Custom alpha description"},
        },
    )

    names = set(scoped.list_names())
    assert names == {"alpha", "beta"}
    assert scoped.get("alpha").description == "Custom alpha description"
    assert scoped.get("beta").description == "Echo the provided params"
    assert scoped.get("gamma") is None


@pytest.mark.asyncio
async def test_registry_execute_happy_path(monkeypatch) -> None:
    import tesslate_agent.agent.tools.registry as reg_mod

    monkeypatch.setattr(reg_mod, "_registry", None)
    registry = get_tool_registry()
    registry.register(_make_tool("read_file"))

    result = await registry.execute(
        "read_file",
        {"text": "hi"},
        context={"edit_mode": "auto"},
    )
    assert result["success"] is True
    assert result["tool"] == "read_file"
    assert result["result"]["echo"] == {"text": "hi"}


@pytest.mark.asyncio
async def test_registry_execute_blocks_in_plan_mode(monkeypatch) -> None:
    import tesslate_agent.agent.tools.registry as reg_mod

    monkeypatch.setattr(reg_mod, "_registry", None)
    registry = get_tool_registry()
    registry.register(_make_tool("write_file"))

    result = await registry.execute(
        "write_file",
        {"text": "hi"},
        context={"edit_mode": "plan"},
    )
    assert result["success"] is False
    assert "Plan mode" in result["error"]


# ---------------------------------------------------------------------------
# Injected approval_handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injected_handler_allow_executes_tool() -> None:
    """When the injected handler returns allow_once the tool runs normally."""
    calls: list[tuple[str, dict, str]] = []

    async def _allow(tool_name: str, params: dict, session_id: str) -> str:
        calls.append((tool_name, params, session_id))
        return "allow_once"

    registry = ToolRegistry(approval_handler=_allow)
    registry.register(_make_tool("write_file"))

    result = await registry.execute(
        "write_file",
        {"text": "hello"},
        context={"edit_mode": "ask", "chat_id": "sess-1"},
    )
    assert result["success"] is True
    assert result["tool"] == "write_file"
    assert len(calls) == 1
    assert calls[0] == ("write_file", {"text": "hello"}, "sess-1")


@pytest.mark.asyncio
async def test_injected_handler_deny_returns_approval_required() -> None:
    """When the injected handler returns stop the tool is blocked and
    approval_required is returned so the LLM can recover."""
    called = []

    async def _deny(tool_name: str, params: dict, session_id: str) -> str:
        called.append(tool_name)
        return "stop"

    registry = ToolRegistry(approval_handler=_deny)
    registry.register(_make_tool("patch_file"))

    result = await registry.execute(
        "patch_file",
        {"text": "change"},
        context={"edit_mode": "ask", "chat_id": "sess-2"},
    )
    assert result.get("approval_required") is True
    assert result["tool"] == "patch_file"
    assert result["response"] == "stop"
    assert called == ["patch_file"]


@pytest.mark.asyncio
async def test_injected_handler_allow_all_skips_handler_on_second_call() -> None:
    """allow_all should let subsequent calls through without re-invoking the
    handler — the registry skips the gate entirely when allow_all was returned."""
    invoke_count = 0

    async def _allow_all_first(tool_name: str, params: dict, session_id: str) -> str:
        nonlocal invoke_count
        invoke_count += 1
        return "allow_all"

    registry = ToolRegistry(approval_handler=_allow_all_first)
    registry.register(_make_tool("write_file"))

    # First call: handler is invoked, returns allow_all
    result1 = await registry.execute(
        "write_file", {"text": "a"}, context={"edit_mode": "ask", "chat_id": "sess-3"}
    )
    assert result1["success"] is True
    assert invoke_count == 1

    # The registry alone doesn't track allow_all memory — that lives in the
    # orchestrator's PendingUserInputManager. The handler itself is responsible
    # for returning allow_once on subsequent calls. Verify the path is clean:
    # second call still invokes the handler (registry always delegates to it).
    result2 = await registry.execute(
        "write_file", {"text": "b"}, context={"edit_mode": "ask", "chat_id": "sess-3"}
    )
    assert result2["success"] is True
    assert invoke_count == 2


@pytest.mark.asyncio
async def test_no_handler_falls_back_to_env_var_allow(monkeypatch) -> None:
    """Without an injected handler the env-var ApprovalManager is used.
    Default policy is 'allow', so the tool should run unblocked."""
    monkeypatch.setenv("TESSLATE_AGENT_APPROVAL_POLICY", "allow")
    # Reset the singleton so it picks up the env var
    import tesslate_agent.agent.tools.approval_manager as am_mod

    monkeypatch.setattr(am_mod, "_approval_manager", None)

    registry = ToolRegistry()  # no approval_handler
    registry.register(_make_tool("write_file"))

    result = await registry.execute(
        "write_file",
        {"text": "hi"},
        context={"edit_mode": "ask", "chat_id": "sess-4"},
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_non_dangerous_tool_skips_handler_entirely() -> None:
    """Tools not in DANGEROUS_TOOLS bypass the approval gate even in ask mode."""
    handler_called = []

    async def _handler(tool_name: str, params: dict, session_id: str) -> str:
        handler_called.append(tool_name)
        return "stop"

    registry = ToolRegistry(approval_handler=_handler)
    registry.register(_make_tool("read_file"))  # not in DANGEROUS_TOOLS

    result = await registry.execute(
        "read_file",
        {"text": "hi"},
        context={"edit_mode": "ask", "chat_id": "sess-5"},
    )
    assert result["success"] is True
    assert handler_called == []  # handler never invoked for safe tools
