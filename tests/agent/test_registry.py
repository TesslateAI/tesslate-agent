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
            "properties": {
                "text": {"type": "string", "description": "Text to echo"}
            },
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
