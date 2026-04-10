"""Integration tests for the ``read_many_files`` file operation tool."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from tesslate_agent.agent.tools.file_ops.read_many import read_many_files_tool
from tesslate_agent.orchestration import (
    DeploymentMode,
    LocalOrchestrator,
    OrchestratorFactory,
)

pytestmark = pytest.mark.asyncio


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def project_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")

    _write(tmp_path / "README.md", "# Project\n\nIntro paragraph.\n")
    _write(tmp_path / "docs" / "guide.md", "# Guide\n\nBody.\n")
    _write(tmp_path / "docs" / "api.md", "# API\n\nMore body.\n")

    _write(tmp_path / "src" / "a.py", "def a():\n    return 1\n")
    _write(tmp_path / "src" / "b.py", "def b():\n    return 2\n")
    _write(tmp_path / "src" / "deep" / "c.py", "def c():\n    return 3\n")
    _write(tmp_path / "src" / "deep" / "d.ts", "export const d = 4;\n")

    _write(tmp_path / "node_modules" / "noise" / "index.js", "// noise\n")
    _write(tmp_path / "dist" / "bundle.js", "// bundled\n")
    _write(tmp_path / "package-lock.json", "{}\n")

    _write(tmp_path / "tests" / "test_a.py", "def test_a():\n    assert True\n")

    return tmp_path


@pytest.fixture
def bound_tree_orchestrator(project_tree: Path):
    OrchestratorFactory.clear_cache()
    orchestrator = LocalOrchestrator()
    for mode in DeploymentMode:
        OrchestratorFactory._instances[mode] = orchestrator
    yield orchestrator
    OrchestratorFactory.clear_cache()


@pytest.fixture
def tool_context() -> dict:
    return {
        "user_id": uuid4(),
        "project_id": uuid4(),
        "project_slug": "test-project",
        "container_name": "main",
        "container_directory": None,
        "edit_mode": "auto",
    }


async def test_single_glob(bound_tree_orchestrator, tool_context) -> None:
    result = await read_many_files_tool({"include": ["**/*.md"]}, tool_context)
    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert "README.md" in paths
    assert "docs/guide.md" in paths
    assert "docs/api.md" in paths
    assert result["total_bytes"] > 0


async def test_multi_glob_include(bound_tree_orchestrator, tool_context) -> None:
    result = await read_many_files_tool(
        {"include": ["src/**/*.py", "src/**/*.ts"]},
        tool_context,
    )
    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert "src/a.py" in paths
    assert "src/b.py" in paths
    assert "src/deep/c.py" in paths
    assert "src/deep/d.ts" in paths


async def test_exclude_list_honored(bound_tree_orchestrator, tool_context) -> None:
    result = await read_many_files_tool(
        {"include": ["**/*.py"], "exclude": ["tests/**"]},
        tool_context,
    )
    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert not any(p.startswith("tests/") for p in paths)
    assert "src/a.py" in paths


async def test_default_excludes_skip_node_modules(
    bound_tree_orchestrator, tool_context
) -> None:
    result = await read_many_files_tool({"include": ["**/*.js"]}, tool_context)
    assert result["success"] is True
    paths = [f["path"] for f in result["files"]]
    assert not any(p.startswith("node_modules/") for p in paths)
    assert not any(p.startswith("dist/") for p in paths)


async def test_disabling_default_excludes_includes_noise(
    bound_tree_orchestrator, tool_context
) -> None:
    with_defaults = await read_many_files_tool(
        {"include": ["package-lock.json"]},
        tool_context,
    )
    assert with_defaults["success"] is True
    assert not any(
        f["path"] == "package-lock.json" for f in with_defaults["files"]
    )

    without_defaults = await read_many_files_tool(
        {"include": ["package-lock.json"], "use_default_excludes": False},
        tool_context,
    )
    assert without_defaults["success"] is True
    assert any(
        f["path"] == "package-lock.json" for f in without_defaults["files"]
    )


async def test_max_bytes_per_file_truncates(
    bound_tree_orchestrator, tool_context
) -> None:
    result = await read_many_files_tool(
        {"include": ["**/*.md"], "max_bytes_per_file": 10},
        tool_context,
    )
    assert result["success"] is True
    for f in result["files"]:
        assert len(f["content"].encode("utf-8")) <= 10
        if f["size"] > 10:
            assert f["truncated"] is True


async def test_max_total_bytes_halts_enumeration(
    bound_tree_orchestrator, tool_context
) -> None:
    result = await read_many_files_tool(
        {
            "include": ["**/*.md", "**/*.py"],
            "max_total_bytes": 20,
            "max_bytes_per_file": 20,
        },
        tool_context,
    )
    assert result["success"] is True
    assert result["total_bytes"] <= 20
    assert result["truncated_overall"] is True
    assert len(result["skipped"]) >= 1


async def test_missing_include_errors(
    bound_tree_orchestrator, tool_context
) -> None:
    result = await read_many_files_tool({}, tool_context)
    assert result["success"] is False
    assert "include" in result["message"].lower()


async def test_empty_include_errors(
    bound_tree_orchestrator, tool_context
) -> None:
    result = await read_many_files_tool({"include": []}, tool_context)
    assert result["success"] is False
    assert "include" in result["message"].lower()
