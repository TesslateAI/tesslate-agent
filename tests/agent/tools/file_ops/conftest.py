"""
Shared fixtures for file-ops tool tests.

Every test runs against a real :class:`LocalOrchestrator` bound to
``tmp_path`` -- no mocked I/O. The orchestrator is registered as the
cached instance for all deployment modes so code that calls
``get_orchestrator()`` picks it up regardless of the configured
``DEPLOYMENT_MODE``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from tesslate_agent.orchestration import (
    DeploymentMode,
    LocalOrchestrator,
    OrchestratorFactory,
)


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
def fops_context() -> dict:
    """Minimal tool-execution context for file-ops tools."""
    return {
        "user_id": uuid4(),
        "project_id": uuid4(),
        "project_slug": "test-project",
        "container_name": None,
        "container_directory": None,
    }


@pytest.fixture(autouse=True)
async def _clear_edit_history():
    """Reset the shared edit history between tests."""
    from tesslate_agent.agent.tools.file_ops.edit_history import EDIT_HISTORY

    await EDIT_HISTORY.clear()
    yield
    await EDIT_HISTORY.clear()
