"""
Shared fixtures for git-ops tool tests.

Each test runs against a real :class:`LocalOrchestrator` bound to a
freshly initialized git repository under ``tmp_path``. The repository
contains three commits, a feature branch, and a mixed worktree state so
that the tools under test have realistic data to chew on.

HOME / GIT_CONFIG_GLOBAL / GIT_CONFIG_SYSTEM are monkey-patched per test
so the local git binary never reads or writes the developer's real git
configuration. ``DEPLOYMENT_MODE`` is forced to ``local`` so
``get_orchestrator()`` resolves the bound :class:`LocalOrchestrator`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from tesslate_agent.orchestration import (
    DeploymentMode,
    LocalOrchestrator,
    OrchestratorFactory,
)


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """
    Build a sanitized environment for git subprocess invocations so the
    test is fully isolated from the developer's global config.
    """
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(fake_home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test Author")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "author@example.com")
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    return dict(os.environ)


@pytest.fixture
def temp_git_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_env: dict[str, str],
) -> Path:
    """
    Create a fresh git repo at ``tmp_path/repo`` with:

    - 3 commits on the default branch
    - a ``feature/extra`` branch with one additional commit
    - 1 staged change
    - 1 unstaged worktree change
    - 1 untracked file

    Returns the repo path, which is also exported as ``PROJECT_ROOT``.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], repo, git_env)

    (repo / "README.md").write_text("# Hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo, git_env)
    _run(
        ["git", "commit", "-q", "-m", "initial: add README"],
        repo,
        git_env,
    )

    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text(
        "def greet(name):\n    return f'hi {name}'\n",
        encoding="utf-8",
    )
    _run(["git", "add", "src/app.py"], repo, git_env)
    _run(
        ["git", "commit", "-q", "-m", "feat: add greet()"],
        repo,
        git_env,
    )

    (repo / "src" / "app.py").write_text(
        "def greet(name):\n    return f'hello {name}'\n",
        encoding="utf-8",
    )
    _run(["git", "add", "src/app.py"], repo, git_env)
    _run(
        ["git", "commit", "-q", "-m", "fix: say hello instead of hi"],
        repo,
        git_env,
    )

    # Feature branch with one extra commit.
    _run(["git", "checkout", "-q", "-b", "feature/extra"], repo, git_env)
    (repo / "src" / "extra.py").write_text(
        "VALUE = 42\n", encoding="utf-8"
    )
    _run(["git", "add", "src/extra.py"], repo, git_env)
    _run(
        ["git", "commit", "-q", "-m", "feat: add extra constant"],
        repo,
        git_env,
    )
    _run(["git", "checkout", "-q", "main"], repo, git_env)

    # Staged change to README.
    (repo / "README.md").write_text("# Hello\n\nSecond line.\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo, git_env)

    # Unstaged worktree change to src/app.py.
    (repo / "src" / "app.py").write_text(
        "def greet(name):\n    return f'hello, {name}!'\n",
        encoding="utf-8",
    )

    # Untracked file.
    (repo / "untracked.txt").write_text("stray\n", encoding="utf-8")

    monkeypatch.setenv("PROJECT_ROOT", str(repo))
    return repo


@pytest.fixture
def bound_orchestrator(temp_git_repo: Path) -> LocalOrchestrator:
    """
    Register a :class:`LocalOrchestrator` rooted at ``temp_git_repo`` as
    the cached instance for every deployment mode.
    """
    OrchestratorFactory.clear_cache()
    orchestrator = LocalOrchestrator()
    for mode in DeploymentMode:
        OrchestratorFactory._instances[mode] = orchestrator
    yield orchestrator
    OrchestratorFactory.clear_cache()


@pytest.fixture
def git_context() -> dict:
    """Minimal tool-execution context for git-ops tools."""
    return {
        "user_id": uuid4(),
        "project_id": uuid4(),
        "project_slug": "test-repo",
        "container_name": None,
        "container_directory": None,
    }
