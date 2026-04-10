"""Tests for the local filesystem orchestrator."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from tesslate_agent.orchestration import (
    DeploymentMode,
    LocalOrchestrator,
    PTY_SESSIONS,
)

USER = uuid4()
PROJECT = uuid4()
CONTAINER = "local"


@pytest.fixture
def orch(tmp_path, monkeypatch) -> LocalOrchestrator:
    """Return a fresh LocalOrchestrator rooted at ``tmp_path``."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    return LocalOrchestrator()


def test_deployment_mode(orch: LocalOrchestrator) -> None:
    assert orch.deployment_mode == DeploymentMode.LOCAL
    assert orch.deployment_mode.is_local is True
    assert str(orch.deployment_mode) == "local"


async def test_write_then_read_roundtrip(
    orch: LocalOrchestrator, tmp_path
) -> None:
    ok = await orch.write_file(USER, PROJECT, CONTAINER, "hello.txt", "hello world")
    assert ok is True
    assert (tmp_path / "hello.txt").read_text() == "hello world"

    content = await orch.read_file(USER, PROJECT, CONTAINER, "hello.txt")
    assert content == "hello world"


async def test_read_missing_file_returns_none(orch: LocalOrchestrator) -> None:
    assert await orch.read_file(USER, PROJECT, CONTAINER, "nope.txt") is None


async def test_write_file_is_atomic_and_preserves_mode(
    orch: LocalOrchestrator, tmp_path
) -> None:
    target = tmp_path / "script.sh"
    target.write_text("original")
    os.chmod(target, 0o755)

    ok = await orch.write_file(USER, PROJECT, CONTAINER, "script.sh", "updated")
    assert ok is True
    assert target.read_text() == "updated"
    # Mode preserved through the atomic replace.
    assert (target.stat().st_mode & 0o777) == 0o755

    # No .tmp stragglers hanging around.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".script.sh.")]
    assert leftovers == []


async def test_delete_file_removes_file(
    orch: LocalOrchestrator, tmp_path
) -> None:
    (tmp_path / "to_delete.txt").write_text("bye")
    ok = await orch.delete_file(USER, PROJECT, CONTAINER, "to_delete.txt")
    assert ok is True
    assert not (tmp_path / "to_delete.txt").exists()

    # Deleting a non-existent file is a no-op that returns False.
    assert (
        await orch.delete_file(USER, PROJECT, CONTAINER, "to_delete.txt") is False
    )


async def test_list_files_returns_sorted_entries(
    orch: LocalOrchestrator, tmp_path
) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("bb")
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").write_text("hidden")

    entries = await orch.list_files(USER, PROJECT, CONTAINER, ".")
    names = [e["name"] for e in entries]
    assert names == ["a.txt", "b.txt", "sub"]

    types = {e["name"]: e["type"] for e in entries}
    assert types["a.txt"] == "file"
    assert types["sub"] == "directory"


async def test_list_tree_excludes_common_junk_dirs(
    orch: LocalOrchestrator, tmp_path
) -> None:
    # Include one excluded directory and one normal one.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module.exports = {}")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    entries = await orch.list_tree(USER, PROJECT, CONTAINER)
    paths = {e["path"] for e in entries}
    assert "src" in paths or "src/main.py" in paths
    assert "src/main.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any("__pycache__" in p for p in paths)


async def test_execute_command_echo(orch: LocalOrchestrator) -> None:
    output = await orch.execute_command(
        USER, PROJECT, CONTAINER, ["/bin/echo", "hello from subprocess"]
    )
    assert "hello from subprocess" in output


async def test_path_escape_is_refused(
    orch: LocalOrchestrator, tmp_path
) -> None:
    # Attempt to escape the project root via .. — should be refused.
    escaped = await orch.read_file(
        USER, PROJECT, CONTAINER, "../../../../etc/passwd"
    )
    assert escaped is None

    ok = await orch.write_file(
        USER, PROJECT, CONTAINER, "../outside.txt", "nope"
    )
    assert ok is False
    assert not (tmp_path.parent / "outside.txt").exists()


async def test_is_container_ready(orch: LocalOrchestrator) -> None:
    ready = await orch.is_container_ready(USER, PROJECT, CONTAINER)
    assert ready["ready"] is True
    assert "local" in ready["message"]


async def test_read_files_batch(
    orch: LocalOrchestrator, tmp_path
) -> None:
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "b.txt").write_text("bbbb")

    files, errors = await orch.read_files_batch(
        USER, PROJECT, CONTAINER, ["a.txt", "b.txt", "missing.txt"]
    )
    assert len(files) == 2
    assert sorted(f["path"] for f in files) == ["a.txt", "b.txt"]
    assert errors == ["missing.txt"]


async def test_pty_session_roundtrip() -> None:
    """Create a PTY session, wait for output, then close it.

    The drain task is scheduled on the current event loop, so we give it
    real wall time to run (``await asyncio.sleep``) rather than polling
    in a tight loop. We keep reading until either the expected bytes are
    collected OR a generous deadline elapses — collecting nothing past
    the drain task completion is a real bug, so we should only break out
    on success.
    """
    session_id = PTY_SESSIONS.create(["/bin/sh", "-c", "printf 'hello-pty'; exit 0"])
    try:
        collected = bytearray()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            chunk = PTY_SESSIONS.read(session_id)
            if chunk:
                collected.extend(chunk)
            if b"hello-pty" in collected:
                break
            await asyncio.sleep(0.05)
        # Final sweep in case the last chunk landed after the loop exit.
        collected.extend(PTY_SESSIONS.read(session_id))
        assert b"hello-pty" in collected
        assert PTY_SESSIONS.has(session_id) is True
    finally:
        PTY_SESSIONS.close(session_id)
        assert PTY_SESSIONS.has(session_id) is False
