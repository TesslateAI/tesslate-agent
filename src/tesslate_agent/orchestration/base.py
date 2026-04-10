"""
Abstract Base Orchestrator.

Defines the slim file-/shell-operations interface that all orchestration
backends must implement for the standalone agent. Project- and container-
lifecycle management are intentionally *not* part of this interface — the
standalone agent operates on a single project root and delegates execution
directly to the underlying backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from .deployment_mode import DeploymentMode


class BaseOrchestrator(ABC):
    """
    Abstract base class for agent-facing orchestration backends.

    Implementations expose filesystem and shell primitives scoped to a
    single project root. Identifiers like ``user_id``, ``project_id``,
    ``container_name``, ``project_slug``, ``subdir``, ``volume_id``, and
    ``cache_node`` are carried through to match the shape the agent tools
    expect; implementations are free to ignore identifiers that have no
    meaning in their execution environment.
    """

    @property
    @abstractmethod
    def deployment_mode(self) -> DeploymentMode:
        """Return the deployment mode this orchestrator handles."""
        ...

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    @abstractmethod
    async def read_file(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        file_path: str,
        project_slug: str | None = None,
        subdir: str | None = None,
        volume_id: str | None = None,
        cache_node: str | None = None,
    ) -> str | None:
        """
        Read a file and return its contents as a string, or ``None`` when
        the file cannot be read (missing, not a file, or refused by the
        backend's containment policy).
        """
        ...

    @abstractmethod
    async def write_file(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        file_path: str,
        content: str,
        project_slug: str | None = None,
        subdir: str | None = None,
        volume_id: str | None = None,
        cache_node: str | None = None,
    ) -> bool:
        """Write ``content`` to ``file_path``. Returns ``True`` on success."""
        ...

    @abstractmethod
    async def delete_file(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        file_path: str,
    ) -> bool:
        """Delete a file. Returns ``True`` on success."""
        ...

    @abstractmethod
    async def list_files(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        directory: str = ".",
    ) -> list[dict[str, Any]]:
        """
        Return the immediate entries in ``directory``.

        Each entry is a dict with ``name``, ``type`` (``"file"`` /
        ``"directory"``), ``size``, and ``path``.
        """
        ...

    @abstractmethod
    async def list_tree(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        subdir: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return a recursive, filtered file tree (excluding ``node_modules``,
        ``.git``, caches, build output, etc.).

        Each entry is a dict with ``path``, ``name``, ``is_dir``, ``size``,
        ``mod_time``.
        """
        ...

    @abstractmethod
    async def read_file_content(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        file_path: str,
        subdir: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Read a single file and return ``{"path", "content", "size"}``, or
        ``None`` when the file cannot be read.
        """
        ...

    @abstractmethod
    async def read_files_batch(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        paths: list[str],
        subdir: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Batch-read multiple files.

        Returns a tuple of ``(successful_reads, failed_paths)`` where each
        successful read matches :meth:`read_file_content`'s shape.
        """
        ...

    # =========================================================================
    # SHELL OPERATIONS
    # =========================================================================

    @abstractmethod
    async def execute_command(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
        command: list[str],
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> str:
        """
        Run ``command`` and return combined stdout + stderr as a string.

        Raises:
            RuntimeError: On spawn failure, timeout, or generic execution error.
        """
        ...

    @abstractmethod
    async def is_container_ready(
        self,
        user_id: UUID,
        project_id: UUID,
        container_name: str,
    ) -> dict[str, Any]:
        """
        Return a readiness snapshot ``{"ready": bool, "message": str, ...}``.
        """
        ...

    # =========================================================================
    # ACTIVITY TRACKING + LOGS (default no-op implementations)
    # =========================================================================

    def track_activity(
        self,
        user_id: UUID,
        project_id: str,
        container_name: str | None = None,
    ) -> None:
        """
        Record activity for idle-cleanup purposes.

        The default implementation is a no-op; backends that cannot
        idle-reap (e.g. the local filesystem backend) simply inherit it.
        """
        return None

    async def stream_logs(
        self,
        project_id: UUID,
        user_id: UUID,
        container_id: UUID | None = None,
        tail_lines: int = 100,
    ) -> AsyncIterator[str]:
        """
        Stream container log lines.

        The default implementation yields nothing — backends without a
        container runtime inherit this.
        """
        if False:  # pragma: no cover - makes this a proper async generator
            yield ""
        return
