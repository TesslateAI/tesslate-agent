"""Per-file write fence — serialize read-modify-write on the same path.

Two concurrent agents editing the same file would otherwise silently
overwrite each other (agent A reads X, agent B reads X, A writes, B writes).
We grab a short distributed lock keyed on (project_id, normalized_path) just
around the RMW sequence. Agents editing DIFFERENT files stay fully parallel.

Falls back to a per-process asyncio.Lock when no distributed lock service is
available (standalone / single-process environments).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

# In-process fallback: one lock per (project_id, normalized_path).
_process_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _normalize_path(path: str) -> str:
    return os.path.normpath(path).lstrip("./").replace("\\", "/")


@contextlib.asynccontextmanager
async def fence_file(project_id: str, file_path: str, *, ttl_seconds: int = 30):
    """Acquire a distributed lock around a single file's RMW.

    Falls back to a per-process asyncio.Lock when the distributed lock
    service is unavailable so single-process environments still get
    serialization without blocking.
    """
    if not project_id or not file_path:
        yield
        return

    name = f"file:{project_id}:{_normalize_path(file_path)}"

    # Try distributed lock first (orchestrator context with Redis).
    try:
        from tesslate_agent.orchestration import get_orchestrator  # noqa: F401

        # Attempt to reach a distributed lock via context injection.
        # If not available, fall through to the in-process fallback.
        _dlock = None
        try:
            from tesslate_agent.orchestration.factory import OrchestratorFactory

            orch = OrchestratorFactory._instances.get(
                OrchestratorFactory.get_deployment_mode()
            )
            _dlock = getattr(orch, "_distributed_lock", None) if orch else None
        except Exception:
            pass

        if _dlock is not None:
            try:
                async with _dlock.wait_for(name, ttl_seconds=ttl_seconds, max_wait_seconds=15.0):
                    yield
                return
            except TimeoutError:
                logger.warning(
                    "[FILE-FENCE] timed out waiting for %s — proceeding without fence",
                    name,
                )
                yield
                return
    except Exception as e:  # noqa: BLE001
        logger.debug("[FILE-FENCE] distributed lock unavailable: %s", e)

    # In-process asyncio.Lock fallback.
    lock = _process_locks[name]
    try:
        async with asyncio.timeout(15.0):
            async with lock:
                yield
    except TimeoutError:
        logger.warning(
            "[FILE-FENCE] in-process lock timed out for %s — proceeding without fence",
            name,
        )
        yield
