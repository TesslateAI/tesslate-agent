"""Orchestration layer — backends for file / shell operations."""

from __future__ import annotations

from .base import BaseOrchestrator
from .deployment_mode import DeploymentMode
from .factory import (
    OrchestratorFactory,
    get_orchestrator,
    is_docker_mode,
    is_kubernetes_mode,
    is_local_mode,
)
from .local import PTY_SESSIONS, LocalOrchestrator, PtySessionRegistry

__all__ = [
    "BaseOrchestrator",
    "DeploymentMode",
    "LocalOrchestrator",
    "PtySessionRegistry",
    "PTY_SESSIONS",
    "OrchestratorFactory",
    "get_orchestrator",
    "is_local_mode",
    "is_docker_mode",
    "is_kubernetes_mode",
]
