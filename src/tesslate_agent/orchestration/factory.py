"""
Orchestrator factory.

Pluggable registry that maps :class:`DeploymentMode` values to
:class:`BaseOrchestrator` implementations. The ``local`` backend is
registered automatically at import time; additional backends (Docker,
Kubernetes, etc.) can be registered by downstream packages.
"""

from __future__ import annotations

import logging
import os

from .base import BaseOrchestrator
from .deployment_mode import DeploymentMode

logger = logging.getLogger(__name__)


class OrchestratorFactory:
    """
    Pluggable factory for orchestration backends.

    Backends register themselves by calling :meth:`register` with their
    :class:`DeploymentMode` and a :class:`BaseOrchestrator` subclass.
    Consumers call :meth:`create_orchestrator` to obtain a cached
    instance for a given mode.
    """

    _registry: dict[DeploymentMode, type[BaseOrchestrator]] = {}
    _instances: dict[DeploymentMode, BaseOrchestrator] = {}

    @classmethod
    def register(
        cls, mode: DeploymentMode, orchestrator_cls: type[BaseOrchestrator]
    ) -> None:
        """
        Register ``orchestrator_cls`` as the implementation for ``mode``.

        Any previously cached instance for ``mode`` is dropped so the next
        call to :meth:`create_orchestrator` picks up the new class.
        """
        cls._registry[mode] = orchestrator_cls
        cls._instances.pop(mode, None)

    @classmethod
    def is_registered(cls, mode: DeploymentMode) -> bool:
        """Return ``True`` when a backend is registered for ``mode``."""
        return mode in cls._registry

    @classmethod
    def get_deployment_mode(cls) -> DeploymentMode:
        """
        Read the active deployment mode from the ``DEPLOYMENT_MODE`` env var.

        Defaults to :attr:`DeploymentMode.LOCAL` when the variable is unset
        or empty.
        """
        value = os.environ.get("DEPLOYMENT_MODE", "local").strip().lower()
        if not value:
            value = "local"
        return DeploymentMode.from_string(value)

    @classmethod
    def create_orchestrator(
        cls, mode: DeploymentMode | None = None
    ) -> BaseOrchestrator:
        """
        Return a cached orchestrator instance for ``mode``.

        When ``mode`` is ``None`` the active deployment mode is read from
        the environment.

        Raises:
            ValueError: If no backend has been registered for ``mode``.
        """
        if mode is None:
            mode = cls.get_deployment_mode()
        if mode in cls._instances:
            return cls._instances[mode]
        if mode not in cls._registry:
            raise ValueError(
                f"No orchestrator registered for deployment mode '{mode}'. "
                f"Registered: {sorted(m.value for m in cls._registry.keys())}"
            )
        instance = cls._registry[mode]()
        cls._instances[mode] = instance
        logger.info("[ORCHESTRATOR] Created %s orchestrator", mode.value)
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """Drop every cached orchestrator instance."""
        cls._instances.clear()


def get_orchestrator(mode: DeploymentMode | None = None) -> BaseOrchestrator:
    """
    Shortcut for :meth:`OrchestratorFactory.create_orchestrator`.

    Args:
        mode: Optional deployment mode override. When ``None``, the mode
            is read from the ``DEPLOYMENT_MODE`` environment variable.
    """
    return OrchestratorFactory.create_orchestrator(mode)


def is_local_mode() -> bool:
    """Return ``True`` when the active deployment mode is ``local``."""
    return OrchestratorFactory.get_deployment_mode() == DeploymentMode.LOCAL


def is_docker_mode() -> bool:
    """Return ``True`` when the active deployment mode is ``docker``."""
    return OrchestratorFactory.get_deployment_mode() == DeploymentMode.DOCKER


def is_kubernetes_mode() -> bool:
    """Return ``True`` when the active deployment mode is ``kubernetes``."""
    return OrchestratorFactory.get_deployment_mode() == DeploymentMode.KUBERNETES


# Register the built-in local backend.
from .local import LocalOrchestrator  # noqa: E402

OrchestratorFactory.register(DeploymentMode.LOCAL, LocalOrchestrator)
