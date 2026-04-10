"""
Tool approval manager.

Policy-gated, synchronous approval resolution for dangerous tool calls.
The policy is read from the ``TESSLATE_AGENT_APPROVAL_POLICY`` environment
variable:

    - ``allow`` (default): every approval request resolves to ``allow_once``.
    - ``deny``: every approval request resolves to ``stop``.

Later waves may plug in an interactive approval UI; this baseline keeps
every agent tool test deterministic and self-contained.
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

logger = logging.getLogger(__name__)


VALID_RESPONSES: frozenset[str] = frozenset({"allow_once", "allow_all", "stop"})


class ApprovalRequest:
    """
    Represents a resolved tool approval request.

    The request resolves synchronously in the constructor based on the
    configured policy; callers that ``await`` on older APIs can still
    inspect :attr:`response` directly.
    """

    def __init__(
        self,
        approval_id: str,
        tool_name: str,
        parameters: dict,
        session_id: str,
        response: str,
    ):
        self.approval_id = approval_id
        self.tool_name = tool_name
        self.parameters = parameters
        self.session_id = session_id
        self.response = response


class ApprovalManager:
    """
    Policy-gated approval manager.

    Tracks per-session "allow all" memory, but relies on the environment
    variable ``TESSLATE_AGENT_APPROVAL_POLICY`` (``allow`` or ``deny``) to
    decide the verdict for each new request.
    """

    POLICY_ENV_VAR = "TESSLATE_AGENT_APPROVAL_POLICY"

    def __init__(self) -> None:
        self._approved_tools: dict[str, set[str]] = {}
        logger.info("[ApprovalManager] Initialized (policy=%s)", self._policy())

    @classmethod
    def _policy(cls) -> str:
        """Return the active approval policy (``"allow"`` or ``"deny"``)."""
        raw = os.environ.get(cls.POLICY_ENV_VAR, "allow").strip().lower()
        if raw not in ("allow", "deny"):
            logger.warning(
                "[ApprovalManager] Unknown policy %r; defaulting to 'allow'", raw
            )
            return "allow"
        return raw

    def is_tool_approved(self, session_id: str, tool_name: str) -> bool:
        """
        Return ``True`` when ``tool_name`` has been approved "for all"
        within ``session_id``.
        """
        if session_id not in self._approved_tools:
            return False
        return tool_name in self._approved_tools[session_id]

    def approve_tool_for_session(self, session_id: str, tool_name: str) -> None:
        """
        Mark ``tool_name`` as approved for every future call in ``session_id``.
        """
        if session_id not in self._approved_tools:
            self._approved_tools[session_id] = set()
        self._approved_tools[session_id].add(tool_name)
        logger.info(
            "[ApprovalManager] Approved %s for session %s", tool_name, session_id
        )

    def clear_session_approvals(self, session_id: str) -> None:
        """Forget every "allow all" approval for ``session_id``."""
        if session_id in self._approved_tools:
            del self._approved_tools[session_id]
            logger.info("[ApprovalManager] Cleared approvals for session %s", session_id)

    def request_approval(
        self, tool_name: str, parameters: dict, session_id: str
    ) -> tuple[str, ApprovalRequest]:
        """
        Resolve an approval request synchronously.

        Under the ``allow`` policy the request resolves to ``allow_once``;
        under the ``deny`` policy it resolves to ``stop``. A previously
        session-approved tool always resolves to ``allow_once`` regardless
        of policy.

        Returns:
            A tuple of ``(approval_id, request)``.
        """
        approval_id = str(uuid4())

        if self.is_tool_approved(session_id, tool_name):
            response = "allow_once"
        else:
            response = "allow_once" if self._policy() == "allow" else "stop"

        request = ApprovalRequest(
            approval_id=approval_id,
            tool_name=tool_name,
            parameters=parameters,
            session_id=session_id,
            response=response,
        )
        logger.info(
            "[ApprovalManager] Request %s for %s resolved to %s",
            approval_id,
            tool_name,
            response,
        )
        return approval_id, request

    def respond_to_approval(self, approval_id: str, response: str) -> None:
        """
        Compatibility shim for older callers that push a response in later.

        Because this manager resolves requests synchronously, the response
        is only used to update the per-session "allow all" memory when the
        user picked ``"allow_all"``. Unknown or stale approval IDs are
        silently ignored.
        """
        if response not in VALID_RESPONSES:
            logger.warning(
                "[ApprovalManager] Ignoring invalid approval response: %r", response
            )
            return
        logger.debug(
            "[ApprovalManager] respond_to_approval id=%s response=%s (no-op)",
            approval_id,
            response,
        )


# Module-level singleton.
_approval_manager: ApprovalManager | None = None


def get_approval_manager() -> ApprovalManager:
    """Return the process-wide :class:`ApprovalManager` singleton."""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager
