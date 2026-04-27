"""Domain errors raised by the tesslate-agent runtime.

These errors carry semantic meaning the host orchestrator (e.g.
``orchestrator/app/services/automations/dispatcher.py`` in OpenSail) can
catch by type. Generic exceptions get retried by the agent loop's
backoff helper; these surface terminal conditions that retries cannot
fix.

Why these live in the submodule
-------------------------------
The agent runs as a standalone package — it MUST NOT import host code
to keep ``pip install tesslate-agent`` self-contained. The host catches
these by either:

  * importing the type:
    ``from tesslate_agent.errors import BudgetExhaustedError``
  * matching by ``isinstance(exc, BudgetExhaustedError)`` after lazy
    import inside the catch block.

OpenSail's ``services.automations.budget.BudgetExhaustedError`` is the
host's own equivalent type carrying additional run-attribution fields
(``key_id``, ``spent_usd``); the orchestrator's catch path translates
between the two when persisting an ``automation_runs.status``.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for tesslate-agent domain errors.

    Anything not derived from this is treated as transient by the
    agent loop's retry helper and may be retried with backoff.
    """


class BudgetExhaustedError(AgentError):
    """The model rejected the request because the per-run / per-key
    budget allocated to this agent invocation is exhausted.

    Raised by :class:`tesslate_agent.agent.models.LiteLLMAdapter` when
    LiteLLM returns a 429 with a ``budget``/``credit``/``quota`` hint in
    the error body. The agent loop does NOT retry this — the host
    orchestrator's dispatcher is expected to catch it, register an
    "extend daily budget?" approval, and rerun once the user grants
    more spend.

    Attributes:
        run_id: The agent run identifier that exhausted its budget,
            sourced from ``X-Tesslate-Run-Id`` (the request-scoped
            header threaded through every LLM call). ``None`` when the
            host did not supply a run id.
        model_name: The model that returned 429 — useful when a host
            wants to fan out to a cheaper fallback model on the next
            attempt.
        message: Human-readable string from the upstream provider.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.model_name = model_name
        self.message = message


__all__ = [
    "AgentError",
    "BudgetExhaustedError",
]
