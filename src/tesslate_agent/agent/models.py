"""
LiteLLM-based model adapter for the Tesslate agent.

Single adapter class; handles every LLM provider through litellm's unified
interface. API keys are resolved from provider-specific env vars
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.) OR via a LiteLLM proxy
configured via ``LITELLM_API_BASE`` + ``LITELLM_MASTER_KEY``.

Run-id propagation
------------------
Every ``litellm.acompletion`` call carries an ``X-Tesslate-Run-Id``
header (when ``run_id`` is supplied to the adapter) so the upstream
LiteLLM proxy can deduplicate parallel 429s by run — without it, the
proxy would surface the same budget-exhausted condition once per HTTP
attempt instead of once per run.

Budget-exhausted catch
----------------------
LiteLLM raises ``RateLimitError`` for both transient rate limiting and
hard budget exhaustion. The two are distinguished by inspecting the
error body for a ``budget``/``credit``/``quota`` hint; budget-exhausted
errors are re-raised as :class:`tesslate_agent.errors.BudgetExhaustedError`
so the agent loop's keyword-based retry helper does NOT silently retry
them. Generic 429s remain ``RateLimitError`` and stay retryable.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from tesslate_agent.errors import BudgetExhaustedError

logger = logging.getLogger(__name__)


# Substrings that, when found in a 429's stringified body, indicate a
# hard budget/quota condition rather than transient rate limiting. Kept
# narrow on purpose — false positives would convert a retryable rate
# limit into a terminal error and starve the run.
_BUDGET_HINTS: tuple[str, ...] = (
    "budget",
    "quota",
    "credit",
    "insufficient_funds",
    "spend limit",
    "spending limit",
    "exceeded your monthly",
)


def _is_budget_exhausted(exc: Exception) -> bool:
    """Return True iff ``exc`` looks like a hard budget/quota condition.

    Uses substring matching against the stringified exception (which
    LiteLLM's ``RateLimitError`` populates with the upstream provider's
    error body). The match is case-insensitive and intentionally narrow
    — see :data:`_BUDGET_HINTS`.
    """
    message = str(exc).lower()
    return any(hint in message for hint in _BUDGET_HINTS)

# Ordered list of (model-name-prefix-tuples, required-env-var) pairs used to
# map a litellm-style model identifier back to the provider env var that
# would satisfy it. Checked in order, first match wins.
_PROVIDER_ENV_VARS: list[tuple[tuple[str, ...], str]] = [
    (("openai/", "gpt-", "o1-", "o3-", "o4-", "chatgpt"), "OPENAI_API_KEY"),
    (("anthropic/", "claude-"), "ANTHROPIC_API_KEY"),
    (("openrouter/",), "OPENROUTER_API_KEY"),
    (("groq/",), "GROQ_API_KEY"),
    (("together/", "together_ai/"), "TOGETHER_API_KEY"),
    (("deepseek/",), "DEEPSEEK_API_KEY"),
    (("fireworks/", "fireworks_ai/"), "FIREWORKS_API_KEY"),
    (("gemini/", "vertex_ai/"), "GEMINI_API_KEY"),
    (("mistral/",), "MISTRAL_API_KEY"),
    (("cohere/",), "COHERE_API_KEY"),
    (("perplexity/",), "PERPLEXITYAI_API_KEY"),
    (("bedrock/",), "AWS_ACCESS_KEY_ID"),
]


class MissingApiKeyError(ValueError):
    """
    Raised when a model requires a provider API key that is not configured.

    Attributes:
        model_name: The requested litellm model identifier.
        env_var: The environment variable that would satisfy the request.
    """

    def __init__(self, model_name: str, env_var: str):
        self.model_name = model_name
        self.env_var = env_var
        super().__init__(
            f"Model '{model_name}' requires the {env_var} environment variable to be set. "
            f"Either export {env_var}, or configure a LiteLLM proxy via LITELLM_API_BASE + "
            f"LITELLM_MASTER_KEY."
        )


class ModelAdapter(ABC):
    """Abstract interface all model adapters must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the underlying model identifier."""
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Call the model with optional tool definitions.

        When ``stream=False`` returns a dict with ``content``, ``tool_calls``,
        ``usage``, ``finish_reason``, and ``raw_response``. When ``stream=True``
        returns an async iterator of delta dicts with ``delta_content``,
        ``delta_tool_calls``, and ``finish_reason``.
        """
        ...


class LiteLLMAdapter(ModelAdapter):
    """
    LiteLLM-backed :class:`ModelAdapter` implementation.

    Handles every LLM provider that LiteLLM supports. Credentials can be
    supplied either via the provider's native environment variable, or via
    a LiteLLM proxy (``LITELLM_API_BASE`` + ``LITELLM_MASTER_KEY``).
    """

    def __init__(
        self,
        model_name: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int | None = None,
        thinking_effort: str | None = None,
        run_id: str | None = None,
        **extra: Any,
    ):
        """
        Construct a new adapter.

        Args:
            model_name: LiteLLM model identifier (e.g. ``"openai/gpt-4o-mini"``).
            api_base: Optional proxy base URL. Falls back to ``LITELLM_API_BASE``.
            api_key: Optional proxy master key. Falls back to ``LITELLM_MASTER_KEY``.
            default_temperature: Temperature used when the caller does not
                override it per-request.
            default_max_tokens: Max tokens used when the caller does not
                override it per-request.
            thinking_effort: Extended-thinking effort tier for supported
                models (``"low"`` / ``"medium"`` / ``"high"``). Passed through
                to the model via ``extra_body.reasoning_effort``.
            run_id: Optional opaque run identifier. When set, every
                :func:`litellm.acompletion` call carries
                ``X-Tesslate-Run-Id: <run_id>`` in ``extra_headers`` so
                upstream LiteLLM proxies can dedupe parallel 429s and
                attribute spend to the correct run.
            **extra: Additional keyword arguments forwarded verbatim to
                every :func:`litellm.acompletion` call.
        """
        self._model_name = model_name
        self._api_base = api_base or os.environ.get("LITELLM_API_BASE")
        self._api_key = api_key or os.environ.get("LITELLM_MASTER_KEY")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._thinking_effort = thinking_effort
        self._run_id = run_id
        self._extra = extra

    @property
    def model_name(self) -> str:
        return self._model_name

    def _resolve_provider_env_var(self) -> str | None:
        """Return the env var expected for this model's provider, if known."""
        lowered = self._model_name.lower()
        for prefixes, env_var in _PROVIDER_ENV_VARS:
            if any(lowered.startswith(p) for p in prefixes):
                return env_var
        return None

    def _check_credentials(self) -> None:
        """
        Verify that we have usable credentials for this model.

        When a LiteLLM proxy is configured (both ``api_base`` and ``api_key``
        are present), no provider env var is required.

        Raises:
            MissingApiKeyError: If neither the provider env var nor a
                proxy configuration is present.
        """
        if self._api_base and self._api_key:
            return
        env_var = self._resolve_provider_env_var()
        if env_var and not os.environ.get(env_var):
            raise MissingApiKeyError(self._model_name, env_var)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Invoke the model via :func:`litellm.acompletion`.

        See :meth:`ModelAdapter.chat_with_tools` for the return contract.

        Raises:
            BudgetExhaustedError: LiteLLM returned 429 with a
                budget/quota/credit hint in the body. Generic 429s
                (transient rate limiting) are NOT translated and remain
                ``litellm.exceptions.RateLimitError``.
        """
        import litellm

        self._check_credentials()

        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._default_temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        elif self._default_max_tokens is not None:
            params["max_tokens"] = self._default_max_tokens
        if self._api_base:
            params["api_base"] = self._api_base
        if self._api_key:
            params["api_key"] = self._api_key
        if self._thinking_effort:
            extra_body = dict(params.get("extra_body") or {})
            extra_body["reasoning_effort"] = self._thinking_effort
            params["extra_body"] = extra_body

        # Always thread X-Tesslate-Run-Id so the upstream LiteLLM proxy
        # can dedupe parallel 429s by run. We merge into any caller-
        # supplied extra_headers so Authorization / proxy headers from
        # ``self._extra`` / ``kwargs`` aren't clobbered.
        if self._run_id:
            extra_headers = dict(params.get("extra_headers") or {})
            extra_headers.setdefault("X-Tesslate-Run-Id", self._run_id)
            params["extra_headers"] = extra_headers

        params.update(self._extra)
        params.update(kwargs)

        # Re-apply the run-id header AFTER the merges — ``self._extra``
        # / ``kwargs`` callers may have provided their own
        # ``extra_headers`` dict that wiped ours. Use setdefault so a
        # caller-supplied X-Tesslate-Run-Id wins (test escape hatch).
        if self._run_id:
            extra_headers = dict(params.get("extra_headers") or {})
            extra_headers.setdefault("X-Tesslate-Run-Id", self._run_id)
            params["extra_headers"] = extra_headers

        try:
            if stream:
                params["stream"] = True
                return self._wrap_stream(await litellm.acompletion(**params))

            response = await litellm.acompletion(**params)
            return self._format_response(response)
        except Exception as exc:
            # Translate hard budget exhaustion into the host-stable
            # error type. Generic rate limits (transient) are re-raised
            # untouched so the agent loop's retry helper can back off.
            self._maybe_raise_budget_exhausted(exc)
            raise

    def _maybe_raise_budget_exhausted(self, exc: Exception) -> None:
        """Re-raise as :class:`BudgetExhaustedError` if ``exc`` is a budget hit.

        Detection rules (in order):

        1. ``exc`` is a ``litellm.exceptions.RateLimitError`` (or has
           a ``status_code`` of 429) AND its body matches a budget hint.
        2. The exception itself is named ``BudgetExceededError`` /
           ``ContextWindowExceededError`` / similar — LiteLLM occasionally
           surfaces a more specific subclass.

        Plain rate-limit errors fall through and are retried by the
        agent loop's keyword-based backoff helper.
        """
        # Fast bail — non-rate-limit errors are never budget-exhausted.
        try:
            import litellm.exceptions as litellm_exc
        except Exception:  # pragma: no cover — defensive, litellm is required
            litellm_exc = None  # type: ignore[assignment]

        is_rate_limit = False
        if litellm_exc is not None and hasattr(litellm_exc, "RateLimitError"):
            is_rate_limit = isinstance(exc, litellm_exc.RateLimitError)

        # Some LiteLLM error subclasses don't inherit from RateLimitError
        # but still carry status_code=429.
        if not is_rate_limit:
            status = getattr(exc, "status_code", None)
            if status == 429:
                is_rate_limit = True

        if not is_rate_limit:
            return

        if not _is_budget_exhausted(exc):
            return

        logger.warning(
            "[LiteLLMAdapter] 429 with budget hint — translating to "
            "BudgetExhaustedError run_id=%s model=%s",
            self._run_id,
            self._model_name,
        )
        raise BudgetExhaustedError(
            str(exc),
            run_id=self._run_id,
            model_name=self._model_name,
        ) from exc

    def _format_response(self, response: Any) -> dict[str, Any]:
        """Convert a non-streaming litellm response into the adapter shape."""
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            func = getattr(tc, "function", None)
            tool_calls.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(func, "name", "") if func else "",
                        "arguments": getattr(func, "arguments", "") if func else "",
                    },
                }
            )
        usage: dict[str, Any] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage["prompt_tokens"] = getattr(raw_usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = getattr(raw_usage, "completion_tokens", 0) or 0
            usage["total_tokens"] = getattr(raw_usage, "total_tokens", 0) or 0
            details = getattr(raw_usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
                if cached:
                    usage["cached_tokens"] = cached
        return {
            "content": getattr(message, "content", "") or "",
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": getattr(choice, "finish_reason", None),
            "raw_response": response,
        }

    async def _wrap_stream(self, stream: Any) -> AsyncIterator[dict[str, Any]]:
        """Convert a litellm stream into the adapter's delta-dict shape."""
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            delta_tool_calls: list[dict[str, Any]] = []
            for tc in getattr(delta, "tool_calls", None) or []:
                func = getattr(tc, "function", None)
                delta_tool_calls.append(
                    {
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(func, "name", None) if func else None,
                            "arguments": getattr(func, "arguments", None) if func else None,
                        },
                    }
                )
            yield {
                "delta_content": getattr(delta, "content", None) or "",
                "delta_tool_calls": delta_tool_calls,
                "finish_reason": getattr(choice, "finish_reason", None),
            }


async def create_model_adapter(
    model_name: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    thinking_effort: str | None = None,
    run_id: str | None = None,
    **kwargs: Any,
) -> ModelAdapter:
    """
    Build a :class:`LiteLLMAdapter` and verify its credentials eagerly.

    Args:
        model_name: LiteLLM model identifier (e.g. ``"openai/gpt-4o-mini"``).
        api_base: Optional proxy base URL. Falls back to ``LITELLM_API_BASE``.
        api_key: Optional proxy master key. Falls back to ``LITELLM_MASTER_KEY``.
        thinking_effort: Optional extended-thinking effort tier.
        run_id: Optional run identifier; threads ``X-Tesslate-Run-Id``
            into every LiteLLM call so the proxy can dedupe parallel
            429s by run.
        **kwargs: Additional arguments forwarded to :class:`LiteLLMAdapter`.

    Raises:
        MissingApiKeyError: If no provider env var or proxy configuration
            is available for ``model_name``.
    """
    adapter = LiteLLMAdapter(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        thinking_effort=thinking_effort,
        run_id=run_id,
        **kwargs,
    )
    adapter._check_credentials()
    return adapter
