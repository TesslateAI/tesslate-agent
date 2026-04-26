"""Tests for BudgetExhaustedError translation in LiteLLMAdapter.

Covers the contract documented on
:class:`tesslate_agent.agent.models.LiteLLMAdapter`:

* 429 with a ``budget`` / ``quota`` / ``credit`` hint in the body →
  re-raised as :class:`tesslate_agent.errors.BudgetExhaustedError`,
  carrying ``run_id`` + ``model_name``.
* Generic 429 (transient rate limiting, no budget hint) → falls through
  as the original ``RateLimitError`` so the agent loop's retry helper
  can still back off.
* Every ``litellm.acompletion`` call carries
  ``X-Tesslate-Run-Id: <run_id>`` in ``extra_headers`` when the adapter
  was constructed with a ``run_id``.
* :func:`tesslate_agent.agent.tesslate_agent._is_retryable_error` returns
  ``False`` for :class:`BudgetExhaustedError` even though its message
  contains "rate limit" / "429".
"""

from __future__ import annotations

from typing import Any

import pytest

from tesslate_agent.agent.models import LiteLLMAdapter, create_model_adapter
from tesslate_agent.agent.tesslate_agent import _is_retryable_error
from tesslate_agent.errors import AgentError, BudgetExhaustedError


@pytest.fixture(autouse=True)
def _set_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter construction validates credentials eagerly; supply a stub."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _fake_response_dict() -> Any:
    """Return a minimal litellm response object the adapter can format."""
    from types import SimpleNamespace

    message = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# 429 → BudgetExhaustedError translation
# ---------------------------------------------------------------------------


async def test_429_with_budget_hint_translates_to_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 + 'budget exceeded' body → BudgetExhaustedError with run_id."""
    import litellm
    import litellm.exceptions as litellm_exc

    async def fake_acompletion(**_kwargs: Any) -> Any:
        # Mimic litellm's RateLimitError with a body that hints budget.
        raise litellm_exc.RateLimitError(
            message="Rate limit exceeded: monthly budget exhausted",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = await create_model_adapter(
        "openai/gpt-4o-mini", run_id="run-abc-123"
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        await adapter.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}]
        )

    assert exc_info.value.run_id == "run-abc-123"
    assert exc_info.value.model_name == "openai/gpt-4o-mini"
    # Original message is preserved in the cause chain.
    assert isinstance(exc_info.value.__cause__, litellm_exc.RateLimitError)
    # And the message is set sensibly.
    assert "budget" in str(exc_info.value).lower()


async def test_429_with_quota_hint_also_translates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'quota'-flavoured 429 also translates — same family of conditions."""
    import litellm
    import litellm.exceptions as litellm_exc

    async def fake_acompletion(**_kwargs: Any) -> Any:
        raise litellm_exc.RateLimitError(
            message="You have exceeded your monthly quota",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter("openai/gpt-4o-mini", run_id="r-2")

    with pytest.raises(BudgetExhaustedError):
        await adapter.chat_with_tools(messages=[{"role": "user", "content": "x"}])


async def test_generic_429_without_budget_hint_does_not_translate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain 'too many requests' 429 must NOT be classified as budget-exhausted.

    Generic rate limits are transient; the agent loop's retry helper
    should still back off and retry.
    """
    import litellm
    import litellm.exceptions as litellm_exc

    async def fake_acompletion(**_kwargs: Any) -> Any:
        raise litellm_exc.RateLimitError(
            message="429 Too Many Requests; please slow down",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter("openai/gpt-4o-mini", run_id="r-3")

    # The original RateLimitError surfaces — NOT BudgetExhaustedError.
    with pytest.raises(litellm_exc.RateLimitError):
        await adapter.chat_with_tools(
            messages=[{"role": "user", "content": "x"}]
        )


async def test_non_429_errors_pass_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout / 500 must propagate as-is; never translated."""
    import litellm

    class FakeTimeout(Exception):
        pass

    async def fake_acompletion(**_kwargs: Any) -> Any:
        raise FakeTimeout("connection timed out — budget mention is irrelevant")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter("openai/gpt-4o-mini", run_id="r-4")

    # Even though the message includes "budget", the exception isn't a
    # 429 / RateLimitError, so it must NOT translate.
    with pytest.raises(FakeTimeout):
        await adapter.chat_with_tools(
            messages=[{"role": "user", "content": "x"}]
        )


# ---------------------------------------------------------------------------
# X-Tesslate-Run-Id header threading
# ---------------------------------------------------------------------------


async def test_run_id_is_injected_as_extra_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_id`` must surface as ``extra_headers['X-Tesslate-Run-Id']``."""
    import litellm

    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response_dict()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter(
        "openai/gpt-4o-mini", run_id="run-XYZ"
    )

    await adapter.chat_with_tools(messages=[{"role": "user", "content": "hi"}])

    extra_headers = captured.get("extra_headers") or {}
    assert extra_headers.get("X-Tesslate-Run-Id") == "run-XYZ"


async def test_no_run_id_means_no_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a ``run_id``, no ``X-Tesslate-Run-Id`` header is added.

    This keeps the host free to opt out (e.g., for ad-hoc CLI calls
    that have no run identity).
    """
    import litellm

    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response_dict()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter("openai/gpt-4o-mini")  # no run_id

    await adapter.chat_with_tools(messages=[{"role": "user", "content": "hi"}])

    extra_headers = captured.get("extra_headers") or {}
    assert "X-Tesslate-Run-Id" not in extra_headers


async def test_caller_supplied_extra_headers_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run-id injection must not clobber caller-supplied extra_headers.

    Keys other than ``X-Tesslate-Run-Id`` must survive verbatim — the
    adapter merges instead of overwriting.
    """
    import litellm

    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response_dict()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter = await create_model_adapter(
        "openai/gpt-4o-mini", run_id="run-Z"
    )

    await adapter.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Custom-Trace": "abc"},
    )

    extra_headers = captured.get("extra_headers") or {}
    # If caller supplied a custom header, it must stick. The run-id
    # header is also preserved (caller's dict + ours both win).
    assert extra_headers.get("X-Custom-Trace") == "abc"


# ---------------------------------------------------------------------------
# Retry classifier interaction
# ---------------------------------------------------------------------------


def test_budget_exhausted_is_never_retryable() -> None:
    """``_is_retryable_error`` must return False for BudgetExhaustedError.

    Even though the underlying 429 message contains "rate limit" / "429"
    / "budget exceeded" — all of which would trigger the keyword
    classifier — the type check short-circuits to False so the agent
    loop hands the error straight back to the host.
    """
    err = BudgetExhaustedError(
        "Rate limit exceeded: budget exhausted (429)",
        run_id="r-1",
        model_name="openai/gpt-4o-mini",
    )
    assert _is_retryable_error(err) is False


def test_generic_rate_limit_string_is_still_retryable() -> None:
    """The classifier still sees plain RateLimitError-shaped strings as
    retryable. We didn't break the existing behaviour for non-budget 429s.
    """

    class FakeRateLimit(Exception):
        pass

    err = FakeRateLimit("429 rate limit exceeded; please retry")
    assert _is_retryable_error(err) is True


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------


def test_budget_exhausted_inherits_agent_error() -> None:
    """The host can catch ``AgentError`` to handle every domain error.

    Keeps a stable umbrella type for callers that don't care to
    distinguish between budget vs. other terminal conditions.
    """
    err = BudgetExhaustedError("x")
    assert isinstance(err, AgentError)
    assert isinstance(err, Exception)
