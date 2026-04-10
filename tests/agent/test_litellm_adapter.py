"""Tests for the LiteLLM model adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from tesslate_agent.agent.models import (
    LiteLLMAdapter,
    MissingApiKeyError,
    ModelAdapter,
    create_model_adapter,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch) -> None:
    """Strip every provider/proxy env var before each test for deterministic checks."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "PERPLEXITYAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "LITELLM_API_BASE",
        "LITELLM_MASTER_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


async def test_create_model_adapter_with_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    adapter = await create_model_adapter("openai/gpt-4o-mini")
    assert isinstance(adapter, LiteLLMAdapter)
    assert isinstance(adapter, ModelAdapter)
    assert adapter.model_name == "openai/gpt-4o-mini"


async def test_missing_key_raises_with_provider_env_var() -> None:
    with pytest.raises(MissingApiKeyError) as exc_info:
        await create_model_adapter("openai/gpt-4o-mini")
    assert exc_info.value.env_var == "OPENAI_API_KEY"
    assert exc_info.value.model_name == "openai/gpt-4o-mini"
    assert "OPENAI_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize(
    "model_name,expected_env",
    [
        ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic/claude-3-5-sonnet", "ANTHROPIC_API_KEY"),
        ("claude-3-5-haiku", "ANTHROPIC_API_KEY"),
        ("openrouter/meta-llama/llama-3", "OPENROUTER_API_KEY"),
        ("groq/llama-3.1-70b", "GROQ_API_KEY"),
        ("together_ai/meta-llama/llama-3", "TOGETHER_API_KEY"),
        ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
        ("gemini/gemini-2.0-flash", "GEMINI_API_KEY"),
    ],
)
async def test_provider_detection(model_name: str, expected_env: str) -> None:
    with pytest.raises(MissingApiKeyError) as exc_info:
        await create_model_adapter(model_name)
    assert exc_info.value.env_var == expected_env


async def test_litellm_proxy_satisfies_credentials(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_BASE", "http://proxy.local:4000")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-proxy")
    adapter = await create_model_adapter("openai/gpt-4o-mini")
    assert adapter.model_name == "openai/gpt-4o-mini"


async def test_proxy_args_override_env(monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    adapter = await create_model_adapter(
        "openai/gpt-4o-mini",
        api_base="http://explicit.local",
        api_key="sk-explicit",
    )
    assert adapter.model_name == "openai/gpt-4o-mini"


def _fake_response(
    content: str = "done",
    tool_call_name: str | None = None,
    tool_call_args: str = "{}",
) -> Any:
    tool_calls = None
    if tool_call_name is not None:
        tool_calls = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name=tool_call_name, arguments=tool_call_args),
            )
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


async def test_chat_with_tools_non_streaming(monkeypatch) -> None:
    import litellm

    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response(content="hello", tool_call_name="write_file")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = await create_model_adapter("openai/gpt-4o-mini")
    result = await adapter.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )

    assert isinstance(result, dict)
    assert result["content"] == "hello"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
    assert result["usage"]["cached_tokens"] == 3
    assert result["finish_reason"] == "stop"
    assert captured["model"] == "openai/gpt-4o-mini"
    assert "tools" in captured
    assert captured["tool_choice"] == "auto"


async def test_chat_with_tools_streaming(monkeypatch) -> None:
    import litellm

    async def fake_stream() -> AsyncIterator[Any]:
        deltas = [
            SimpleNamespace(
                content="Hel",
                tool_calls=None,
            ),
            SimpleNamespace(
                content="lo",
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_x",
                        function=SimpleNamespace(name="bash_exec", arguments='{"cmd":'),
                    )
                ],
            ),
        ]
        for i, delta in enumerate(deltas):
            finish = "stop" if i == len(deltas) - 1 else None
            choice = SimpleNamespace(delta=delta, finish_reason=finish)
            yield SimpleNamespace(choices=[choice])

    async def fake_acompletion(**kwargs: Any) -> Any:
        assert kwargs.get("stream") is True
        return fake_stream()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = await create_model_adapter("openai/gpt-4o-mini")
    stream = await adapter.chat_with_tools(
        messages=[{"role": "user", "content": "go"}],
        stream=True,
    )

    collected: list[dict[str, Any]] = []
    async for chunk in stream:
        collected.append(chunk)

    assert len(collected) == 2
    assert collected[0]["delta_content"] == "Hel"
    assert collected[1]["delta_content"] == "lo"
    assert collected[1]["delta_tool_calls"][0]["function"]["name"] == "bash_exec"
    assert collected[1]["finish_reason"] == "stop"


async def test_thinking_effort_adds_extra_body(monkeypatch) -> None:
    import litellm

    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = await create_model_adapter(
        "openai/gpt-4o-mini", thinking_effort="high"
    )
    await adapter.chat_with_tools(messages=[{"role": "user", "content": "hi"}])

    assert captured["extra_body"] == {"reasoning_effort": "high"}


async def test_missing_key_exception_includes_proxy_hint() -> None:
    with pytest.raises(MissingApiKeyError) as exc_info:
        await create_model_adapter("anthropic/claude-3-5-sonnet")
    message = str(exc_info.value)
    assert "LITELLM_API_BASE" in message
    assert "LITELLM_MASTER_KEY" in message


async def test_unknown_model_without_key_does_not_raise() -> None:
    """An unknown provider has no env var mapping — should not raise on construction."""
    adapter = await create_model_adapter("unknown-custom/my-model")
    assert adapter.model_name == "unknown-custom/my-model"
