"""Tests for the ``web_search`` tool — provider calls are fully mocked."""

from __future__ import annotations

from typing import Any

import pytest

from tesslate_agent.agent.tools.web_ops import providers as providers_module
from tesslate_agent.agent.tools.web_ops.providers import SearchResult
from tesslate_agent.agent.tools.web_ops.search import web_search_tool

pytestmark = pytest.mark.asyncio


class _StubProvider:
    """Deterministic stand-in for any concrete ``SearchProvider``."""

    def __init__(self, results: list[SearchResult] | Exception) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls.append((query, max_results))
        if isinstance(self._results, Exception):
            raise self._results
        return list(self._results)


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> _StubProvider:
    """Install a stub provider by patching ``get_search_provider``."""
    stub = _StubProvider(
        [
            SearchResult(
                title="Example A",
                url="https://example.com/a",
                snippet="First result snippet.",
            ),
            SearchResult(
                title="Example B",
                url="https://example.com/b",
                snippet="Second result snippet.",
            ),
        ]
    )

    def _get() -> Any:
        return stub

    monkeypatch.setattr(providers_module, "get_search_provider", _get)
    # The tool imports ``get_search_provider`` lazily inside its executor,
    # so patching the module attribute is sufficient.
    return stub


async def test_search_happy_path(stub_provider: _StubProvider) -> None:
    result = await web_search_tool(
        {"query": "react hooks", "max_results": 2}, {}
    )

    assert result["success"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Example A"
    assert result["results"][0]["url"] == "https://example.com/a"
    assert "snippet" in result["results"][0]

    assert stub_provider.calls == [("react hooks", 2)]


async def test_search_caps_max_results_at_ten(stub_provider: _StubProvider) -> None:
    await web_search_tool({"query": "xyz", "max_results": 999}, {})

    assert stub_provider.calls[-1] == ("xyz", 10)


async def test_search_empty_results_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers_module,
        "get_search_provider",
        lambda: _StubProvider([]),
    )

    result = await web_search_tool({"query": "no results here"}, {})
    assert result["success"] is True
    assert result["results"] == []
    assert "No results" in result["message"]


async def test_search_requires_query() -> None:
    with pytest.raises(ValueError, match="query"):
        await web_search_tool({}, {})


async def test_search_empty_query_returns_error(
    stub_provider: _StubProvider,
) -> None:
    result = await web_search_tool({"query": "   "}, {})
    assert result["success"] is False
    assert "empty" in result["message"].lower()


async def test_search_provider_selection_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any API key env vars, the factory must fall back to DuckDuckGo."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    provider = providers_module.get_search_provider()
    assert isinstance(provider, providers_module.DuckDuckGoProvider)


async def test_search_provider_selection_prefers_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "fake-brave-key")

    provider = providers_module.get_search_provider()
    assert isinstance(provider, providers_module.TavilyProvider)
    assert provider.api_key == "fake-tavily-key"


async def test_search_provider_selection_falls_back_to_brave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "fake-brave-key")

    provider = providers_module.get_search_provider()
    assert isinstance(provider, providers_module.BraveSearchProvider)
    assert provider.api_key == "fake-brave-key"
