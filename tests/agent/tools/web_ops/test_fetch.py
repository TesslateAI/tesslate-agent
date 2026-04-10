"""Tests for the ``web_fetch`` tool — HTTP I/O is fully mocked."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tesslate_agent.agent.tools.web_ops import fetch as fetch_module
from tesslate_agent.agent.tools.web_ops.fetch import web_fetch_tool

pytestmark = pytest.mark.asyncio


class _FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.reason_phrase = "OK"

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                "error",
                request=SimpleNamespace(),  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


class _FakeAsyncClient:
    """Drop-in replacement for ``httpx.AsyncClient`` used by the tool."""

    _response_factory: Any = None
    _captured_url: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        _FakeAsyncClient._captured_url = url
        if _FakeAsyncClient._response_factory is None:
            raise AssertionError("FakeAsyncClient: no response factory configured")
        result = _FakeAsyncClient._response_factory(url)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect every ``httpx.AsyncClient`` lookup through our fake."""
    _FakeAsyncClient._response_factory = None
    _FakeAsyncClient._captured_url = None
    monkeypatch.setattr(fetch_module.httpx, "AsyncClient", _FakeAsyncClient)


async def test_fetch_happy_path_with_mocked_html() -> None:
    html = "<html><body>hello world</body></html>"
    _FakeAsyncClient._response_factory = lambda url: _FakeResponse(
        text=html,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
    )

    result = await web_fetch_tool(
        {"url": "https://example.com/page"}, {}
    )

    assert result["success"] is True
    assert result["content"] == html
    assert result["details"]["status_code"] == 200
    assert "text/html" in result["details"]["content_type"]
    assert result["details"]["truncated"] is False
    assert _FakeAsyncClient._captured_url == "https://example.com/page"


async def test_fetch_truncates_large_responses() -> None:
    big_text = "a" * 60_000
    _FakeAsyncClient._response_factory = lambda url: _FakeResponse(
        text=big_text,
        status_code=200,
        headers={"content-type": "text/plain"},
    )

    result = await web_fetch_tool(
        {"url": "https://example.com/big"}, {}
    )

    assert result["success"] is True
    assert result["details"]["truncated"] is True
    # 50_000 chars of content + the "... (truncated)" marker.
    assert len(result["content"]) >= 50_000
    assert "(truncated)" in result["content"]


async def test_fetch_rejects_non_http_url() -> None:
    result = await web_fetch_tool({"url": "ftp://example.com"}, {})

    assert result["success"] is False
    assert "http" in result["message"].lower()


async def test_fetch_requires_url() -> None:
    with pytest.raises(ValueError, match="url"):
        await web_fetch_tool({}, {})


async def test_fetch_http_error_404() -> None:
    _FakeAsyncClient._response_factory = lambda url: _FakeResponse(
        text="not found",
        status_code=404,
    )

    result = await web_fetch_tool(
        {"url": "https://example.com/missing"}, {}
    )

    assert result["success"] is False
    assert "404" in result["message"]
    assert result["details"]["status_code"] == 404


async def test_fetch_timeout_returns_structured_error() -> None:
    def _raise_timeout(url: str) -> Any:
        return httpx.TimeoutException("slow")

    _FakeAsyncClient._response_factory = _raise_timeout

    result = await web_fetch_tool(
        {"url": "https://example.com/slow", "timeout": 1}, {}
    )

    assert result["success"] is False
    assert "timed out" in result["message"].lower()
    assert result["url"] == "https://example.com/slow"
