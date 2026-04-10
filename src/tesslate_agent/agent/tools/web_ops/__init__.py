"""
Web Operations Module

Tools for fetching web content and searching the web.
"""

from tesslate_agent.agent.tools.web_ops.fetch import register_web_fetch_tool
from tesslate_agent.agent.tools.web_ops.search import register_web_search_tool


def register_web_ops_tools(registry) -> None:
    """Register web fetch and search tools (2 tools)."""
    register_web_fetch_tool(registry)
    register_web_search_tool(registry)


__all__ = [
    "register_web_ops_tools",
    "register_web_fetch_tool",
    "register_web_search_tool",
]
