"""Web search.

Primary:  SearXNG (self-hosted via docker-compose, no key, no quota).
Fallback: Tavily search API (requires `TAVILY_API_KEY`, quota-tracked at
          1000/month free).

Output is a list of `{title, url, snippet}` dicts so downstream agents
can pick which results to fetch full content from via `web_fetch`.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.config import get_settings
from src.tools._common import (
    ToolResult,
    error_result,
    http_client,
    ok_result,
    safe_call,
    with_quota,
)

log = logging.getLogger(__name__)


async def _search_searxng(query: str, limit: int) -> ToolResult:
    settings = get_settings()

    async def _call() -> ToolResult:
        params = {"q": query, "format": "json"}
        async with http_client() as client:
            resp = await client.get(f"{settings.searxng_base_url}/search", params=params)
            resp.raise_for_status()
            payload = resp.json()

        results = payload.get("results") or []
        items = [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
            for r in results[:limit]
        ]
        if not items:
            return error_result("searxng", "no_results", f"No results for {query!r}")
        return ok_result("searxng", items)

    return await safe_call("searxng", _call)


async def _search_tavily(query: str, limit: int) -> ToolResult:
    settings = get_settings()
    if not settings.tavily_api_key:
        return error_result(
            "tavily",
            "missing_config",
            "TAVILY_API_KEY not set; cannot use Tavily fallback",
        )

    async def _call() -> ToolResult:
        body: dict[str, Any] = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
        }
        async with http_client() as client:
            resp = await client.post("https://api.tavily.com/search", json=body)
            resp.raise_for_status()
            payload = resp.json()

        results = payload.get("results") or []
        items = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
            }
            for r in results
        ]
        if not items:
            return error_result("tavily", "no_results", f"No results for {query!r}")
        return ok_result("tavily", items)

    return await with_quota("tavily", _call)


@tool
async def web_search(query: str, limit: int = 5) -> ToolResult:
    """Search the web for `query`.

    SearXNG is tried first (self-hosted, no key required). On failure,
    falls back to Tavily if `TAVILY_API_KEY` is set.
    """
    log.info("web_search %r (limit=%d)", query, limit)
    primary = await _search_searxng(query, limit)
    if primary["ok"]:
        return primary

    log.info("searxng failed (%s); trying Tavily", primary.get("error_type"))
    fallback = await _search_tavily(query, limit)
    return fallback if fallback["ok"] else primary
