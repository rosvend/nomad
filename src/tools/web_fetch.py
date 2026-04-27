"""Fetch and extract text content from a URL.

Default path is plain `httpx` + a small HTML→text pass — fast, no JS.
For pages that require JS rendering (e.g. SPAs), pass `render=True` to
opt into the `crawl4ai` browser-driven path; that's heavier (Playwright)
but produces far cleaner output.
"""

from __future__ import annotations

import logging
import re

from langchain_core.tools import tool

from src.tools._common import (
    ToolResult,
    error_result,
    http_client,
    ok_result,
)

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_CHARS = 20_000  # cap returned text so it fits in an LLM prompt


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


async def _fetch_static(url: str) -> ToolResult:
    async with http_client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = _strip_html(resp.text)[:_MAX_CHARS]
    return ok_result(
        "httpx",
        {"url": url, "title": None, "text": text, "char_count": len(text)},
    )


async def _fetch_rendered(url: str) -> ToolResult:
    """JS-rendered fetch via crawl4ai. Heavier; only when caller asks."""
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import-not-found]
    except ImportError:
        return error_result(
            "crawl4ai",
            "missing_config",
            "crawl4ai not installed; install or set render=False",
        )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        text = (result.markdown or result.cleaned_html or "")[:_MAX_CHARS]
    return ok_result(
        "crawl4ai",
        {
            "url": url,
            "title": getattr(result, "metadata", {}).get("title"),
            "text": text,
            "char_count": len(text),
        },
    )


@tool
async def fetch_page(url: str, render: bool = False) -> ToolResult:
    """Fetch a URL and return its plain-text content.

    Args:
        url: full URL to fetch.
        render: if True, use crawl4ai (browser) for JS-heavy pages.
                Otherwise httpx fetches the raw HTML and we strip tags.
    """
    log.info("fetch_page %s render=%s", url, render)
    if render:
        return await _fetch_rendered(url)
    try:
        return await _fetch_static(url)
    except Exception as e:  # noqa: BLE001
        return error_result("httpx", "network_error", str(e))
