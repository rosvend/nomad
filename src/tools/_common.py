"""Shared types & helpers for the tools layer.

Every tool returns a `ToolResult` dict in one of two shapes:

    success:  {"ok": True,  "provider": "<name>", "data": <payload>}
    failure:  {"ok": False, "provider": "<name>", "error_type": "...",
               "message": "...", "detail": <optional dict>}

This contract is what `graph/edges.py` predicates branch on, and what
agents copy into `state["errors"]` when something goes wrong. Tools
**never raise** — all paths funnel through `ok_result` / `error_result`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict

import httpx

from src.tools.quota import QuotaExceededError, quota_tracker

log = logging.getLogger("src.tools")

DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
USER_AGENT = "nomad-travel-planner/0.1 (https://github.com/rosvend/nomad)"

# Strip API keys from any string headed for a log or error_result. Most
# Google APIs auth via `?key=AIza...` query param; httpx exception messages
# echo the full URL, which means an unredacted error leaks the key into
# the conversation. Redact aggressively.
_KEY_RE = re.compile(
    r"(?P<param>(?:^|[?&\s])(?:key|api[_-]?key|apikey))=[A-Za-z0-9_\-]+",
    re.IGNORECASE,
)


def redact(s: str) -> str:
    """Replace `key=...` / `api_key=...` patterns with `key=REDACTED`."""
    return _KEY_RE.sub(lambda m: f"{m.group('param')}=REDACTED", s)


class ToolSuccess(TypedDict):
    ok: Literal[True]
    provider: str
    data: Any


class ToolError(TypedDict, total=False):
    ok: Literal[False]
    provider: str
    error_type: str  # "quota_exceeded" | "network_error" | "provider_error" | "no_results" | "missing_config"
    message: str
    detail: dict[str, Any]


ToolResult = ToolSuccess | ToolError


def ok_result(provider: str, data: Any) -> ToolSuccess:
    return {"ok": True, "provider": provider, "data": data}


def error_result(
    provider: str,
    error_type: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> ToolError:
    err: ToolError = {
        "ok": False,
        "provider": provider,
        "error_type": error_type,
        "message": message,
    }
    if detail is not None:
        err["detail"] = detail
    return err


def http_client(**kwargs: Any) -> httpx.AsyncClient:
    """Construct an httpx.AsyncClient with sane defaults (UA, timeout)."""
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    return httpx.AsyncClient(
        headers=headers,
        timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT),
        follow_redirects=True,
        **kwargs,
    )


async def safe_call(
    provider: str,
    fn: Callable[[], Awaitable[ToolResult]],
) -> ToolResult:
    """Run `fn`, converting any exception into a structured error dict.

    Use this to wrap every external call from a tool — it guarantees the
    "tools never raise" contract.
    """
    try:
        return await fn()
    except httpx.HTTPError as e:
        msg = redact(str(e))
        log.warning("network error calling %s: %s", provider, msg)
        return error_result(provider, "network_error", msg)
    except Exception as e:  # noqa: BLE001 — last-resort guard, see CLAUDE.md
        log.exception("unexpected error in tool %s", provider)
        return error_result(provider, "provider_error", redact(str(e)))


async def with_quota(
    provider: str,
    fn: Callable[[], Awaitable[ToolResult]],
) -> ToolResult:
    """Reserve a quota slot for `provider` and then run `fn` via `safe_call`.

    Use for paid / rate-limited providers (SerpApi, Tavily, Google Maps).
    """
    try:
        quota_tracker.check_and_increment(provider)
    except QuotaExceededError as e:
        log.warning("quota exceeded: %s", e)
        return error_result(
            provider,
            "quota_exceeded",
            str(e),
            detail={"used": e.used, "limit": e.limit},
        )

    return await safe_call(provider, fn)
