"""Reviews — structured user reviews and grounded narrative summaries.

Two tools, two distinct providers, two distinct quotas:

- `get_reviews(query, max_reviews)` — **primary**. Google Maps Places API
  Place Details, returning real user-written reviews with author / rating /
  timestamp / text + aggregate rating, review count, address, hours,
  price level, website. Quota provider: `google_places` (each lookup costs
  2 API calls: findplacefromtext + details).

- `get_grounded_summary(place, question)` — **opt-in narrative**. Gemini
  with the `googleMaps` tool (Maps Grounding Lite). Returns LLM-mediated
  prose grounded on Maps data — useful for trip-summary blurbs, *not* for
  raw review snippets. Quota provider: `google_maps_grounding` (consumes
  the Gemini API quota, which is shared with every other Gemini call in
  the app — use sparingly).

Both tools cache results in-process with TTLs so repeated lookups in the
same trip-planning session don't burn quota. The cache is intentionally
in-memory only; persistence is left as a future enhancement.

Notes on key handling:
- The Places Maps API only supports `?key=...` query auth. Errors are
  routed through `_common.safe_call`, which redacts the key from log /
  error_result messages via the shared `redact()` helper.
- The Gemini API supports header auth; we use `x-goog-api-key` so the
  key never appears in URLs in the first place.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from langchain_core.tools import tool

from src.config import get_settings
from src.tools._common import (
    ToolResult,
    error_result,
    http_client,
    ok_result,
    redact,
    safe_call,
    with_quota,
)

log = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────

_FIND_PLACE_TTL_S = 30 * 24 * 3600  # place_ids are essentially permanent
_DETAILS_TTL_S = 24 * 3600           # reviews refresh slowly; 24h is safe
_GROUNDING_TTL_S = 12 * 3600          # narrative summaries change rarely

_find_cache: dict[str, tuple[float, str]] = {}
_details_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_grounding_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _norm_key(s: str) -> str:
    return " ".join(s.lower().split())


def _cache_get(cache: dict, key: str) -> Any | None:
    item = cache.get(key)
    if item and item[0] > time.time():
        return item[1]
    if item:
        cache.pop(key, None)
    return None


def _cache_put(cache: dict, key: str, value: Any, ttl: int) -> None:
    cache[key] = (time.time() + ttl, value)


def cache_stats() -> dict[str, int]:
    """Cheap visibility for tests / debugging."""
    return {
        "find_place": len(_find_cache),
        "details": len(_details_cache),
        "grounding": len(_grounding_cache),
    }


# ── Path B: Google Places Details API ────────────────────────────────

PLACES_BASE = "https://maps.googleapis.com/maps/api/place"
DETAILS_FIELDS = (
    "name,rating,user_ratings_total,reviews,price_level,"
    "formatted_address,website,opening_hours,url"
)


async def _find_place(
    query: str,
    key: str,
    lat: float | None = None,
    lon: float | None = None,
    bias_radius_m: int = 5000,
) -> ToolResult:
    """Resolve a free-form query to a `place_id`. Quota: 1 google_places call.

    When `lat`/`lon` are provided, the request includes `locationbias` so
    Google's matcher prefers nearby candidates. Without it, common names
    ("Frutoss Restaurante Vegetariano") match the most popular global
    instance and pollute results with places from other cities.
    """
    bias_suffix = (
        f"@{round(lat, 3)},{round(lon, 3)}r{bias_radius_m}"
        if lat is not None and lon is not None
        else ""
    )
    cache_key = _norm_key(query) + bias_suffix
    cached = _cache_get(_find_cache, cache_key)
    if cached is not None:
        log.info("places find_place cache HIT for %r", query)
        return ok_result("google_places", {"place_id": cached, "cached": True})

    async def _call() -> ToolResult:
        params: dict[str, Any] = {
            "input": query,
            "inputtype": "textquery",
            "fields": "place_id,name",
            "key": key,
        }
        if lat is not None and lon is not None:
            params["locationbias"] = f"circle:{bias_radius_m}@{lat},{lon}"
        async with http_client() as client:
            resp = await client.get(
                f"{PLACES_BASE}/findplacefromtext/json",
                params=params,
            )
            resp.raise_for_status()
            payload = resp.json()

        status = payload.get("status")
        if status != "OK":
            return error_result(
                "google_places",
                "no_results" if status == "ZERO_RESULTS" else "provider_error",
                f"findplace returned {status}: {payload.get('error_message', '')}",
            )
        cands = payload.get("candidates") or []
        if not cands:
            return error_result(
                "google_places", "no_results", f"No place matches {query!r}"
            )
        place_id = cands[0]["place_id"]
        _cache_put(_find_cache, cache_key, place_id, _FIND_PLACE_TTL_S)
        return ok_result("google_places", {"place_id": place_id, "cached": False})

    return await with_quota("google_places", _call)


async def _place_details(place_id: str, key: str) -> ToolResult:
    """Fetch reviews + metadata for a place_id. Quota: 1 google_places call."""
    cached = _cache_get(_details_cache, place_id)
    if cached is not None:
        log.info("places details cache HIT for %s", place_id)
        return ok_result("google_places", {**cached, "cached": True})

    async def _call() -> ToolResult:
        async with http_client() as client:
            resp = await client.get(
                f"{PLACES_BASE}/details/json",
                params={
                    "place_id": place_id,
                    "fields": DETAILS_FIELDS,
                    "key": key,
                    "reviews_no_translations": "true",
                },
            )
            resp.raise_for_status()
            payload = resp.json()

        status = payload.get("status")
        if status != "OK":
            return error_result(
                "google_places",
                "provider_error",
                f"details returned {status}: {payload.get('error_message', '')}",
            )

        result = payload.get("result") or {}
        normalized = _normalize_details(result, place_id)
        _cache_put(_details_cache, place_id, normalized, _DETAILS_TTL_S)
        return ok_result("google_places", {**normalized, "cached": False})

    return await with_quota("google_places", _call)


def _normalize_details(result: dict[str, Any], place_id: str) -> dict[str, Any]:
    return {
        "place_id": place_id,
        "name": result.get("name"),
        "rating": result.get("rating"),
        "review_count": result.get("user_ratings_total"),
        "address": result.get("formatted_address"),
        "website": result.get("website"),
        "google_url": result.get("url"),
        "price_level": result.get("price_level"),
        "open_now": (result.get("opening_hours") or {}).get("open_now"),
        "weekday_hours": (result.get("opening_hours") or {}).get("weekday_text") or [],
        "reviews": [
            {
                "author": r.get("author_name"),
                "rating": r.get("rating"),
                "time": r.get("relative_time_description"),
                "text": r.get("text"),
                "language": r.get("language"),
            }
            for r in (result.get("reviews") or [])
        ],
    }


# ── Path C: Maps Grounding Lite (Gemini googleMaps tool) ─────────────

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


async def _maps_grounding(prompt: str, key: str) -> ToolResult:
    """Single-shot Gemini call with the `googleMaps` tool enabled."""
    cache_key = _norm_key(prompt)
    cached = _cache_get(_grounding_cache, cache_key)
    if cached is not None:
        log.info("maps_grounding cache HIT")
        return ok_result("google_maps_grounding", {**cached, "cached": True})

    async def _call() -> ToolResult:
        # Header auth = no key in URL = nothing to leak through httpx errors.
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"googleMaps": {}}],  # camelCase variant — the snake one is rejected
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(GEMINI_URL, headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()

        cands = payload.get("candidates") or [{}]
        cand = cands[0]
        text = "".join(
            p.get("text", "") for p in (cand.get("content", {}).get("parts") or [])
        )
        gmeta = cand.get("groundingMetadata") or {}
        chunks = gmeta.get("groundingChunks") or []
        if not text and not chunks:
            return error_result(
                "google_maps_grounding",
                "no_results",
                "Gemini returned no grounded answer",
            )

        out = {
            "answer": text,
            "sources": [
                {
                    "title": (ch.get("maps") or ch.get("web") or {}).get("title"),
                    "uri": (ch.get("maps") or ch.get("web") or {}).get("uri"),
                }
                for ch in chunks
            ],
        }
        _cache_put(_grounding_cache, cache_key, out, _GROUNDING_TTL_S)
        return ok_result("google_maps_grounding", {**out, "cached": False})

    return await with_quota("google_maps_grounding", _call)


# ── Public @tool surface ─────────────────────────────────────────────

@tool
async def get_reviews(
    query: str,
    max_reviews: int = 5,
    lat: float | None = None,
    lon: float | None = None,
) -> ToolResult:
    """Fetch user reviews + metadata for a place.

    Args:
        query: free-form name, e.g. "Ichiran Shibuya".
        max_reviews: cap on returned review snippets (1-5 — Google
            returns at most 5 per place).
        lat, lon: optional caller-supplied coordinates for location bias.
            When set, Google's matcher prefers nearby candidates within
            ~5km. Strongly recommended when enriching OSM POIs since
            common names match the most popular global instance otherwise
            (e.g. "Frutoss Restaurante Vegetariano" → Cali instead of
            Santa Marta).

    Returns structured data: rating, review_count, top reviews with
    author/rating/time/text, address, hours, price_level, website,
    google_url, open_now status. `cached=True` flag is set when the
    response is served from the in-process cache (no quota consumed).

    Costs 2 quota slots on `google_places` per uncached lookup
    (findplacefromtext + place details). Cached lookups consume zero.
    """
    settings = get_settings()
    if not settings.google_maps_api_key:
        return error_result(
            "google_places",
            "missing_config",
            "GOOGLE_MAPS_API_KEY not set",
        )

    log.info("get_reviews %r (max=%d)", query, max_reviews)

    found = await _find_place(
        query, settings.google_maps_api_key, lat=lat, lon=lon
    )
    if not found["ok"]:
        return found

    place_id = found["data"]["place_id"]
    details = await _place_details(place_id, settings.google_maps_api_key)
    if not details["ok"]:
        return details

    payload = dict(details["data"])
    payload["reviews"] = payload.get("reviews", [])[: max(1, min(max_reviews, 5))]
    payload["query"] = query
    return ok_result("google_places", payload)


@tool
async def get_grounded_summary(place: str, question: str | None = None) -> ToolResult:
    """Grounded narrative summary of a place via Maps Grounding Lite.

    **Opt-in.** Consumes the Gemini API quota — which is the same quota
    every other Gemini call in this app uses. Prefer `get_reviews` for
    raw review data; reach for this only when an LLM-mediated paragraph
    is what you actually need (e.g., the Synthesizer composing trip
    blurbs).

    Args:
        place: name of the place, e.g. "Park Hyatt Tokyo".
        question: optional specific angle. Defaults to a general "what
            do visitors think" prompt.

    Returns `{answer: str, sources: [{title, uri}, ...], cached: bool}`.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        return error_result(
            "google_maps_grounding",
            "missing_config",
            "GEMINI_API_KEY not set",
        )

    q = question or (
        f"Summarize what visitors and reviewers say about {place}. "
        f"Include the typical rating, the most common pros and cons, "
        f"and cite Google Maps places where useful."
    )
    log.info("get_grounded_summary %r question=%r", place, question)
    return await _maps_grounding(q, settings.gemini_api_key)
