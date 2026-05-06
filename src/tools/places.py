"""Place lookup, split by job:

- `geocode(query)` — string → (lat, lon, display_name) via OSM Nominatim.
- `search_pois(lat, lon, category, radius_m, ...)` — spatial+tag query
  via the Overpass API. Returns POIs with rich OSM tags (cuisine, brand,
  opening hours, multilingual names, contact info).
- `search_places(destination, category, ...)` — convenience wrapper that
  composes the two: geocode the destination, then run `search_pois`
  around the result.

Why two providers? Nominatim is a *geocoder*: brilliant at "string →
coords", weak at "everything tagged X near Y". Overpass is a query
engine over OSM tags: brilliant at the spatial+tag job, but it doesn't
parse free-form addresses. Together they cover both halves cleanly.

Usage policy: Nominatim caps callers at ~1 req/s and requires a custom
User-Agent (`_common.http_client()` sets it; we serialise with an asyncio
lock). Overpass tolerates more concurrent load but encourages callers
to keep timeouts modest and avoid hammering it; default timeout is 25s.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

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

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
OVERPASS_BASE = "https://overpass-api.de/api/interpreter"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

PoiCategory = Literal["hotel", "restaurant", "cafe", "attraction", "any"]

# Map a coarse category → Overpass tag selectors. Each list of selectors
# is OR-ed together inside the Overpass union block so multiple OSM tag
# values can map onto one user-facing category (e.g. "attraction" covers
# tourism=attraction|museum|viewpoint).
#
# Motels are filtered client-side by hotel_agent._is_motel rather than
# server-side: a `["name"!~"motel|...",i]` clause forces Overpass into a
# full scan (negative regex + case-insensitive flag bypass the tag
# indexes), which reliably blew past the 25 s server timeout / 30 s
# client timeout and starved every hotel run of results.
_OVERPASS_SELECTORS: dict[PoiCategory, list[str]] = {
    "hotel": ['["tourism"~"^(hotel|hostel|guest_house|apartment)$"]'],
    # Fast-food spots (Burger King, taquerías, etc.) are tagged
    # `amenity=fast_food`, not `restaurant`, in OSM — include both so a
    # "burger places" query actually returns burger places.
    "restaurant": ['["amenity"~"^(restaurant|fast_food)$"]'],
    "cafe": ['["amenity"~"^(cafe|coffee_shop)$"]'],
    "attraction": ['["tourism"~"^(attraction|museum|viewpoint|gallery|artwork)$"]'],
    "any": [
        '["amenity"~"^(restaurant|cafe|bar)$"]',
        '["tourism"~"^(hotel|attraction|museum|viewpoint)$"]',
    ],
}


# ── Nominatim plumbing ────────────────────────────────────────────────

# Nominatim's polite-use limit is 1 req/s. Serialise from this process.
_nominatim_lock = asyncio.Lock()
_min_interval_s = 1.05
_last_call_at = 0.0


async def _rate_limit_nominatim() -> None:
    global _last_call_at
    async with _nominatim_lock:
        now = asyncio.get_event_loop().time()
        wait = _min_interval_s - (now - _last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_at = asyncio.get_event_loop().time()


async def _nominatim_search(query: str, limit: int) -> list[dict[str, Any]]:
    await _rate_limit_nominatim()
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": str(limit),
        "addressdetails": "1",
    }
    async with http_client() as client:
        resp = await client.get(f"{NOMINATIM_BASE}/search", params=params)
        resp.raise_for_status()
        return resp.json()


async def _google_geocode(query: str, key: str) -> ToolResult:
    """Geocode via Google's Geocoding API. Quota: 1 google_places call.

    Reaches places Nominatim doesn't index — particularly Latin American
    `carrera #X-Y` notation, conjuntos, and other building-level
    addresses. Counts against `google_places` since it's the same
    monthly Maps Platform credit as findplace/details lookups.
    """
    async def _call() -> ToolResult:
        async with http_client() as client:
            resp = await client.get(
                GOOGLE_GEOCODE_URL,
                params={"address": query, "key": key},
            )
            resp.raise_for_status()
            payload = resp.json()
        status = payload.get("status")
        if status == "ZERO_RESULTS":
            return error_result(
                "google_geocode", "no_results", f"Could not geocode {query!r}"
            )
        if status != "OK":
            return error_result(
                "google_geocode",
                "provider_error",
                f"geocode returned {status}: {payload.get('error_message', '')}",
            )
        results = payload.get("results") or []
        if not results:
            return error_result(
                "google_geocode", "no_results", f"Could not geocode {query!r}"
            )
        top = results[0]
        loc = (top.get("geometry") or {}).get("location") or {}
        return ok_result(
            "google_geocode",
            {
                "lat": float(loc["lat"]),
                "lon": float(loc["lng"]),
                "display_name": top.get("formatted_address"),
            },
        )

    return await with_quota("google_places", _call)


async def _geocode_impl(query: str) -> ToolResult:
    """Geocode `query`. Nominatim primary; Google Geocoding API fallback.

    Why two providers: Nominatim's address index is great for cities and
    landmarks but thin on Latin American building-level addresses
    ("cra 66 #48-106"). Google's Geocoding API handles those cleanly.
    The fallback only fires when Nominatim returns no results AND the
    user has set `GOOGLE_MAPS_API_KEY`; otherwise we surface the original
    Nominatim miss.
    """
    async def _call() -> ToolResult:
        items = await _nominatim_search(query, limit=1)
        if items:
            item = items[0]
            return ok_result(
                "nominatim",
                {
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "display_name": item.get("display_name"),
                },
            )
        return error_result(
            "nominatim", "no_results", f"Could not geocode {query!r}"
        )

    primary = await safe_call("nominatim", _call)
    if primary["ok"]:
        return primary

    settings = get_settings()
    if primary.get("error_type") == "no_results" and settings.google_maps_api_key:
        log.info("geocode: nominatim missed %r — trying Google Geocoding", query)
        google = await _google_geocode(query, settings.google_maps_api_key)
        if google["ok"]:
            return google
        log.info(
            "geocode: Google fallback also failed (%s)",
            google.get("error_type"),
        )
    return primary


# ── Overpass plumbing ─────────────────────────────────────────────────

def _sanitize_overpass_regex(raw: str) -> str:
    """Strip characters that would break an Overpass regex literal.

    Overpass regex syntax is roughly POSIX ERE inside double quotes. The
    characters most likely to slip in from user-supplied free-form text
    are quote marks (would close the literal early) and backslashes.
    Pipes are intentionally allowed so callers can pass alternations
    like "ramen|sushi".
    """
    return raw.replace('"', '').replace('\\', '').strip()


def _build_overpass_query(
    lat: float,
    lon: float,
    category: PoiCategory,
    radius_m: int,
    cuisine: str | None,
    limit: int,
    name_hint: str | None = None,
) -> str:
    selectors = _OVERPASS_SELECTORS[category]
    cuisine_filter = f'["cuisine"~"{cuisine}",i]' if cuisine else ""
    name_filter = ""
    if name_hint:
        clean = _sanitize_overpass_regex(name_hint)
        if clean:
            name_filter = f'["name"~"{clean}",i]'

    body_parts: list[str] = []
    for sel in selectors:
        full = f"{sel}{cuisine_filter}" if category in {"restaurant", "cafe"} else sel
        full = f"{full}{name_filter}"
        body_parts.append(f"  node{full}(around:{radius_m},{lat},{lon});")
        body_parts.append(f"  way{full}(around:{radius_m},{lat},{lon});")
    body = "\n".join(body_parts)

    return (
        f"[out:json][timeout:25];\n"
        f"(\n{body}\n);\n"
        f"out tags center {limit};"
    )


def _normalize_overpass(item: dict[str, Any], category: PoiCategory) -> dict[str, Any]:
    tags = item.get("tags", {}) or {}
    center = item.get("center") or {}
    lat = item.get("lat") or center.get("lat")
    lon = item.get("lon") or center.get("lon")

    name = (
        tags.get("name")
        or tags.get("name:en")
        or tags.get("brand")
        or "(unnamed)"
    )

    address_parts = [
        tags.get(k)
        for k in (
            "addr:housenumber",
            "addr:street",
            "addr:suburb",
            "addr:city",
            "addr:postcode",
            "addr:country",
        )
        if tags.get(k)
    ]
    address = ", ".join(address_parts) if address_parts else None

    return {
        "name": name,
        "name_en": tags.get("name:en"),
        "name_local": tags.get("name") if tags.get("name") != name else None,
        "lat": lat,
        "lon": lon,
        "category": category,
        "osm_type": item.get("type"),
        "osm_id": item.get("id"),
        "address": address,
        "cuisine": tags.get("cuisine"),
        "brand": tags.get("brand"),
        # Raw tag values surfaced for downstream classifiers (e.g. motel
        # detection wants the literal `tourism` value, not just "hotel"
        # the category).
        "tourism": tags.get("tourism"),
        "amenity": tags.get("amenity"),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": tags.get("website") or tags.get("contact:website"),
        "opening_hours": tags.get("opening_hours"),
        "wheelchair": tags.get("wheelchair"),
        "stars": tags.get("stars"),  # hotels often expose this
        "rooms": tags.get("rooms"),
        "tags": tags,  # raw OSM tags for downstream LLM use
    }


async def _search_pois_impl(
    lat: float,
    lon: float,
    category: PoiCategory,
    radius_m: int,
    limit: int,
    cuisine: str | None,
    name_hint: str | None = None,
) -> ToolResult:
    query = _build_overpass_query(
        lat, lon, category, radius_m, cuisine, limit, name_hint=name_hint,
    )

    async def _call() -> ToolResult:
        async with http_client(timeout=30.0) as client:
            resp = await client.post(OVERPASS_BASE, data={"data": query})
            resp.raise_for_status()
            payload = resp.json()
        elements = payload.get("elements") or []
        if not elements:
            return error_result(
                "overpass",
                "no_results",
                f"No {category} POIs within {radius_m}m of ({lat},{lon})",
            )
        return ok_result(
            "overpass",
            [_normalize_overpass(e, category) for e in elements[:limit]],
        )

    return await safe_call("overpass", _call)


# ── Public @tool surface ──────────────────────────────────────────────

@tool
async def geocode(query: str) -> ToolResult:
    """Resolve a free-form location string to (lat, lon) via Nominatim."""
    log.info("geocode %r", query)
    return await _geocode_impl(query)


@tool
async def search_pois(
    lat: float,
    lon: float,
    category: PoiCategory = "any",
    radius_m: int = 1000,
    limit: int = 20,
    cuisine: str | None = None,
    name_hint: str | None = None,
) -> ToolResult:
    """Find POIs near a point via OSM Overpass, filtered by tag category.

    Args:
        lat, lon: query centre.
        category: "hotel" | "restaurant" | "cafe" | "attraction" | "any".
        radius_m: search radius in metres.
        limit: max POIs returned.
        cuisine: optional regex to filter restaurants/cafes by cuisine
                 (e.g. "japanese", "italian|french", "ramen").
        name_hint: optional case-insensitive regex applied to the OSM
                 `name` tag. Useful for surfacing query-specific results
                 like "rooftop" or "vegetariano" that aren't captured by
                 the cuisine tag. Quotes/backslashes are stripped; pipes
                 are kept so callers may pass alternations.

    Returns rich OSM tag data — name (multilingual), address, cuisine,
    brand, opening hours, contact info, accessibility, plus the raw tag
    dict under `tags` for downstream LLM use.
    """
    log.info(
        "search_pois (%s,%s) cat=%s r=%dm limit=%d cuisine=%s name_hint=%s",
        lat, lon, category, radius_m, limit, cuisine, name_hint,
    )
    return await _search_pois_impl(
        lat, lon, category, radius_m, limit, cuisine, name_hint=name_hint,
    )


@tool
async def search_places(
    destination: str,
    category: PoiCategory = "any",
    limit: int = 10,
    radius_m: int | None = None,
    cuisine: str | None = None,
) -> ToolResult:
    """Convenience: geocode a destination string, then run `search_pois`.

    Args:
        destination: free-form text, e.g. "Shibuya, Tokyo".
        category: see `search_pois`.
        limit: max POIs returned.
        radius_m: optional override; defaults to 2000m for hotels (people
                  pick lodging within walking-distance of an area) and
                  1000m for everything else.
        cuisine: optional cuisine filter (restaurants/cafes only).
    """
    log.info("search_places %r cat=%s", destination, category)

    geo = await _geocode_impl(destination)
    if not geo["ok"]:
        return geo  # already a structured error_result

    lat = geo["data"]["lat"]
    lon = geo["data"]["lon"]
    radius = radius_m or (2000 if category == "hotel" else 1000)
    return await _search_pois_impl(lat, lon, category, radius, limit, cuisine)
