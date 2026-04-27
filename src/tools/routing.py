"""Routing — directions between two coordinates.

Primary:  OSRM public demo server (free, no key, no signup).
Fallback: Google Maps Directions API (requires `GOOGLE_MAPS_API_KEY`,
          quota-tracked at 10k/month free tier).

OSRM modes are encoded in the URL path: `driving`, `walking`, `cycling`.
We map the `mode` Literal to those.

Caveat: the public OSRM demo at router.project-osrm.org typically only
ships the `car` profile, so requesting `foot`/`bike` may silently return
car-speed durations. For accurate non-driving routes either self-host
OSRM with the relevant profiles or configure the Google Maps fallback.
"""

from __future__ import annotations

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

OSRM_BASE = "https://router.project-osrm.org"

TransportMode = Literal["walk", "drive", "bike", "transit", "taxi"]

_OSRM_PROFILE: dict[TransportMode, str] = {
    "walk": "foot",
    "drive": "car",
    "bike": "bike",
    "taxi": "car",
    # OSRM doesn't have transit; callers should fall back to Google for that.
    "transit": "car",
}


async def _route_osrm(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    mode: TransportMode,
) -> ToolResult:
    profile = _OSRM_PROFILE[mode]

    async def _call() -> ToolResult:
        url = (
            f"{OSRM_BASE}/route/v1/{profile}/"
            f"{from_lon},{from_lat};{to_lon},{to_lat}"
        )
        params = {"overview": "false", "alternatives": "false", "steps": "false"}
        async with http_client() as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        routes = payload.get("routes") or []
        if not routes:
            return error_result("osrm", "no_results", "No route found")

        r = routes[0]
        return ok_result(
            "osrm",
            {
                "from_stop": f"{from_lat},{from_lon}",
                "to_stop": f"{to_lat},{to_lon}",
                "mode": mode,
                "duration_minutes": round(r["duration"] / 60, 1),
                "distance_km": round(r["distance"] / 1000, 2),
                "instructions_url": (
                    f"https://www.openstreetmap.org/directions?"
                    f"engine=fossgis_osrm_{profile}"
                    f"&route={from_lat},{from_lon};{to_lat},{to_lon}"
                ),
            },
        )

    return await safe_call("osrm", _call)


async def _route_google(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    mode: TransportMode,
) -> ToolResult:
    settings = get_settings()
    if not settings.google_maps_api_key:
        return error_result(
            "google_maps_directions",
            "missing_config",
            "GOOGLE_MAPS_API_KEY not set; cannot use Google Directions fallback",
        )

    google_mode = {
        "walk": "walking",
        "drive": "driving",
        "bike": "bicycling",
        "transit": "transit",
        "taxi": "driving",
    }[mode]

    async def _call() -> ToolResult:
        params = {
            "origin": f"{from_lat},{from_lon}",
            "destination": f"{to_lat},{to_lon}",
            "mode": google_mode,
            "key": settings.google_maps_api_key,
        }
        async with http_client() as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/directions/json", params=params
            )
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("status") != "OK":
            return error_result(
                "google_maps_directions",
                "provider_error",
                payload.get("status", "unknown"),
                detail={"error_message": payload.get("error_message")},
            )

        route = payload["routes"][0]
        leg = route["legs"][0]
        return ok_result(
            "google_maps_directions",
            {
                "from_stop": leg.get("start_address", f"{from_lat},{from_lon}"),
                "to_stop": leg.get("end_address", f"{to_lat},{to_lon}"),
                "mode": mode,
                "duration_minutes": round(leg["duration"]["value"] / 60, 1),
                "distance_km": round(leg["distance"]["value"] / 1000, 2),
                "instructions_url": None,
            },
        )

    return await with_quota("google_maps_grounding", _call)


@tool
async def get_route(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    mode: TransportMode = "walk",
) -> ToolResult:
    """Compute a route between two points.

    OSRM is tried first (no key). If it errors AND a Google Maps key is
    configured, the Google Directions API is used as a fallback. Use
    `mode="transit"` only if Google is configured — OSRM has no transit
    profile.
    """
    log.info(
        "get_route (%s,%s)->(%s,%s) mode=%s",
        from_lat, from_lon, to_lat, to_lon, mode,
    )
    if mode == "transit":
        return await _route_google(from_lat, from_lon, to_lat, to_lon, mode)

    primary = await _route_osrm(from_lat, from_lon, to_lat, to_lon, mode)
    if primary["ok"]:
        return primary

    log.info("OSRM failed (%s); trying Google Directions", primary.get("error_type"))
    fallback = await _route_google(from_lat, from_lon, to_lat, to_lon, mode)
    return fallback if fallback["ok"] else primary
