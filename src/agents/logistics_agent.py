"""Logistics agent — computes transit between the hotel and key stops.

Pipeline
--------
1. Resolve a starting point: the top-ranked hotel (`state["hotels"][0]`)
   if Hotel produced anything; otherwise the geocoded destination as a
   centroid fallback.
2. Build a stop list:
     - top N restaurants from `state["restaurants"]` (in-rank order)
     - top M attractions discovered via Overpass (around the start)
3. Compute a walking route from start → each stop in parallel via
   `get_route` (OSRM). Failed routes are kept with a `notes` field
   explaining why; the agent never raises.
4. Sort by distance ascending — closest first is the most useful order
   for someone planning their day.

Graph contract: this node has incoming edges from both `hotel` and
`food`, so LangGraph waits for both to write their state before invoking
this agent. That's why we can read `state["hotels"]` and
`state["restaurants"]` here even though they didn't exist when Router ran.

Reads:  destination, hotels, restaurants
Writes: logistics, errors
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.state.trip_state import TripState
from src.tools import geocode, get_route, search_pois

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

MAX_RESTAURANTS = 3              # cap restaurants in stop list
MAX_ATTRACTIONS = 3              # cap attractions in stop list
ATTRACTION_RADIUS_M = 5000       # how far to look for attractions
DEFAULT_MODE = "walk"            # OSRM public demo only ships car-speed,
                                 # so this is "best-effort" walking.


# ── Helpers ──────────────────────────────────────────────────────────

def _has_coords(item: dict[str, Any]) -> bool:
    return item.get("lat") is not None and item.get("lon") is not None


async def _resolve_starting_point(state: TripState) -> dict[str, Any] | None:
    """Pick the lat/lon to route every leg from.

    Order of preference:
      1. User-provided lodging (geocoded). Anchored to the destination so
         partial addresses like "cra 66 #48-106" resolve in the right city.
      2. The top-ranked hotel that has coordinates (typically state["hotels"][0]).
      3. Geocoded destination centroid.
      4. None — the agent will surface a helpful error.
    """
    user_lodging = state.get("user_lodging")
    destination = state.get("destination")
    if user_lodging:
        # Anchor the geocode query to the destination so partial addresses
        # ("cra 66 #48-106") don't match a same-named street in another city.
        query = ", ".join(p for p in (user_lodging, destination) if p)
        geo = await geocode.ainvoke({"query": query})
        if geo["ok"]:
            return {
                "name": user_lodging,
                "lat": geo["data"]["lat"],
                "lon": geo["data"]["lon"],
                "kind": "user_lodging",
            }
        log.warning(
            "logistics: could not geocode user_lodging %r — falling back to hotel/destination",
            user_lodging,
        )

    for h in state.get("hotels") or []:
        if _has_coords(h):
            return {
                "name": h.get("name") or "Hotel",
                "lat": h["lat"],
                "lon": h["lon"],
                "kind": "hotel",
            }

    if destination:
        geo = await geocode.ainvoke({"query": destination})
        if geo["ok"]:
            return {
                "name": f"{destination} (city centre)",
                "lat": geo["data"]["lat"],
                "lon": geo["data"]["lon"],
                "kind": "destination",
            }

    return None


def _restaurant_stops(state: TripState) -> list[dict[str, Any]]:
    """Convert the top restaurants into the agent's stop schema."""
    out: list[dict[str, Any]] = []
    for r in (state.get("restaurants") or [])[:MAX_RESTAURANTS]:
        if _has_coords(r):
            out.append({
                "name": r.get("name") or "(restaurant)",
                "lat": r["lat"],
                "lon": r["lon"],
                "category": "restaurant",
            })
    return out


async def _fetch_attractions(lat: float, lon: float) -> list[dict[str, Any]]:
    """Pull a small set of nearby tourist attractions via Overpass."""
    res = await search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "attraction",
        "radius_m": ATTRACTION_RADIUS_M,
        # Over-fetch so that filtering for valid coords still yields enough.
        "limit": MAX_ATTRACTIONS * 2,
    })
    if not res["ok"]:
        log.info("logistics: attractions lookup failed (%s)", res.get("error_type"))
        return []
    valid = [p for p in res["data"] if _has_coords(p)]
    return [
        {
            "name": p.get("name") or "(attraction)",
            "lat": p["lat"],
            "lon": p["lon"],
            "category": "attraction",
        }
        for p in valid[:MAX_ATTRACTIONS]
    ]


async def _route_one_leg(
    start: dict[str, Any],
    stop: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Compute one walking leg start → stop. Returns a dict matching
    `LogisticsLeg`. On failure, returns the same shape with no
    duration/distance and a `notes` line explaining why."""
    res = await get_route.ainvoke({
        "from_lat": start["lat"],
        "from_lon": start["lon"],
        "to_lat": stop["lat"],
        "to_lon": stop["lon"],
        "mode": mode,
    })
    base = {
        "from_stop": start["name"],
        "to_stop": stop["name"],
        "mode": mode,
        "from_lat": start["lat"],
        "from_lon": start["lon"],
        "to_lat": stop["lat"],
        "to_lon": stop["lon"],
        "category": f"{start.get('kind', 'start')}→{stop['category']}",
    }
    if not res["ok"]:
        return {
            **base,
            "duration_minutes": None,
            "distance_km": None,
            "instructions_url": None,
            "notes": f"route failed: {res.get('error_type')}",
        }
    d = res["data"]
    return {
        **base,
        "duration_minutes": d.get("duration_minutes"),
        "distance_km": d.get("distance_km"),
        "instructions_url": d.get("instructions_url"),
    }


# ── Agent entry point ────────────────────────────────────────────────

async def logistics_agent(state: TripState) -> dict:
    """Compute walking legs between the hotel and key stops."""
    destination = state.get("destination")
    if not destination:
        log.warning("logistics_agent: no destination — skipping")
        return {
            "errors": [{"agent": "logistics", "stage": "input", "message": "no destination"}],
            "logistics": [],
        }

    log.info(
        "logistics: destination=%r, %d hotel(s) and %d restaurant(s) in state",
        destination,
        len(state.get("hotels") or []),
        len(state.get("restaurants") or []),
    )

    start = await _resolve_starting_point(state)
    if not start:
        return {
            "errors": [{
                "agent": "logistics",
                "stage": "resolve_start",
                "message": "could not resolve a starting point — no hotel and geocode failed",
            }],
            "logistics": [],
        }
    log.info(
        "logistics: starting point %r (%s) @ (%.4f, %.4f)",
        start["name"], start["kind"], start["lat"], start["lon"],
    )

    # Build stop list. Restaurants come from upstream agents; attractions
    # we fetch ourselves so the agent stays useful even if Food returned
    # nothing (e.g. it errored, or the destination has no tagged restaurants).
    rest_stops = _restaurant_stops(state)
    attr_stops = await _fetch_attractions(start["lat"], start["lon"])
    all_stops = rest_stops + attr_stops

    log.info(
        "logistics: %d restaurant stop(s) from state, %d attraction stop(s) from Overpass",
        len(rest_stops), len(attr_stops),
    )

    if not all_stops:
        return {
            "errors": [{
                "agent": "logistics",
                "stage": "stops",
                "message": "no stops to route to (no restaurants in state and no attractions found)",
            }],
            "logistics": [],
        }

    # Compute every start→stop leg in parallel. `get_route` calls OSRM
    # (free, no key); each call is independent so concurrency is a
    # straight win. `safe_call` inside the tool ensures failures become
    # error_results, not exceptions.
    legs = await asyncio.gather(
        *(_route_one_leg(start, stop, DEFAULT_MODE) for stop in all_stops)
    )

    # Sort by distance ascending. Failed legs (distance_km=None) sink to
    # the end so they don't crowd out usable routes.
    legs.sort(key=lambda l: (l.get("distance_km") is None, l.get("distance_km") or 0))

    succeeded = sum(1 for l in legs if l.get("duration_minutes") is not None)
    log.info("logistics_agent: returning %d legs (%d successful)", len(legs), succeeded)
    return {"logistics": legs}
