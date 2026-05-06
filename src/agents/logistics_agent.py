"""Logistics agent — computes transit between the hotel and key stops.

Pipeline
--------
1. Resolve a starting point: the top-ranked hotel (`state["hotels"][0]`)
   if Hotel produced anything; otherwise the geocoded destination as a
   centroid fallback.
2. Build a stop list:
     - top N restaurants from `state["restaurants"]` (in-rank order)
     - up to M attractions: prefer the pool food_agent already wrote to
       `state["attractions"]`; only fetch our own via Overpass if state
       is empty (food may have errored).
3. Compute a walking route from start → each stop in parallel via
   `get_route` (OSRM). Failed routes are kept with a `notes` field
   explaining why; the agent never raises.
4. Sort by distance ascending — closest first is the most useful order
   for someone planning their day.

Graph contract: this node has incoming edges from both `hotel` and
`food`, so LangGraph waits for both to write their state before invoking
this agent. That's why we can read `state["hotels"]` and
`state["restaurants"]` here even though they didn't exist when Router ran.

Reads:  destination, dates, hotels, restaurants, attractions
Writes: logistics, errors
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date
from typing import Any

from src.state.trip_state import TripState
from src.tools import geocode, get_route, search_pois

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

MAX_RESTAURANTS = 3              # cap restaurants in stop list
MIN_ATTRACTIONS = 6              # floor for attractions in stop list
ATTRACTION_RADIUS_M = 5000       # how far to look for attractions when we
                                 # have to fetch our own
WALK_MAX_KM = 1.5                # straight-line distance under which we
                                 # suggest walking (~18 min). Beyond this
                                 # we suggest driving — telling someone to
                                 # walk 9km across a city is unhelpful.


def _attractions_to_visit(num_days: int) -> int:
    """How many attractions the itinerary can absorb for an N-day trip.

    Middle days each consume 2 (morning + afternoon); the arrival day has
    none and the departure day has one. We add a slack so the synthesizer
    has spares when it sorts/dedupes by distance.
    """
    middle = max(0, num_days - 2)
    departure = 1 if num_days > 1 else 0
    target = 2 * middle + departure
    return max(MIN_ATTRACTIONS, target + 2)


def _trip_num_days(state: TripState) -> int:
    dates = state.get("dates") or {}
    try:
        start = date.fromisoformat(dates["start"])
        end = date.fromisoformat(dates["end"])
        return max(1, (end - start).days + 1)
    except (KeyError, TypeError, ValueError):
        return 3


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


async def _fetch_attractions(lat: float, lon: float, want: int) -> list[dict[str, Any]]:
    """Pull nearby tourist attractions via Overpass — only used when state
    didn't already supply a pool from the food agent."""
    res = await search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "attraction",
        "radius_m": ATTRACTION_RADIUS_M,
        # Over-fetch so that filtering for valid coords still yields enough.
        "limit": want * 2,
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
            "address": p.get("address"),
            "category": "attraction",
        }
        for p in valid[:want]
    ]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance in km — used to pick the closest
    attractions out of the food-agent pool without a routing call."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _attractions_from_state(
    state: TripState,
    start_lat: float,
    start_lon: float,
    want: int,
) -> list[dict[str, Any]]:
    """Take the food-agent pool and pick the closest `want` to the start."""
    pool = state.get("attractions") or []
    valid = [a for a in pool if _has_coords(a) and a.get("name")]
    valid.sort(key=lambda a: _haversine_km(start_lat, start_lon, a["lat"], a["lon"]))
    return [
        {
            "name": a.get("name") or "(attraction)",
            "lat": a["lat"],
            "lon": a["lon"],
            "address": a.get("address"),
            "category": "attraction",
        }
        for a in valid[:want]
    ]


def _pick_mode(start: dict[str, Any], stop: dict[str, Any]) -> str:
    """Pick walk or drive based on straight-line distance.

    Tells the user to walk only when it's actually a reasonable walk; for
    cross-city distances we recommend driving instead. Haversine is a
    cheap lower bound — the road-network distance is always ≥ this.
    """
    d_km = _haversine_km(start["lat"], start["lon"], stop["lat"], stop["lon"])
    return "walk" if d_km <= WALK_MAX_KM else "drive"


async def _route_one_leg(
    start: dict[str, Any],
    stop: dict[str, Any],
    mode: str | None = None,
) -> dict[str, Any]:
    """Compute one leg start → stop in the appropriate mode. Returns a
    dict matching `LogisticsLeg`. On failure, returns the same shape with
    no duration/distance and a `notes` line explaining why.
    """
    if mode is None:
        mode = _pick_mode(start, stop)
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

    # Build stop list. Restaurants come from upstream agents. For attractions,
    # prefer the pool food_agent already wrote to state (~15 items); only
    # call Overpass ourselves when state is empty (food may have errored).
    rest_stops = _restaurant_stops(state)
    want_attractions = _attractions_to_visit(_trip_num_days(state))

    attr_stops = _attractions_from_state(state, start["lat"], start["lon"], want_attractions)
    attr_source = "state"
    if not attr_stops:
        attr_stops = await _fetch_attractions(start["lat"], start["lon"], want_attractions)
        attr_source = "overpass"
    all_stops = rest_stops + attr_stops

    log.info(
        "logistics: %d restaurant stop(s) from state, %d attraction stop(s) from %s "
        "(target=%d for %d-day trip)",
        len(rest_stops), len(attr_stops), attr_source,
        want_attractions, _trip_num_days(state),
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
    # Mode is decided per-leg by haversine distance: short legs walk,
    # cross-city legs drive. Avoids "walk 9 km" recommendations.
    legs = await asyncio.gather(
        *(_route_one_leg(start, stop) for stop in all_stops)
    )

    # Sort by distance ascending. Failed legs (distance_km=None) sink to
    # the end so they don't crowd out usable routes.
    legs.sort(key=lambda l: (l.get("distance_km") is None, l.get("distance_km") or 0))

    succeeded = sum(1 for l in legs if l.get("duration_minutes") is not None)
    log.info("logistics_agent: returning %d legs (%d successful)", len(legs), succeeded)
    return {"logistics": legs}
