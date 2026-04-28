"""Flight agent — searches and ranks flight options.

Pipeline
--------
1. Resolve `state["origin"]` (or `settings.default_origin` if missing)
   and `state["destination"]` to 3-letter IATA codes via an in-process
   city map. A free-form 3-letter input is treated as IATA directly.
2. Call `search_flights` (Fli primary, SerpApi fallback) for the date
   range with the requested traveler count.
3. Score each result on a composite of:
     - **price** (cheaper better, normalized within this batch)
     - **stops** (non-stop best; 1-stop OK; 2+ heavily penalized)
     - **duration** (shorter better, normalized within this batch)
   Weights are **tier-aware**: budget travelers prioritize price; luxury
   travelers prioritize stops and speed.
4. Return the top N flights sorted by composite score (default 5).

Reads:  origin, destination, dates, travelers, budget_tier
Writes: flights (list of dicts), errors (on failures)
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.state.trip_state import TripState
from src.tools import search_flights

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

MAX_RESULTS = 5
DEFAULT_SEAT = "economy"
DEFAULT_TIER = "mid"

# Composite-score weights per budget tier: (price, stops, duration). Sum to 1.0.
_WEIGHT_PROFILES: dict[str, tuple[float, float, float]] = {
    "budget": (0.50, 0.30, 0.20),  # price-driven
    "mid": (0.35, 0.35, 0.30),     # balanced
    "luxury": (0.20, 0.40, 0.40),  # convenience-driven
}

# City → primary IATA. Multi-airport cities pick the most-used hub.
# Free-form text outside this map can still resolve if the user passes
# a 3-letter IATA directly.
_CITY_TO_IATA: dict[str, str] = {
    # Asia
    "tokyo": "HND", "osaka": "KIX", "kyoto": "KIX",
    "seoul": "ICN", "beijing": "PEK", "shanghai": "PVG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK",
    "dubai": "DXB", "delhi": "DEL", "mumbai": "BOM",
    "taipei": "TPE", "manila": "MNL", "kuala lumpur": "KUL",
    "ho chi minh city": "SGN", "hanoi": "HAN", "jakarta": "CGK",
    # Europe
    "london": "LHR", "paris": "CDG", "rome": "FCO",
    "madrid": "MAD", "barcelona": "BCN", "berlin": "BER",
    "amsterdam": "AMS", "frankfurt": "FRA", "munich": "MUC",
    "zurich": "ZRH", "vienna": "VIE", "istanbul": "IST",
    "athens": "ATH", "lisbon": "LIS", "dublin": "DUB",
    "copenhagen": "CPH", "stockholm": "ARN", "oslo": "OSL",
    "moscow": "SVO", "warsaw": "WAW", "prague": "PRG",
    "budapest": "BUD",
    # Americas — North
    "new york": "JFK", "los angeles": "LAX", "san francisco": "SFO",
    "chicago": "ORD", "miami": "MIA", "boston": "BOS",
    "washington": "IAD", "seattle": "SEA", "atlanta": "ATL",
    "dallas": "DFW", "denver": "DEN", "las vegas": "LAS",
    "houston": "IAH", "philadelphia": "PHL", "phoenix": "PHX",
    "toronto": "YYZ", "vancouver": "YVR", "montreal": "YUL",
    "mexico city": "MEX",
    # Americas — South
    "sao paulo": "GRU", "rio de janeiro": "GIG",
    "buenos aires": "EZE", "lima": "LIM",
    "bogota": "BOG", "santiago": "SCL", "medellin": "MDE",
    # Oceania
    "sydney": "SYD", "melbourne": "MEL", "auckland": "AKL",
    # Africa & Middle East
    "cairo": "CAI", "johannesburg": "JNB", "nairobi": "NBO",
    "casablanca": "CMN", "tel aviv": "TLV", "doha": "DOH",
}


# ── IATA resolution ──────────────────────────────────────────────────

def _resolve_iata(value: str | None) -> str | None:
    """Normalise a free-form input to a 3-letter IATA code.

    - Already an IATA code (3 alphabetic chars): uppercase and return it
      verbatim. We *don't* validate against fli's full Airport enum here;
      the search_flights tool catches unknown codes and returns
      `missing_config` cleanly.
    - Known city in `_CITY_TO_IATA`: return its primary airport.
    - Otherwise: None — caller surfaces a helpful error.
    """
    if not value:
        return None
    v = value.strip()
    if len(v) == 3 and v.isalpha():
        return v.upper()
    return _CITY_TO_IATA.get(v.lower())


# ── Scoring helpers (in-batch normalization) ─────────────────────────

def _stops_score(stops: int | None) -> float:
    """Fewer stops = higher score. Non-linear: 0 stops is decisively best."""
    if stops is None:
        return 0.5
    if stops <= 0:
        return 1.0
    if stops == 1:
        return 0.5
    return 0.1


def _normalize_inverse_in_batch(values: list[float | None]) -> list[float]:
    """Lower input → higher score (1.0 for the smallest in-batch value).

    Returns 0.5 (neutral) for any None / non-positive inputs and when the
    batch has no spread. Used for both price and duration where 'less is
    better'.
    """
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return [0.5] * len(values)
    lo = min(valid)
    hi = max(valid)
    if hi == lo:
        # Everything at the same value — give the valid ones max score
        return [1.0 if (v is not None and v > 0) else 0.5 for v in values]
    out: list[float] = []
    span = hi - lo
    for v in values:
        if v is None or v <= 0:
            out.append(0.5)
        else:
            out.append(round(1.0 - (v - lo) / span, 4))
    return out


# ── Output shaping ───────────────────────────────────────────────────

def _flight_dict(
    flight: dict[str, Any],
    breakdown: dict[str, float],
    composite: float,
) -> dict[str, Any]:
    """Convert a scored flight into the dict shape `Flight` validates."""
    return {
        "airline": flight.get("airline"),
        "flight_number": flight.get("flight_number"),
        "origin": flight.get("origin"),
        "destination": flight.get("destination"),
        "depart_at": flight.get("depart_at"),
        "arrive_at": flight.get("arrive_at"),
        "duration_minutes": flight.get("duration_minutes"),
        "price": flight.get("price"),
        "currency": flight.get("currency"),
        "stops": int(flight.get("stops") or 0),
        "legs": flight.get("legs") or [],
        "score": round(composite, 3),
        "score_breakdown": breakdown,
    }


# ── Agent entry point ────────────────────────────────────────────────

async def flights_agent(state: TripState) -> dict:
    """Find and rank candidate flights."""
    settings = get_settings()

    destination = state.get("destination")
    dates = state.get("dates") or {}
    travelers = state.get("travelers") or 1
    budget_tier = state.get("budget_tier") or DEFAULT_TIER
    origin_raw = state.get("origin") or settings.default_origin

    if not destination:
        log.warning("flights_agent: no destination in state — skipping")
        return {
            "errors": [{"agent": "flights", "stage": "input", "message": "no destination"}],
            "flights": [],
        }
    if not origin_raw:
        log.warning("flights_agent: no origin in state and no default_origin in config")
        return {
            "errors": [{
                "agent": "flights",
                "stage": "input",
                "message": "no origin — set state['origin'] or DEFAULT_ORIGIN in .env",
            }],
            "flights": [],
        }
    depart_date = dates.get("start")
    return_date = dates.get("end")
    if not depart_date:
        return {
            "errors": [{"agent": "flights", "stage": "input", "message": "no depart date in state['dates']"}],
            "flights": [],
        }

    origin_iata = _resolve_iata(origin_raw)
    dest_iata = _resolve_iata(destination)
    if not origin_iata:
        return {
            "errors": [{
                "agent": "flights",
                "stage": "resolve_iata",
                "message": (
                    f"could not resolve origin {origin_raw!r} to an IATA code; "
                    f"pass a 3-letter code or a known city name"
                ),
            }],
            "flights": [],
        }
    if not dest_iata:
        return {
            "errors": [{
                "agent": "flights",
                "stage": "resolve_iata",
                "message": (
                    f"could not resolve destination {destination!r} to an IATA code"
                ),
            }],
            "flights": [],
        }

    log.info(
        "flights: %s -> %s on %s (return=%s) for %d adult(s), tier=%s",
        origin_iata, dest_iata, depart_date, return_date, travelers, budget_tier,
    )

    # Step 2 — call the tool. Fli is async-wrapped via to_thread; SerpApi is
    # the fallback and only kicks in if Fli fails AND SERPAPI_API_KEY is set.
    res = await search_flights.ainvoke({
        "origin": origin_iata,
        "destination": dest_iata,
        "depart_date": depart_date,
        "return_date": return_date,
        "adults": travelers,
        "seat": DEFAULT_SEAT,
    })

    if not res["ok"]:
        return {
            "errors": [{"agent": "flights", "stage": "search_flights", "details": res}],
            "flights": [],
        }

    raw: list[dict[str, Any]] = list(res["data"])
    if not raw:
        return {
            "errors": [{"agent": "flights", "stage": "search_flights", "message": "no flights returned"}],
            "flights": [],
        }

    # Fli's round-trip search returns many (outbound, return) combinations
    # that share the same outbound flight. Once normalized, those collapse
    # to identical dicts — present them as one option.
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for c in raw:
        key = (
            c.get("airline"),
            c.get("flight_number"),
            c.get("depart_at"),
            c.get("arrive_at"),
            c.get("duration_minutes"),
            c.get("stops"),
            c.get("price"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)

    log.info(
        "flights: %d candidates (%d unique) from %s",
        len(raw), len(candidates), res.get("provider"),
    )

    # Step 3 — score. In-batch normalisation for price and duration so the
    # ranking adapts to whatever Google Flights gave us today.
    weights = _WEIGHT_PROFILES.get(budget_tier, _WEIGHT_PROFILES[DEFAULT_TIER])
    w_price, w_stops, w_duration = weights

    prices = [f.get("price") for f in candidates]
    durations = [f.get("duration_minutes") for f in candidates]
    price_scores = _normalize_inverse_in_batch(prices)
    duration_scores = _normalize_inverse_in_batch(durations)

    scored: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for i, f in enumerate(candidates):
        ps = price_scores[i]
        ss = _stops_score(f.get("stops"))
        ds = duration_scores[i]
        composite = w_price * ps + w_stops * ss + w_duration * ds
        breakdown = {
            "price": round(ps, 3),
            "stops": round(ss, 3),
            "duration": round(ds, 3),
        }
        scored.append((composite, breakdown, f))

    scored.sort(key=lambda triple: triple[0], reverse=True)
    final = [_flight_dict(f, br, sc) for sc, br, f in scored[:MAX_RESULTS]]

    log.info(
        "flights_agent: returning %d flights (top score=%.3f, tier=%s, weights=p%.2f/s%.2f/d%.2f)",
        len(final),
        final[0]["score"] if final else 0,
        budget_tier, w_price, w_stops, w_duration,
    )
    return {"flights": final}
