"""Hotel agent — recommends accommodation matching the trip's budget.

Pipeline
--------
1. Geocode the destination via Nominatim (free).
2. In parallel, fetch:
     - hotel candidates (Overpass `tourism=hotel|hostel|guest_house|apartment`)
     - tourist attractions (Overpass `tourism=attraction|museum|viewpoint|...`)
   Both calls are free; no quota consumed.
3. Enrich the top-N hotel candidates with Google Places Details to add
   rating, review_count, price_level, address, website. Costs 2
   `google_places` quota slots per hotel (findplace + details). Defaults
   keep this conservative: N=6 → 12 quota slots per agent run.
4. Score each candidate as a weighted composite of:
     - rating         (weight 0.30) — Google's 5-star rating
     - popularity     (weight 0.20) — log10(review_count) saturating at 1k
     - proximity      (weight 0.30) — mean distance to top tourist attractions
     - budget match   (weight 0.20) — price_level vs requested budget tier
5. Sort descending by composite score; return at least 3.

Reads:  destination, dates, travelers, budget_tier
Writes: hotels (list of dicts), errors (on failures)
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from src.state.trip_state import TripState
from src.tools import geocode, get_reviews, search_pois

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

HOTEL_RADIUS_M = 3000           # search this far from destination centroid
ATTRACTION_RADIUS_M = 5000      # tourist attractions can be a bit further
MAX_CANDIDATES = 8              # cap on Overpass hotel results we consider
MAX_ENRICHED = 6                # cap on Places API enrichment (cost = 2 each)
MIN_RESULTS = 3                 # always return at least this many
PROXIMITY_TOP_K = 5             # average distance to top-K nearest attractions

# Composite-score weights (must sum to 1.0).
W_RATING = 0.30
W_POPULARITY = 0.20
W_PROXIMITY = 0.30
W_BUDGET = 0.20

# Budget tier → preferred Google `price_level` range (1=$ … 4=$$$$).
_BUDGET_PRICE_PREFS: dict[str, tuple[int, int]] = {
    "budget": (1, 2),
    "mid": (2, 3),
    "luxury": (3, 4),
}

# Proximity score: 1.0 within this radius (km); 0.0 beyond `_PROX_FAR_KM`.
_PROX_NEAR_KM = 1.0
_PROX_FAR_KM = 5.0


# ── Geometry / scoring helpers ───────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in km."""
    rad = math.pi / 180
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _proximity_score(
    hotel: tuple[float, float] | None,
    attractions: list[tuple[float, float]],
) -> float:
    """0.0 (far) … 1.0 (close to attraction cluster). 0.5 when undefined."""
    if hotel is None or not attractions:
        return 0.5
    distances = sorted(
        _haversine_km(hotel[0], hotel[1], a[0], a[1]) for a in attractions
    )
    top = distances[:PROXIMITY_TOP_K]
    mean_d = sum(top) / len(top)
    if mean_d <= _PROX_NEAR_KM:
        return 1.0
    if mean_d >= _PROX_FAR_KM:
        return 0.0
    return 1.0 - (mean_d - _PROX_NEAR_KM) / (_PROX_FAR_KM - _PROX_NEAR_KM)


def _popularity_score(review_count: int | None) -> float:
    """Log-scaled: 1 review ≈ 0, 1k+ reviews ≈ 1."""
    if not review_count or review_count <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log10(review_count) / 3.0))


def _budget_match_score(price_level: int | None, budget_tier: str) -> float:
    """1.0 if price_level in tier's preferred range, 0.5 if adjacent, 0.0 otherwise."""
    if price_level is None:
        return 0.5  # neutral when Google doesn't know
    lo, hi = _BUDGET_PRICE_PREFS.get(budget_tier, (2, 3))
    if lo <= price_level <= hi:
        return 1.0
    if price_level == lo - 1 or price_level == hi + 1:
        return 0.5
    return 0.0


def _rating_score(rating: float | None) -> float:
    if rating is None:
        return 0.5  # neutral
    return max(0.0, min(1.0, rating / 5.0))


# ── Enrichment ───────────────────────────────────────────────────────

async def _enrich_with_places(candidate: dict[str, Any], destination: str) -> dict[str, Any]:
    """Look up Google Places Details for one OSM hotel candidate.

    Mutates the input dict with rating / review_count / price_level / etc.
    On failure (quota, 404, network), the candidate is returned unmodified
    plus an `enrichment_error` field for transparency. We *never* raise —
    a failed enrichment just means that candidate is scored with neutral
    defaults (handled in the ranking step).
    """
    name = candidate.get("name") or candidate.get("brand") or ""
    if not name or name == "(unnamed)":
        candidate["enrichment_error"] = "no_name"
        return candidate

    # Bias the query with location context so Places picks the right place.
    city_hint = candidate.get("tags", {}).get("addr:city") or destination
    query = f"{name} hotel {city_hint}".strip()

    res = await get_reviews.ainvoke({"query": query, "max_reviews": 1})
    if not res["ok"]:
        candidate["enrichment_error"] = res.get("error_type")
        return candidate

    d = res["data"]
    candidate["rating"] = d.get("rating")
    candidate["review_count"] = d.get("review_count")
    candidate["price_level"] = d.get("price_level")
    # Prefer Google's clean address/name over OSM's local-language version
    # only when Google actually returned them.
    if d.get("address"):
        candidate["address"] = d["address"]
    candidate["website"] = d.get("website")
    candidate["google_url"] = d.get("google_url")
    return candidate


# ── Output shaping ───────────────────────────────────────────────────

def _hotel_dict(candidate: dict[str, Any], breakdown: dict[str, float], composite: float) -> dict[str, Any]:
    """Convert a scored candidate into the dict shape expected by `Hotel`."""
    tags = candidate.get("tags") or {}
    amenities = [k.replace("_", " ") for k, v in tags.items() if v == "yes"][:8]

    notes_parts: list[str] = []
    if candidate.get("brand"):
        notes_parts.append(f"brand: {candidate['brand']}")
    if candidate.get("stars"):
        notes_parts.append(f"OSM stars: {candidate['stars']}")
    if candidate.get("rooms"):
        notes_parts.append(f"{candidate['rooms']} rooms")
    if candidate.get("enrichment_error"):
        notes_parts.append(f"places: {candidate['enrichment_error']}")

    return {
        "name": candidate.get("name") or "(unnamed hotel)",
        "address": candidate.get("address"),
        "lat": candidate.get("lat"),
        "lon": candidate.get("lon"),
        "rating": candidate.get("rating"),
        "review_count": candidate.get("review_count"),
        "price_level": candidate.get("price_level"),
        "website": candidate.get("website"),
        "amenities": amenities,
        "score": round(composite, 3),
        "score_breakdown": breakdown,
        "notes": "; ".join(notes_parts) or None,
    }


# ── Agent entry point ────────────────────────────────────────────────

async def hotel_agent(state: TripState) -> dict:
    """Find and rank candidate hotels.

    Async because every step talks to an external service via the tools
    layer. LangGraph awaits async node functions natively when the graph
    is invoked via `ainvoke()`.
    """
    destination = state.get("destination")
    if not destination:
        log.warning("hotel_agent: no destination in state — skipping")
        return {
            "errors": [{"agent": "hotel", "stage": "input", "message": "no destination"}],
            "hotels": [],
        }

    budget_tier = state.get("budget_tier") or "mid"
    log.info("hotel: searching destination=%r tier=%r", destination, budget_tier)

    # Step 1 — geocode
    geo = await geocode.ainvoke({"query": destination})
    if not geo["ok"]:
        return {
            "errors": [{"agent": "hotel", "stage": "geocode", "details": geo}],
            "hotels": [],
        }
    lat = geo["data"]["lat"]
    lon = geo["data"]["lon"]
    log.info("hotel: geocoded %r -> (%.4f, %.4f)", destination, lat, lon)

    # Step 2 — fetch candidates and attractions concurrently
    hotels_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "hotel",
        "radius_m": HOTEL_RADIUS_M,
        "limit": MAX_CANDIDATES,
    })
    attractions_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "attraction",
        "radius_m": ATTRACTION_RADIUS_M,
        "limit": 15,
    })
    hotels_result, attractions_result = await asyncio.gather(
        hotels_task, attractions_task
    )

    if not hotels_result["ok"]:
        return {
            "errors": [{"agent": "hotel", "stage": "search_pois.hotel", "details": hotels_result}],
            "hotels": [],
        }

    candidates = list(hotels_result["data"])[:MAX_CANDIDATES]
    if not candidates:
        return {
            "errors": [{"agent": "hotel", "stage": "search_pois.hotel", "message": "no hotels found"}],
            "hotels": [],
        }

    if attractions_result["ok"]:
        attraction_positions = [
            (a["lat"], a["lon"])
            for a in attractions_result["data"]
            if a.get("lat") is not None and a.get("lon") is not None
        ]
    else:
        log.info("hotel: no attractions found — proximity score will be neutral")
        attraction_positions = []

    log.info(
        "hotel: %d candidates from Overpass, %d attractions for proximity",
        len(candidates), len(attraction_positions),
    )

    # Step 3 — enrich top-N candidates with Places API (parallel)
    to_enrich = candidates[:MAX_ENRICHED]
    enrich_tasks = [_enrich_with_places(dict(c), destination) for c in to_enrich]
    enriched = await asyncio.gather(*enrich_tasks)

    # Candidates beyond MAX_ENRICHED keep their OSM-only data and a flag.
    leftover = [
        {**dict(c), "enrichment_error": "skipped_to_save_quota"}
        for c in candidates[MAX_ENRICHED:]
    ]
    all_candidates = enriched + leftover

    # Step 4 — score
    ranked: list[dict[str, Any]] = []
    for c in all_candidates:
        pos = (
            (c["lat"], c["lon"])
            if c.get("lat") is not None and c.get("lon") is not None
            else None
        )
        rating_n = _rating_score(c.get("rating"))
        pop_n = _popularity_score(c.get("review_count"))
        prox_n = _proximity_score(pos, attraction_positions)
        budget_n = _budget_match_score(c.get("price_level"), budget_tier)

        composite = (
            W_RATING * rating_n
            + W_POPULARITY * pop_n
            + W_PROXIMITY * prox_n
            + W_BUDGET * budget_n
        )
        breakdown = {
            "rating": round(rating_n, 3),
            "popularity": round(pop_n, 3),
            "proximity": round(prox_n, 3),
            "budget": round(budget_n, 3),
        }
        ranked.append(_hotel_dict(c, breakdown, composite))

    # Step 5 — sort descending; ensure we return at least MIN_RESULTS.
    ranked.sort(key=lambda h: h["score"], reverse=True)
    if len(ranked) < MIN_RESULTS:
        log.warning(
            "hotel_agent: only %d hotels found (target %d)",
            len(ranked), MIN_RESULTS,
        )

    # Cap at the larger of MIN_RESULTS or the number of enriched candidates.
    # We never want to flood the synthesizer with low-information leftovers
    # unless we have nothing better.
    cap = max(MIN_RESULTS, MAX_ENRICHED)
    final = ranked[:cap]

    log.info(
        "hotel_agent: returning %d hotels (top score=%.3f, bottom score=%.3f)",
        len(final),
        final[0]["score"] if final else 0,
        final[-1]["score"] if final else 0,
    )
    return {"hotels": final}
