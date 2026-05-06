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
from typing import Any

from src.agents._scoring import (
    budget_match_score,
    diversify_mmr,
    haversine_km,
    popularity_score,
    proximity_score,
    rating_score,
)
from src.state.trip_state import TripState
from src.tools import geocode, get_reviews, search_pois

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

HOTEL_RADIUS_M = 3000           # search this far from destination centroid
ATTRACTION_RADIUS_M = 5000      # tourist attractions can be a bit further
MAX_CANDIDATES = 8              # cap on Overpass hotel results we consider
MAX_ENRICHED = 6                # cap on Places API enrichment (cost = 2 each)
MIN_RESULTS = 3                 # always return at least this many

# Composite-score weights (must sum to 1.0).
W_RATING = 0.30
W_POPULARITY = 0.20
W_PROXIMITY = 0.30
W_BUDGET = 0.20

# Proximity-curve parameters: hotels can be a bit further from attractions
# than restaurants, since you commute from them once or twice a day.
_PROX_NEAR_KM = 1.0
_PROX_FAR_KM = 5.0
_PROX_TOP_K = 5


# ── Motel exclusion ──────────────────────────────────────────────────
#
# In Latin America (and project's home region), "motel" connotes
# short-stay / by-the-hour use, not roadside lodging. We filter them out
# of standard travel recommendations using three independent signals:
#
#   1. Name patterns — covers OSM hotels whose `tourism` tag is just
#      "hotel" but whose name reveals the truth (very common in LATAM).
#   2. OSM `tourism` value — direct hit when OSM has tagged the place
#      `tourism=motel`. The Overpass selector already excludes these,
#      but we re-check defensively in case the selector is bypassed.
#   3. Google Place `types` — Google sometimes types a venue as `motel`
#      even when OSM had it as a hotel; this catches the rare case the
#      first two layers miss.

_MOTEL_NAME_PATTERNS: tuple[str, ...] = (
    "motel", "moteles",
    "auto hotel", "auto-hotel", "autohotel",
    "love hotel", "love-hotel", "lovehotel",
    "by the hour",
)
_MOTEL_GOOGLE_TYPES: frozenset[str] = frozenset({"motel"})


def _hotel_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """1.0 if two hotels are essentially equivalent picks (same brand, or
    within ~200 m of each other); 0.0 otherwise. Used by MMR so the
    final list spans different chains/blocks rather than five Hilton
    properties on the same avenue.
    """
    brand_a = (a.get("brand") or "").strip().lower()
    brand_b = (b.get("brand") or "").strip().lower()
    if brand_a and brand_a == brand_b:
        return 1.0
    la, lo_a = a.get("lat"), a.get("lon")
    lb, lo_b = b.get("lat"), b.get("lon")
    if la is not None and lo_a is not None and lb is not None and lo_b is not None:
        if haversine_km(la, lo_a, lb, lo_b) <= 0.20:
            return 1.0
    return 0.0


def _is_motel(candidate: dict[str, Any]) -> bool:
    """Heuristic: true if the candidate looks like a short-stay motel.

    Layered checks: cheap name match first (no enrichment needed),
    then OSM `tourism` tag, then Google place types if present.
    """
    name = (candidate.get("name") or "").lower()
    if any(p in name for p in _MOTEL_NAME_PATTERNS):
        return True
    osm_tourism = (candidate.get("tourism") or "").lower()
    if osm_tourism == "motel":
        return True
    google_types = candidate.get("google_types") or []
    if any(t in _MOTEL_GOOGLE_TYPES for t in google_types):
        return True
    return False


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

    # Pass the OSM candidate's coords through so Google constrains its
    # match to the same neighborhood — otherwise common names match the
    # most popular global instance (e.g. a Pereira hotel matched into
    # a Santa Marta plan).
    res = await get_reviews.ainvoke({
        "query": query,
        "max_reviews": 1,
        "lat": candidate.get("lat"),
        "lon": candidate.get("lon"),
    })
    if not res["ok"]:
        candidate["enrichment_error"] = res.get("error_type")
        return candidate

    d = res["data"]
    candidate["rating"] = d.get("rating")
    candidate["review_count"] = d.get("review_count")
    candidate["price_level"] = d.get("price_level")
    # Prefer Google's clean address only if it actually mentions the
    # destination — guard against the rare case where locationbias still
    # returns a wrong-city match (e.g. tiny radius miss). Fallback keeps
    # OSM's address, which we know is correct because OSM is what we
    # geocoded against in the first place.
    google_addr = d.get("address")
    if google_addr and destination.lower() in google_addr.lower():
        candidate["address"] = google_addr
    elif google_addr:
        candidate["enrichment_error"] = "address_other_city"
    candidate["website"] = d.get("website")
    candidate["google_url"] = d.get("google_url")
    candidate["google_types"] = d.get("google_types") or []
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
        "brand": candidate.get("brand"),
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

    # User told us where they're staying — don't compete with their choice.
    # Logistics will geocode the address and use it as its routing origin.
    user_lodging = state.get("user_lodging")
    if user_lodging:
        log.info("hotel: user_lodging set (%r) — skipping hotel search", user_lodging)
        return {"hotels": []}

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

    # Drop motels before we burn Google enrichment quota on them. The
    # Overpass selector already excludes the obvious cases server-side;
    # this catches anything the regex didn't (multilingual names,
    # punctuation variants, hostels named "Hostal Motel X").
    pre_motel = len(candidates)
    candidates = [c for c in candidates if not _is_motel(c)]
    if pre_motel != len(candidates):
        log.info(
            "hotel: pre-enrichment dropped %d motel candidate(s)",
            pre_motel - len(candidates),
        )

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

    # Re-check for motels post-enrichment: Google's `types` array can
    # reveal classifications the OSM tags concealed.
    post_motel = len(all_candidates)
    all_candidates = [c for c in all_candidates if not _is_motel(c)]
    if post_motel != len(all_candidates):
        log.info(
            "hotel: post-enrichment dropped %d motel candidate(s) flagged by Google types",
            post_motel - len(all_candidates),
        )

    # Step 4 — score
    ranked: list[dict[str, Any]] = []
    for c in all_candidates:
        pos = (
            (c["lat"], c["lon"])
            if c.get("lat") is not None and c.get("lon") is not None
            else None
        )
        rating_n = rating_score(c.get("rating"))
        pop_n = popularity_score(c.get("review_count"))
        prox_n = proximity_score(
            pos,
            attraction_positions,
            near_km=_PROX_NEAR_KM,
            far_km=_PROX_FAR_KM,
            top_k=_PROX_TOP_K,
        )
        budget_n = budget_match_score(c.get("price_level"), budget_tier)

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

    # Step 5 — diversify with MMR so the final list spans different
    # chains/blocks rather than five Hilton properties on the same
    # avenue. Falls back to plain top-k when there are fewer than `cap`
    # ranked candidates.
    if len(ranked) < MIN_RESULTS:
        log.warning(
            "hotel_agent: only %d hotels found (target %d)",
            len(ranked), MIN_RESULTS,
        )

    cap = max(MIN_RESULTS, MAX_ENRICHED)
    final = diversify_mmr(
        ranked,
        k=cap,
        score_key="score",
        similarity=_hotel_similarity,
        lambda_=0.75,
    )

    log.info(
        "hotel_agent: returning %d hotels (top score=%.3f, bottom score=%.3f)",
        len(final),
        final[0]["score"] if final else 0,
        final[-1]["score"] if final else 0,
    )
    return {"hotels": final}
