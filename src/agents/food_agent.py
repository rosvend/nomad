"""Food agent — discovers restaurants matching destination, preferences,
and budget tier.

Pipeline
--------
1. Geocode the destination via Nominatim (free).
2. In parallel:
     - restaurant candidates (Overpass `amenity=restaurant`, no cuisine
       filter so the candidate pool stays diverse)
     - tourist attractions (used to score proximity — eating near
       sightseeing spots is a UX win)
3. Enrich the top-N candidates with Google Places Details: rating,
   review_count, price_level, website, address. Costs 2 `google_places`
   slots per restaurant (findplace + details). Defaults keep it
   conservative: N=6 → 12 quota slots per agent run, matching the hotel
   agent.
4. Score each candidate as a weighted composite:
     - rating         (0.25) — Google's 5-star rating
     - popularity     (0.15) — log10(review_count) saturating at 1k
     - proximity      (0.20) — mean km to top-3 attractions, NEAR=0.3, FAR=2
     - cuisine match  (0.25) — vs the user's cuisine preferences
     - budget match   (0.15) — price_level vs the requested tier
5. Sort descending by composite score; return at least 3.

Reads:  destination, preferences, budget_tier
Writes: restaurants, attractions (shared pool fetched here for proximity
        scoring; logistics + synthesizer consume it), errors (on failures)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.agents._scoring import (
    budget_match_score,
    popularity_score,
    proximity_score,
    rating_score,
)
from src.state.trip_state import TripState
from src.tools import geocode, get_reviews, search_pois

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

RESTAURANT_RADIUS_M = 2000      # restaurants must be walkable
ATTRACTION_RADIUS_M = 5000      # attractions can be a bit further
MAX_CANDIDATES = 12             # bigger pool than hotels — food has more variety
MAX_ENRICHED = 6                # cap Places API calls (2 each → 12 quota)
MIN_RESULTS = 3

# Composite-score weights (must sum to 1.0).
W_RATING = 0.25
W_POPULARITY = 0.15
W_PROXIMITY = 0.20
W_CUISINE = 0.25
W_BUDGET = 0.15

# Restaurants: tighter proximity than hotels — you walk to lunch.
_PROX_NEAR_KM = 0.3
_PROX_FAR_KM = 2.0
_PROX_TOP_K = 3

# Curated whitelist of cuisine-ish words to look for in `state["preferences"]`.
# Matches OSM `cuisine` tag values commonly seen in real data. Free-form
# preference text ("nice atmosphere", "outdoor seating") is intentionally
# ignored — too brittle for v1.
_CUISINE_VOCAB: frozenset[str] = frozenset({
    # broad cuisines
    "japanese", "chinese", "indian", "thai", "vietnamese", "korean",
    "italian", "french", "spanish", "mexican", "american", "german",
    "mediterranean", "greek", "turkish", "lebanese", "ethiopian",
    "asian", "european", "latin_american",
    # specific dishes / styles
    "ramen", "sushi", "tempura", "tonkatsu", "yakiniku", "udon",
    "pizza", "pasta", "burger", "barbecue", "bbq", "steak", "seafood",
    "noodle", "noodles", "curry", "tapas", "kebab",
    # dietary
    "vegetarian", "vegan", "halal", "kosher", "gluten_free",
    # venue types
    "cafe", "bakery", "dessert", "ice_cream",
})

# Fallback hints when the OSM `cuisine` tag is missing — we look for these
# substrings in the restaurant name. Keep small and obvious; a name match
# only confirms intent, never overrides a present-but-different OSM tag.
_CUISINE_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "vegetarian": ("vegetarian", "vegetariano", "vegetariana", "veggie", "veg "),
    "vegan":      ("vegan", "vegano", "vegana", "plant-based", "plant based"),
}

# Names that strongly imply meat-only menus. Used to hard-exclude obviously
# inappropriate places when the user has a vegetarian/vegan preference. The
# user can still get them surfaced for non-restricted searches.
_MEAT_KEYWORDS_NAME: tuple[str, ...] = (
    "chicken", "pollo", "cangrejo", "crab", "seafood", "mariscos",
    "pescaderia", "pescadería", "pescados", "fish",
    "bbq", "barbecue", "parrilla", "asadero", "asados",
    "steak", "steakhouse", "carne", "carnes", "pork", "cerdo",
    "ribs", "rotisserie", "wings", "alitas",
)
_MEAT_CUISINE_TAGS: frozenset[str] = frozenset({
    "steak_house", "barbecue", "chicken", "seafood", "fish",
})
_DIETARY_RESTRICTED: frozenset[str] = frozenset({"vegetarian", "vegan"})


# ── Preference parsing & cuisine matching ───────────────────────────

def _extract_cuisine_prefs(prefs: list[str] | None) -> set[str]:
    """Normalise the user's preference list down to known cuisine terms."""
    out: set[str] = set()
    for raw in prefs or []:
        for piece in str(raw).lower().replace(",", " ").replace("/", " ").split():
            term = piece.strip().replace("-", "_")
            if term in _CUISINE_VOCAB:
                out.add(term)
    return out


def _split_cuisine(cuisine_str: str | None) -> set[str]:
    """OSM cuisine tags are `;` or `,` separated, lowercase already."""
    if not cuisine_str:
        return set()
    return {
        t.strip().lower().replace("-", "_")
        for t in cuisine_str.replace(",", ";").split(";")
        if t.strip()
    }


def _name_matches_pref(name: str | None, prefs: set[str]) -> bool:
    """Look for cuisine-pref hint substrings in a restaurant name.

    OSM `cuisine` tags are often missing, so a place literally named
    "Restaurante Vegetariano" deserves credit even with no tag.
    """
    if not name:
        return False
    lowered = name.lower()
    for pref in prefs:
        for hint in _CUISINE_NAME_HINTS.get(pref, ()):
            if hint in lowered:
                return True
    return False


def _cuisine_match_score(
    place_cuisine: str | None,
    prefs: set[str],
    *,
    name: str | None = None,
) -> float:
    """1.0 on tag/name overlap, 0.5 on partial or unknown, 0.0 on conflict.
    Returns 0.5 (neutral) when the user expressed no cuisine preference at all.
    """
    if not prefs:
        return 0.5
    place_terms = _split_cuisine(place_cuisine)
    if place_terms & prefs:
        return 1.0
    if _name_matches_pref(name, prefs):
        return 1.0
    if not place_terms:
        # No tag and no name hint — neutral, not a penalty. Otherwise an
        # unrelated tagged cuisine outranks every untagged candidate.
        return 0.5
    for pref in prefs:
        for term in place_terms:
            if pref in term or term in pref:
                return 0.5
    return 0.0


def _is_meat_only(candidate: dict[str, Any]) -> bool:
    """Heuristic: name or OSM cuisine tag strongly implies a meat-only menu."""
    name = (candidate.get("name") or "").lower()
    cuisine = (candidate.get("cuisine") or "").lower()
    if any(kw in name for kw in _MEAT_KEYWORDS_NAME):
        return True
    cuisine_terms = _split_cuisine(cuisine)
    if cuisine_terms & _MEAT_CUISINE_TAGS:
        return True
    return False


# ── Enrichment ───────────────────────────────────────────────────────

async def _enrich_with_places(candidate: dict[str, Any], destination: str) -> dict[str, Any]:
    """Fetch Google Places Details for one OSM restaurant candidate.

    Mutates the candidate with rating / review_count / price_level / etc.
    On failure (quota, no match, network), the candidate is returned with
    an `enrichment_error` field. We never raise.
    """
    name = candidate.get("name") or candidate.get("brand") or ""
    if not name or name == "(unnamed)":
        candidate["enrichment_error"] = "no_name"
        return candidate

    city_hint = candidate.get("tags", {}).get("addr:city") or destination
    query = f"{name} restaurant {city_hint}".strip()

    # See hotel_agent._enrich_with_places for why coords matter here:
    # without locationbias, "Frutoss Restaurante Vegetariano" matches a
    # popular Cali restaurant when the user is planning Santa Marta.
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
    google_addr = d.get("address")
    if google_addr and destination.lower() in google_addr.lower():
        candidate["address"] = google_addr
    elif google_addr:
        # Google's address points at a different city — keep OSM's. The
        # mismatch is signal enough to warn downstream consumers.
        candidate["enrichment_error"] = "address_other_city"
    candidate["website"] = d.get("website")
    candidate["google_url"] = d.get("google_url")
    return candidate


# ── Output shaping ───────────────────────────────────────────────────

def _restaurant_dict(
    candidate: dict[str, Any],
    breakdown: dict[str, float],
    composite: float,
) -> dict[str, Any]:
    """Convert a scored candidate into the shape expected by `Restaurant`."""
    tags = candidate.get("tags") or {}
    amenities: list[str] = []
    if tags.get("outdoor_seating") == "yes":
        amenities.append("outdoor seating")
    if tags.get("takeaway") == "yes":
        amenities.append("takeaway")
    if tags.get("delivery") == "yes":
        amenities.append("delivery")
    if tags.get("wheelchair") == "yes":
        amenities.append("wheelchair")
    if tags.get("reservation") in {"yes", "required", "recommended"}:
        amenities.append(f"reservation: {tags['reservation']}")

    notes_parts: list[str] = []
    if candidate.get("brand"):
        notes_parts.append(f"brand: {candidate['brand']}")
    if tags.get("opening_hours"):
        notes_parts.append(f"hours: {tags['opening_hours'][:60]}")
    if candidate.get("enrichment_error"):
        notes_parts.append(f"places: {candidate['enrichment_error']}")

    return {
        "name": candidate.get("name") or "(unnamed restaurant)",
        "cuisine": candidate.get("cuisine"),
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

async def food_agent(state: TripState) -> dict:
    """Find and rank candidate restaurants."""
    destination = state.get("destination")
    if not destination:
        log.warning("food_agent: no destination in state — skipping")
        return {
            "errors": [{"agent": "food", "stage": "input", "message": "no destination"}],
            "restaurants": [],
        }

    budget_tier = state.get("budget_tier") or "mid"
    cuisine_prefs = _extract_cuisine_prefs(state.get("preferences"))
    log.info(
        "food: destination=%r tier=%r cuisine_prefs=%s",
        destination, budget_tier, sorted(cuisine_prefs) or "(none)",
    )

    # 1 — geocode
    geo = await geocode.ainvoke({"query": destination})
    if not geo["ok"]:
        return {
            "errors": [{"agent": "food", "stage": "geocode", "details": geo}],
            "restaurants": [],
        }
    lat = geo["data"]["lat"]
    lon = geo["data"]["lon"]
    log.info("food: geocoded %r -> (%.4f, %.4f)", destination, lat, lon)

    # 2 — fetch candidates and attractions concurrently
    restaurants_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "restaurant",
        "radius_m": RESTAURANT_RADIUS_M,
        "limit": MAX_CANDIDATES,
    })
    attractions_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "attraction",
        "radius_m": ATTRACTION_RADIUS_M,
        "limit": 15,
    })
    rest_result, attr_result = await asyncio.gather(restaurants_task, attractions_task)

    if not rest_result["ok"]:
        return {
            "errors": [{"agent": "food", "stage": "search_pois.restaurant", "details": rest_result}],
            "restaurants": [],
        }

    candidates = list(rest_result["data"])[:MAX_CANDIDATES]
    if not candidates:
        return {
            "errors": [{"agent": "food", "stage": "search_pois.restaurant", "message": "no restaurants found"}],
            "restaurants": [],
        }

    if attr_result["ok"]:
        attractions_pool = [
            {
                "name": a.get("name") or "(attraction)",
                "lat": a.get("lat"),
                "lon": a.get("lon"),
                "address": a.get("address"),
            }
            for a in attr_result["data"]
            if a.get("name")
            and a.get("name") != "(unnamed)"
            and a.get("lat") is not None
            and a.get("lon") is not None
        ]
        attraction_positions = [(a["lat"], a["lon"]) for a in attractions_pool]
    else:
        log.info("food: no attractions found — proximity score will be neutral")
        attractions_pool = []
        attraction_positions = []

    log.info(
        "food: %d candidates from Overpass, %d attractions for proximity",
        len(candidates), len(attraction_positions),
    )

    # Hard-exclude obvious meat venues when user has dietary restrictions.
    # Soft signals (rating/popularity) easily push a "Tropical Chicken"-type
    # place to the top otherwise.
    if cuisine_prefs & _DIETARY_RESTRICTED:
        before = len(candidates)
        candidates = [c for c in candidates if not _is_meat_only(c)]
        excluded = before - len(candidates)
        if excluded:
            log.info(
                "food: excluded %d meat-only candidate(s) for %s prefs",
                excluded, sorted(cuisine_prefs & _DIETARY_RESTRICTED),
            )

    # 3 — enrich top-N with Places API (parallel)
    to_enrich = candidates[:MAX_ENRICHED]
    enrich_tasks = [_enrich_with_places(dict(c), destination) for c in to_enrich]
    enriched = await asyncio.gather(*enrich_tasks)

    leftover = [
        {**dict(c), "enrichment_error": "skipped_to_save_quota"}
        for c in candidates[MAX_ENRICHED:]
    ]
    all_candidates = enriched + leftover

    # 4 — score
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
        cuisine_n = _cuisine_match_score(
            c.get("cuisine"), cuisine_prefs, name=c.get("name")
        )
        budget_n = budget_match_score(c.get("price_level"), budget_tier)

        composite = (
            W_RATING * rating_n
            + W_POPULARITY * pop_n
            + W_PROXIMITY * prox_n
            + W_CUISINE * cuisine_n
            + W_BUDGET * budget_n
        )
        breakdown = {
            "rating": round(rating_n, 3),
            "popularity": round(pop_n, 3),
            "proximity": round(prox_n, 3),
            "cuisine": round(cuisine_n, 3),
            "budget": round(budget_n, 3),
        }
        ranked.append(_restaurant_dict(c, breakdown, composite))

    # 5 — sort and cap
    ranked.sort(key=lambda r: r["score"], reverse=True)
    if len(ranked) < MIN_RESULTS:
        log.warning(
            "food_agent: only %d restaurants found (target %d)",
            len(ranked), MIN_RESULTS,
        )

    cap = max(MIN_RESULTS, MAX_ENRICHED)
    final = ranked[:cap]

    log.info(
        "food_agent: returning %d restaurants (top score=%.3f, bottom score=%.3f), "
        "%d attractions written to state",
        len(final),
        final[0]["score"] if final else 0,
        final[-1]["score"] if final else 0,
        len(attractions_pool),
    )
    return {"restaurants": final, "attractions": attractions_pool}
