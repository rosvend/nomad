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

RESTAURANT_RADIUS_M = 2000      # restaurants must be walkable
RESTAURANT_RADIUS_LODGING_M = 6000  # when user_lodging anchors the search,
                                    # widen so both the lodging neighborhood
                                    # and the city centre are covered.
ATTRACTION_RADIUS_M = 5000      # attractions can be a bit further
ATTRACTION_RADIUS_LODGING_M = 8000  # same idea — cover lodging + centre.
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


# Stop-words filtered out of the free-form name hint so a preference like
# "places with outdoor seating" doesn't degenerate into a name regex
# matching every restaurant called "the". Also strips climate, landscape,
# vibe, and budget descriptors that describe the *trip* — not what we
# want to find inside a restaurant's name (e.g. "warm beach weather").
_NAME_HINT_STOPWORDS: frozenset[str] = frozenset({
    # grammar
    "a", "an", "the", "and", "or", "of", "with", "for", "in", "on",
    "at", "to", "from", "by", "near", "places", "place", "spot",
    "spots", "good", "nice", "great", "best", "cheap", "expensive",
    "food", "restaurant", "restaurants", "want", "looking", "find",
    "some", "any", "very", "really", "no", "not",
    # climate / weather
    "warm", "warmth", "hot", "cold", "cool", "chilly", "snowy", "snow",
    "sunny", "rainy", "tropical", "weather", "climate",
    # landscape / setting
    "beach", "beaches", "mountain", "mountains", "lake", "river", "forest",
    "desert", "island", "city", "rural", "urban", "countryside",
    # vibe / generic descriptors
    "quiet", "lively", "romantic", "adventure", "adventurous", "relaxing",
    "relax", "luxury", "luxurious", "premium", "elegant", "fancy",
    "budget", "midrange", "mid", "affordable",
})


def _build_name_hint(prefs: list[str] | None, cuisine_prefs: set[str]) -> str | None:
    """Build an Overpass `name~` regex alternation from user preferences.

    Strategy: take every word in `prefs` that isn't already captured by
    the cuisine whitelist or the stop-word list, and OR them together.
    For "rooftop, vegetarian, ramen" with cuisine_prefs={vegetarian, ramen}
    this yields "rooftop". With nothing left, returns None and the
    Overpass query stays unfiltered.
    """
    if not prefs:
        return None
    words: list[str] = []
    seen: set[str] = set()
    for raw in prefs:
        for piece in str(raw).lower().replace(",", " ").replace("/", " ").split():
            term = piece.strip().replace("-", "_")
            if not term or len(term) < 3:
                continue
            if term in _NAME_HINT_STOPWORDS:
                continue
            if term in cuisine_prefs:
                continue
            if term in _CUISINE_VOCAB:
                # Belongs in the cuisine_prefs path even if not chosen by
                # this user — don't pollute name search with broad terms.
                continue
            if term in seen:
                continue
            seen.add(term)
            words.append(term)
    if not words:
        return None
    # Cap at ~4 alternatives so the regex stays small and Overpass-friendly.
    return "|".join(words[:4])


def _restaurant_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """1.0 if two restaurants are essentially duplicates from a diversity
    standpoint (same cuisine OR very close geographically); 0.0 otherwise.

    Used by MMR to spread the final picks across distinct cuisines and
    blocks rather than returning five Italian places on the same street.
    """
    cuisines_a = _split_cuisine(a.get("cuisine"))
    cuisines_b = _split_cuisine(b.get("cuisine"))
    if cuisines_a and cuisines_b and (cuisines_a & cuisines_b):
        return 1.0
    la, lo_a = a.get("lat"), a.get("lon")
    lb, lo_b = b.get("lat"), b.get("lon")
    if la is not None and lo_a is not None and lb is not None and lo_b is not None:
        if haversine_km(la, lo_a, lb, lo_b) <= 0.15:  # ~150 m
            return 1.0
    return 0.0


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
    raw_prefs = state.get("preferences")
    cuisine_prefs = _extract_cuisine_prefs(raw_prefs)
    name_hint = _build_name_hint(raw_prefs, cuisine_prefs)
    cuisine_filter = "|".join(sorted(cuisine_prefs)) if cuisine_prefs else None
    user_lodging = state.get("user_lodging")
    log.info(
        "food: destination=%r tier=%r cuisine_prefs=%s name_hint=%s lodging=%r",
        destination, budget_tier, sorted(cuisine_prefs) or "(none)",
        name_hint, user_lodging,
    )

    # 1 — geocode. Prefer user_lodging when provided so search anchors on
    # the actual stay; widen radii to still cover the city centre. Anchor
    # the geocode query to the destination so partial addresses resolve
    # in the right city.
    if user_lodging:
        geo_query = ", ".join(p for p in (user_lodging, destination) if p)
        geo = await geocode.ainvoke({"query": geo_query})
        if geo["ok"]:
            lat = geo["data"]["lat"]
            lon = geo["data"]["lon"]
            rest_radius = RESTAURANT_RADIUS_LODGING_M
            attr_radius = ATTRACTION_RADIUS_LODGING_M
            log.info(
                "food: geocoded lodging %r -> (%.4f, %.4f); search r=%dm/%dm",
                user_lodging, lat, lon, rest_radius, attr_radius,
            )
        else:
            log.warning(
                "food: lodging geocode failed (%s); falling back to destination centre",
                geo.get("error_type"),
            )
            geo = await geocode.ainvoke({"query": destination})
            if not geo["ok"]:
                return {
                    "errors": [{"agent": "food", "stage": "geocode", "details": geo}],
                    "restaurants": [],
                }
            lat = geo["data"]["lat"]
            lon = geo["data"]["lon"]
            rest_radius = RESTAURANT_RADIUS_M
            attr_radius = ATTRACTION_RADIUS_M
    else:
        geo = await geocode.ainvoke({"query": destination})
        if not geo["ok"]:
            return {
                "errors": [{"agent": "food", "stage": "geocode", "details": geo}],
                "restaurants": [],
            }
        lat = geo["data"]["lat"]
        lon = geo["data"]["lon"]
        rest_radius = RESTAURANT_RADIUS_M
        attr_radius = ATTRACTION_RADIUS_M
        log.info("food: geocoded %r -> (%.4f, %.4f)", destination, lat, lon)

    # 2 — fetch candidates and attractions concurrently. The Overpass
    # cuisine filter narrows the pool to the user's preference (e.g.
    # "burger" → only burger places); name_hint adds free-form keyword
    # filtering ("rooftop"). Either filter that returns too few hits is
    # retried without the filter so the pool never starves.
    restaurants_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "restaurant",
        "radius_m": rest_radius,
        "limit": MAX_CANDIDATES,
        "cuisine": cuisine_filter,
        "name_hint": name_hint,
    })
    attractions_task = search_pois.ainvoke({
        "lat": lat, "lon": lon,
        "category": "attraction",
        "radius_m": attr_radius,
        "limit": 15,
    })
    rest_result, attr_result = await asyncio.gather(restaurants_task, attractions_task)

    # First fallback: if a cuisine filter was applied and starved the pool,
    # retry without it. Better to show non-matching restaurants than nothing.
    if cuisine_filter and (
        not rest_result["ok"]
        or len(rest_result.get("data") or []) < MIN_RESULTS
    ):
        log.info(
            "food: cuisine=%r yielded too few results — retrying without cuisine filter",
            cuisine_filter,
        )
        rest_result = await search_pois.ainvoke({
            "lat": lat, "lon": lon,
            "category": "restaurant",
            "radius_m": rest_radius,
            "limit": MAX_CANDIDATES,
            "name_hint": name_hint,
        })

    # Second fallback: if the name_hint is still cutting too deep, drop it.
    if name_hint and (
        not rest_result["ok"]
        or len(rest_result.get("data") or []) < MIN_RESULTS
    ):
        log.info(
            "food: name_hint=%r yielded too few results — falling back to unfiltered search",
            name_hint,
        )
        rest_result = await search_pois.ainvoke({
            "lat": lat, "lon": lon,
            "category": "restaurant",
            "radius_m": rest_radius,
            "limit": MAX_CANDIDATES,
        })

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

    # 5 — diversify via MMR, then cap. MMR keeps the highest-scored
    # candidate first, then prefers candidates that aren't in the same
    # cuisine bucket or within 150 m of one already picked. Deterministic
    # for identical inputs; produces noticeable variety for typical real
    # candidate pools.
    if len(ranked) < MIN_RESULTS:
        log.warning(
            "food_agent: only %d restaurants found (target %d)",
            len(ranked), MIN_RESULTS,
        )

    cap = max(MIN_RESULTS, MAX_ENRICHED)
    final = diversify_mmr(
        ranked,
        k=cap,
        score_key="score",
        similarity=_restaurant_similarity,
        lambda_=0.7,
    )

    log.info(
        "food_agent: returning %d restaurants (top score=%.3f, bottom score=%.3f), "
        "%d attractions written to state",
        len(final),
        final[0]["score"] if final else 0,
        final[-1]["score"] if final else 0,
        len(attractions_pool),
    )
    return {"restaurants": final, "attractions": attractions_pool}
