"""Shared scoring primitives for specialist agents.

All functions return values in `[0.0, 1.0]` (higher = better) and accept
`None` inputs gracefully — they return a neutral 0.5 when there's not
enough information to score, rather than raising.

Pure functions, no I/O. Safe to import from any agent.
"""

from __future__ import annotations

import math

# Budget tier → preferred Google `price_level` range (1=$ … 4=$$$$).
_BUDGET_PRICE_PREFS: dict[str, tuple[int, int]] = {
    "budget": (1, 2),
    "mid": (2, 3),
    "luxury": (3, 4),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in km."""
    rad = math.pi / 180
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def proximity_score(
    pos: tuple[float, float] | None,
    targets: list[tuple[float, float]],
    near_km: float,
    far_km: float,
    top_k: int,
) -> float:
    """1.0 when `pos` is within `near_km` of the average of its top-K
    nearest targets; 0.0 when beyond `far_km`; linearly interpolated in
    between. Returns 0.5 when undefined (no position or no targets).
    """
    if pos is None or not targets:
        return 0.5
    distances = sorted(
        haversine_km(pos[0], pos[1], t[0], t[1]) for t in targets
    )
    top = distances[:top_k]
    mean_d = sum(top) / len(top)
    if mean_d <= near_km:
        return 1.0
    if mean_d >= far_km:
        return 0.0
    return 1.0 - (mean_d - near_km) / (far_km - near_km)


def popularity_score(review_count: int | None) -> float:
    """Log-scaled: 1 review ≈ 0, 1k+ reviews ≈ 1."""
    if not review_count or review_count <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log10(review_count) / 3.0))


def budget_match_score(price_level: int | None, budget_tier: str) -> float:
    """1.0 if `price_level` is in the tier's preferred range, 0.5 if
    adjacent, 0.0 otherwise. Neutral 0.5 when `price_level` is missing.
    """
    if price_level is None:
        return 0.5
    lo, hi = _BUDGET_PRICE_PREFS.get(budget_tier, (2, 3))
    if lo <= price_level <= hi:
        return 1.0
    if price_level == lo - 1 or price_level == hi + 1:
        return 0.5
    return 0.0


def rating_score(rating: float | None) -> float:
    """Map a 0-5 rating onto 0-1. Neutral 0.5 when missing."""
    if rating is None:
        return 0.5
    return max(0.0, min(1.0, rating / 5.0))
