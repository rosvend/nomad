"""Hotel agent — recommends accommodation matching the trip's budget.

Reads:  destination, dates, travelers, budget_tier
Writes: hotels
"""

from __future__ import annotations

import logging

from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def hotel_agent(state: TripState) -> dict:
    """Find candidate hotels for the trip.

    TODO: call a places/hotel tool (Google Maps Grounding Lite or OSM)
    filtered by `budget_tier`. Return validated `Hotel` dicts.
    """
    log.info(
        "hotel: searching for destination=%s tier=%s",
        state.get("destination"),
        state.get("budget_tier"),
    )
    return {"hotels": []}
