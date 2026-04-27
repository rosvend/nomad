"""Logistics agent — computes transit between hotel and itinerary stops.

Reads:  destination, hotels, itinerary_stops
Writes: logistics
"""

from __future__ import annotations

import logging

from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def logistics_agent(state: TripState) -> dict:
    """Compute travel legs between stops.

    TODO: call OSM Overpass / Google Maps Directions to produce
    `LogisticsLeg` dicts for each (from, to) pair.
    """
    log.info(
        "logistics: computing routes for destination=%s",
        state.get("destination"),
    )
    return {"logistics": []}
