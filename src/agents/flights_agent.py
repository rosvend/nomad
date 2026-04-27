"""Flight agent — searches and ranks flight options.

Reads:  destination, dates, travelers
Writes: flights
"""

from __future__ import annotations

import logging

from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def flights_agent(state: TripState) -> dict:
    """Find candidate flights for the trip.

    TODO: call a flight-search tool (fast-flights / fli / SerpApi fallback)
    and return validated `Flight` dicts. On tool error, append to `errors`
    and return an empty list rather than raising.
    """
    log.info("flights: searching for destination=%s", state.get("destination"))
    return {"flights": []}
