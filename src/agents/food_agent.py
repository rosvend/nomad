"""Food agent — discovers restaurants and cuisine highlights.

Reads:  destination, preferences
Writes: restaurants
"""

from __future__ import annotations

import logging

from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def food_agent(state: TripState) -> dict:
    """Find restaurants matching the destination and preferences.

    TODO: call places tool, filter by `preferences` (e.g. "vegetarian"),
    and return validated `Restaurant` dicts.
    """
    log.info(
        "food: searching for destination=%s prefs=%s",
        state.get("destination"),
        state.get("preferences"),
    )
    return {"restaurants": []}
