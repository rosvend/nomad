"""Router agent — parses the user's raw query into structured trip parameters.

Reads:  raw_query
Writes: destination, dates, travelers, budget_tier, preferences
"""

from __future__ import annotations

import logging

from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def router_agent(state: TripState) -> dict:
    """Parse the raw query and populate the structured intent fields.

    TODO: replace placeholder values with an LLM call (`get_llm()`) that
    produces a structured response (Pydantic-validated) from `raw_query`.
    """
    log.info("router: parsing query=%r", state.get("raw_query"))
    return {
        "destination": "Tokyo",
        "dates": {"start": "2026-05-01", "end": "2026-05-08"},
        "travelers": 1,
        "budget_tier": "mid",
        "preferences": [],
    }
