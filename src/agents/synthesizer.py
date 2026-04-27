"""Synthesizer — joins specialist outputs into a single TravelPlan.

Reads:  every field
Writes: final_plan, errors
"""

from __future__ import annotations

import logging

from src.output.schemas import TravelPlan
from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def synthesizer_agent(state: TripState) -> dict:
    """Assemble the final plan from whatever the specialists returned.

    Partial results are fine: a missing `flights` list yields an empty
    Flights section in the markdown, not an error. Real synthesis logic
    (LLM-generated summary, ranking, deduping) lands in a later pass.
    """
    log.info("synthesizer: assembling final plan")
    plan = TravelPlan(
        destination=state.get("destination") or "Unknown",
        dates=state.get("dates"),
        travelers=state.get("travelers") or 1,
        budget_tier=state.get("budget_tier"),
        # Specialists currently return [] — once they emit dicts, validate via
        # `Flight.model_validate(d)` etc. before appending.
        flights=[],
        hotels=[],
        restaurants=[],
        itinerary=[],
        logistics=[],
        errors=state.get("errors", []),
    )
    return {"final_plan": plan.model_dump(mode="json")}
