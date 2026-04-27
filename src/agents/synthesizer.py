"""Synthesizer — joins specialist outputs into a single TravelPlan.

Reads:  every field
Writes: final_plan, errors
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from src.output.schemas import (
    Flight,
    Hotel,
    ItineraryStop,
    LogisticsLeg,
    Restaurant,
    TravelPlan,
)
from src.state.trip_state import TripState

log = logging.getLogger(__name__)


def _validate_list(
    items: list[dict[str, Any]] | None,
    model: type[BaseModel],
    field_name: str,
) -> tuple[list[BaseModel], list[dict[str, Any]]]:
    """Validate each dict against `model`. Bad rows go to errors, not the plan."""
    valid: list[BaseModel] = []
    errors: list[dict[str, Any]] = []
    for i, raw in enumerate(items or []):
        try:
            valid.append(model.model_validate(raw))
        except ValidationError as e:
            log.warning("synthesizer: dropped invalid %s[%d]: %s", field_name, i, e)
            errors.append({
                "agent": "synthesizer",
                "stage": f"validate.{field_name}",
                "index": i,
                "message": str(e),
            })
    return valid, errors


def synthesizer_agent(state: TripState) -> dict:
    """Assemble the final plan from whatever the specialists returned.

    Partial results are fine: a missing `flights` list yields an empty
    Flights section in the markdown, not an error. Each specialist list
    is validated through its Pydantic model so malformed rows are dropped
    (and recorded under `errors`) rather than crashing the synthesis.
    """
    log.info("synthesizer: assembling final plan")

    flights, e1 = _validate_list(state.get("flights"), Flight, "flights")
    hotels, e2 = _validate_list(state.get("hotels"), Hotel, "hotels")
    restaurants, e3 = _validate_list(state.get("restaurants"), Restaurant, "restaurants")
    itinerary, e4 = _validate_list(state.get("itinerary_stops"), ItineraryStop, "itinerary")
    logistics, e5 = _validate_list(state.get("logistics"), LogisticsLeg, "logistics")

    plan = TravelPlan(
        destination=state.get("destination") or "Unknown",
        dates=state.get("dates"),
        travelers=state.get("travelers") or 1,
        budget_tier=state.get("budget_tier"),
        flights=flights,
        hotels=hotels,
        restaurants=restaurants,
        itinerary=itinerary,
        logistics=logistics,
        errors=[*state.get("errors", []), *e1, *e2, *e3, *e4, *e5],
    )
    return {"final_plan": plan.model_dump(mode="json")}
