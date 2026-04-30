"""Synthesizer — joins specialist outputs into a single TravelPlan.

Pipeline:
1. Validate every specialist list against its Pydantic model. Bad rows
   go into `errors[]` rather than crashing the synthesis.
2. Extract attraction names from `state["logistics"]` (the Logistics
   agent is the only place attractions live today — it queried Overpass
   while computing routes from the hotel).
3. Build an algorithmic itinerary distributing restaurants + attractions
   across the trip's days at stable times. Pure function in
   `_itinerary.build_itinerary`.
4. Optionally call the LLM for a narrative summary paragraph. Failure
   here is graceful — the plan just renders without a summary.

Reads:  every field
Writes: final_plan, errors
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from src.agents._itinerary import build_itinerary
from src.config import get_llm
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


def _attractions_from_logistics(
    logistics: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Pull deduped attraction stops out of logistics legs.

    Logistics legs have `category` like `"hotel→attraction"` or
    `"destination→attraction"`. The attraction's name is the leg's
    `to_stop`. Order of first appearance is preserved (legs come
    distance-sorted, so closer attractions appear first).
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for leg in logistics or []:
        cat = leg.get("category") or ""
        if not cat.endswith("attraction"):
            continue
        name = leg.get("to_stop")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "address": None,
            "lat": leg.get("to_lat"),
            "lon": leg.get("to_lon"),
        })
    return out


def _pick_attractions(state: TripState) -> list[dict[str, Any]]:
    """Prefer the pool food_agent wrote to state; fall back to extracting
    attractions out of logistics legs (older behaviour, kept for safety)."""
    direct = state.get("attractions") or []
    if direct:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for a in direct:
            name = a.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(a)
        if out:
            return out
    return _attractions_from_logistics(state.get("logistics"))


_SUMMARY_PROMPT = """\
Write a concise 3-5 sentence travel summary blurb for the trip described
below. Mention the destination, dates if present, the chosen hotel, and
1-2 highlights from the attractions or cuisine. Tone: warm and useful,
not marketing copy. Plain prose only — no markdown, no bullet points.
{inferred_note}
Trip:
- Destination: {destination}
- Dates: {dates}
- Travelers: {travelers}
- Budget tier: {budget_tier}
- Preferences: {preferences}
- Top hotel: {top_hotel}
- Restaurants picked: {n_restaurants} (top: {top_restaurant})
- Attractions identified: {n_attractions} (top: {top_attraction})
- Itinerary length: {num_days} day(s)
"""


async def _generate_summary(
    plan: TravelPlan,
    itinerary: list[dict[str, Any]],
    state: TripState,
) -> str | None:
    """Best-effort LLM-generated summary. Returns None on any failure."""
    try:
        llm = get_llm()
    except Exception as e:  # noqa: BLE001
        log.info("synthesizer: skipping summary — LLM unavailable (%s)", e)
        return None

    days = {stop["day"] for stop in itinerary}
    if plan.user_lodging:
        top_hotel = f"user-provided lodging at {plan.user_lodging}"
    elif plan.hotels:
        top_hotel = plan.hotels[0].name
    else:
        top_hotel = "(none chosen)"
    top_restaurant = plan.restaurants[0].name if plan.restaurants else "—"
    attractions = _pick_attractions(state)
    top_attraction = attractions[0]["name"] if attractions else "—"

    inferred_note = ""
    if state.get("destination_was_inferred"):
        inferred_note = (
            f"\nIMPORTANT: The user did not name a destination. We picked "
            f"{plan.destination} based on their preferences. Open the "
            f"summary by acknowledging that you chose {plan.destination} "
            f"for them and briefly say why.\n"
        )

    prompt = _SUMMARY_PROMPT.format(
        inferred_note=inferred_note,
        destination=plan.destination,
        dates=plan.dates or "(unspecified)",
        travelers=plan.travelers,
        budget_tier=plan.budget_tier or "mid",
        preferences=", ".join(state.get("preferences") or []) or "(none)",
        top_hotel=top_hotel,
        n_restaurants=len(plan.restaurants),
        top_restaurant=top_restaurant,
        n_attractions=len(attractions),
        top_attraction=top_attraction,
        num_days=len(days) if days else 0,
    )

    try:
        resp = await llm.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001
        log.warning("synthesizer: summary call failed (%s)", e)
        return None

    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip() if isinstance(text, str) else None
    return text or None


async def synthesizer_agent(state: TripState) -> dict:
    """Assemble the final plan, build itinerary, generate summary."""
    log.info("synthesizer: assembling final plan")

    flights, e1 = _validate_list(state.get("flights"), Flight, "flights")
    hotels, e2 = _validate_list(state.get("hotels"), Hotel, "hotels")
    restaurants, e3 = _validate_list(state.get("restaurants"), Restaurant, "restaurants")
    logistics, e5 = _validate_list(state.get("logistics"), LogisticsLeg, "logistics")

    # If the router couldn't extract dates, the itinerary builder silently
    # falls back to date.today(). Surface that so the rendered output
    # makes the fallback visible instead of misleading the user.
    e_dates: list[dict[str, Any]] = []
    if not state.get("dates"):
        e_dates.append({
            "agent": "synthesizer",
            "stage": "dates",
            "message": "no dates resolved from query — itinerary defaulted to today + 3 days",
        })

    # Itinerary skeleton — algorithmic distribution, no LLM.
    attractions = _pick_attractions(state)
    itinerary_dicts = build_itinerary(
        dates=state.get("dates"),
        restaurants=[
            {"name": r.name, "address": r.address}
            for r in restaurants
        ],
        attractions=attractions,
        hotel_name=hotels[0].name if hotels else None,
    )
    itinerary, e4 = _validate_list(itinerary_dicts, ItineraryStop, "itinerary")

    plan = TravelPlan(
        destination=state.get("destination") or "Unknown",
        dates=state.get("dates"),
        travelers=state.get("travelers") or 1,
        budget_tier=state.get("budget_tier"),
        user_lodging=state.get("user_lodging"),
        flights=flights,
        hotels=hotels,
        restaurants=restaurants,
        itinerary=itinerary,
        logistics=logistics,
        destination_was_inferred=bool(state.get("destination_was_inferred")),
        errors=[*state.get("errors", []), *e1, *e2, *e3, *e4, *e5, *e_dates],
    )

    # Optional narrative summary on top. Skipped for per-leg runs in
    # multi-city trips — the orchestrator writes one trip-level summary.
    if state.get("skip_summary"):
        plan.summary = None
    else:
        plan.summary = await _generate_summary(plan, itinerary_dicts, state)

    log.info(
        "synthesizer: plan ready — %d itinerary stops, summary=%s",
        len(itinerary), "yes" if plan.summary else "no",
    )
    return {"final_plan": plan.model_dump(mode="json")}
