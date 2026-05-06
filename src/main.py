"""CLI entry point.

For single-destination trips this is a thin wrapper around
``graph.ainvoke`` — same as before. For multi-leg trips (e.g. "3 days in
Bogota then 4 days in Cartagena") the router emits ``state["legs"]`` and
this module loops the graph once per leg, then merges results into a
single ``TravelPlan`` with per-leg sections.

The graph itself is unchanged — per-leg invocations call it with
``raw_query=""`` and pre-filled scalar fields, so the router
short-circuits and the rest of the pipeline behaves identically to a
normal one-city run.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv

from src.agents.destination_suggester import destination_suggester_agent
from src.agents.flights_agent import _resolve_iata
from src.agents.router import router_agent
from src.config import configure_logging, get_settings
from src.graph.builder import build_graph
from src.output.formatter import render_terminal
from src.output.schemas import Flight, LegPlan, TravelPlan
from src.tools import search_flights

DEFAULT_QUERY = "Plan a 5-day trip to Tokyo for one person, mid budget."

log = logging.getLogger(__name__)


def _leg_dates(leg: dict[str, Any]) -> dict[str, str] | None:
    s, e = leg.get("start"), leg.get("end")
    if s and e:
        return {"start": s, "end": e}
    return None


def _next_day(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        return (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    except ValueError:
        return None


def _leg_day_count(leg: dict[str, Any], lp: LegPlan) -> int:
    """How many wall-clock days this leg occupies.

    Prefer the leg's start/end window (authoritative), falling back to
    the itinerary's day-range if dates are missing. Itineraries can
    silently drop a day (e.g. Day 1 only had a check-in stop and the
    hotel search returned nothing), so the date span is the safer source
    of truth for day-numbering across legs.
    """
    s, e = leg.get("start"), leg.get("end")
    if s and e:
        try:
            return (date.fromisoformat(e) - date.fromisoformat(s)).days + 1
        except ValueError:
            pass
    if leg.get("days"):
        return int(leg["days"])
    if lp.itinerary:
        return max(s.day for s in lp.itinerary) - min(s.day for s in lp.itinerary) + 1
    return 0


async def _run_single_leg_keep_summary(
    graph: Any,
    base_state: dict[str, Any],
    leg: dict[str, Any],
    origin: str | None,
    keep_summary: bool = False,
) -> tuple[TravelPlan | None, dict[str, Any]]:
    """Run the graph for one leg and return (parsed plan, raw final state).

    Each leg uses its own ``leg.start``/``leg.end`` window so the
    itinerary builder produces day-counts that match the leg, not the
    whole trip.
    """
    leg_state: dict[str, Any] = {
        "raw_query": "",  # router short-circuits
        "origin": origin,
        "destination": leg["destination"],
        "dates": _leg_dates(leg),
        "travelers": base_state.get("travelers"),
        "budget_tier": base_state.get("budget_tier"),
        "budget_amount": base_state.get("budget_amount"),
        "budget_currency": base_state.get("budget_currency"),
        "budget_scope": base_state.get("budget_scope"),
        "preferences": base_state.get("preferences") or [],
        "user_lodging": leg.get("lodging"),
        "legs": [leg],
        "destination_was_inferred": bool(base_state.get("destination_was_inferred")),
        "skip_summary": not keep_summary,
    }
    final = await graph.ainvoke(leg_state)
    plan_data = final.get("final_plan") or {}
    plan = TravelPlan.model_validate(plan_data) if plan_data else None
    return plan, final


async def _search_return_flight(
    last_dest: str, origin_city: str, depart_iso: str | None, travelers: int,
) -> list[dict[str, Any]]:
    """Best-effort one-way flight search for the trip's return leg."""
    if not depart_iso:
        return []
    last_iata = await _resolve_iata(last_dest)
    home_iata = await _resolve_iata(origin_city)
    if not last_iata or not home_iata:
        log.warning(
            "return-flight: could not resolve IATA (%r->%r, %r->%r)",
            last_dest, last_iata, origin_city, home_iata,
        )
        return []
    res = await search_flights.ainvoke({
        "origin": last_iata,
        "destination": home_iata,
        "depart_date": depart_iso,
        "return_date": None,
        "adults": travelers,
        "seat": "economy",
    })
    if not res.get("ok"):
        return []
    raw = list(res.get("data") or [])
    # Cheapest first; mirror the simple shape the synthesizer ultimately renders.
    raw.sort(key=lambda f: (f.get("price") or 1e12, f.get("stops") or 0))
    return raw[:3]


def _leg_to_legplan(leg: dict[str, Any], plan: TravelPlan | None) -> LegPlan:
    if plan is None:
        return LegPlan(
            destination=leg["destination"],
            dates=_leg_dates(leg),
            user_lodging=leg.get("lodging"),
        )
    return LegPlan(
        destination=leg["destination"],
        dates=_leg_dates(leg) or plan.dates,
        user_lodging=leg.get("lodging") or plan.user_lodging,
        hotels=plan.hotels,
        restaurants=plan.restaurants,
        itinerary=plan.itinerary,
        logistics=plan.logistics,
    )


def _renumber_itinerary_days(plan: TravelPlan, day_offset: int) -> None:
    """Shift all itinerary stops on `plan` by `day_offset` days. Mutates in place."""
    if day_offset == 0:
        return
    for stop in plan.itinerary:
        stop.day = stop.day + day_offset


async def _run(query: str) -> TravelPlan:
    graph = build_graph()
    settings = get_settings()

    # Step 1: run the router standalone. This parses the query into legs
    # without dispatching specialists, so we can run each leg with its own
    # (leg-local) dates instead of the overall window.
    parsed = await router_agent({"raw_query": query})

    # Step 2: if the router produced no destination, run the destination
    # suggester (interactive). Pass raw_query so region hints like "asia"
    # that didn't make it into preferences still reach the suggester.
    has_dest = bool(parsed.get("destination") or (parsed.get("legs") or []))
    if not has_dest:
        chosen = await destination_suggester_agent({**parsed, "raw_query": query})
        # Drop the now-stale router error before merging the suggester's result.
        if chosen.get("destination") or chosen.get("legs"):
            parsed["errors"] = [
                e for e in (parsed.get("errors") or [])
                if not (e.get("agent") == "router"
                        and e.get("stage") == "extract"
                        and "destination" in (e.get("message") or ""))
            ]
        parsed = {**parsed, **chosen}
        has_dest = bool(parsed.get("destination") or (parsed.get("legs") or []))

    # Step 2b: if we *still* have no destination (suggester failed or
    # returned nothing), short-circuit instead of geocoding "Unknown".
    if not has_dest:
        return TravelPlan(
            destination="(unresolved)",
            errors=[*(parsed.get("errors") or []), {
                "agent": "main",
                "stage": "destination",
                "message": (
                    "Could not determine a destination. Please re-run with a "
                    "city, country, or specific preferences (e.g. 'beach', 'museums')."
                ),
            }],
        )

    legs = parsed.get("legs") or []
    user_origin = parsed.get("origin") or settings.default_origin
    base = {
        "travelers": parsed.get("travelers"),
        "budget_tier": parsed.get("budget_tier"),
        "budget_amount": parsed.get("budget_amount"),
        "budget_currency": parsed.get("budget_currency"),
        "budget_scope": parsed.get("budget_scope"),
        "preferences": parsed.get("preferences") or [],
    }

    # Single-leg trip: one graph invocation, return its plan as-is.
    if len(legs) <= 1:
        leg = legs[0] if legs else {
            "destination": parsed.get("destination") or "Unknown",
            "start": (parsed.get("dates") or {}).get("start"),
            "end": (parsed.get("dates") or {}).get("end"),
            "lodging": parsed.get("user_lodging"),
        }
        single_state: dict[str, Any] = {
            "raw_query": "",  # router will short-circuit
            "origin": user_origin,
            "destination": leg["destination"],
            "dates": _leg_dates(leg) or parsed.get("dates"),
            "travelers": base["travelers"],
            "budget_tier": base["budget_tier"],
            "budget_amount": base["budget_amount"],
            "budget_currency": base["budget_currency"],
            "budget_scope": base["budget_scope"],
            "preferences": base["preferences"],
            "user_lodging": leg.get("lodging") or parsed.get("user_lodging"),
            "legs": [leg],
            "destination_was_inferred": bool(parsed.get("destination_was_inferred")),
            "skip_summary": False,
        }
        final = await graph.ainvoke(single_state)
        plan_data = final.get("final_plan") or {}
        if plan_data:
            plan = TravelPlan.model_validate(plan_data)
            # Combine router-stage errors with whatever the graph collected.
            for err in parsed.get("errors") or []:
                if err not in plan.errors:
                    plan.errors.append(err)
            return plan
        return TravelPlan(destination=leg["destination"] or "Unknown")

    # Multi-leg trip: run each leg with its own dates and stitch results.
    leg_plans: list[LegPlan] = []
    all_flights: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = list(parsed.get("errors") or [])
    leg1_summary: str | None = None
    leg1_budget_assessment = None

    for i, leg in enumerate(legs):
        origin_for_leg = user_origin if i == 0 else legs[i - 1]["destination"]
        log.info(
            "orchestrator: running leg %d/%d (%s -> %s, %s)",
            i + 1, len(legs), origin_for_leg, leg["destination"], _leg_dates(leg),
        )
        # Leg 1 keeps the LLM summary; others skip to save calls.
        leg_state_overrides = {
            **base,
            "destination_was_inferred": bool(parsed.get("destination_was_inferred")) if i == 0 else False,
        }
        leg_plan, _ = await _run_single_leg_keep_summary(
            graph, leg_state_overrides, leg, origin_for_leg, keep_summary=(i == 0),
        )
        if leg_plan:
            if i == 0:
                leg1_summary = leg_plan.summary
                leg1_budget_assessment = leg_plan.budget_assessment
            all_flights.extend(f.model_dump(mode="json") for f in leg_plan.flights)
            all_errors.extend(leg_plan.errors)
            leg_plans.append(_leg_to_legplan(leg, leg_plan))
        else:
            all_errors.append({
                "agent": "orchestrator",
                "stage": "leg_run",
                "message": f"leg {i + 1} ({leg['destination']}) produced no plan",
            })
            leg_plans.append(_leg_to_legplan(leg, None))

    # Renumber itinerary days so day numbers are continuous across legs.
    # Each leg's itinerary uses local days 1..K; we shift by the running
    # total of leg-spans so leg N starts on (sum of prior leg lengths) + 1.
    # Use the leg's date span (not the itinerary day-range) — itineraries
    # can drop a day when there's no hotel/restaurant data, but the trip
    # still occupies that day in wall-clock terms.
    cumulative = 0
    for leg, lp in zip(legs, leg_plans):
        if cumulative:
            for stop in lp.itinerary:
                stop.day = stop.day + cumulative
        cumulative += _leg_day_count(leg, lp)

    # Return flight: last leg's destination → user's origin, day after last.end.
    last_leg = legs[-1]
    return_depart = _next_day(last_leg.get("end")) or last_leg.get("end")
    if user_origin and return_depart:
        log.info(
            "orchestrator: searching return flight %s -> %s on %s",
            last_leg["destination"], user_origin, return_depart,
        )
        return_raw = await _search_return_flight(
            last_dest=last_leg["destination"],
            origin_city=user_origin,
            depart_iso=return_depart,
            travelers=parsed.get("travelers") or 1,
        )
        all_flights.extend(return_raw)

    # De-duplicate flights by (airline, flight_number, depart_at, price).
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for f in all_flights:
        key = (
            f.get("airline"),
            f.get("flight_number"),
            f.get("depart_at"),
            f.get("price"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    flights_models = [Flight.model_validate(f) for f in deduped]

    overall_dates = parsed.get("dates")
    if not overall_dates and legs[0].get("start") and legs[-1].get("end"):
        overall_dates = {"start": legs[0]["start"], "end": legs[-1]["end"]}

    plan = TravelPlan(
        destination=" → ".join(l["destination"] for l in legs),
        dates=overall_dates,
        travelers=parsed.get("travelers") or 1,
        budget_tier=parsed.get("budget_tier"),
        budget_assessment=leg1_budget_assessment,
        user_lodging=parsed.get("user_lodging"),
        flights=flights_models,
        legs=leg_plans,
        destination_was_inferred=bool(parsed.get("destination_was_inferred")),
        summary=leg1_summary,
        errors=all_errors,
    )
    return plan


def main() -> None:
    load_dotenv()
    configure_logging()
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    plan = asyncio.run(_run(query))
    render_terminal(plan)


if __name__ == "__main__":
    main()
