"""Shared state contract for all agents in the travel-planning graph.

`TripState` is a TypedDict that every node reads from and returns partial
updates to. LangGraph merges those partial updates into the running state
according to each field's reducer (default: overwrite; lists use additive
reducers so parallel specialists can each contribute results).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class TripState(TypedDict, total=False):
    """The single source of truth passed between every node in the graph."""

    # --- inputs (Router reads, user provides) ---
    raw_query: str  # the user's natural-language request

    # --- parsed intent (Router writes) ---
    origin: str | None  # free-form city name or 3-letter IATA — Flight agent resolves
    destination: str | None  # mirrors legs[0]["destination"] for back-compat
    dates: dict[str, str] | None  # mirrors legs[0] window for back-compat
    travelers: int | None
    budget_tier: str | None  # "budget" | "mid" | "luxury"
    # Optional absolute budget. When the user mentions a number ("1 million
    # COP", "2000 USD"), the router populates these so the flights agent
    # can compare against per-route price priors and surface a feasibility
    # verdict. None means "no absolute budget given — use tier only".
    budget_amount: float | None
    budget_currency: str | None  # ISO 4217, e.g. "COP", "USD", "EUR"
    budget_scope: str | None     # "flights" | "trip" | None (defaults to "trip")
    # Verdict written by the flights agent: "ok" | "tight" | "infeasible".
    # Surfaced in the rendered output as a warning banner when not "ok".
    budget_assessment: dict[str, Any] | None
    preferences: list[str]  # ["vegetarian", "museums", "no-redeye", ...]
    user_lodging: str | None  # Router writes when user said "I'm staying at <X>";
                              # Hotel agent skips search and Logistics uses it as
                              # the routing origin.

    # --- multi-leg trips (Router writes) ---
    # A normal one-city trip is `legs=[{<one>}]`. Each leg dict has:
    #   destination: str
    #   start: str | None  (YYYY-MM-DD)
    #   end:   str | None
    #   lodging: str | None  (per-leg user_lodging override)
    # main.py runs the existing graph once per leg with leg-local fields
    # populated into a fresh TripState. Specialists are unchanged.
    legs: list[dict[str, Any]] | None

    # Set by destination_suggester when it filled in a destination from
    # vague preferences. Synthesizer mentions this in the summary so the
    # user knows the destination wasn't given.
    destination_was_inferred: bool

    # main.py sets this on per-leg graph runs (legs 2..N) so the
    # synthesizer skips the narrative summary LLM call. The orchestrator
    # generates a single trip-level summary itself.
    skip_summary: bool

    # --- specialist outputs (each specialist writes its own field) ---
    flights: Annotated[list[dict[str, Any]], operator.add]  # Flight Agent
    hotels: Annotated[list[dict[str, Any]], operator.add]  # Hotel Agent
    restaurants: Annotated[list[dict[str, Any]], operator.add]  # Food Agent
    attractions: Annotated[list[dict[str, Any]], operator.add]  # Food agent writes
                                                                # the pool from Overpass;
                                                                # Logistics & Synthesizer
                                                                # consume it.
    itinerary_stops: Annotated[list[dict[str, Any]], operator.add]  # Food/Logistics
    logistics: Annotated[list[dict[str, Any]], operator.add]  # Logistics Agent

    # --- final output (Synthesizer writes) ---
    final_plan: dict[str, Any] | None  # serialized TravelPlan

    # --- error channel (any node may append) ---
    errors: Annotated[list[dict[str, Any]], operator.add]
