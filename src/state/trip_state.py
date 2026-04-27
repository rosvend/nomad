"""Shared state contract for all agents in the travel-planning graph.

`TripState` is a TypedDict that every node reads from and returns partial
updates to. LangGraph merges those partial updates into the running state
according to each field's reducer (default: overwrite; lists use additive
reducers so parallel specialists can each contribute results).

Rules (see CLAUDE.md):
- All output fields are Optional with a None / empty default.
- Agents return a dict of *only* the keys they update — never mutate state.
- Add the field here before any agent code references it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class TripState(TypedDict, total=False):
    """The single source of truth passed between every node in the graph."""

    # --- inputs (Router reads, user provides) ---
    raw_query: str  # the user's natural-language request

    # --- parsed intent (Router writes) ---
    destination: str | None
    dates: dict[str, str] | None  # {"start": "2026-05-01", "end": "2026-05-08"}
    travelers: int | None
    budget_tier: str | None  # "budget" | "mid" | "luxury"
    preferences: list[str]  # ["vegetarian", "museums", "no-redeye", ...]

    # --- specialist outputs (each specialist writes its own field) ---
    flights: Annotated[list[dict[str, Any]], operator.add]  # Flight Agent
    hotels: Annotated[list[dict[str, Any]], operator.add]  # Hotel Agent
    restaurants: Annotated[list[dict[str, Any]], operator.add]  # Food Agent
    itinerary_stops: Annotated[list[dict[str, Any]], operator.add]  # Food/Logistics
    logistics: Annotated[list[dict[str, Any]], operator.add]  # Logistics Agent

    # --- final output (Synthesizer writes) ---
    final_plan: dict[str, Any] | None  # serialized TravelPlan

    # --- error channel (any node may append) ---
    errors: Annotated[list[dict[str, Any]], operator.add]
