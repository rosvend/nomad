"""Conditional edge logic for the travel-planning graph.

The Router fans out to all four specialists in parallel. Conditional
branching for fallback / retry behaviour will be added later (e.g. if
flight search errors and a fallback provider is configured, route there
before continuing to Synthesizer).
"""

from __future__ import annotations

from langgraph.types import Send

from src.state.trip_state import TripState

SPECIALIST_NODES = ("flights", "hotel", "food", "logistics")


def fan_out_to_specialists(state: TripState) -> list[Send]:
    """Dispatch the same state to every specialist node in parallel."""
    return [Send(node, state) for node in SPECIALIST_NODES]


def has_errors(state: TripState) -> bool:
    """Edge predicate stub for future fallback routing."""
    return bool(state.get("errors"))
