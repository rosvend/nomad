"""Conditional edge logic for the travel-planning graph.

Three specialists fan out from the Router and run in parallel: Flights,
Hotel, Food. Logistics is *not* in the initial fan-out — it depends on
the merged state of Hotel + Food (it routes between hotels and the
restaurants those agents discovered), so it has explicit incoming edges
from each of them. LangGraph waits for both before invoking Logistics.
"""

from __future__ import annotations

from langgraph.types import Send

from src.state.trip_state import TripState

# Initial parallel specialists dispatched via Send() from the Router.
# Logistics is intentionally absent — see this module's docstring.
INITIAL_SPECIALISTS = ("flights", "hotel", "food")

# Backward-compat alias for any external callers that referenced the old
# name. Always reflects the initial fan-out set.
SPECIALIST_NODES = INITIAL_SPECIALISTS


def fan_out_to_specialists(state: TripState) -> list[Send]:
    """Dispatch the same state to every initial specialist in parallel."""
    return [Send(node, state) for node in INITIAL_SPECIALISTS]


def has_errors(state: TripState) -> bool:
    """Edge predicate stub for future fallback routing."""
    return bool(state.get("errors"))
