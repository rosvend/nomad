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


def route_after_router(state: TripState) -> str | list[Send]:
    """Route to the destination_suggester if the router couldn't extract any
    destination but the user did express preferences. Otherwise fan out to
    specialists as usual.

    Returning ``"destination_suggester"`` triggers the suggester node;
    returning a list of ``Send`` objects fires the parallel fan-out.
    """
    has_dest = bool(state.get("destination") or (state.get("legs") or []))
    has_prefs = bool(state.get("preferences") or [])
    if not has_dest and has_prefs:
        return "destination_suggester"
    return fan_out_to_specialists(state)


def has_errors(state: TripState) -> bool:
    """Edge predicate stub for future fallback routing."""
    return bool(state.get("errors"))
