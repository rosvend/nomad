"""Assemble the LangGraph StateGraph that orchestrates every agent.

Topology::

    START → router → (Send) → {flights, hotel, food}
                               flights ─────────────────→ synthesizer
                               hotel  ──┐
                                        ├── logistics ──→ synthesizer
                               food   ──┘
                                              synthesizer → END

The Router fans out three specialists in parallel via `Send()`. Logistics
has explicit incoming edges from `hotel` and `food`; LangGraph joins on
incoming edges, so Logistics runs once after both have written their
state. The Synthesizer is registered with `defer=True` so it runs exactly
once after every other node has settled, regardless of which superstep
each upstream finishes in. Without `defer`, a fast-failing `flights` path
fires the synthesizer before `logistics` completes, producing two passes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.destination_suggester import destination_suggester_agent
from src.agents.flights_agent import flights_agent
from src.agents.food_agent import food_agent
from src.agents.hotel_agent import hotel_agent
from src.agents.logistics_agent import logistics_agent
from src.agents.router import router_agent
from src.agents.synthesizer import synthesizer_agent
from src.graph.edges import (
    INITIAL_SPECIALISTS,
    fan_out_to_specialists,
    route_after_router,
)
from src.state.trip_state import TripState


def build_graph():
    """Compile and return the travel-planning graph."""
    graph = StateGraph(TripState)

    graph.add_node("router", router_agent)
    graph.add_node("destination_suggester", destination_suggester_agent)
    graph.add_node("flights", flights_agent)
    graph.add_node("hotel", hotel_agent)
    graph.add_node("food", food_agent)
    graph.add_node("logistics", logistics_agent)
    graph.add_node("synthesizer", synthesizer_agent, defer=True)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        [*INITIAL_SPECIALISTS, "destination_suggester"],
    )
    # Once a destination has been picked, fan out exactly like the router.
    graph.add_conditional_edges(
        "destination_suggester",
        fan_out_to_specialists,
        list(INITIAL_SPECIALISTS),
    )

    # Logistics joins on hotel + food (LangGraph waits for both).
    graph.add_edge("hotel", "logistics")
    graph.add_edge("food", "logistics")

    # Synthesizer joins on flights + logistics. defer=True on the node
    # ensures it runs exactly once at the end, even when upstream nodes
    # complete in different supersteps.
    graph.add_edge("flights", "synthesizer")
    graph.add_edge("logistics", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()
