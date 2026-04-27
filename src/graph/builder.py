"""Assemble the LangGraph StateGraph that orchestrates every agent.

Topology:

    START → router → (Send) → {flights, hotel, food, logistics}
                                         ↓ (join)
                                    synthesizer → END

The four specialists are dispatched in parallel via `Send()` from a
conditional edge after the router. Each specialist edges directly into
the synthesizer; LangGraph waits until all incoming edges have produced
state before running the synthesizer node.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.flights_agent import flights_agent
from src.agents.food_agent import food_agent
from src.agents.hotel_agent import hotel_agent
from src.agents.logistics_agent import logistics_agent
from src.agents.router import router_agent
from src.agents.synthesizer import synthesizer_agent
from src.graph.edges import SPECIALIST_NODES, fan_out_to_specialists
from src.state.trip_state import TripState


def build_graph():
    """Compile and return the travel-planning graph."""
    graph = StateGraph(TripState)

    graph.add_node("router", router_agent)
    graph.add_node("flights", flights_agent)
    graph.add_node("hotel", hotel_agent)
    graph.add_node("food", food_agent)
    graph.add_node("logistics", logistics_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        fan_out_to_specialists,
        list(SPECIALIST_NODES),
    )
    for node in SPECIALIST_NODES:
        graph.add_edge(node, "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
