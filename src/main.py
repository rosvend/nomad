"""CLI entry point. Build the graph, run it, print the markdown plan.

Usage:
    uv run python -m src.main "Plan a 5-day trip to Tokyo"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from src.config import configure_logging
from src.graph.builder import build_graph
from src.output.formatter import to_markdown
from src.output.schemas import TravelPlan

DEFAULT_QUERY = "Plan a 5-day trip to Tokyo for one person, mid budget."


def main() -> None:
    load_dotenv()
    configure_logging()

    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    graph = build_graph()
    final_state = graph.invoke({"raw_query": query})

    plan_data = final_state.get("final_plan") or {}
    plan = TravelPlan.model_validate(plan_data) if plan_data else TravelPlan(destination="Unknown")
    print(to_markdown(plan))


if __name__ == "__main__":
    main()
