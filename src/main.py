"""CLI entry point. Build the graph, run it, print the markdown plan.

Usage:
    uv run python -m src.main "Plan a 5-day trip to Tokyo"

The graph is invoked via ``ainvoke`` because some specialist nodes
(e.g. the Hotel agent) are async — they call the tools layer, which is
async throughout. ``ainvoke`` awaits sync nodes natively, so we don't
need a separate sync code path.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from src.config import configure_logging
from src.graph.builder import build_graph
from src.output.formatter import to_markdown
from src.output.schemas import TravelPlan

DEFAULT_QUERY = "Plan a 5-day trip to Tokyo for one person, mid budget."


async def _run(query: str) -> TravelPlan:
    graph = build_graph()
    final_state = await graph.ainvoke({"raw_query": query})
    plan_data = final_state.get("final_plan") or {}
    if plan_data:
        return TravelPlan.model_validate(plan_data)
    return TravelPlan(destination="Unknown")


def main() -> None:
    load_dotenv()
    configure_logging()
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    plan = asyncio.run(_run(query))
    print(to_markdown(plan))


if __name__ == "__main__":
    main()
