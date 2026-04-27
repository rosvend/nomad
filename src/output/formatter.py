"""Render a TravelPlan as markdown or JSON for the CLI / API consumers.

These are deliberately minimal stubs — real templating (with sections,
emoji bullets, tables of flights, etc.) lands once specialists return data.
"""

from __future__ import annotations

import json

from src.output.schemas import TravelPlan


def to_json(plan: TravelPlan) -> str:
    """Serialize the plan to pretty-printed JSON."""
    return plan.model_dump_json(indent=2)


def to_markdown(plan: TravelPlan) -> str:
    """Render the plan as a markdown document."""
    lines: list[str] = []
    lines.append(f"# Travel Plan — {plan.destination or 'Unknown destination'}")
    if plan.dates:
        lines.append(f"_Dates: {plan.dates.get('start')} → {plan.dates.get('end')}_")
    lines.append(f"_Travelers: {plan.travelers}  ·  Budget: {plan.budget_tier or 'n/a'}_")
    lines.append("")

    lines.append("## Flights")
    lines.append(_section(plan.flights, "No flights found."))
    lines.append("")

    lines.append("## Hotels")
    lines.append(_section(plan.hotels, "No hotels found."))
    lines.append("")

    lines.append("## Restaurants")
    lines.append(_section(plan.restaurants, "No restaurants found."))
    lines.append("")

    lines.append("## Itinerary")
    lines.append(_section(plan.itinerary, "No itinerary stops yet."))
    lines.append("")

    lines.append("## Logistics")
    lines.append(_section(plan.logistics, "No logistics computed yet."))
    lines.append("")

    if plan.errors:
        lines.append("## Errors")
        for err in plan.errors:
            lines.append(f"- `{err}`")

    return "\n".join(lines)


def _section(items: list, empty_msg: str) -> str:
    if not items:
        return f"_{empty_msg}_"
    return "\n".join(f"- `{json.dumps(item.model_dump(mode='json'))}`" for item in items)
