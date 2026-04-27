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
    lines.append(_hotels_section(plan.hotels))
    lines.append("")

    lines.append("## Restaurants")
    lines.append(_restaurants_section(plan.restaurants))
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


def _hotels_section(hotels: list) -> str:
    """Human-readable hotel listing (the other sections stay JSON until
    their agents are filled in)."""
    if not hotels:
        return "_No hotels found._"

    lines: list[str] = []
    for i, h in enumerate(hotels, 1):
        meta_parts: list[str] = []
        if h.rating is not None:
            meta_parts.append(f"{h.rating}★")
        if h.review_count:
            meta_parts.append(f"{h.review_count:,} reviews")
        if h.price_level is not None:
            meta_parts.append("$" * max(1, h.price_level))
        if h.score is not None:
            meta_parts.append(f"score {h.score:.2f}")
        meta = "  ·  ".join(meta_parts) if meta_parts else "—"

        lines.append(f"### {i}. {h.name}")
        lines.append(f"_{meta}_")
        if h.address:
            lines.append(f"📍 {h.address}")
        if h.website:
            lines.append(f"🔗 {h.website}")
        if h.score_breakdown:
            sb = h.score_breakdown
            lines.append(
                f"`rating={sb.get('rating', 0):.2f}  "
                f"popularity={sb.get('popularity', 0):.2f}  "
                f"proximity={sb.get('proximity', 0):.2f}  "
                f"budget={sb.get('budget', 0):.2f}`"
            )
        if h.notes:
            lines.append(f"> {h.notes}")
        lines.append("")
    return "\n".join(lines)


def _restaurants_section(restaurants: list) -> str:
    """Human-readable restaurant listing, parallel to `_hotels_section`."""
    if not restaurants:
        return "_No restaurants found._"

    lines: list[str] = []
    for i, r in enumerate(restaurants, 1):
        meta_parts: list[str] = []
        if r.rating is not None:
            meta_parts.append(f"{r.rating}★")
        if r.review_count:
            meta_parts.append(f"{r.review_count:,} reviews")
        if r.price_level is not None:
            meta_parts.append("$" * max(1, r.price_level))
        if r.score is not None:
            meta_parts.append(f"score {r.score:.2f}")
        meta = "  ·  ".join(meta_parts) if meta_parts else "—"

        lines.append(f"### {i}. {r.name}")
        lines.append(f"_{meta}_")
        if r.cuisine:
            lines.append(f"🍴 {r.cuisine}")
        if r.address:
            lines.append(f"📍 {r.address}")
        if r.website:
            lines.append(f"🔗 {r.website}")
        if r.amenities:
            lines.append(f"_amenities: {', '.join(r.amenities)}_")
        if r.score_breakdown:
            sb = r.score_breakdown
            lines.append(
                f"`rating={sb.get('rating', 0):.2f}  "
                f"popularity={sb.get('popularity', 0):.2f}  "
                f"proximity={sb.get('proximity', 0):.2f}  "
                f"cuisine={sb.get('cuisine', 0):.2f}  "
                f"budget={sb.get('budget', 0):.2f}`"
            )
        if r.notes:
            lines.append(f"> {r.notes}")
        lines.append("")
    return "\n".join(lines)
