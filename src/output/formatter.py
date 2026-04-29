"""Render a TravelPlan as markdown or JSON for the CLI / API consumers.

These are deliberately minimal stubs — real templating (with sections,
emoji bullets, tables of flights, etc.) lands once specialists return data.
"""

from __future__ import annotations

import json
from urllib.parse import quote

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

    if plan.summary:
        lines.append("## Summary")
        lines.append(plan.summary)
        lines.append("")

    lines.append("## Flights")
    deeplink = _flights_search_url(plan)
    if deeplink:
        lines.append(f"_Verify on Google Flights: {deeplink}_")
        lines.append("")
    lines.append(_flights_section(plan.flights))
    lines.append("")

    if plan.user_lodging:
        lines.append("## Lodging")
        lines.append(f"_Provided by user: {plan.user_lodging}_")
        lines.append("")
    else:
        lines.append("## Hotels")
        lines.append(_hotels_section(plan.hotels))
        lines.append("")

    lines.append("## Restaurants")
    lines.append(_restaurants_section(plan.restaurants))
    lines.append("")

    lines.append("## Itinerary")
    lines.append(_itinerary_section(plan.itinerary))
    lines.append("")

    lines.append("## Logistics")
    lines.append(_logistics_section(plan.logistics))
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


def _fmt_minutes(mins: int | None) -> str:
    if not mins:
        return "—"
    h, m = divmod(int(mins), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _fmt_dt(dt) -> str:
    """Render a datetime field compactly: `2026-05-01 00:05`."""
    if dt is None:
        return "—"
    s = str(dt)
    # Python datetime → "2026-05-01T00:05:00" or "2026-05-01 00:05:00"
    return s.replace("T", " ").rsplit(":", 1)[0]


def _flights_search_url(plan: TravelPlan) -> str | None:
    """Build a Google Flights deeplink users can click to sanity-check results.

    Origin comes from the first flight's IATA (correct for direct + multi-leg).
    Destination comes from `plan.destination` (the user's requested city)
    rather than the first flight's `destination` field, because for multi-stop
    routes `fli`'s normalizer can surface the layover airport — leading to a
    misleading deeplink like "MDE → GYE" when the user asked for Santa Marta.
    """
    if not plan.flights or not plan.destination:
        return None
    origin = plan.flights[0].origin
    if not origin:
        return None
    parts = [f"Flights from {origin} to {plan.destination}"]
    if plan.dates and plan.dates.get("start"):
        parts.append(f"on {plan.dates['start']}")
    if plan.dates and plan.dates.get("end"):
        parts.append(f"through {plan.dates['end']}")
    return f"https://www.google.com/travel/flights?q={quote(' '.join(parts))}"


def _flights_section(flights: list) -> str:
    """Human-readable flight listing, parallel to `_hotels_section`."""
    if not flights:
        return "_No flights found._"

    lines: list[str] = []
    for i, f in enumerate(flights, 1):
        # Header: "1. UA  LAX → HND  ·  2026-05-01 00:05 → 2026-05-02 20:20"
        airline = f.airline or "—"
        flight_no = f"  ({f.flight_number})" if f.flight_number else ""
        route = f"{f.origin or '—'} → {f.destination or '—'}"
        depart = _fmt_dt(f.depart_at)
        arrive = _fmt_dt(f.arrive_at)
        lines.append(f"### {i}. {airline}{flight_no}")
        lines.append(f"_{route}  ·  {depart} → {arrive}_")

        meta_parts: list[str] = []
        if f.duration_minutes is not None:
            meta_parts.append(_fmt_minutes(f.duration_minutes))
        if f.stops is not None:
            meta_parts.append("non-stop" if f.stops == 0 else f"{f.stops} stop{'s' if f.stops != 1 else ''}")
        if f.price is not None:
            currency = f.currency or ""
            meta_parts.append(f"{f.price:.0f} {currency}".strip())
        if f.score is not None:
            meta_parts.append(f"score {f.score:.2f}")
        if meta_parts:
            lines.append(f"_{'  ·  '.join(meta_parts)}_")

        if f.score_breakdown:
            sb = f.score_breakdown
            lines.append(
                f"`price={sb.get('price', 0):.2f}  "
                f"stops={sb.get('stops', 0):.2f}  "
                f"duration={sb.get('duration', 0):.2f}`"
            )
        if f.notes:
            lines.append(f"> {f.notes}")
        lines.append("")
    return "\n".join(lines)


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


def _itinerary_section(stops: list) -> str:
    """Day-by-day rendering of `ItineraryStop` entries."""
    if not stops:
        return "_No itinerary stops yet._"

    # Group by day, preserving stop order within each day.
    by_day: dict[int, list] = {}
    for s in stops:
        by_day.setdefault(s.day, []).append(s)

    lines: list[str] = []
    for day in sorted(by_day):
        items = by_day[day]
        # Pull the date from the first item with a start_time, if any.
        date_str = ""
        for it in items:
            if it.start_time is not None:
                date_str = f" — {_fmt_dt(it.start_time).split(' ')[0]}"
                break
        lines.append(f"### Day {day}{date_str}")
        for it in items:
            time_str = _fmt_dt(it.start_time).split(" ", 1)[1] if it.start_time else "—"
            duration = _fmt_minutes(it.duration_minutes) if it.duration_minutes else ""
            tag = f"  ·  {duration}" if duration else ""
            lines.append(f"- **{time_str}**  {it.name}{tag}")
            if it.address:
                lines.append(f"  📍 {it.address}")
            if it.notes:
                lines.append(f"  > {it.notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _logistics_section(legs: list) -> str:
    """Walking-leg listing grouped by category (hotel→restaurant, etc.)."""
    if not legs:
        return "_No logistics computed yet._"

    # Group preserving order of first appearance.
    grouped: dict[str, list] = {}
    for leg in legs:
        cat = leg.category or "other"
        grouped.setdefault(cat, []).append(leg)

    lines: list[str] = []
    for cat, group in grouped.items():
        # Pretty label: "hotel→restaurant" → "Hotel → Restaurant"
        label = " → ".join(p.replace("_", " ").title() for p in cat.split("→"))
        lines.append(f"_{label}_")
        for leg in group:
            duration = _fmt_minutes(leg.duration_minutes) if leg.duration_minutes else "—"
            distance = f"{leg.distance_km:.1f} km" if leg.distance_km is not None else "—"
            lines.append(
                f"- **{leg.from_stop}** → **{leg.to_stop}**  ·  "
                f"{leg.mode}  ·  {duration}  ·  {distance}"
            )
            if leg.instructions_url:
                lines.append(f"  🗺 {leg.instructions_url}")
            if leg.notes:
                lines.append(f"  > {leg.notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
