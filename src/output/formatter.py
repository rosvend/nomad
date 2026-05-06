"""Render a TravelPlan for CLI / API consumers.

Two renderers:

- ``render_terminal(plan)`` — terminal-friendly. Uses `rich` to draw
  aligned tables and colored panels when stdout is a TTY, and degrades
  to clean plain text (no ANSI escapes) when it isn't (pipes, files,
  CI logs). This is what the CLI in `main.py` calls.

- ``to_markdown(plan)`` — Markdown rendering kept for API consumers
  that want to embed the plan into a document or web view. Not used by
  the CLI anymore (Markdown source rendered raw is unreadable in a
  plain terminal).

- ``to_json(plan)`` — schema-faithful JSON dump for programmatic use.
"""

from __future__ import annotations

import json
import sys
from urllib.parse import quote

from src.output.schemas import BudgetAssessment, TravelPlan


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

    if plan.legs:
        # Multi-leg trip — render one block per destination.
        for i, leg in enumerate(plan.legs, 1):
            lines.append(f"# Leg {i} — {leg.destination}")
            if leg.dates:
                lines.append(
                    f"_Dates: {leg.dates.get('start')} → {leg.dates.get('end')}_"
                )
            lines.append("")

            if leg.user_lodging:
                lines.append("## Lodging")
                lines.append(f"_Provided by user: {leg.user_lodging}_")
                lines.append("")
            else:
                lines.append("## Hotels")
                lines.append(_hotels_section(leg.hotels))
                lines.append("")

            lines.append("## Restaurants")
            lines.append(_restaurants_section(leg.restaurants))
            lines.append("")

            lines.append("## Itinerary")
            lines.append(_itinerary_section(leg.itinerary))
            lines.append("")

            lines.append("## Logistics")
            lines.append(_logistics_section(leg.logistics))
            lines.append("")
    else:
        # Single-leg trip — unchanged layout.
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


# ── Terminal renderer (rich) ─────────────────────────────────────────
#
# The CLI uses this. It auto-detects whether stdout is a TTY:
#   - TTY: bold/underlined section headers, colored budget banner,
#     box-drawn tables.
#   - Non-TTY (pipe/file/CI): no ANSI escapes, plain text fallback that
#     stays grep-friendly. `rich.Console(force_terminal=None)` handles
#     this automatically.
#
# Markdown sigils are deliberately absent: no `#`/`**`/`-`/backticks.
# Terminals that don't render Markdown would otherwise show the raw
# punctuation as noise.

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _make_console(file=None) -> Console:
    """Build a Console that auto-degrades to plain text on non-TTY.

    `force_terminal=None` lets rich detect TTY-ness via isatty(); when
    output is being piped or redirected it strips ANSI and produces
    clean ASCII. `soft_wrap=True` lets long URLs flow to the next line
    instead of being truncated.
    """
    return Console(file=file, force_terminal=None, soft_wrap=True, highlight=False)


def _section_header(console: Console, text: str) -> None:
    """A blank line, then the section title in bold + underline.

    On non-TTY this collapses to plain text on its own line.
    """
    console.print()
    console.print(f"[bold underline]{text}[/]")


def _render_header(console: Console, plan: TravelPlan) -> None:
    title = f"TRAVEL PLAN  —  {plan.destination or 'Unknown destination'}"
    console.print(f"[bold]{title}[/]")
    if plan.dates:
        s = plan.dates.get("start") or "?"
        e = plan.dates.get("end") or "?"
        console.print(f"[dim]Dates:    {s} -> {e}[/]")
    console.print(
        f"[dim]Travelers:[/] {plan.travelers}    "
        f"[dim]Budget:[/] {plan.budget_tier or 'n/a'}"
    )


def _render_budget_banner(console: Console, ba: BudgetAssessment | None) -> None:
    """Show a colored panel when the user's budget is tight or infeasible.

    Stays silent when verdict is "ok" or no assessment exists.
    """
    if ba is None or ba.verdict == "ok":
        return
    lines: list[str] = []
    amount_str = (
        f"{ba.budget_amount:,.0f} {ba.budget_currency}"
        if ba.budget_amount and ba.budget_currency else "?"
    )
    usd_str = f"~${ba.budget_usd:,.0f} USD" if ba.budget_usd is not None else ""
    prior_str = (
        f"${ba.prior_usd_low:,}-${ba.prior_usd_high:,} USD"
        if ba.prior_usd_low and ba.prior_usd_high else "(no prior available)"
    )
    if ba.verdict == "infeasible":
        title = "BUDGET WARNING — likely infeasible"
        style = "red"
        lines.append(f"Your budget:        {amount_str}  ({usd_str})")
        lines.append(f"Typical RT fare:    {prior_str}")
        lines.append("Verdict:            below the typical low end for this route.")
    else:  # tight
        title = "BUDGET NOTE — tight"
        style = "yellow"
        lines.append(f"Your budget:        {amount_str}  ({usd_str})")
        lines.append(f"Typical RT fare:    {prior_str}")
        lines.append("Verdict:            within reach but close to the floor.")
    if ba.cheapest_found is not None:
        currency = ba.cheapest_found_currency or ""
        lines.append(
            f"Cheapest found:     {ba.cheapest_found:,.0f} {currency}".rstrip()
        )
    lines.append("")
    lines.append("Results may rely on multi-stop routings, off-peak dates, or")
    lines.append("budget carriers. Consider relaxing the budget, choosing a")
    lines.append("closer destination, or shifting dates.")
    console.print()
    console.print(Panel("\n".join(lines), title=title, border_style=style, expand=True))


def _render_summary(console: Console, summary: str | None) -> None:
    if not summary:
        return
    _section_header(console, "Summary")
    console.print(summary)


def _render_flights(console: Console, plan: TravelPlan) -> None:
    _section_header(console, "Flights")
    deeplink = _flights_search_url(plan)
    if deeplink:
        console.print(f"[dim]Verify on Google Flights:[/] {deeplink}")
    if not plan.flights:
        console.print("[dim](no flights found)[/]")
        return
    table = Table(box=SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Airline", no_wrap=True)
    table.add_column("Route", no_wrap=True)
    table.add_column("Depart", no_wrap=True)
    table.add_column("Arrive", no_wrap=True)
    table.add_column("Dur", no_wrap=True)
    table.add_column("Stops", justify="right", no_wrap=True)
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    for i, f in enumerate(plan.flights, 1):
        airline = f.airline or "—"
        if f.flight_number:
            airline = f"{airline} {f.flight_number}"
        route = f"{f.origin or '—'} -> {f.destination or '—'}"
        depart = _fmt_dt(f.depart_at)
        arrive = _fmt_dt(f.arrive_at)
        duration = _fmt_minutes(f.duration_minutes)
        if f.stops is None:
            stops = "—"
        elif f.stops == 0:
            stops = "0"
        else:
            stops = str(f.stops)
        if f.price is not None:
            currency = f.currency or ""
            price = f"{f.price:,.0f} {currency}".strip()
        else:
            price = "—"
        score = f"{f.score:.2f}" if f.score is not None else "—"
        table.add_row(str(i), airline, route, depart, arrive, duration, stops, price, score)
    console.print(table)


def _render_hotels(console: Console, hotels: list, header: str = "Hotels") -> None:
    _section_header(console, header)
    if not hotels:
        console.print("[dim](no hotels found)[/]")
        return
    table = Table(box=SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Rating", justify="right", no_wrap=True)
    table.add_column("Reviews", justify="right", no_wrap=True)
    table.add_column("Price", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Address")
    for i, h in enumerate(hotels, 1):
        rating = f"{h.rating}*" if h.rating is not None else "—"
        reviews = f"{h.review_count:,}" if h.review_count else "—"
        if h.price_level is not None:
            price = "$" * max(1, h.price_level)
        else:
            price = "—"
        score = f"{h.score:.2f}" if h.score is not None else "—"
        address = h.address or "—"
        table.add_row(str(i), h.name, rating, reviews, price, score, address)
    console.print(table)


def _render_restaurants(console: Console, restaurants: list) -> None:
    _section_header(console, "Restaurants")
    if not restaurants:
        console.print("[dim](no restaurants found)[/]")
        return
    table = Table(box=SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Cuisine", no_wrap=True)
    table.add_column("Rating", justify="right", no_wrap=True)
    table.add_column("Reviews", justify="right", no_wrap=True)
    table.add_column("Price", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    for i, r in enumerate(restaurants, 1):
        rating = f"{r.rating}*" if r.rating is not None else "—"
        reviews = f"{r.review_count:,}" if r.review_count else "—"
        if r.price_level is not None:
            price = "$" * max(1, r.price_level)
        else:
            price = "—"
        score = f"{r.score:.2f}" if r.score is not None else "—"
        cuisine = r.cuisine or "—"
        table.add_row(str(i), r.name, cuisine, rating, reviews, price, score)
    console.print(table)


def _render_itinerary(console: Console, stops: list) -> None:
    _section_header(console, "Itinerary")
    if not stops:
        console.print("[dim](no itinerary stops)[/]")
        return
    by_day: dict[int, list] = {}
    for s in stops:
        by_day.setdefault(s.day, []).append(s)
    for day in sorted(by_day):
        items = by_day[day]
        date_str = ""
        for it in items:
            if it.start_time is not None:
                date_str = f"  ({_fmt_dt(it.start_time).split(' ')[0]})"
                break
        console.print(f"[bold]Day {day}{date_str}[/]")
        for it in items:
            time_str = (
                _fmt_dt(it.start_time).split(" ", 1)[1]
                if it.start_time else "  —  "
            )
            duration = _fmt_minutes(it.duration_minutes) if it.duration_minutes else ""
            tail = f"  ({duration})" if duration else ""
            console.print(f"  {time_str}  {it.name}{tail}")
            if it.address:
                console.print(f"    [dim]{it.address}[/]")
            if it.notes:
                console.print(f"    [dim]{it.notes}[/]")


def _render_logistics(console: Console, legs: list) -> None:
    _section_header(console, "Logistics")
    if not legs:
        console.print("[dim](no logistics computed)[/]")
        return
    grouped: dict[str, list] = {}
    for leg in legs:
        cat = leg.category or "other"
        grouped.setdefault(cat, []).append(leg)
    for cat, group in grouped.items():
        label = " -> ".join(p.replace("_", " ").title() for p in cat.split("→"))
        console.print(f"[dim]{label}[/]")
        for leg in group:
            duration = _fmt_minutes(leg.duration_minutes) if leg.duration_minutes else "—"
            distance = (
                f"{leg.distance_km:.1f} km"
                if leg.distance_km is not None else "—"
            )
            console.print(
                f"  {leg.from_stop}  ->  {leg.to_stop}    "
                f"[dim]{leg.mode}  ·  {duration}  ·  {distance}[/]"
            )
            if leg.notes:
                console.print(f"    [dim]{leg.notes}[/]")


def _render_errors(console: Console, errors: list[dict]) -> None:
    if not errors:
        return
    _section_header(console, "Errors")
    for err in errors:
        agent = err.get("agent", "?")
        stage = err.get("stage", "?")
        msg = err.get("message") or err.get("details") or ""
        # `[agent/stage]` would be parsed as rich markup and silently
        # consumed; build the line via Text so the brackets are literal.
        line = Text("  ")
        line.append("* ", style="red")
        line.append(f"[{agent}/{stage}] ")
        line.append(str(msg))
        console.print(line)


def render_terminal(plan: TravelPlan, file=None) -> None:
    """Render `plan` to a terminal. Pass `file=sys.stderr` to redirect.

    On a TTY this produces colored, bordered tables; on a non-TTY it
    produces clean plain text with no ANSI escapes (rich detects this).
    """
    console = _make_console(file=file or sys.stdout)
    _render_header(console, plan)
    _render_budget_banner(console, plan.budget_assessment)
    _render_summary(console, plan.summary)
    _render_flights(console, plan)

    if plan.legs:
        for i, leg in enumerate(plan.legs, 1):
            console.print()
            console.print(f"[bold]LEG {i} — {leg.destination}[/]")
            if leg.dates:
                s = leg.dates.get("start") or "?"
                e = leg.dates.get("end") or "?"
                console.print(f"[dim]Dates:    {s} -> {e}[/]")
            if leg.user_lodging:
                _section_header(console, "Lodging")
                console.print(f"[dim]Provided by user:[/] {leg.user_lodging}")
            else:
                _render_hotels(console, leg.hotels)
            _render_restaurants(console, leg.restaurants)
            _render_itinerary(console, leg.itinerary)
            _render_logistics(console, leg.logistics)
    else:
        if plan.user_lodging:
            _section_header(console, "Lodging")
            console.print(f"[dim]Provided by user:[/] {plan.user_lodging}")
        else:
            _render_hotels(console, plan.hotels)
        _render_restaurants(console, plan.restaurants)
        _render_itinerary(console, plan.itinerary)
        _render_logistics(console, plan.logistics)

    _render_errors(console, plan.errors)
    console.print()
