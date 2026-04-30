"""Pure itinerary distribution logic.

Given a fixed number of trip days and the lists of restaurants /
attractions / a primary hotel, produce a list of `ItineraryStop` dicts
spread across days at stable times of day. No I/O, no LLM — this is the
deterministic skeleton that an LLM-generated narrative summary later
sits on top of.

Algorithm
---------
- Day 1 (arrival)              : check-in note · 19:00 dinner
- Days 2..N-1 (middle days)    : 10:00 attraction · 13:00 lunch ·
                                 15:00 attraction · 19:00 dinner
- Day N (departure)            : 10:00 attraction · 13:00 lunch · check-out note

Resources are consumed in the order provided (which the upstream agents
already ranked). Attractions are placed exactly once — if the pool is
shorter than the number of slots, remaining slots become "Free time"
rather than repeats. Restaurants still wrap (eating at a great place
twice over a trip is fine).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any


def _parse_start_date(dates: dict[str, str] | None) -> date:
    """Pull the trip's starting date from the state, falling back to today."""
    if dates and dates.get("start"):
        try:
            return date.fromisoformat(dates["start"])
        except ValueError:
            pass
    return date.today()


def _num_days(dates: dict[str, str] | None, default: int = 3) -> int:
    if not dates:
        return default
    try:
        start = date.fromisoformat(dates["start"])
        end = date.fromisoformat(dates["end"])
    except (KeyError, ValueError, TypeError):
        return default
    return max(1, (end - start).days + 1)


def _pick(items: list[dict[str, Any]], index: int, *, wrap: bool) -> dict[str, Any] | None:
    if not items:
        return None
    if wrap:
        return items[index % len(items)]
    if index >= len(items):
        return None
    return items[index]


def _free_time_stop(day: int, when: datetime, notes: str) -> dict[str, Any]:
    return _stop(
        name="Free time / explore the area",
        day=day,
        when=when,
        duration_minutes=120,
        notes=notes,
    )


def _stop(
    name: str,
    day: int,
    when: datetime,
    *,
    duration_minutes: int | None = None,
    address: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "day": day,
        "start_time": when.isoformat(),
        "duration_minutes": duration_minutes,
        "address": address,
        "notes": notes,
    }


def build_itinerary(
    *,
    dates: dict[str, str] | None,
    restaurants: list[dict[str, Any]],
    attractions: list[dict[str, Any]],
    hotel_name: str | None = None,
) -> list[dict[str, Any]]:
    """Distribute restaurants + attractions across the trip days.

    Args:
        dates: ``state["dates"]`` shape — `{"start": ISO, "end": ISO}` or None.
        restaurants: ranked list of restaurant dicts (Food agent output
            shape; we read `name`, `address`).
        attractions: ranked list of attraction dicts (extracted by the
            synthesizer from logistics legs; we read `name`).
        hotel_name: name of the chosen hotel for the check-in/out notes.

    Returns a flat list of dicts each conforming to `ItineraryStop`.
    """
    n = _num_days(dates)
    start = _parse_start_date(dates)
    out: list[dict[str, Any]] = []

    # Counters so we walk through resources in rank order and only wrap
    # when truly out of distinct picks.
    rest_i = 0
    attr_i = 0

    def _next_rest() -> dict[str, Any] | None:
        nonlocal rest_i
        # Restaurants may wrap — repeating a great spot is fine.
        item = _pick(restaurants, rest_i, wrap=True)
        rest_i += 1
        return item

    def _next_attr() -> dict[str, Any] | None:
        nonlocal attr_i
        # Attractions don't wrap — None signals "leave a free-time slot".
        item = _pick(attractions, attr_i, wrap=False)
        attr_i += 1
        return item

    def _ts(day_offset: int, hour: int, minute: int = 0) -> datetime:
        return datetime.combine(start + timedelta(days=day_offset), time(hour, minute))

    for d in range(n):
        day = d + 1
        is_arrival = d == 0
        is_departure = d == n - 1 and n > 1

        if is_arrival:
            if hotel_name:
                out.append(_stop(
                    name=f"Check in at {hotel_name}",
                    day=day,
                    when=_ts(d, 16, 0),
                    duration_minutes=60,
                    notes="Arrival day — hotel check-in",
                ))
            r = _next_rest()
            if r:
                out.append(_stop(
                    name=r.get("name") or "(restaurant)",
                    day=day,
                    when=_ts(d, 19, 0),
                    duration_minutes=90,
                    address=r.get("address"),
                    notes="Dinner",
                ))
            continue

        if is_departure:
            a = _next_attr()
            if a:
                out.append(_stop(
                    name=a.get("name") or "(attraction)",
                    day=day,
                    when=_ts(d, 10, 0),
                    duration_minutes=120,
                    address=a.get("address"),
                    notes="Morning visit",
                ))
            else:
                out.append(_free_time_stop(
                    day, _ts(d, 10, 0), "Open morning before departure"
                ))
            r = _next_rest()
            if r:
                out.append(_stop(
                    name=r.get("name") or "(restaurant)",
                    day=day,
                    when=_ts(d, 13, 0),
                    duration_minutes=75,
                    address=r.get("address"),
                    notes="Lunch",
                ))
            if hotel_name:
                out.append(_stop(
                    name=f"Check out from {hotel_name}",
                    day=day,
                    when=_ts(d, 15, 0),
                    duration_minutes=30,
                    notes="Departure day — hotel check-out",
                ))
            continue

        # Middle day — full plan
        a1 = _next_attr()
        if a1:
            out.append(_stop(
                name=a1.get("name") or "(attraction)",
                day=day,
                when=_ts(d, 10, 0),
                duration_minutes=150,
                address=a1.get("address"),
                notes="Morning attraction",
            ))
        else:
            out.append(_free_time_stop(
                day, _ts(d, 10, 0), "Open slot — wander, rest, or revisit a favourite"
            ))
        r1 = _next_rest()
        if r1:
            out.append(_stop(
                name=r1.get("name") or "(restaurant)",
                day=day,
                when=_ts(d, 13, 0),
                duration_minutes=75,
                address=r1.get("address"),
                notes="Lunch",
            ))
        a2 = _next_attr()
        if a2:
            out.append(_stop(
                name=a2.get("name") or "(attraction)",
                day=day,
                when=_ts(d, 15, 0),
                duration_minutes=150,
                address=a2.get("address"),
                notes="Afternoon attraction",
            ))
        else:
            out.append(_free_time_stop(
                day, _ts(d, 15, 0), "Open slot — wander, rest, or revisit a favourite"
            ))
        r2 = _next_rest()
        if r2:
            out.append(_stop(
                name=r2.get("name") or "(restaurant)",
                day=day,
                when=_ts(d, 19, 0),
                duration_minutes=90,
                address=r2.get("address"),
                notes="Dinner",
            ))

    return out
