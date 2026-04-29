"""Flight search.

Primary:  `fli` (PyPI: `flights`) — scrapes Google Flights, no API key.
          Returns typed Pydantic models with per-leg detail (airline,
          flight_number, departure/arrival airports + datetimes).
Fallback: SerpApi google_flights endpoint (requires `SERPAPI_API_KEY`,
          quota-tracked at 250/month free tier).

`fli`'s `SearchFlights().search()` is sync, so we run it in a worker
thread via `asyncio.to_thread` to keep the calling agent's event loop
unblocked. Result rows are normalized into the dict shape expected by
`output.schemas.Flight`.

Caveat shared with any Google Flights scraper: the upstream price is in
the locale Google guesses from our IP. Fli at least exposes the
`currency` field so callers can detect this; FX normalization is a
downstream concern.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from langchain_core.tools import tool

from src.config import get_settings
from src.tools._common import (
    ToolResult,
    error_result,
    http_client,
    ok_result,
    safe_call,
    with_quota,
)

log = logging.getLogger(__name__)

SeatClass = Literal["economy", "premium-economy", "business", "first"]


def _resolve_airport(code: str) -> Any:
    """Map a 3-letter IATA string to fli's `Airport` enum member."""
    from fli.models import Airport

    code = code.upper().strip()
    try:
        return Airport[code]
    except KeyError as e:
        raise ValueError(f"Unknown IATA code: {code!r}") from e


def _resolve_seat(seat: SeatClass) -> Any:
    from fli.models import SeatType

    return {
        "economy": SeatType.ECONOMY,
        "premium-economy": SeatType.PREMIUM_ECONOMY,
        "business": SeatType.BUSINESS,
        "first": SeatType.FIRST,
    }[seat]


async def _search_flights_fli(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None,
    adults: int,
    seat: SeatClass,
) -> ToolResult:
    """Primary path — Fli wrapped in a thread.

    Two-stage call: try `MaxStops.NON_STOP` first; if no nonstops exist
    on this date, fall back to `MaxStops.ANY`. This avoids the failure
    mode where fli returned a 5h MDE→GYE→SMR layover at COP 1,405,980
    while $295 1h20 nonstops were available on Google Flights.
    """
    from fli.models import (
        FlightSearchFilters,
        FlightSegment,
        MaxStops,
        PassengerInfo,
        SortBy,
        TripType,
    )
    from fli.search import SearchFlights

    try:
        origin_ap = _resolve_airport(origin)
        dest_ap = _resolve_airport(destination)
        seat_type = _resolve_seat(seat)
    except ValueError as e:
        return error_result("fli", "missing_config", str(e))

    segments = [
        FlightSegment(
            departure_airport=[[origin_ap, 0]],
            arrival_airport=[[dest_ap, 0]],
            travel_date=depart_date,
        )
    ]
    trip = TripType.ONE_WAY
    if return_date:
        segments.append(
            FlightSegment(
                departure_airport=[[dest_ap, 0]],
                arrival_airport=[[origin_ap, 0]],
                travel_date=return_date,
            )
        )
        trip = TripType.ROUND_TRIP

    def _build_filters(stops: Any) -> Any:
        return FlightSearchFilters(
            trip_type=trip,
            passenger_info=PassengerInfo(adults=adults),
            flight_segments=segments,
            stops=stops,
            seat_type=seat_type,
            sort_by=SortBy.CHEAPEST,
        )

    def _run(filters: Any) -> Any:
        return SearchFlights().search(filters)

    async def _try(stops: Any) -> ToolResult:
        async def _call() -> ToolResult:
            results = await asyncio.to_thread(_run, _build_filters(stops))
            if not results:
                return error_result("fli", "no_results", "No flights returned")
            flights = [
                _normalize_fli_flight(r, requested_destination=destination)
                for r in results
            ]
            # Drop flights whose actual outbound destination differs from the
            # requested IATA. fli sometimes returns flights to entirely
            # different cities for poorly-indexed routes (e.g. MDE→SMR
            # surfaced flights to GYE/Guayaquil). Better to fall through to
            # SerpApi than mislead the user.
            requested = destination.upper()
            relevant = [
                f for f in flights
                if (f.get("destination") or "").upper() == requested
            ]
            dropped = len(flights) - len(relevant)
            if dropped:
                log.info(
                    "fli: dropped %d/%d results whose destination wasn't %s",
                    dropped, len(flights), requested,
                )
            if not relevant:
                return error_result(
                    "fli",
                    "no_results",
                    f"No flights to {requested} (fli returned only off-route results)",
                )
            return ok_result("fli", relevant)
        return await safe_call("fli", _call)

    nonstop = await _try(MaxStops.NON_STOP)
    if nonstop["ok"]:
        return nonstop
    if nonstop.get("error_type") != "no_results":
        return nonstop  # network/auth/etc — bubble up immediately
    log.info("fli: no nonstops for %s->%s on %s — retrying stops=ANY",
             origin, destination, depart_date)
    return await _try(MaxStops.ANY)


def _ap_code(ap: Any) -> str | None:
    if ap is None:
        return None
    return ap.name if hasattr(ap, "name") else str(ap)


def _airline_name(a: Any) -> str | None:
    if a is None:
        return None
    return a.value if hasattr(a, "value") else str(a)


def _leg_dict(leg: Any, leg_type: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "airline": _airline_name(leg.airline),
        "flight_number": getattr(leg, "flight_number", None),
        "from": _ap_code(leg.departure_airport),
        "to": _ap_code(leg.arrival_airport),
        "depart_at": (
            leg.departure_datetime.isoformat()
            if leg.departure_datetime
            else None
        ),
        "arrive_at": (
            leg.arrival_datetime.isoformat()
            if leg.arrival_datetime
            else None
        ),
        "duration_minutes": getattr(leg, "duration", None),
    }
    if leg_type:
        d["leg_type"] = leg_type
    return d


def _normalize_one_way(r: Any, requested_destination: str | None = None) -> dict[str, Any]:  # noqa: ARG001
    """Normalize a one-way FlightResult.

    `requested_destination` is accepted for signature parity with the
    round-trip path and to support future filtering, but we no longer
    override `destination` from it: the upstream filter in
    `_search_flights_fli` drops flights whose actual final stop doesn't
    match the request, which is the truthful behaviour.
    """
    legs = list(r.legs or [])
    first = legs[0] if legs else None
    last = legs[-1] if legs else None
    return {
        "airline": _airline_name(first.airline) if first else None,
        "flight_number": getattr(first, "flight_number", None) if first else None,
        "origin": _ap_code(first.departure_airport) if first else None,
        "destination": _ap_code(last.arrival_airport) if last else None,
        "depart_at": (
            first.departure_datetime.isoformat()
            if first and first.departure_datetime
            else None
        ),
        "arrive_at": (
            last.arrival_datetime.isoformat()
            if last and last.arrival_datetime
            else None
        ),
        "duration_minutes": getattr(r, "duration", None),
        "stops": getattr(r, "stops", 0),
        "price": getattr(r, "price", None),
        "currency": getattr(r, "currency", None),
        "trip_type": "one-way",
        "legs": [_leg_dict(leg) for leg in legs],
    }


def _normalize_round_trip(
    outbound: Any, ret: Any, requested_destination: str | None = None  # noqa: ARG001
) -> dict[str, Any]:
    """Combine an (outbound, return) FlightResult pair into one Flight dict.

    Fli returns round-trip results as `list[tuple[FlightResult, FlightResult]]`
    where both halves carry the same total trip price. We surface the
    outbound's depart/arrive (the user's primary leg) and stash the return
    in `legs` tagged `leg_type="return"` so renderers can group them.
    """
    out_legs = list(outbound.legs or [])
    ret_legs = list(ret.legs or [])
    out_first = out_legs[0] if out_legs else None
    out_last = out_legs[-1] if out_legs else None
    return {
        "airline": _airline_name(out_first.airline) if out_first else None,
        "flight_number": getattr(out_first, "flight_number", None) if out_first else None,
        "origin": _ap_code(out_first.departure_airport) if out_first else None,
        "destination": _ap_code(out_last.arrival_airport) if out_last else None,
        "depart_at": (
            out_first.departure_datetime.isoformat()
            if out_first and out_first.departure_datetime
            else None
        ),
        "arrive_at": (
            out_last.arrival_datetime.isoformat()
            if out_last and out_last.arrival_datetime
            else None
        ),
        "duration_minutes": (
            (getattr(outbound, "duration", 0) or 0)
            + (getattr(ret, "duration", 0) or 0)
        ),
        "stops": max(
            getattr(outbound, "stops", 0) or 0,
            getattr(ret, "stops", 0) or 0,
        ),
        "price": getattr(outbound, "price", None),  # Fli reports same on both
        "currency": getattr(outbound, "currency", None),
        "trip_type": "round-trip",
        "legs": [
            *[_leg_dict(leg, leg_type="outbound") for leg in out_legs],
            *[_leg_dict(leg, leg_type="return") for leg in ret_legs],
        ],
    }


def _normalize_fli_flight(
    r: Any, requested_destination: str | None = None
) -> dict[str, Any]:
    """Normalize one Fli result row.

    Fli returns:
    - one-way searches → `list[FlightResult]` (one per option)
    - round-trip searches → `list[tuple[FlightResult, FlightResult]]`
      (outbound + return paired). We collapse the pair into one Flight
      dict so downstream agents reason about complete trip options.
    """
    if isinstance(r, tuple) and len(r) == 2:
        return _normalize_round_trip(r[0], r[1], requested_destination)
    return _normalize_one_way(r, requested_destination)


async def _search_flights_serpapi(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None,
    adults: int,
) -> ToolResult:
    """Fallback path — SerpApi google_flights. Quota: 250/month free."""
    settings = get_settings()
    if not settings.serpapi_api_key:
        return error_result(
            "serpapi",
            "missing_config",
            "SERPAPI_API_KEY not set; cannot use SerpApi fallback",
        )

    async def _call() -> ToolResult:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": depart_date,
            "currency": "USD",
            "adults": adults,
            "api_key": settings.serpapi_api_key,
        }
        if return_date:
            params["return_date"] = return_date
            params["type"] = "1"
        else:
            params["type"] = "2"

        async with http_client() as client:
            resp = await client.get("https://serpapi.com/search.json", params=params)
            resp.raise_for_status()
            payload = resp.json()

        best = payload.get("best_flights") or payload.get("other_flights") or []
        flights = [_normalize_serpapi_flight(f, origin, destination) for f in best]
        if not flights:
            return error_result("serpapi", "no_results", "No flights returned")
        return ok_result("serpapi", flights)

    return await with_quota("serpapi", _call)


def _normalize_serpapi_flight(item: dict, origin: str, destination: str) -> dict[str, Any]:
    legs = item.get("flights") or []
    first = legs[0] if legs else {}
    last = legs[-1] if legs else {}
    return {
        "airline": first.get("airline", "Unknown"),
        "flight_number": first.get("flight_number"),
        "origin": origin,
        "destination": destination,
        "depart_at": (first.get("departure_airport") or {}).get("time"),
        "arrive_at": (last.get("arrival_airport") or {}).get("time"),
        "duration_minutes": item.get("total_duration"),
        "stops": max(0, len(legs) - 1),
        "price": item.get("price"),
        "currency": "USD",
    }


@tool
async def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    adults: int = 1,
    seat: SeatClass = "economy",
) -> ToolResult:
    """Search for flights between two airports.

    Args:
        origin: IATA airport code, e.g. "LAX".
        destination: IATA airport code, e.g. "HND".
        depart_date: ISO date "YYYY-MM-DD".
        return_date: optional ISO date for round-trips.
        adults: passenger count.
        seat: cabin class.

    Tries Fli first (primary, no key). If Fli yields no results / errors
    AND `SERPAPI_API_KEY` is configured, falls back to SerpApi.
    """
    log.info("search_flights %s -> %s on %s", origin, destination, depart_date)
    primary = await _search_flights_fli(
        origin, destination, depart_date, return_date, adults, seat
    )
    if primary["ok"]:
        return primary

    log.info("fli failed (%s); trying SerpApi", primary.get("error_type"))
    fallback = await _search_flights_serpapi(
        origin, destination, depart_date, return_date, adults
    )
    if fallback["ok"]:
        return fallback
    # Both failed — surface the primary error since it's the default path.
    return primary
