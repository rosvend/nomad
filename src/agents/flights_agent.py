"""Flight agent — searches and ranks flight options.

Pipeline
--------
1. Resolve `state["origin"]` (or `settings.default_origin` if missing)
   and `state["destination"]` to 3-letter IATA codes via an in-process
   city map. A free-form 3-letter input is treated as IATA directly.
2. Call `search_flights` (Fli primary, SerpApi fallback) for the date
   range with the requested traveler count.
3. Score each result on a composite of:
     - **price** (cheaper better, normalized within this batch)
     - **stops** (non-stop best; 1-stop OK; 2+ heavily penalized)
     - **duration** (shorter better, normalized within this batch)
   Weights are **tier-aware**: budget travelers prioritize price; luxury
   travelers prioritize stops and speed.
4. Return the top N flights sorted by composite score (default 5).

Reads:  origin, destination, dates, travelers, budget_tier
Writes: flights (list of dicts), errors (on failures)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents._priors import (
    estimate_rt_price_usd,
    feasibility_verdict,
    to_usd,
)
from src.config import get_llm, get_settings
from src.state.trip_state import TripState
from src.tools import search_flights

log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

MAX_RESULTS = 5
DEFAULT_SEAT = "economy"
DEFAULT_TIER = "mid"

# Composite-score weights per budget tier: (price, stops, duration). Sum to 1.0.
_WEIGHT_PROFILES: dict[str, tuple[float, float, float]] = {
    "budget": (0.50, 0.30, 0.20),  # price-driven
    "mid": (0.35, 0.35, 0.30),     # balanced
    "luxury": (0.20, 0.40, 0.40),  # convenience-driven
}

# City → primary IATA. Multi-airport cities pick the most-used hub.
# Free-form text outside this map can still resolve if the user passes
# a 3-letter IATA directly.
_CITY_TO_IATA: dict[str, str] = {
    # Asia
    "tokyo": "HND", "osaka": "KIX", "kyoto": "KIX",
    "seoul": "ICN", "beijing": "PEK", "shanghai": "PVG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK",
    "dubai": "DXB", "delhi": "DEL", "mumbai": "BOM",
    "taipei": "TPE", "manila": "MNL", "kuala lumpur": "KUL",
    "ho chi minh city": "SGN", "hanoi": "HAN", "jakarta": "CGK",
    # Europe
    "london": "LHR", "paris": "CDG", "rome": "FCO",
    "madrid": "MAD", "barcelona": "BCN", "berlin": "BER",
    "amsterdam": "AMS", "frankfurt": "FRA", "munich": "MUC",
    "zurich": "ZRH", "vienna": "VIE", "istanbul": "IST",
    "athens": "ATH", "lisbon": "LIS", "dublin": "DUB",
    "copenhagen": "CPH", "stockholm": "ARN", "oslo": "OSL",
    "moscow": "SVO", "warsaw": "WAW", "prague": "PRG",
    "budapest": "BUD",
    # Americas — North
    "new york": "JFK", "los angeles": "LAX", "san francisco": "SFO",
    "chicago": "ORD", "miami": "MIA", "boston": "BOS",
    "washington": "IAD", "seattle": "SEA", "atlanta": "ATL",
    "dallas": "DFW", "denver": "DEN", "las vegas": "LAS",
    "houston": "IAH", "philadelphia": "PHL", "phoenix": "PHX",
    "toronto": "YYZ", "vancouver": "YVR", "montreal": "YUL",
    "mexico city": "MEX",
    # Americas — South
    "sao paulo": "GRU", "rio de janeiro": "GIG",
    "buenos aires": "EZE", "lima": "LIM",
    "bogota": "BOG", "santiago": "SCL", "medellin": "MDE",
    # Colombia (project hometown — keep this thorough)
    "cartagena": "CTG", "cali": "CLO", "barranquilla": "BAQ",
    "santa marta": "SMR", "pereira": "PEI", "bucaramanga": "BGA",
    "san andres": "ADZ", "san andrés": "ADZ", "armenia": "AXM",
    "monteria": "MTR", "montería": "MTR", "valledupar": "VUP",
    "riohacha": "RCH", "cucuta": "CUC", "cúcuta": "CUC",
    "pasto": "PSO", "ibague": "IBE", "ibagué": "IBE",
    "leticia": "LET", "manizales": "MZL", "neiva": "NVA",
    # Central America & nearby gaps
    "quito": "UIO", "guayaquil": "GYE", "panama city": "PTY",
    "san jose": "SJO", "tegucigalpa": "TGU", "guatemala city": "GUA",
    "san salvador": "SAL", "managua": "MGA", "havana": "HAV",
    # Oceania
    "sydney": "SYD", "melbourne": "MEL", "auckland": "AKL",
    # Africa & Middle East
    "cairo": "CAI", "johannesburg": "JNB", "nairobi": "NBO",
    "casablanca": "CMN", "tel aviv": "TLV", "doha": "DOH",
}


# IATA → ISO 3166-1 alpha-2 country code, scoped to the airports we
# resolve via `_CITY_TO_IATA` (plus a few common LATAM gaps). Used by
# the budget feasibility check; falls back to None for codes we don't
# know, which silences the verdict (better than guessing wrong).
_IATA_TO_COUNTRY: dict[str, str] = {
    # Asia
    "HND": "JP", "NRT": "JP", "KIX": "JP", "ITM": "JP",
    "ICN": "KR", "GMP": "KR",
    "PEK": "CN", "PVG": "CN", "PKX": "CN", "SHA": "CN",
    "HKG": "HK", "TPE": "TW",
    "SIN": "SG", "BKK": "TH", "DMK": "TH",
    "DXB": "AE", "AUH": "AE", "DOH": "QA",
    "DEL": "IN", "BOM": "IN",
    "MNL": "PH", "KUL": "MY", "SGN": "VN", "HAN": "VN",
    "CGK": "ID", "TLV": "IL",
    # Europe
    "LHR": "GB", "LGW": "GB", "STN": "GB",
    "CDG": "FR", "ORY": "FR",
    "FCO": "IT", "MXP": "IT",
    "MAD": "ES", "BCN": "ES",
    "BER": "DE", "FRA": "DE", "MUC": "DE",
    "AMS": "NL", "ZRH": "CH", "VIE": "AT",
    "IST": "TR", "ATH": "GR", "LIS": "PT", "DUB": "IE",
    "CPH": "DK", "ARN": "SE", "OSL": "NO",
    "SVO": "RU", "WAW": "PL", "PRG": "CZ", "BUD": "HU",
    # North America
    "JFK": "US", "LGA": "US", "EWR": "US",
    "LAX": "US", "SFO": "US", "ORD": "US", "MIA": "US",
    "BOS": "US", "IAD": "US", "DCA": "US", "SEA": "US",
    "ATL": "US", "DFW": "US", "DEN": "US", "LAS": "US",
    "IAH": "US", "PHL": "US", "PHX": "US", "JAX": "US",
    "YYZ": "CA", "YVR": "CA", "YUL": "CA", "YYC": "CA",
    "MEX": "MX",
    # South America
    "GRU": "BR", "GIG": "BR", "BSB": "BR",
    "EZE": "AR", "AEP": "AR",
    "LIM": "PE", "SCL": "CL", "UIO": "EC", "GYE": "EC",
    # Colombia
    "BOG": "CO", "MDE": "CO", "CTG": "CO", "CLO": "CO",
    "BAQ": "CO", "SMR": "CO", "PEI": "CO", "BGA": "CO",
    "ADZ": "CO", "AXM": "CO", "MTR": "CO", "VUP": "CO",
    "RCH": "CO", "CUC": "CO", "PSO": "CO", "IBE": "CO",
    "LET": "CO", "MZL": "CO", "NVA": "CO",
    # Central America & Caribbean
    "PTY": "PA", "SJO": "CR", "TGU": "HN", "GUA": "GT",
    "SAL": "SV", "MGA": "NI", "HAV": "CU",
    # Oceania
    "SYD": "AU", "MEL": "AU", "BNE": "AU", "AKL": "NZ",
    # Africa
    "CAI": "EG", "JNB": "ZA", "NBO": "KE", "CMN": "MA",
}


def _country_for_iata(code: str | None) -> str | None:
    if not code:
        return None
    return _IATA_TO_COUNTRY.get(code.upper())


# ── IATA resolution ──────────────────────────────────────────────────

async def _resolve_iata(value: str | None) -> str | None:
    """Normalise a free-form input to a 3-letter IATA code.

    - Already an IATA code (3 alphabetic chars): uppercase and return it
      verbatim. We *don't* validate against fli's full Airport enum here;
      the search_flights tool catches unknown codes and returns
      `missing_config` cleanly.
    - Known city in `_CITY_TO_IATA`: return its primary airport.
    - Otherwise: ask the LLM for a best guess. Returns None if the LLM
      can't help (offline, schema-violating reply, etc.).
    """
    if not value:
        return None
    v = value.strip()
    if len(v) == 3 and v.isalpha():
        return v.upper()
    # Country-qualified inputs like "Cartagena, Colombia" come from the
    # router when it disambiguates ambiguous city names. The dict is keyed
    # on bare city names, so try the head segment too before LLM fallback.
    candidates = [v.lower()]
    if "," in v:
        candidates.append(v.split(",", 1)[0].strip().lower())
    for c in candidates:
        hit = _CITY_TO_IATA.get(c)
        if hit:
            return hit
    return await _llm_iata_fallback(v)


async def _llm_iata_fallback(city: str) -> str | None:
    """Last-resort: ask the LLM for the primary IATA. None on any failure.

    Cheap to add — one extra LLM call per query, only triggered when the
    dict misses. If the LLM hallucinates an unknown code, the downstream
    `_resolve_airport` in `tools/flights.py` returns `missing_config`
    cleanly, so a bad guess degrades to the same error path as before.
    """
    try:
        llm = get_llm()
    except Exception as e:  # noqa: BLE001
        log.info("flights: LLM unavailable for IATA fallback (%s)", e)
        return None
    prompt = (
        f"Return ONLY the 3-letter IATA airport code for the primary "
        f"international airport serving '{city}'. Reply with exactly 3 "
        f"uppercase letters and nothing else. If no major airport exists "
        f"or you don't know, reply NONE."
    )
    try:
        resp = await llm.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001
        log.info("flights: IATA fallback call failed (%s: %s)", type(e).__name__, e)
        return None
    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip().upper() if isinstance(text, str) else ""
    text = text.split()[0] if text else ""  # trim accidental prose
    if len(text) == 3 and text.isalpha() and text != "NON":
        log.info("flights: IATA fallback resolved %r -> %s", city, text)
        return text
    return None


# ── Scoring helpers (in-batch normalization) ─────────────────────────

def _stops_score(stops: int | None) -> float:
    """Fewer stops = higher score. Non-linear: 0 stops is decisively best."""
    if stops is None:
        return 0.5
    if stops <= 0:
        return 1.0
    if stops == 1:
        return 0.5
    return 0.1


def _normalize_inverse_in_batch(values: list[float | None]) -> list[float]:
    """Lower input → higher score (1.0 for the smallest in-batch value).

    Returns 0.5 (neutral) for any None / non-positive inputs and when the
    batch has no spread. Used for both price and duration where 'less is
    better'.
    """
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return [0.5] * len(values)
    lo = min(valid)
    hi = max(valid)
    if hi == lo:
        # Everything at the same value — give the valid ones max score
        return [1.0 if (v is not None and v > 0) else 0.5 for v in values]
    out: list[float] = []
    span = hi - lo
    for v in values:
        if v is None or v <= 0:
            out.append(0.5)
        else:
            out.append(round(1.0 - (v - lo) / span, 4))
    return out


# ── Output shaping ───────────────────────────────────────────────────

def _flight_dict(
    flight: dict[str, Any],
    breakdown: dict[str, float],
    composite: float,
) -> dict[str, Any]:
    """Convert a scored flight into the dict shape `Flight` validates."""
    return {
        "airline": flight.get("airline"),
        "flight_number": flight.get("flight_number"),
        "origin": flight.get("origin"),
        "destination": flight.get("destination"),
        "depart_at": flight.get("depart_at"),
        "arrive_at": flight.get("arrive_at"),
        "duration_minutes": flight.get("duration_minutes"),
        "price": flight.get("price"),
        "currency": flight.get("currency"),
        "stops": int(flight.get("stops") or 0),
        "legs": flight.get("legs") or [],
        "score": round(composite, 3),
        "score_breakdown": breakdown,
    }


# ── Agent entry point ────────────────────────────────────────────────

async def flights_agent(state: TripState) -> dict:
    """Find and rank candidate flights."""
    settings = get_settings()

    destination = state.get("destination")
    dates = state.get("dates") or {}
    travelers = state.get("travelers") or 1
    budget_tier = state.get("budget_tier") or DEFAULT_TIER
    origin_raw = state.get("origin") or settings.default_origin

    if not destination:
        log.warning("flights_agent: no destination in state — skipping")
        return {
            "errors": [{"agent": "flights", "stage": "input", "message": "no destination"}],
            "flights": [],
        }
    if not origin_raw:
        log.warning("flights_agent: no origin in state and no default_origin in config")
        return {
            "errors": [{
                "agent": "flights",
                "stage": "input",
                "message": "no origin — set state['origin'] or DEFAULT_ORIGIN in .env",
            }],
            "flights": [],
        }
    depart_date = dates.get("start")
    return_date = dates.get("end")
    if not depart_date:
        return {
            "errors": [{"agent": "flights", "stage": "input", "message": "no depart date in state['dates']"}],
            "flights": [],
        }

    origin_iata = await _resolve_iata(origin_raw)
    dest_iata = await _resolve_iata(destination)
    if not origin_iata:
        return {
            "errors": [{
                "agent": "flights",
                "stage": "resolve_iata",
                "message": (
                    f"could not resolve origin {origin_raw!r} to an IATA code; "
                    f"pass a 3-letter code or a known city name"
                ),
            }],
            "flights": [],
        }
    if not dest_iata:
        return {
            "errors": [{
                "agent": "flights",
                "stage": "resolve_iata",
                "message": (
                    f"could not resolve destination {destination!r} to an IATA code"
                ),
            }],
            "flights": [],
        }

    log.info(
        "flights: %s -> %s on %s (return=%s) for %d adult(s), tier=%s",
        origin_iata, dest_iata, depart_date, return_date, travelers, budget_tier,
    )

    # Budget feasibility — compute *before* the API call so the verdict
    # surfaces even if search_flights returns no results. Per product
    # decision, we always proceed with the search and only warn the user;
    # we never block the API on a feasibility failure.
    budget_amount = state.get("budget_amount")
    budget_currency = state.get("budget_currency")
    budget_scope = state.get("budget_scope") or "trip"
    budget_assessment: dict[str, Any] | None = None
    if budget_amount and budget_currency:
        budget_usd = to_usd(float(budget_amount), budget_currency)
        prior = estimate_rt_price_usd(
            _country_for_iata(origin_iata),
            _country_for_iata(dest_iata),
        )
        verdict = feasibility_verdict(budget_usd, prior)
        budget_assessment = {
            "verdict": verdict,
            "budget_amount": float(budget_amount),
            "budget_currency": budget_currency.upper(),
            "budget_usd": round(budget_usd, 2) if budget_usd is not None else None,
            "prior_usd_low": prior[0] if prior else None,
            "prior_usd_high": prior[1] if prior else None,
            "scope": budget_scope,
            "origin_country": _country_for_iata(origin_iata),
            "dest_country": _country_for_iata(dest_iata),
        }
        log.info(
            "flights: budget verdict=%s (budget=%.0f %s ≈ $%s USD vs prior $%s-$%s)",
            verdict,
            float(budget_amount), budget_currency.upper(),
            f"{budget_usd:.0f}" if budget_usd is not None else "?",
            prior[0] if prior else "?",
            prior[1] if prior else "?",
        )

    # Step 2 — call the tool. Fli is async-wrapped via to_thread; SerpApi is
    # the fallback and only kicks in if Fli fails AND SERPAPI_API_KEY is set.
    res = await search_flights.ainvoke({
        "origin": origin_iata,
        "destination": dest_iata,
        "depart_date": depart_date,
        "return_date": return_date,
        "adults": travelers,
        "seat": DEFAULT_SEAT,
    })

    if not res["ok"]:
        out_err: dict[str, Any] = {
            "errors": [{"agent": "flights", "stage": "search_flights", "details": res}],
            "flights": [],
        }
        if budget_assessment is not None:
            out_err["budget_assessment"] = budget_assessment
        return out_err

    raw: list[dict[str, Any]] = list(res["data"])
    if not raw:
        out_empty: dict[str, Any] = {
            "errors": [{"agent": "flights", "stage": "search_flights", "message": "no flights returned"}],
            "flights": [],
        }
        if budget_assessment is not None:
            out_empty["budget_assessment"] = budget_assessment
        return out_empty

    # Fli's round-trip search returns many (outbound, return) combinations
    # that share the same outbound flight. Once normalized, those collapse
    # to identical dicts — present them as one option.
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for c in raw:
        key = (
            c.get("airline"),
            c.get("flight_number"),
            c.get("depart_at"),
            c.get("arrive_at"),
            c.get("duration_minutes"),
            c.get("stops"),
            c.get("price"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)

    log.info(
        "flights: %d candidates (%d unique) from %s",
        len(raw), len(candidates), res.get("provider"),
    )

    # Step 3 — score. In-batch normalisation for price and duration so the
    # ranking adapts to whatever Google Flights gave us today.
    weights = _WEIGHT_PROFILES.get(budget_tier, _WEIGHT_PROFILES[DEFAULT_TIER])
    w_price, w_stops, w_duration = weights

    prices = [f.get("price") for f in candidates]
    durations = [f.get("duration_minutes") for f in candidates]
    price_scores = _normalize_inverse_in_batch(prices)
    duration_scores = _normalize_inverse_in_batch(durations)

    scored: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for i, f in enumerate(candidates):
        ps = price_scores[i]
        ss = _stops_score(f.get("stops"))
        ds = duration_scores[i]
        composite = w_price * ps + w_stops * ss + w_duration * ds
        breakdown = {
            "price": round(ps, 3),
            "stops": round(ss, 3),
            "duration": round(ds, 3),
        }
        scored.append((composite, breakdown, f))

    scored.sort(key=lambda triple: triple[0], reverse=True)
    final = [_flight_dict(f, br, sc) for sc, br, f in scored[:MAX_RESULTS]]

    log.info(
        "flights_agent: returning %d flights (top score=%.3f, tier=%s, weights=p%.2f/s%.2f/d%.2f)",
        len(final),
        final[0]["score"] if final else 0,
        budget_tier, w_price, w_stops, w_duration,
    )
    out: dict[str, Any] = {"flights": final}
    if budget_assessment is not None:
        # Record the cheapest realised price alongside the prior so the
        # renderer can show "your budget is X, prior was $A-$B, cheapest
        # fare we actually found was Y". We don't auto-downgrade the
        # verdict here because the flight's native currency may differ
        # from the user's budget currency, and FX-converting per row
        # belongs in the renderer where formatting decisions live.
        cheapest = min(
            (f.get("price") for f in final
             if isinstance(f.get("price"), (int, float)) and f["price"] > 0),
            default=None,
        )
        if cheapest is not None and final:
            budget_assessment["cheapest_found"] = cheapest
            budget_assessment["cheapest_found_currency"] = final[0].get("currency")
        out["budget_assessment"] = budget_assessment
    return out
