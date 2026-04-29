"""Router agent — parses the user's raw query into structured trip parameters.

Strategy
--------
Single LLM call via `get_llm().with_structured_output(RouterOutput)`. The
prompt embeds today's date so the model can resolve relative phrases
("next week", "5-day trip") into concrete ISO dates without
hallucinating. On any failure (LLM unreachable, schema-violating reply,
JSON parse error) the agent falls back to a sensible default block so
the graph still produces a plan rather than crashing.

Explicit state values take precedence: if a caller (test, CLI, future
UI) already populated `state["origin"]` or `state["destination"]`, the
LLM only fills the remaining gaps.

Reads:  raw_query, plus any partially-populated intent fields
Writes: origin, destination, dates, travelers, budget_tier, preferences,
        errors (when LLM fails or destination still missing)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

from dateutil import parser as dateparser

from src.agents._router_schema import RouterOutput
from src.config import get_llm, get_settings
from src.state.trip_state import TripState

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a trip-planning assistant. Extract structured trip parameters from
the user's request and return ONLY a JSON object matching the provided
schema. Do not include prose, markdown fences, or commentary.

TODAY'S DATE: {today}

ORIGIN / DESTINATION
The user's request may use any of these phrasings:
- "from medellin to bogota"           → origin=medellin, destination=bogota
- "medellin to bogota"                → origin=medellin, destination=bogota
- "trip to bogota from medellin"      → origin=medellin, destination=bogota
- "fly to Tokyo"                      → origin=null, destination=Tokyo
- "NYC to LAX"                        → origin=NYC, destination=LAX (IATA-like)

When the query contains TWO `from` clauses (one for cities, one for dates),
the first `from <X> to <Y>` is the city pair and the second `from <date> to
<date>` is the dates. Example:
  "from medellin to bogota from may 27th to may 30th 2026"
    → origin=medellin, destination=bogota,
      dates={{"start": "2026-05-27", "end": "2026-05-30"}}

If the user did not specify origin/destination, leave them null — do NOT
invent one.

DATES (resolve relative to TODAY's date above)
- "next week"                         → start = today + 7 days
- "in two months"                     → start = today + 60 days
- "5-day trip" with no anchor         → start = today + 14 days, end = start + 5 days
- "May 1-8 2026"                      → start=2026-05-01, end=2026-05-08
- "may 27th to may 30th 2026"         → start=2026-05-27, end=2026-05-30
- "from May 27 to May 30, 2026"       → start=2026-05-27, end=2026-05-30
- "2026-12-15 to 2026-12-22"          → start=2026-12-15, end=2026-12-22
- "Dec 15 - Dec 22 2026"              → start=2026-12-15, end=2026-12-22

Lowercase month names and ordinal day numbers (1st, 2nd, 27th) are
common — handle them all. If the user gave no temporal hint at all, set
dates to null.

PREFERENCES
Free-form list of interests, dietary restrictions, cuisines, etc. extracted
from the request (e.g. ["vegetarian", "museums", "ramen"]). If no
preferences are stated, return an empty list. NEVER pad with empty strings
or whitespace — every item must be a non-empty descriptive phrase.

USER LODGING
If the user told you where they'll be staying — a friend's or relative's
address, an Airbnb, a specific hotel they've already booked, etc. — extract
just the address or place name into `user_lodging`. Examples:
- "staying at my grandmas: cra 66 #48-106 conjunto san lorenzo"
    → user_lodging="cra 66 #48-106 conjunto san lorenzo"
- "I booked the Park Hyatt"           → user_lodging="Park Hyatt"
- "my Airbnb is on Calle 53"          → user_lodging="Calle 53"
- "we're staying with friends in Chapinero"
    → user_lodging="Chapinero"
If the user did not mention a specific lodging, set user_lodging to null —
do NOT make one up.

User's request:
{raw_query}
"""


# ── Regex backfill (safety net for fields the LLM left null) ─────────

# Captures "from <X> to <Y>" where the lookahead stops at a date marker
# (second `from`, `on`, `for`, etc.) or punctuation/end. Case-insensitive.
_CITIES_RE = re.compile(
    r"\bfrom\s+([a-z][a-z\s]*?)\s+to\s+([a-z][a-z\s]*?)"
    r"(?=\s+(?:from|on|for|next|in|during|starting|between|departing)\b|\s*[,.]|$)",
    re.IGNORECASE,
)

# Date phrases: ISO, or "<month> <day>[<ord>][ <year>]". Year is optional.
_MONTH = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)"
)
_DATE_RE = re.compile(
    rf"(\d{{4}}-\d{{2}}-\d{{2}})"
    rf"|({_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)"
    rf"|(\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?)",
    re.IGNORECASE,
)

# "3-day", "5 day", "3day" — duration without an anchor date.
_DURATION_RE = re.compile(r"\b(\d+)[-\s]?day\b", re.IGNORECASE)


def _parse_date_phrases(raw_query: str, today: date) -> list[date]:
    """Extract and parse all date-like substrings from the query."""
    out: list[date] = []
    default = datetime(today.year, 1, 1)
    for m in _DATE_RE.finditer(raw_query):
        phrase = next((g for g in m.groups() if g), None)
        if not phrase:
            continue
        try:
            parsed = dateparser.parse(phrase, default=default, fuzzy=False)
        except (ValueError, OverflowError):
            continue
        if parsed:
            out.append(parsed.date())
    return out


def _regex_backfill(raw_query: str, parsed: RouterOutput) -> RouterOutput:
    """Fill ONLY null fields the LLM missed. Regex is the safety net.

    Never overwrites a field the LLM populated — if the LLM said
    `origin="Paris"` we trust it even if the regex would extract
    something different from a malformed query.
    """
    today = date.today()

    # Cities
    if parsed.origin is None or parsed.destination is None:
        m = _CITIES_RE.search(raw_query)
        if m:
            origin_match = m.group(1).strip()
            dest_match = m.group(2).strip()
            if parsed.origin is None and origin_match:
                parsed.origin = origin_match
            if parsed.destination is None and dest_match:
                parsed.destination = dest_match

    # Dates
    if parsed.dates is None:
        dates = _parse_date_phrases(raw_query, today)
        if len(dates) >= 2 and dates[1] >= dates[0]:
            parsed.dates = {
                "start": dates[0].isoformat(),
                "end": dates[1].isoformat(),
            }
        elif len(dates) == 1:
            dur_m = _DURATION_RE.search(raw_query)
            if dur_m:
                n = max(1, int(dur_m.group(1)))
                end = dates[0] + timedelta(days=max(0, n - 1))
                parsed.dates = {
                    "start": dates[0].isoformat(),
                    "end": end.isoformat(),
                }

    return parsed


def _defaults_from(state: TripState) -> dict:
    """Sensible defaults applied when the LLM call fails outright.

    Preserves anything the caller already populated so a test that
    pre-fills state still works in offline mode.
    """
    settings = get_settings()
    return {
        "origin": state.get("origin") or settings.default_origin,
        "destination": state.get("destination"),
        "dates": state.get("dates"),
        "travelers": state.get("travelers") or 1,
        "budget_tier": state.get("budget_tier") or "mid",
        "preferences": state.get("preferences") or [],
        "user_lodging": state.get("user_lodging"),
    }


def _clean_preferences(prefs: list[str] | None) -> list[str]:
    """Drop empty / whitespace-only entries; trim survivors."""
    return [p.strip() for p in (prefs or []) if p and p.strip()]


def _clean_user_lodging(raw: str | None) -> str | None:
    """Trim whitespace; return None for empty / placeholder strings."""
    if not raw:
        return None
    s = raw.strip()
    if not s or s.lower() in {"none", "null", "n/a", "(none)"}:
        return None
    return s


def _merge_with_state(parsed: RouterOutput, state: TripState) -> dict:
    """Caller-provided state wins; LLM fills gaps."""
    settings = get_settings()
    return {
        "origin": state.get("origin") or parsed.origin or settings.default_origin,
        "destination": state.get("destination") or parsed.destination,
        "dates": state.get("dates") or parsed.dates,
        "travelers": state.get("travelers") or parsed.travelers,
        "budget_tier": state.get("budget_tier") or parsed.budget_tier,
        "preferences": state.get("preferences") or _clean_preferences(parsed.preferences),
        "user_lodging": state.get("user_lodging") or _clean_user_lodging(parsed.user_lodging),
    }


async def router_agent(state: TripState) -> dict:
    """Parse raw_query (LLM-driven) and write structured intent fields."""
    raw_query = (state.get("raw_query") or "").strip()
    log.info("router: parsing query=%r", raw_query[:120])

    if not raw_query:
        # Nothing to parse — keep whatever the caller passed in.
        log.info("router: empty raw_query; using caller-provided state + defaults")
        out = _defaults_from(state)
        if not out["destination"]:
            return {
                **out,
                "errors": [{
                    "agent": "router",
                    "stage": "input",
                    "message": "no raw_query and no destination in initial state",
                }],
            }
        return out

    prompt = _PROMPT_TEMPLATE.format(today=date.today().isoformat(), raw_query=raw_query)

    parsed: RouterOutput | None = None
    llm_error: dict | None = None
    try:
        llm = get_llm()
        # `with_structured_output` is supported by both ChatGoogleGenerativeAI
        # and ChatOllama. It validates the model's reply against RouterOutput
        # and raises if the schema is violated.
        structured_llm = llm.with_structured_output(RouterOutput)
        parsed = await structured_llm.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001
        # Last-resort guard: surface a graceful error rather than crashing.
        # Common failure modes: LLM 503/quota, schema mismatch, network blip.
        log.warning("router: LLM extraction failed (%s: %s)", type(e).__name__, e)
        llm_error = {
            "agent": "router",
            "stage": "llm",
            "message": f"{type(e).__name__}: {e}"[:300],
        }

    if parsed is None:
        out = _defaults_from(state)
        out["errors"] = [llm_error] if llm_error else []
        return out

    parsed = _regex_backfill(raw_query, parsed)

    log.info(
        "router: parsed origin=%r destination=%r dates=%r tier=%r prefs=%s lodging=%r",
        parsed.origin, parsed.destination, parsed.dates,
        parsed.budget_tier, parsed.preferences, parsed.user_lodging,
    )

    out = _merge_with_state(parsed, state)
    if not out["destination"]:
        out["errors"] = [{
            "agent": "router",
            "stage": "extract",
            "message": "could not determine destination from query",
        }]
    return out
