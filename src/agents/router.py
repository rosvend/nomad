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

ORIGIN / DESTINATION / LEGS
The user's request may use any of these phrasings:
- "from medellin to bogota"           → origin=medellin, legs=[{{bogota}}]
- "medellin to bogota"                → origin=medellin, legs=[{{bogota}}]
- "trip to bogota from medellin"      → origin=medellin, legs=[{{bogota}}]
- "fly to Tokyo"                      → origin=null, legs=[{{Tokyo}}]
- "NYC to LAX"                        → origin=NYC, legs=[{{LAX}}]

When the query contains TWO `from` clauses (one for cities, one for dates),
the first `from <X> to <Y>` is the city pair and the second `from <date> to
<date>` is the dates. Example:
  "from medellin to bogota from may 27th to may 30th 2026"
    → origin=medellin, legs=[{{destination=bogota, start=2026-05-27,
      end=2026-05-30}}], dates={{"start": "2026-05-27", "end": "2026-05-30"}}

MULTI-CITY TRIPS
If the user names more than one destination, return ONE entry in `legs`
per destination, in the order they said them. Set `destination` to the
FIRST leg's city for back-compat. Examples:
- "trip starting in Medellin. 3 days in Bogota, then 4 days in Cartagena
   before flying back, starting May 15"
    → origin=Medellin,
      legs=[
        {{destination=Bogota,    start=2026-05-15, end=2026-05-17, days=3}},
        {{destination=Cartagena, start=2026-05-18, end=2026-05-21, days=4}},
      ],
      destination=Bogota,
      dates={{"start": "2026-05-15", "end": "2026-05-21"}}
- "Tokyo for a week then Kyoto for 3 days"
    → legs=[
        {{destination=Tokyo, days=7}},
        {{destination=Kyoto, days=3}},
      ],
      destination=Tokyo
When per-leg dates are derivable from a global start + day counts,
distribute them sequentially with no gap (leg N+1 starts on leg N's end+1).

If the user did not specify origin/destination, leave origin null and
return an EMPTY legs list — do NOT invent destinations.

DISAMBIGUATING CITY NAMES
Many city names exist in multiple countries (Cartagena → Colombia AND
Spain; Cordoba → Argentina AND Spain; Santiago → Chile AND Spain;
Granada → Spain AND Nicaragua). When the surrounding context makes the
country unambiguous, append it explicitly to the destination string so
downstream geocoding hits the right city. Example:
  "trip starting in Medellin. 3 days in Bogota, then 4 days in Cartagena"
    → legs=[{{destination="Bogota, Colombia"}},
            {{destination="Cartagena, Colombia"}}]
Do this whenever the origin or the OTHER destinations clearly identify
a country. If a city name is unique or context is missing, use the
plain name.

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

BUDGET AMOUNT
If the user mentioned a numeric budget, extract it into `budget_amount`
(the number alone, in major units) and `budget_currency` (ISO 4217 code).
Examples:
- "1 million COP"          → budget_amount=1000000, budget_currency="COP"
- "1,000,000 pesos"        → budget_amount=1000000, budget_currency="COP"
- "2k USD"                 → budget_amount=2000, budget_currency="USD"
- "$2000"                  → budget_amount=2000, budget_currency="USD"
- "presupuesto de 3 millones de pesos"
                           → budget_amount=3000000, budget_currency="COP"
- "trip under €1500"       → budget_amount=1500, budget_currency="EUR"
Set `budget_scope` to "flights" if the budget specifically refers to
airfare ("flight budget", "airfare under X"), or "trip" otherwise. Leave
all three fields null when no number was mentioned.

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

# Budget amount + currency. Captures patterns like:
#   "1,000,000 COP", "1.5M COP", "2k USD", "$2000", "€500",
#   "3 millones de pesos", "presupuesto de 1 millón cop"
# Returns the lexical pieces; downstream code normalises into a number +
# ISO currency code.
#
# Number token: either thousand-separated (1,000 / 1.000.000 / 1,234,567)
# or a plain digit run, optionally with a decimal tail. The two alternatives
# are ordered to prefer the longest match.
_NUMBER_TOKEN = (
    r"(\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d+)?"   # 1,000  | 1.000.000  | 1,234.5
    r"|\d+(?:[\.,]\d+)?)"                       # 2000   | 2000.5
)
_MULTIPLIER = r"(?:\s*(k|m|mm|millones|millon(?:es)?|millón|million|mil)\b)?"
# Allow stop-word filler (de, of, en) between multiplier and the currency
# word, since users naturally say "3 millones DE pesos" / "1 million OF
# dollars". Kept short to avoid over-matching.
_CURRENCY_FILLER = r"(?:\s+(?:de|del|en|of))?"
_CURRENCY_WORD = (
    r"(cop|usd|eur|gbp|jpy|mxn|brl|ars|clp|pen|"
    r"pesos\s*colombianos|pesos|peso|euros?|euro|dollars?|dollar|"
    r"d[oó]lares|d[oó]lar|reales|reais|real|yen|libras|sterling)"
)
_BUDGET_RE = re.compile(
    rf"(?:budget|presupuesto|under|menos\s+de|less\s+than|hasta|up\s+to|"
    rf"around|cerca\s+de|about|approximately|por)?\s*"
    rf"(?:[\$€£¥]\s*)?{_NUMBER_TOKEN}{_MULTIPLIER}{_CURRENCY_FILLER}\s+{_CURRENCY_WORD}\b",
    re.IGNORECASE,
)
# Plain "$2000" / "€500" with no trailing currency word.
_BUDGET_SYMBOL_RE = re.compile(
    rf"([\$€£¥])\s*{_NUMBER_TOKEN}{_MULTIPLIER}",
)
_SYMBOL_TO_ISO: dict[str, str] = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
}
_WORD_TO_ISO: dict[str, str] = {
    "cop": "COP", "usd": "USD", "eur": "EUR", "gbp": "GBP", "jpy": "JPY",
    "mxn": "MXN", "brl": "BRL", "ars": "ARS", "clp": "CLP", "pen": "PEN",
    "pesos colombianos": "COP",
    "pesos": "COP",  # Colombian project default; users mention COP most often
    "peso": "COP",
    "euro": "EUR", "euros": "EUR",
    "dollar": "USD", "dollars": "USD",
    "dolar": "USD", "dolares": "USD", "dólar": "USD", "dólares": "USD",
    "real": "BRL", "reales": "BRL", "reais": "BRL",
    "yen": "JPY",
    "libras": "GBP", "sterling": "GBP",
}


def _normalize_budget_amount(num_str: str, multiplier: str | None) -> float | None:
    """Convert "1,000,000" / "1.5M" / "2k" → a float in major currency units."""
    s = num_str.strip().lower()
    # Decide the decimal separator. Latin-American numeric notation uses
    # "." as thousands separator and "," as decimal; English uses the
    # opposite. Heuristic: the LAST separator is the decimal one if the
    # group after it is 1–2 digits long; otherwise both separators are
    # treated as thousands.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Single comma — decimal if 1-2 trailing digits, else thousands.
        tail = s.rsplit(",", 1)[1]
        if len(tail) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        tail = s.rsplit(".", 1)[1]
        if len(tail) > 2:
            s = s.replace(".", "")
    try:
        amount = float(s)
    except ValueError:
        return None
    mult = (multiplier or "").lower().strip()
    if mult in {"k", "mil"}:
        amount *= 1_000
    elif mult in {"m", "mm", "million", "millon", "millones", "millón"}:
        amount *= 1_000_000
    return amount


def _extract_budget(raw_query: str) -> tuple[float | None, str | None]:
    """Best-effort: find one (amount, ISO currency) pair in the query.

    Returns (None, None) when nothing is extractable. Used as a backfill
    when the router LLM didn't populate budget_amount itself.
    """
    m = _BUDGET_RE.search(raw_query)
    if m:
        amount = _normalize_budget_amount(m.group(1), m.group(2))
        if amount is None:
            return None, None
        cur_word = (m.group(3) or "").strip().lower()
        cur_word = " ".join(cur_word.split())
        iso = _WORD_TO_ISO.get(cur_word)
        if iso:
            return amount, iso
    m2 = _BUDGET_SYMBOL_RE.search(raw_query)
    if m2:
        amount = _normalize_budget_amount(m2.group(2), m2.group(3))
        if amount is None:
            return None, None
        iso = _SYMBOL_TO_ISO.get(m2.group(1))
        if iso:
            return amount, iso
    return None, None


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

    # Budget amount/currency. Backfill only — never overwrite the LLM.
    if parsed.budget_amount is None or parsed.budget_currency is None:
        amount, currency = _extract_budget(raw_query)
        if parsed.budget_amount is None and amount is not None:
            parsed.budget_amount = amount
        if parsed.budget_currency is None and currency is not None:
            parsed.budget_currency = currency

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
        else:
            # No explicit date phrase. The prompt asks the LLM to infer
            # "today + 14 days" as the anchor when only a duration is
            # given ("7-day trip", "5 day getaway"); both Ollama and the
            # OpenAI models routinely ignore that rule, so we enforce it
            # here as a deterministic fallback. This is what unblocks the
            # flights agent for queries like "plan a 7 day trip to X".
            dur_m = _DURATION_RE.search(raw_query)
            if dur_m:
                n = max(1, int(dur_m.group(1)))
                start = today + timedelta(days=14)
                end = start + timedelta(days=max(0, n - 1))
                parsed.dates = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }

    return parsed


def _defaults_from(state: TripState) -> dict:
    """Sensible defaults applied when the LLM call fails outright.

    Preserves anything the caller already populated so a test that
    pre-fills state still works in offline mode.
    """
    settings = get_settings()
    dest = state.get("destination")
    legs = state.get("legs") or ([{"destination": dest}] if dest else [])
    return {
        "origin": state.get("origin") or settings.default_origin,
        "destination": dest,
        "dates": state.get("dates"),
        "travelers": state.get("travelers") or 1,
        "budget_tier": state.get("budget_tier") or "mid",
        "budget_amount": state.get("budget_amount"),
        "budget_currency": state.get("budget_currency"),
        "budget_scope": state.get("budget_scope"),
        "preferences": state.get("preferences") or [],
        "user_lodging": state.get("user_lodging"),
        "legs": legs,
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


def _normalize_legs(parsed: RouterOutput) -> list[dict]:
    """Convert RouterOutput.legs (Pydantic) into TripState legs (dicts) and
    fill missing per-leg start/end from the overall window + day counts.

    Rules:
    - If parsed.legs is empty but parsed.destination is set, synthesize
      one leg from destination + parsed.dates.
    - If a leg has explicit start/end, use them.
    - If a leg has only `days` and we know the previous leg's end (or the
      overall start for the first leg), derive its start = prev.end + 1
      (or overall start), end = start + days - 1.
    - Otherwise leave start/end null and let the synthesizer default.
    """
    legs_raw = list(parsed.legs or [])
    if not legs_raw and parsed.destination:
        legs_raw = [type("L", (), {  # minimal stand-in
            "destination": parsed.destination,
            "start": (parsed.dates or {}).get("start"),
            "end": (parsed.dates or {}).get("end"),
            "days": None,
            "lodging": None,
        })()]

    overall_start = (parsed.dates or {}).get("start") if parsed.dates else None
    legs: list[dict] = []
    cursor: date | None = None
    if overall_start:
        try:
            cursor = date.fromisoformat(overall_start)
        except ValueError:
            cursor = None

    for i, leg in enumerate(legs_raw):
        leg_start = getattr(leg, "start", None)
        leg_end = getattr(leg, "end", None)
        leg_days = getattr(leg, "days", None)

        if leg_start and not leg_end and leg_days:
            try:
                s = date.fromisoformat(leg_start)
                leg_end = (s + timedelta(days=max(0, leg_days - 1))).isoformat()
            except ValueError:
                pass
        elif not leg_start and leg_days and cursor is not None:
            s = cursor if i == 0 else cursor + timedelta(days=1)
            e = s + timedelta(days=max(0, leg_days - 1))
            leg_start, leg_end = s.isoformat(), e.isoformat()

        # Advance cursor for next leg.
        if leg_end:
            try:
                cursor = date.fromisoformat(leg_end)
            except ValueError:
                pass

        legs.append({
            "destination": getattr(leg, "destination", None),
            "start": leg_start,
            "end": leg_end,
            "days": leg_days,
            "lodging": getattr(leg, "lodging", None),
        })

    return [l for l in legs if l.get("destination")]


def _merge_with_state(parsed: RouterOutput, state: TripState) -> dict:
    """Caller-provided state wins; LLM fills gaps."""
    settings = get_settings()
    legs = state.get("legs") or _normalize_legs(parsed)
    # Mirror first leg into scalar `destination` for back-compat with any
    # caller that pre-populated state.destination.
    first_leg_dest = legs[0]["destination"] if legs else None
    return {
        "origin": state.get("origin") or parsed.origin or settings.default_origin,
        "destination": state.get("destination") or parsed.destination or first_leg_dest,
        "dates": state.get("dates") or parsed.dates,
        "travelers": state.get("travelers") or parsed.travelers,
        "budget_tier": state.get("budget_tier") or parsed.budget_tier,
        "budget_amount": state.get("budget_amount") or parsed.budget_amount,
        "budget_currency": state.get("budget_currency") or parsed.budget_currency,
        "budget_scope": state.get("budget_scope") or parsed.budget_scope,
        "preferences": state.get("preferences") or _clean_preferences(parsed.preferences),
        "user_lodging": state.get("user_lodging") or _clean_user_lodging(parsed.user_lodging),
        "legs": legs,
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
        "router: parsed origin=%r destination=%r legs=%s dates=%r tier=%r prefs=%s lodging=%r",
        parsed.origin, parsed.destination,
        [l.destination for l in (parsed.legs or [])],
        parsed.dates, parsed.budget_tier, parsed.preferences, parsed.user_lodging,
    )

    out = _merge_with_state(parsed, state)
    if not out["destination"]:
        out["errors"] = [{
            "agent": "router",
            "stage": "extract",
            "message": "could not determine destination from query",
        }]
    return out
