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
from datetime import date

from src.agents._router_schema import RouterOutput
from src.config import get_llm, get_settings
from src.state.trip_state import TripState

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a trip-planning assistant. Extract structured trip parameters from
the user's request and return ONLY a JSON object matching the provided
schema. Do not include prose, markdown fences, or commentary.

TODAY'S DATE: {today}

Use TODAY's date to resolve relative time references:
- "next week" → start = today + 7 days
- "in two months" → start = today + 60 days
- "5-day trip" with no anchor → start = today + 14 days, end = start + 5 days
- "May 1-8 2026" → start = 2026-05-01, end = 2026-05-08

If the user did not give any temporal hint at all, set dates to null.
If the user did not specify origin/destination, leave them null — do
NOT invent one.

User's request:
{raw_query}
"""


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
    }


def _merge_with_state(parsed: RouterOutput, state: TripState) -> dict:
    """Caller-provided state wins; LLM fills gaps."""
    settings = get_settings()
    return {
        "origin": state.get("origin") or parsed.origin or settings.default_origin,
        "destination": state.get("destination") or parsed.destination,
        "dates": state.get("dates") or parsed.dates,
        "travelers": state.get("travelers") or parsed.travelers,
        "budget_tier": state.get("budget_tier") or parsed.budget_tier,
        "preferences": state.get("preferences") or list(parsed.preferences),
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

    log.info(
        "router: parsed origin=%r destination=%r dates=%r tier=%r prefs=%s",
        parsed.origin, parsed.destination, parsed.dates,
        parsed.budget_tier, parsed.preferences,
    )

    out = _merge_with_state(parsed, state)
    if not out["destination"]:
        out["errors"] = [{
            "agent": "router",
            "stage": "extract",
            "message": "could not determine destination from query",
        }]
    return out
