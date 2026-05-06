"""Destination suggester — fills in a destination when the user was vague.

Activates when the router couldn't extract any destination from the
query but the user did express preferences (e.g. "send me somewhere warm
with a beach"). Asks the LLM for 2-3 ranked candidates with one-line
reasoning, prints them, and reads the user's pick from stdin. If stdin
is not a TTY (piped, scripted), defaults to candidate #1 with a logged
warning so headless runs still produce a plan.

Reads:  origin, budget_tier, preferences, dates
Writes: destination, legs, destination_was_inferred, errors
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from pydantic import BaseModel, Field

from src.config import get_llm
from src.state.trip_state import TripState

log = logging.getLogger(__name__)


class Candidate(BaseModel):
    destination: str = Field(description="City name (e.g. 'Cartagena').")
    reason: str = Field(
        description="One-sentence justification grounded in the user's "
        "preferences, origin, and dates (mention season fit, travel time, "
        "or budget alignment)."
    )


class SuggesterOutput(BaseModel):
    candidates: list[Candidate] = Field(
        description="2-3 ranked destinations, best first.",
        min_length=2,
        max_length=3,
    )


_PROMPT = """\
You are a travel-planning assistant. The user described what kind of
trip they want without naming a specific city. Suggest 2 or 3 concrete
city-level destinations that fit, ranked best first.

USER PROFILE
- Origin: {origin}
- Travel dates: {dates}
- Budget tier: {budget_tier}
- Preferences / interests: {preferences}
- User said (verbatim): {raw_query}
- Today's date: {today}

GUIDELINES
- Pick concrete cities or well-known towns (not regions or countries).
- If the user named a region or continent (e.g. "Asia", "the Caribbean"),
  pick cities within that region.
- Favor destinations within reasonable travel distance from the origin
  unless the user clearly wants a long-haul trip.
- For climate-related preferences ("warm", "cold", "beach", "snow"),
  consider what the climate is actually like during the user's dates.
- Match the budget tier — for "luxury" prefer cities with strong premium
  hotel/dining markets.
- Each `reason` is ONE sentence — no fluff, no marketing copy. Mention
  the specific feature that matches (beach, museums, cuisine, etc.).
- Return ONLY a JSON object matching the provided schema.
"""


def _print_candidates(candidates: list[Candidate]) -> None:
    print()
    print("The query didn't name a destination. Here are some matches:")
    print()
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c.destination}")
        print(f"     {c.reason}")
    print()


def _read_choice(num_candidates: int) -> int:
    """Return a 0-indexed candidate index; default to 0 in non-tty/error."""
    if not sys.stdin.isatty():
        log.warning(
            "destination_suggester: stdin not a tty — defaulting to candidate #1"
        )
        return 0
    try:
        raw = input(f"Pick one [1-{num_candidates}, default 1]: ").strip()
    except EOFError:
        return 0
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        log.warning("destination_suggester: invalid choice %r — using #1", raw)
        return 0
    if 1 <= n <= num_candidates:
        return n - 1
    log.warning("destination_suggester: out-of-range %d — using #1", n)
    return 0


async def destination_suggester_agent(state: TripState) -> dict:
    """Suggest a destination when the router couldn't extract one."""
    prefs = state.get("preferences") or []
    origin = state.get("origin") or "(unspecified)"
    dates = state.get("dates")
    dates_str = f"{dates.get('start')} → {dates.get('end')}" if dates else "(unspecified)"
    budget = state.get("budget_tier") or "mid"
    raw_query = (state.get("raw_query") or "").strip() or "(not provided)"

    log.info(
        "destination_suggester: prefs=%s origin=%r dates=%s tier=%s raw_query=%r",
        prefs, origin, dates_str, budget, raw_query,
    )

    prompt = _PROMPT.format(
        origin=origin,
        dates=dates_str,
        budget_tier=budget,
        preferences=", ".join(prefs) if prefs else "(none)",
        raw_query=raw_query,
        today=date.today().isoformat(),
    )

    try:
        llm = get_llm()
        structured = llm.with_structured_output(SuggesterOutput)
        result: SuggesterOutput = await structured.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001 — LLM wrapper, see CLAUDE.md
        log.warning(
            "destination_suggester: LLM failed (%s: %s)", type(e).__name__, e
        )
        return {
            "errors": [{
                "agent": "destination_suggester",
                "stage": "llm",
                "message": f"{type(e).__name__}: {e}"[:300],
            }],
        }

    if not result.candidates:
        return {
            "errors": [{
                "agent": "destination_suggester",
                "stage": "empty",
                "message": "LLM returned no destination candidates",
            }],
        }

    _print_candidates(result.candidates)
    idx = _read_choice(len(result.candidates))
    chosen = result.candidates[idx]
    log.info(
        "destination_suggester: picked %r (%d/%d)",
        chosen.destination, idx + 1, len(result.candidates),
    )

    leg = {
        "destination": chosen.destination,
        "start": (dates or {}).get("start"),
        "end": (dates or {}).get("end"),
        "days": None,
        "lodging": None,
    }
    return {
        "destination": chosen.destination,
        "legs": [leg],
        "destination_was_inferred": True,
    }
