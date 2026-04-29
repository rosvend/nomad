"""Pydantic schema for Router output.

Used both as the LangChain `with_structured_output` target (the LLM
returns JSON conforming to this) and as the contract for the dict the
Router agent writes back into TripState. Field types and constraints
match `TripState` exactly so the values flow through unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    """Structured trip parameters extracted from a user's natural-language query."""

    origin: str | None = Field(
        default=None,
        description=(
            "Where the traveler is flying from. Either a city name "
            "(e.g. 'Los Angeles') or a 3-letter IATA airport code "
            "(e.g. 'LAX'). Null if the user didn't specify."
        ),
    )
    destination: str | None = Field(
        default=None,
        description=(
            "Where the traveler is going. City name preferred "
            "(e.g. 'Tokyo'). Null if the user didn't specify."
        ),
    )
    dates: dict[str, str] | None = Field(
        default=None,
        description=(
            "Trip dates as {\"start\": \"YYYY-MM-DD\", \"end\": \"YYYY-MM-DD\"}. "
            "Resolve relative phrases (\"next week\", \"5-day trip\") against "
            "TODAY's date which is provided in the prompt. Null if the user "
            "gave no temporal hint."
        ),
    )
    travelers: int = Field(
        default=1, ge=1, le=12,
        description="Number of adult travelers. Default 1.",
    )
    budget_tier: Literal["budget", "mid", "luxury"] = Field(
        default="mid",
        description=(
            "Budget tier. Map words like 'cheap', 'backpack' to 'budget'; "
            "'comfortable', 'standard', 'mid-range' to 'mid'; "
            "'luxury', 'splurge', '5-star' to 'luxury'."
        ),
    )
    preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form list of preferences/interests/dietary restrictions/"
            "cuisines, e.g. ['vegetarian', 'museums', 'ramen', 'no red-eye']. "
            "Empty list if no preferences expressed."
        ),
    )
    user_lodging: str | None = Field(
        default=None,
        description=(
            "If the user mentioned where they'll be staying (a friend's or "
            "family member's address, an Airbnb, a specific hotel name they "
            "have already booked, etc.), extract the address or place name. "
            "Examples: 'staying at my grandmas: cra 66 #48-106' → "
            "'cra 66 #48-106'. 'I booked the Park Hyatt' → 'Park Hyatt'. "
            "'my Airbnb is on Calle 53' → 'Calle 53'. "
            "Null if the user did not mention a specific lodging."
        ),
    )
