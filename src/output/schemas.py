"""Pydantic models that describe the typed shape of every output artifact.

Specialist agents store dicts in `TripState` (TypedDicts allow loose shapes
for flexibility), but the Synthesizer validates them through these models
before assembling `TravelPlan`. The CLI/formatter consume `TravelPlan`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Flight(BaseModel):
    airline: str
    flight_number: str | None = None
    origin: str  # IATA airport code
    destination: str
    depart_at: datetime
    arrive_at: datetime
    price: float | None = None
    currency: str = "USD"
    booking_url: str | None = None
    stops: int = 0


class Hotel(BaseModel):
    name: str
    address: str | None = None
    rating: float | None = None  # 0.0 - 5.0 (Google's user rating)
    review_count: int | None = None  # number of Google reviews
    price_level: int | None = None  # Google's 0-4 scale (0=free, 4=very expensive)
    lat: float | None = None
    lon: float | None = None
    nightly_price: float | None = None
    currency: str = "USD"
    check_in: date | None = None
    check_out: date | None = None
    booking_url: str | None = None
    website: str | None = None
    amenities: list[str] = Field(default_factory=list)
    # Composite ranking score (0.0-1.0) computed by the Hotel agent.
    score: float | None = None
    # Per-component scores (rating / popularity / proximity / budget) for
    # transparency on why a hotel was ranked where it was.
    score_breakdown: dict[str, float] | None = None
    notes: str | None = None


class Restaurant(BaseModel):
    name: str
    cuisine: str | None = None
    address: str | None = None
    rating: float | None = None
    price_level: Literal["$", "$$", "$$$", "$$$$"] | None = None
    notes: str | None = None


class ItineraryStop(BaseModel):
    """A single point of interest the traveler will visit."""

    name: str
    day: int  # 1-indexed day of the trip
    start_time: datetime | None = None
    duration_minutes: int | None = None
    address: str | None = None
    notes: str | None = None


class LogisticsLeg(BaseModel):
    """A transit leg between two stops (walk, metro, taxi, etc.)."""

    from_stop: str
    to_stop: str
    mode: Literal["walk", "transit", "drive", "bike", "taxi"]
    duration_minutes: int
    distance_km: float | None = None
    instructions_url: str | None = None


class TravelPlan(BaseModel):
    """The final structured artifact returned to the user."""

    destination: str
    dates: dict[str, str] | None = None
    travelers: int = 1
    budget_tier: str | None = None

    flights: list[Flight] = Field(default_factory=list)
    hotels: list[Hotel] = Field(default_factory=list)
    restaurants: list[Restaurant] = Field(default_factory=list)
    itinerary: list[ItineraryStop] = Field(default_factory=list)
    logistics: list[LogisticsLeg] = Field(default_factory=list)

    summary: str | None = None
    errors: list[dict] = Field(default_factory=list)
