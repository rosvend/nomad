"""Central settings + provider factory.

Reads `.env` via pydantic-settings. Every external dependency is opt-in:
the defaults point at free / self-hosted services (Ollama, SearXNG) so a
fresh clone runs without any signup. Setting an API key in `.env`
flips the relevant provider to its hosted free-tier fallback.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:12b"
    gemini_api_key: str | None = None  # fallback when set

    # --- web search ---
    searxng_base_url: str = "http://localhost:8080"
    tavily_api_key: str | None = None
    serpapi_api_key: str | None = None

    # --- places ---
    google_maps_api_key: str | None = None

    # --- flight defaults ---
    # Free-form origin (city name or 3-letter IATA) used by the Flight agent
    # when state["origin"] isn't populated by the Router. Without this, the
    # Flight agent has no idea where you're flying from and returns a
    # graceful no-op error.
    default_origin: str | None = None

    # --- quotas (monthly) ---
    serpapi_monthly_limit: int = 250
    # Google Maps Platform $200/mo credit covers ~10k calls; we cap at 8k
    # to leave headroom for non-reviews Maps usage. Each `get_reviews` call
    # consumes 2 Places API requests (findplace + details), so 8k = ~4k
    # reviews lookups per month.
    google_places_monthly_limit: int = 8000
    google_maps_grounding_monthly_limit: int = 10000
    tavily_monthly_limit: int = 1000

    # --- runtime ---
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton; safe to call from anywhere."""
    return Settings()


def get_llm() -> Any:
    """Return a configured chat model.

    Default: local Ollama. If `GEMINI_API_KEY` is set, fall back to Gemini.
    The actual import is lazy so users without a given provider's package
    still get a clean error.
    """
    settings = get_settings()
    if settings.gemini_api_key:
        # Lazy import — keeps `langchain-google-genai` optional.
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )


def configure_logging() -> None:
    """Set up root logging once at startup. Call from main()."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
