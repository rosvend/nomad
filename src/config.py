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
    openai_api_key: str | None = None  # highest priority when set
    openai_model: str = "gpt-4o-mini"  # cheap + fast; override to gpt-4o etc.

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
    """Return a chat model with automatic fallback through a chain.

    Order tried (each previous one's failure routes to the next):
      1. OpenAI (only when OPENAI_API_KEY is set).
      2. Gemini (only when GEMINI_API_KEY is set).
      3. Local Ollama with the configured model (`OLLAMA_MODEL`).
      4. Local Ollama with `llama3.1:8b` — a small, widely-pulled model
         we use as a last-resort safety net so a missing configured
         model doesn't take the whole system down.

    LangChain's `with_fallbacks` handles errors transparently — a
    `429 RESOURCE_EXHAUSTED` on Gemini, a `model not found` on Ollama,
    or a network error in any case all silently route to the next
    candidate.
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()

    primary_ollama = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    safety_net = ChatOllama(
        base_url=settings.ollama_base_url,
        model="llama3.1:8b",
    )

    ollama_chain = [primary_ollama, safety_net] if settings.ollama_model != "llama3.1:8b" else [safety_net]

    # Optional Gemini step — built lazily so a missing import doesn't kill us.
    gemini = None
    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
            gemini = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=settings.gemini_api_key,
            )
        except ImportError:
            gemini = None

    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        openai = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
        # Order: OpenAI → Gemini (if available) → Ollama chain.
        chain = []
        if gemini is not None:
            chain.append(gemini)
        chain.extend(ollama_chain)
        return openai.with_fallbacks(chain)

    if gemini is not None:
        return gemini.with_fallbacks(ollama_chain)

    return primary_ollama.with_fallbacks([safety_net])


def configure_logging() -> None:
    """Set up root logging once at startup. Call from main()."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
