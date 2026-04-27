"""In-process quota tracker for paid / rate-limited providers.

Every tool that hits an external paid API must call
`quota_tracker.check_and_increment(provider)` *before* the network call.
If the monthly quota is exceeded, the tracker raises `QuotaExceededError`,
which the calling tool catches and converts into a structured error dict
the graph can route on (see CLAUDE.md → "Tools").

This implementation is in-memory only — counters reset every process start.
Persistence to `~/.nomad/quota.json` is a TODO once the first real tool
ships.
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

from src.config import get_settings


class QuotaExceededError(RuntimeError):
    """Raised when a provider's monthly quota has been hit."""

    def __init__(self, provider: str, used: int, limit: int) -> None:
        self.provider = provider
        self.used = used
        self.limit = limit
        super().__init__(f"Quota exceeded for {provider}: {used}/{limit}")


class QuotaTracker:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def _limit_for(self, provider: str) -> int | None:
        s = get_settings()
        return {
            "serpapi": s.serpapi_monthly_limit,
            "tavily": s.tavily_monthly_limit,
            "google_places": s.google_places_monthly_limit,
            "google_maps_grounding": s.google_maps_grounding_monthly_limit,
        }.get(provider)

    def check_and_increment(self, provider: str) -> None:
        """Reserve one call against `provider`'s quota; raise if over."""
        limit = self._limit_for(provider)
        with self._lock:
            used = self._counts[provider]
            if limit is not None and used >= limit:
                raise QuotaExceededError(provider, used, limit)
            self._counts[provider] = used + 1

    def usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


quota_tracker = QuotaTracker()
