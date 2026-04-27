"""Tools layer — the only place that talks to external APIs.

Every tool returns a `ToolResult` (see `_common.py`): a dict with
`ok: bool`, `provider`, and either `data` or `error_type`/`message`.
Tools never raise — they convert exceptions into structured errors so
the graph's edges can route on them.
"""

from src.tools._common import ToolError, ToolResult, ToolSuccess
from src.tools.flights import search_flights
from src.tools.places import geocode, search_places, search_pois
from src.tools.quota import QuotaExceededError, quota_tracker
from src.tools.reviews import get_grounded_summary, get_reviews
from src.tools.routing import get_route
from src.tools.web_fetch import fetch_page
from src.tools.web_search import web_search

ALL_TOOLS = [
    search_flights,
    geocode,
    search_pois,
    search_places,
    get_reviews,
    get_grounded_summary,
    get_route,
    web_search,
    fetch_page,
]

__all__ = [
    "ALL_TOOLS",
    "QuotaExceededError",
    "ToolError",
    "ToolResult",
    "ToolSuccess",
    "fetch_page",
    "geocode",
    "get_grounded_summary",
    "get_reviews",
    "get_route",
    "quota_tracker",
    "search_flights",
    "search_places",
    "search_pois",
    "web_search",
]
