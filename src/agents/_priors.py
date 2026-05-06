"""Static price priors and FX for budget feasibility checks.

The flights agent uses these to classify a user's stated budget against
typical round-trip economy fares for a given (origin_country, dest_country)
pair, and surface a verdict so the renderer can warn the user when the
constraint is unrealistic.

Pure data + helpers; no I/O. Numbers are intentionally conservative
ranges from public fare aggregators as of late 2025/early 2026 — meant
as a *prior*, not a quote. Replace with a live source later if needed.
"""

from __future__ import annotations

from typing import Literal

# ── FX (1 unit of currency → USD) ────────────────────────────────────
#
# Static rates for v1. Refreshed manually; volatility is low enough that
# the tier verdict ("ok"/"tight"/"infeasible") doesn't flip on day-to-day
# moves. If a user's currency isn't here, we fall back to assuming USD
# and emit no verdict (better silent than wrong).

_FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "COP": 0.00025,   # ≈ 4000 COP per USD
    "MXN": 0.058,     # ≈ 17 MXN per USD
    "BRL": 0.20,      # ≈ 5 BRL per USD
    "ARS": 0.001,     # highly volatile; conservative
    "CLP": 0.0011,
    "PEN": 0.27,
    "CAD": 0.74,
    "AUD": 0.66,
    "CHF": 1.13,
}


def to_usd(amount: float, currency: str | None) -> float | None:
    """Convert `amount` in `currency` to USD using static rates.

    Returns None when the currency isn't covered — caller should treat
    that as "no feasibility verdict possible" rather than guessing.
    """
    if amount is None:
        return None
    if not currency:
        return None
    rate = _FX_TO_USD.get(currency.upper())
    if rate is None:
        return None
    return amount * rate


# ── Price priors (typical round-trip ECONOMY fare in USD) ────────────
#
# Keyed by (origin_country_iso2, dest_country_iso2). Order matters; the
# table is asymmetric for routes where outbound vs return seasonality
# differs significantly. Looked up bidirectionally for any pair not
# explicitly enumerated.

_RT_PRICE_PRIORS_USD: dict[tuple[str, str], tuple[int, int]] = {
    # From Colombia (project's home country — keep thorough)
    ("CO", "JP"): (1100, 1900),
    ("CO", "US"): (350, 800),
    ("CO", "MX"): (300, 700),
    ("CO", "BR"): (400, 900),
    ("CO", "AR"): (500, 1100),
    ("CO", "CL"): (450, 950),
    ("CO", "PE"): (250, 600),
    ("CO", "EC"): (200, 500),
    ("CO", "ES"): (700, 1300),
    ("CO", "FR"): (800, 1500),
    ("CO", "GB"): (800, 1500),
    ("CO", "DE"): (850, 1600),
    ("CO", "IT"): (850, 1600),
    ("CO", "PA"): (180, 450),
    ("CO", "CR"): (250, 600),
    ("CO", "CU"): (300, 700),
    # From the US
    ("US", "JP"): (900, 1700),
    ("US", "GB"): (450, 1100),
    ("US", "FR"): (500, 1200),
    ("US", "ES"): (500, 1200),
    ("US", "IT"): (550, 1300),
    ("US", "MX"): (250, 600),
    ("US", "CO"): (350, 800),
    ("US", "BR"): (700, 1500),
    ("US", "AR"): (800, 1700),
    ("US", "AU"): (1000, 2000),
    ("US", "TH"): (900, 1800),
    # From Europe
    ("GB", "JP"): (700, 1500),
    ("GB", "US"): (450, 1100),
    ("FR", "JP"): (700, 1500),
    ("DE", "JP"): (700, 1500),
    ("ES", "JP"): (800, 1600),
    # From Mexico
    ("MX", "JP"): (1000, 1900),
    ("MX", "ES"): (700, 1400),
    ("MX", "US"): (250, 600),
    ("MX", "CO"): (300, 700),
}

_DEFAULT_INTERCONTINENTAL_USD: tuple[int, int] = (900, 1700)
_DEFAULT_REGIONAL_USD: tuple[int, int] = (200, 600)
_DEFAULT_DOMESTIC_USD: tuple[int, int] = (100, 400)

# Coarse continent groupings used when no exact prior exists. Keeps the
# table small without inventing precision we don't have.
_CONTINENT: dict[str, str] = {
    # North America
    "US": "NA", "CA": "NA", "MX": "NA",
    # Central America & Caribbean (group with NA for routing distance)
    "PA": "NA", "CR": "NA", "GT": "NA", "HN": "NA", "NI": "NA",
    "SV": "NA", "CU": "NA", "DO": "NA",
    # South America
    "CO": "SA", "BR": "SA", "AR": "SA", "CL": "SA", "PE": "SA",
    "EC": "SA", "BO": "SA", "UY": "SA", "PY": "SA", "VE": "SA",
    # Europe
    "GB": "EU", "FR": "EU", "DE": "EU", "ES": "EU", "IT": "EU",
    "PT": "EU", "NL": "EU", "BE": "EU", "CH": "EU", "AT": "EU",
    "PL": "EU", "SE": "EU", "NO": "EU", "DK": "EU", "FI": "EU",
    "IE": "EU", "GR": "EU", "CZ": "EU", "HU": "EU", "TR": "EU",
    # Asia
    "JP": "AS", "KR": "AS", "CN": "AS", "HK": "AS", "TW": "AS",
    "TH": "AS", "VN": "AS", "ID": "AS", "MY": "AS", "PH": "AS",
    "SG": "AS", "IN": "AS", "AE": "AS", "QA": "AS", "IL": "AS",
    # Oceania
    "AU": "OC", "NZ": "OC",
    # Africa
    "ZA": "AF", "EG": "AF", "MA": "AF", "KE": "AF", "NG": "AF",
}


def estimate_rt_price_usd(
    origin_country: str | None,
    dest_country: str | None,
) -> tuple[int, int] | None:
    """Best-effort typical round-trip economy fare prior.

    Returns (low, high) in USD, or None if either country is unknown.
    Lookup order:
      1. Exact (origin, dest) pair.
      2. Reverse pair (dest, origin) — usually within ±10% of forward.
      3. Continent-pair default (intercontinental / regional / domestic).
    """
    if not origin_country or not dest_country:
        return None
    o = origin_country.upper()
    d = dest_country.upper()
    if (o, d) in _RT_PRICE_PRIORS_USD:
        return _RT_PRICE_PRIORS_USD[(o, d)]
    if (d, o) in _RT_PRICE_PRIORS_USD:
        return _RT_PRICE_PRIORS_USD[(d, o)]
    if o == d:
        return _DEFAULT_DOMESTIC_USD
    co = _CONTINENT.get(o)
    cd = _CONTINENT.get(d)
    if co and cd:
        if co == cd:
            return _DEFAULT_REGIONAL_USD
        return _DEFAULT_INTERCONTINENTAL_USD
    return None


# ── Verdict ──────────────────────────────────────────────────────────

Verdict = Literal["ok", "tight", "infeasible"]

# Severity thresholds, expressed as a fraction of the prior's *low* end.
# "tight" : budget covers ≥70% of the cheapest typical fare → likely
#           possible with effort (off-peak, multi-stop, awkward times).
# "infeasible": budget covers <70% of the cheapest typical fare →
#           almost certainly not enough; user should know up front.
_TIGHT_FRAC = 0.70


def feasibility_verdict(
    budget_usd: float | None,
    prior: tuple[int, int] | None,
) -> Verdict:
    """Classify a budget against a price prior.

    Returns "ok" when there's nothing to warn about (either the budget
    covers the typical low end, or we lack enough information to judge).
    The renderer treats "ok" as silent and "tight"/"infeasible" as
    warning conditions.
    """
    if budget_usd is None or prior is None:
        return "ok"
    low, _ = prior
    if budget_usd >= low:
        return "ok"
    if budget_usd >= _TIGHT_FRAC * low:
        return "tight"
    return "infeasible"
