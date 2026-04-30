# Nomad
Multi-agent travel planning system built with LangGraph that automates end-to-end trip organization. From one natural-language request — single-city, multi-city, or even "send me somewhere warm with a beach" — coordinated AI agents handle flight search, hotel recommendations, restaurant discovery, walkable-route logistics, and a day-by-day itinerary, producing a structured travel plan grounded in real-time data and user preferences.

## What it can plan

| Prompt style | Example | What you get |
|---|---|---|
| **Single-leg, explicit** | `"Plan a trip to Santa Marta from June 1 to June 5; I'll be staying at <address>"` | Flights from your origin, restaurants matching your preferences, day-by-day itinerary anchored to the address you provided (the hotel search is skipped). |
| **Multi-leg** | `"Plan a 1-week trip starting in Medellin. 3 days in Bogota, then 4 days in Cartagena before flying back."` | Per-leg hotels/restaurants/itinerary, plus outbound + inter-leg + return flights. Day numbers are continuous across the whole trip. |
| **Vague** | `"Send me somewhere warm with a beach. I have 4 days off next week."` | Either the LLM router infers a destination directly from your origin + preferences, or the destination suggester prints 2-3 candidates and waits for you to pick one. Either way you get a full plan. |


## Repo Structure

```
nomad/            
├── state/
│   └── trip_state.py       # TripState TypedDict — the contract between all agents
├── agents/                 # One file per agent
├── graph/
│   ├── builder.py          # Assembles the StateGraph
│   └── edges.py            # Conditional edge logic
├── tools/                  
└── output/
    ├── schemas.py          # Pydantic output models
    └── formatter.py        # TravelPlan → markdown / JSON
```

## Tech Stack

| Concern | Primary (free/default) | Fallback / Upgrade |
|---|---|---|
| Orchestration | LangGraph | — |
| LLM | Ollama (local) | Gemini free tier |
| Web search | SearXNG/Crawl4AI | Tavily |
| Flight data | `fli`/`fast-flights`| SerpApi |
| Places/reviews | Google Maps Grounding Lite | — |
| Logistics/routing | OpenStreetMap/Overpass | Google Maps Directions API |
| Package manager | `uv` | — |

## How to run

### Prerequisites 

* Python +3.11 
* uv (package manager)
* Ollama

### Quick start

```bash
# 1. install dependencies
uv sync

# 2. seed your env (defaults point at local Ollama + SearXNG; no signup required)
cp .env.example .env

# 3. pull the local LLM (safety-net required; configured default optional)
ollama pull llama3.1:8b      # safety-net fallback — required
ollama pull gemma3:12b       # configured default — optional, skip if disk-constrained

# 4. Run the SearXNG container
docker compose up -d searxng

# 5. Install Playwright's Chromium
uv run playwright install chromium

# 6. run the planner
uv run python -m src.main "Plan a 5-day trip to Santa Marta, Colombia from Medellin starting next monday"
uv run python -m src.main "Plan a 1-week trip starting in Medellin. 3 days in Bogota, then 4 days in Cartagena before flying back. I want to depart may 10th"
uv run python -m src.main "Send me somewhere warm with a beach. I have 4 days off next week."
```


### What needs a key?

Every `.env` field is optional. The table below shows what each one
unlocks and how the system degrades when it's blank.

| `.env` field | Unlocks | If blank |
|---|---|---|
| `DEFAULT_ORIGIN` | Flight search when the user query has no "from X" | Flight section is empty (not an error) |
| `OPENAI_API_KEY` | Fastest LLM path — every router/suggester/synthesizer call goes to OpenAI (default `gpt-4o-mini`; set `OPENAI_MODEL` for `gpt-4o` etc.) | Skipped; chain falls through to Gemini or Ollama |
| `GEMINI_API_KEY` | Free-tier hosted LLM as a step between OpenAI and local Ollama | Skipped; chain falls through to Ollama |
| `GOOGLE_MAPS_API_KEY` | Real ratings + review counts in hotel & restaurant ranking | Ranking still works via OSM/Overpass; rating signal absent |
| `TAVILY_API_KEY` / `SERPAPI_API_KEY` | Hosted web search & flight fallback | Falls back to SearXNG (if up) or returns no results |

The default config talks to a local Ollama (`http://localhost:11434`,
model `gemma3:12b`, with `llama3.1:8b` as a safety net). Setting any
hosted-LLM key flips the chain in priority order **OpenAI → Gemini →
Ollama-primary → Ollama-safety-net**, and `with_fallbacks` cascades on
quota / network / schema errors so a flaky upstream doesn't take the
whole system down. Recommended for tomorrow's class demo: set
`OPENAI_API_KEY` so a full plan runs in ~30 seconds instead of the 5-10
minutes Ollama needs.

### Visualize the graph

```bash
uv run langgraph dev --no-reload --allow-blocking
```
