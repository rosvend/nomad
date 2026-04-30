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

### Quick start

```bash
# 1. install dependencies
uv sync

# 2. seed your env (defaults point at local Ollama + SearXNG; no signup required)
cp .env.example .env

# 3. pull the local LLM (safety-net required; configured default optional)
ollama pull llama3.1:8b      # safety-net fallback — required
ollama pull gemma3:12b       # configured default — optional, skip if disk-constrained

# 4. (optional, for web search) start a local SearXNG
docker compose up -d searxng

# 5. (optional, for JS-rendered page fetching) install Playwright's Chromium
uv run playwright install chromium

# 6. run the planner — try a single-city, then a multi-city, then a vague prompt
uv run python -m src.main "Plan a 5-day trip to Tokyo from LAX"
uv run python -m src.main "Plan a 1-week trip starting in Medellin. 3 days in Bogota, then 4 days in Cartagena before flying back."
uv run python -m src.main "Send me somewhere warm with a beach. I have 4 days off next week."
```

Steps 4 and 5 are only needed when the agents actually call `web_search`
or `fetch_page(render=True)`. Skip them for a flights-only / places-only
run.

### Notes on vague prompts

When the router can't extract a destination (e.g. *"send me somewhere
warm with a beach"*), the **destination suggester** runs: it asks the
LLM for 2-3 candidates that fit your origin + preferences + dates,
prints them, and reads a number from stdin. In a normal terminal the
session is interactive; in piped / non-tty contexts (CI, scripts) the
suggester silently picks candidate #1 so headless runs still produce a
plan. The router's LLM may also infer a destination directly when the
context is unambiguous, in which case the suggester is skipped.

### What needs a key?

Every `.env` field is optional. The table below shows what each one
unlocks and how the system degrades when it's blank.

| `.env` field | Unlocks | If blank |
|---|---|---|
| `DEFAULT_ORIGIN` | Flight search when the user query has no "from X" | Flight section is empty (not an error) |
| `GEMINI_API_KEY` | Faster, higher-quality Router & Synthesizer LLM calls | All LLM work goes to local Ollama |
| `GOOGLE_MAPS_API_KEY` | Real ratings + review counts in hotel & restaurant ranking | Ranking still works via OSM/Overpass; rating signal absent |
| `TAVILY_API_KEY` / `SERPAPI_API_KEY` | Hosted web search & flight fallback | Falls back to SearXNG (if up) or returns no results |

The default config talks to a local Ollama (`http://localhost:11434`,
model `gemma3:12b`, with `llama3.1:8b` as a safety net). Setting
`GEMINI_API_KEY` flips the LLM chain to Gemini first, with the local
Ollama models still available as automatic fallbacks on quota / network
errors.

### Visualize the graph

```bash
uv run langgraph dev
```

This opens LangGraph Studio against the compiled graph from
`src/graph/builder.py:build_graph` so you can step through agent
execution interactively.