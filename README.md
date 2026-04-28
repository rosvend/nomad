# Nomad
Multi-agent travel planning system built with LangGraph that automates end-to-end trip organization. Given a destination (e.g., Tokyo), coordinated AI agents handle flight search, hotel recommendations, itinerary generation, and review aggregation—combining real-time data and user preferences to produce a structured, optimized travel plan.


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

# 6. run the planner
uv run python -m src.main "Plan a 5-day trip to Tokyo from LAX"
```

Steps 4 and 5 are only needed when the agents actually call `web_search`
or `fetch_page(render=True)`. Skip them for a flights-only / places-only
run.

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