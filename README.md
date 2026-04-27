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

# 3. run the planner
uv run python -m src.main "Plan a 5-day trip to Tokyo"
```

The default config talks to a local Ollama (`http://localhost:11434`,
model `gemma3:12b`). To switch to Gemini's free tier, set `GEMINI_API_KEY`
in `.env` — the provider factory in `src/config.py` will pick it up
automatically.

### Visualize the graph

```bash
uv run langgraph dev
```

This opens LangGraph Studio against the compiled graph from
`src/graph/builder.py:build_graph` so you can step through agent
execution interactively.