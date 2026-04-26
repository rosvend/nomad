# Nomad
Multi-agent travel planning system built with LangGraph that automates end-to-end trip organization. Given a destination (e.g., Tokyo), coordinated AI agents handle flight search, hotel recommendations, itinerary generation, and review aggregation—combining real-time data and user preferences to produce a structured, optimized travel plan.


## Repo Structure

```
nomad/
├── config.py               # Settings (pydantic-settings), provider factory
├── state/
│   └── trip_state.py       # TripState TypedDict — the contract between all agents
├── agents/                 # One file per agent
├── graph/
│   ├── builder.py          # Assembles the StateGraph
│   └── edges.py            # Conditional edge logic, fallback routing
├── tools/                  # LangGraph tool wrappers (quota-aware, error-safe)
└── output/
    ├── schemas.py          # Pydantic output models
    └── formatter.py        # TravelPlan → markdown / JSON
```

## Tech Stack

| Concern | Primary (free/default) | Fallback / Upgrade |
|---|---|---|
| Orchestration | LangGraph | — |
| LLM | Ollama (local) | Gemini free tier |
| Web search | SearXNG (self-hosted)/Crawl4AI | Tavily |
| Flight data | `fast-flights` (no key) | SerpApi |
| Places/reviews | Google Maps Grounding Lite | — |
| Logistics/routing | Google Maps Directions API | — |
| Deep scraping | Crawl4AI | — |
| Package manager | `uv` | — |
| Infra (local) | Docker Compose (SearXNG + Ollama) | — |

## How to run

### Prerequisites 

* Python +3.11 
* uv (package manager)

### Copy .env.example to your .env 

### Quick start