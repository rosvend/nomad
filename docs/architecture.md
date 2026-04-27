# Architecture

## Topology

```
[User Input: raw_query]
          │
          ▼
   [Router Agent]            parses raw_query → fills destination,
          │                  dates, travelers, budget_tier, preferences
          │
          ▼  Send() ×4 (parallel fan-out)
 ┌────────┬────────┬────────┐
 ▼        ▼        ▼        ▼
[Flight][Hotel ][Food  ][Logistics]
 │        │        │        │
 └────────┴────┬───┴────────┘   (LangGraph joins on incoming edges)
               ▼
       [Synthesizer Agent]      assembles TravelPlan
               │
               ▼
            [END]               final_plan available in TripState
```

The four specialists run in parallel because the Router emits four `Send()`
objects from `fan_out_to_specialists` (`src/graph/edges.py`). Each specialist
adds an edge directly into the Synthesizer, and LangGraph blocks on all
inbound edges before invoking it. Partial failures (e.g. a flight tool
erroring) become entries in `state["errors"]` rather than exceptions, so
the Synthesizer always runs.

## Repo layout

```
nomad/
├── src/
│   ├── main.py               # CLI entrypoint
│   ├── config.py             # Settings + provider factory (Ollama / Gemini)
│   ├── state/
│   │   └── trip_state.py     # TripState TypedDict — shared contract
│   ├── agents/               # one thin function per agent
│   │   ├── router.py
│   │   ├── flights_agent.py
│   │   ├── hotel_agent.py
│   │   ├── food_agent.py
│   │   ├── logistics_agent.py
│   │   └── synthesizer.py
│   ├── graph/
│   │   ├── builder.py        # build_graph() — assembles StateGraph
│   │   └── edges.py          # fan-out / conditional edges
│   ├── tools/
│   │   ├── quota.py          # QuotaTracker, QuotaExceededError
│   │   └── __init__.py       # re-exports quota_tracker singleton
│   └── output/
│       ├── schemas.py        # Pydantic: Flight, Hotel, Restaurant, ...
│       └── formatter.py      # TravelPlan → markdown / JSON
├── docs/
│   └── architecture.md       # (this file)
├── pyproject.toml
└── .env.example
```

## Agent responsibilities

| Agent       | Reads from TripState                                | Writes to TripState                                                            |
| ----------- | --------------------------------------------------- | ------------------------------------------------------------------------------ |
| Router      | `raw_query`                                         | `destination`, `dates`, `travelers`, `budget_tier`, `preferences`              |
| Flight      | `destination`, `dates`, `travelers`                 | `flights`                                                                      |
| Hotel       | `destination`, `dates`, `travelers`, `budget_tier`  | `hotels`                                                                       |
| Food        | `destination`, `preferences`                        | `restaurants`                                                                  |
| Logistics   | `destination`, `hotels`, `itinerary_stops`          | `logistics`                                                                    |
| Synthesizer | all fields                                          | `final_plan`, `errors`                                                         |

## How the graph executes

1. `main.py` loads `.env`, calls `build_graph()`, invokes the compiled graph
   with `{"raw_query": "..."}`.
2. **Router** populates the structured intent fields and returns. Its
   conditional edge emits four `Send` instances, one per specialist.
3. **Specialists** run concurrently. Each returns a partial state update
   for its own field (`flights`, `hotels`, `restaurants`, `logistics`).
   List fields use `operator.add` reducers so parallel writes merge cleanly.
4. **Synthesizer** runs once all four specialist edges have delivered
   state. It validates each list against the Pydantic schemas in
   `output/schemas.py` and writes `final_plan`.
5. `main.py` reads `final_plan` and prints it via `formatter.to_markdown`.

## Extending the graph

- **New specialist?** Add a node + edge from Router (via `SPECIALIST_NODES`
  in `graph/edges.py`) + edge into Synthesizer + a `TripState` field with
  an additive reducer.
- **Fallback provider?** Add a conditional edge from the failing specialist
  to a fallback node that retries with the alternate provider, then edges
  to Synthesizer.
- **New tool?** Put it in `src/tools/`, decorate with `@tool`, and call
  `quota_tracker.check_and_increment(provider)` before the network call.
