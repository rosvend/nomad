# Architecture

## Topology

```
[User Input: raw_query]
          │
          ▼
   [Router Agent]            parses raw_query → legs, dates, travelers,
          │                  budget_tier, preferences, user_lodging
          │
          │   conditional edge (route_after_router):
          │     • destination missing AND preferences set → suggester
          │     • otherwise → fan-out
          │
          ├──────────────► [Destination Suggester]   (only on vague prompts)
          │                       │   LLM picks 2-3 candidates;
          │                       │   user picks via stdin (or default in non-tty)
          │                       │
          ▼  Send() ×3 ◄──────────┘
 ┌────────┬────────┐
 ▼        ▼        ▼
[Flight][Hotel ][Food ]
 │        │        │
 │        └────┬───┘
 │             ▼
 │        [Logistics]   (joins on hotel + food)
 │             │
 └─────────────┴──► [Synthesizer]   defer=True; runs once after
                          │         flights + logistics both settle
                          ▼
                        [END]       final_plan in TripState
```

The Router fans out three specialists — Flights, Hotel, Food — via
`fan_out_to_specialists` in `src/graph/edges.py`. Logistics is **not**
in the initial fan-out: it has explicit incoming edges from Hotel and
Food because it routes between hotels and the restaurants those agents
discovered. The Synthesizer is registered with `defer=True` so it runs
exactly once after every upstream node has settled, regardless of which
superstep finishes first.

When the user's query gives preferences but no destination ("somewhere
warm with a beach"), the conditional edge routes the request through
the **Destination Suggester** before fan-out. The suggester writes a
single chosen destination into `state["legs"]` and flips
`destination_was_inferred=True` so the Synthesizer's summary
acknowledges the inference.

Partial failures (e.g. a flight tool erroring) become entries in
`state["errors"]` rather than exceptions, so the Synthesizer always
runs and the user gets whatever the rest of the graph produced.

## Multi-leg execution

Multi-city trips ("3 days in Bogota then 4 days in Cartagena") are
handled by `src/main.py`, **not** by changes to the graph. The flow:

1. `main.py` calls `router_agent` standalone to parse the query into
   `legs: list[Leg]` plus an overall date window. Each leg has its own
   destination + start/end + optional lodging.
2. For each leg, `main.py` invokes the compiled graph with
   `raw_query=""` and the leg's local destination/dates/origin
   pre-populated. The router short-circuits, the specialists run as in
   a normal single-city trip, and the synthesizer produces a per-leg
   `TravelPlan`. Origin for leg N is leg N-1's destination so the
   Flights agent searches the inter-leg hop.
3. After the last leg, `main.py` fires one more flights-tool call for
   the return hop (last destination → user's origin).
4. Per-leg `TravelPlan`s are merged into a single `TravelPlan.legs:
   list[LegPlan]`. Day numbers are renumbered to be continuous across
   the whole trip; flights are de-duplicated and surfaced at the
   top level.

Specialist agents stay completely unchanged — each leg invocation looks
like a single-city trip from their perspective. Single-leg trips skip
the loop entirely and run the graph once.

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
│   │   ├── destination_suggester.py
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

| Agent                  | Reads from TripState                                | Writes to TripState                                                                    |
| ---------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Router                 | `raw_query`                                         | `origin`, `destination`, `legs`, `dates`, `travelers`, `budget_tier`, `preferences`, `user_lodging` |
| Destination Suggester  | `origin`, `preferences`, `dates`, `budget_tier`     | `destination`, `legs`, `destination_was_inferred` (only when router left destination null) |
| Flight                 | `origin`, `destination`, `dates`, `travelers`       | `flights`                                                                              |
| Hotel                  | `destination`, `dates`, `travelers`, `budget_tier`, `user_lodging` | `hotels`                                                                |
| Food                   | `destination`, `preferences`                        | `restaurants`, `attractions`                                                           |
| Logistics              | `destination`, `hotels`, `restaurants`, `attractions`, `user_lodging` | `logistics`                                                          |
| Synthesizer            | all fields                                          | `final_plan`, `errors`                                                                 |

## How the graph executes

1. `main.py` loads `.env` and calls `router_agent` standalone with
   `{"raw_query": "..."}` to parse the query into structured intent
   (origin, legs, dates, preferences, etc.).
2. If the router returned no destination but the user expressed
   preferences, `main.py` runs the **Destination Suggester** to fill
   one in.
3. For each leg, `main.py` invokes the compiled graph (built once via
   `build_graph()`) with the leg's local destination/dates pre-populated
   and `raw_query=""`. The router short-circuits and dispatches the
   conditional edge straight to fan-out.
4. **Specialists** run concurrently — Flight, Hotel, Food in parallel;
   Logistics joins on Hotel + Food. Each returns a partial state update
   for its own field. List fields use `operator.add` reducers so
   parallel writes merge cleanly.
5. **Synthesizer** runs once Flights and Logistics have both delivered.
   It validates each list against the Pydantic schemas in
   `output/schemas.py` and writes `final_plan`.
6. After all legs finish, `main.py` searches one more flight (last
   destination → user's origin) for the return hop, deduplicates
   flights, renumbers itinerary days continuously across legs, and
   prints the merged `TravelPlan` via `formatter.to_markdown`.

## Extending the graph

- **New specialist?** Add a node + edge from Router (via `SPECIALIST_NODES`
  in `graph/edges.py`) + edge into Synthesizer + a `TripState` field with
  an additive reducer.
- **Fallback provider?** Add a conditional edge from the failing specialist
  to a fallback node that retries with the alternate provider, then edges
  to Synthesizer.
- **New tool?** Put it in `src/tools/`, decorate with `@tool`, and call
  `quota_tracker.check_and_increment(provider)` before the network call.
