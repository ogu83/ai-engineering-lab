# ai-engineering-lab

> **AI Engineering for Real Apps** — A YouTube series building production AI patterns from scratch.

Each episode adds one focused module to this repo. By episode 4 you have a working multi-component AI system, not four disconnected demos.

---

## Episode 1 — LangGraph: Multi-Agent Report Pipeline

**Branch:** `ep1-langgraph-pipeline`

A `PipelineGraph` with four specialized agents wired together via LangGraph's `StateGraph`:

```
PlannerAgent → ResearchAgent → WriterAgent → QAAgent
                    ↑                            │
                    └──── retry if score < 0.7 ──┘
```

| Agent | Responsibility |
|---|---|
| `PlannerAgent` | Turns a user question into a structured `ResearchPlan` (Pydantic) |
| `ResearchAgent` | Executes search queries against a document store |
| `WriterAgent` | Synthesizes research into a cited `Report` (Pydantic) |
| `QAAgent` | Scores the report; triggers a retry loop if confidence is too low |

### Key concepts demonstrated
- **TypedDict state** — all agents share one typed `PipelineState`; nothing is hidden in local variables
- **Conditional edges** — the retry loop is a graph edge, not an `if/else` inside an agent
- **Structured output** — every LLM call uses Claude's `tool_use` to return a typed Pydantic model, not a string
- **Eval harness** — shape, grounding, and threshold tests that work without mocking LLM content

### Setup

```bash
cd ai-engineering-lab
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env
# edit .env — add your ANTHROPIC_API_KEY
```

### Run

```bash
python -m ep1_langgraph.run "Which product categories had the highest return rates last quarter?"
```

### Test

```bash
# Offline — mocks all LLM calls
pytest ep1_langgraph/eval/ -v

# Integration — calls real Claude API (requires ANTHROPIC_API_KEY in .env)
pytest ep1_langgraph/eval/ -v -m integration
```

---

## Episode 2 — FastAPI + Pydantic + Claude: Structured Enrichment API

A production-style REST service that accepts raw text and returns a validated, structured enrichment object — enforced at every layer.

### Three validation gates

```
Client → EnrichmentRequest (Pydantic, extra=forbid)
       → Claude tool_use (forces structured JSON)
       → EnrichmentResponse (Pydantic, field + model validators)
       → Client
```

| Component | Responsibility |
|---|---|
| `EnrichmentRequest` | Validates inbound payload; rejects unexpected fields (422) |
| `ClaudeClient` | Calls Claude with `tool_choice: {type: tool}` to guarantee typed output |
| `EnrichmentResponse` | Validates genres list, strips unknown LLM fields, appends warning when confidence < 0.5 |
| `routes.py` | Maps upstream errors to 502 (bad LLM output) vs 503 (API unreachable) |

### Key concepts demonstrated
- **Strict input validation** — `extra="forbid"` rejects any field the caller shouldn't send
- **Output envelope hardening** — `extra="ignore"` silently drops unexpected LLM fields instead of crashing
- **Multi-gate validation** — Pydantic fires three times before a response leaves the service
- **Dependency injection** — `Depends(get_claude_client)` makes the LLM client swappable in tests
- **Error taxonomy** — 422 (client mistake), 502 (LLM output invalid), 503 (upstream unreachable)

### Run

```bash
uvicorn ep2_structured_api.main:app --reload
# POST http://localhost:8000/api/enrich
# GET  http://localhost:8000/health
```

Example request:

```json
{
  "title": "Inception",
  "context": "A thief who steals corporate secrets through dream-sharing technology."
}
```

Example response:

```json
{
  "title": "Inception",
  "summary": "A skilled thief uses experimental dream-sharing technology to steal corporate secrets.",
  "genres": ["sci-fi", "thriller"],
  "confidence": 0.92
}
```

### Test

```bash
# Offline — mocks all Claude calls
pytest ep2_structured_api/tests/ -v

# Integration — calls real Claude API (requires ANTHROPIC_API_KEY in .env)
pytest ep2_structured_api/tests/ -v -m integration
```

---

## Episode 3 — Playwright + Claude: Browser Automation Agent

An agent that navigates a real web UI and completes multi-step goals using Claude for decision-making and Playwright for execution — with no hardcoded selectors.

### Observe → Decide → Act loop

```
UserGoal → [observe page state] → [Claude decides next action] → [Playwright executes]
                ↑                                                          │
                └────────────────── loop until done ──────────────────────┘
```

| Module | Responsibility |
|---|---|
| `browser.py` | `get_page_state()` — extracts visible text + element descriptors (~1–3KB, not full HTML) |
| `llm.py` | `decide()` — Claude tool_use forced, returns structured `Action` |
| `loop.py` | `act()` with semantic locator fallbacks; `run_loop()` (injectable); `run_agent()` (browser lifecycle) |
| `actions.py` | `Action` Pydantic model with action-dependent validation |

### Key concepts demonstrated
- **Semantic locators** — `get_by_role`, `get_by_label`, `get_by_placeholder` with `or_()` fallback chains; survive UI refactors
- **Constrained vocabulary** — `Literal["click", "type", "scroll", "scroll_up", "done"]` means Claude can only return actions Playwright can execute
- **Cheap observation** — visible text + element labels, not full DOM; ~50× cheaper per loop iteration
- **Testable loop** — `run_loop(page)` is separated from browser lifecycle so tests inject a mock page; Playwright is a lazy import in `run_agent`
- **Three test signals** — outcome (did it finish?), trace structure (valid types + non-empty reasons), cap-never-hit (earliest stuck detector)

### Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium   # one-time: download browser binary
```

### Run

```bash
python -m ep3_playwright_agent.run https://example.com "Click the More information link"
python -m ep3_playwright_agent.run https://example.com "Click the More information link" --verbose
```

Verbose output shows the full action trace:

```
[1] click        → More information
     reason: The goal is to click the More information link
[2] done         →
     reason: Clicked the link successfully

Result: Clicked the link successfully
```

### Test

```bash
# Offline — no API key, no browser binary required
pytest ep3_playwright_agent/tests/ -v

# Integration — real Claude + real Playwright (requires ANTHROPIC_API_KEY in .env)
pytest ep3_playwright_agent/tests/ -v -m integration
```

---

## Series Overview

| Episode | Module | Topic |
|---|---|---|
| 1 | `ep1_langgraph` | LangGraph multi-agent pipeline |
| 2 | `ep2_structured_api` | FastAPI + Pydantic + Claude structured API |
| 3 | `ep3_playwright_agent` | Playwright browser automation agent |
| 4 | `ep4_nlsql` | NL-to-SQL with Vanna + DuckDB |

---

**GitHub:** [github.com/ogu83/ai-engineering-lab](https://github.com/ogu83/ai-engineering-lab)
