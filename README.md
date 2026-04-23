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

## Series Overview

| Episode | Module | Topic |
|---|---|---|
| 1 | `ep1_langgraph` | LangGraph multi-agent pipeline |
| 2 | `ep2_structured_api` | FastAPI + Pydantic + Claude structured API |
| 3 | `ep3_playwright_agent` | Playwright browser automation agent |
| 4 | `ep4_nlsql` | NL-to-SQL with Vanna + DuckDB |

---

**GitHub:** [github.com/ogu83/ai-engineering-lab](https://github.com/ogu83/ai-engineering-lab)
