# Copilot Instructions — ai-engineering-lab

This repo is a YouTube series ("AI Engineering for Real Apps") where each episode adds one self-contained module. Each module lives in its own directory (`ep1_langgraph/`, `ep2_structured_api/`, …) and has its own `requirements.txt` and eval harness. All modules share the repo root.

## Commands

```bash
# Setup (run once per episode module)
py -3.13 -m venv .venv          # Python 3.11+ required; 3.13 is available on this machine
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Run the ep1 pipeline (CLI)
python -m ep1_langgraph.run "Your question here"
python -m ep1_langgraph.run "Your question here" --verbose

# Run all offline tests (no API key needed)
pytest ep1_langgraph/eval/ -v

# Run a single test class
pytest ep1_langgraph/eval/test_pipeline.py::TestRetryLoop -v

# Run a single test
pytest ep1_langgraph/eval/test_pipeline.py::TestRetryLoop::test_pipeline_caps_retries_at_max -v

# Run integration tests (requires ANTHROPIC_API_KEY in .env)
pytest ep1_langgraph/eval/ -v -m integration
```

## Architecture — ep1_langgraph

The pipeline is a LangGraph `StateGraph` with four nodes:

```
PlannerAgent → ResearchAgent → WriterAgent → QAAgent
                    ↑                            │
                    └──── retry if score < 0.7 ──┘
```

**Data flow through `PipelineState` (TypedDict):**

| Field | Owner | Purpose |
|---|---|---|
| `question` | caller | Immutable input — never mutated by any agent |
| `plan` | `planner_agent` | `ResearchPlan` Pydantic model |
| `research_results` | `researcher_agent` | `list[str]` of retrieved passages |
| `report` | `writer_agent` | `Report` Pydantic model |
| `qa_score` | `qa_agent` | Float 0–1; drives the conditional edge |
| `qa_feedback` | `qa_agent` | String feedback; read by researcher on retry |
| `retry_count` | `qa_agent` | Incremented on each failed QA round |

**The retry loop lives in the graph, not in agent code:**
- `route_after_qa()` in `graph.py` is the conditional edge function — it reads `qa_score` and `retry_count` and returns either `"researcher"` or `END`
- Agents never call each other directly; they return dicts that merge into `PipelineState`

**All LLM calls go through `llm.call_with_tool()`:**
- Forces Claude's `tool_use` with `tool_choice: {"type": "tool", "name": ...}` — no free-text fallback
- The Pydantic model's `model_json_schema()` is used directly as the tool's `input_schema`
- Raises `RuntimeError` if no `tool_use` block is returned

## Key Conventions

**Agent functions return partial state dicts, not full state:**
```python
# Correct — return only the fields this agent owns
def planner_agent(state: PipelineState) -> dict:
    return {"plan": plan}

# Wrong — don't return the full state
def planner_agent(state: PipelineState) -> PipelineState:
    state["plan"] = plan
    return state  # mutating shared state
```

**State is initialized with all fields at the call site** (`run.py`). `TypedDict` has no default values, so callers must supply every key.

**Mock patches go at the agent module level, not the llm module:**
```python
# Correct — patches the name as bound in the agent module
patch("ep1_langgraph.agents.planner.call_with_tool", return_value=MOCK_PLAN)

# Wrong — doesn't affect already-imported references in agent modules
patch("ep1_langgraph.llm.call_with_tool", return_value=MOCK_PLAN)
```

**Eval harness test categories (see `eval/test_pipeline.py`):**
1. `TestReportShape` — structural assertions, always deterministic
2. `TestCitationGrounding` — each citation must be a substring of a research passage
3. `TestQualityThresholds` — `confidence >= 0.65`, `qa_score >= RETRY_THRESHOLD`
4. `TestRetryLoop` — verifies conditional edge routing behavior
5. `TestPipelineIntegration` — marked `@pytest.mark.integration`, skipped without `ANTHROPIC_API_KEY`

**`ResearchAgent` wires `scope` to search depth:** `"focused"` → top 3 results, `"broad"` → top 5. Don't remove `scope` from `ResearchPlan` — it's intentionally connected to retrieval behavior.

**`QAAgent` owns `retry_count` increments.** The routing function `route_after_qa()` only reads it — it never writes state.

## Environment

Copy `.env.example` to `.env`. Required key: `ANTHROPIC_API_KEY`. Optional: `ANTHROPIC_MODEL` (defaults to `claude-3-5-sonnet-20241022`).

The `llm.py` module calls `load_dotenv()` at import time, so the `.env` file is loaded automatically.

## Adding a New Episode Module

New episodes follow the same pattern:
1. Create `epN_<name>/` with its own `__init__.py`, agents, state, models, graph, run, and eval
2. Add an `eval/conftest.py` with the `integration` marker (copy from `ep1_langgraph/eval/conftest.py`)
3. Update the root `README.md` series table
4. The module must be importable as a package from the repo root (no `sys.path` hacks)
