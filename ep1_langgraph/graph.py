from __future__ import annotations

from langgraph.graph import END, StateGraph

from ep1_langgraph.agents.planner import planner_agent
from ep1_langgraph.agents.qa import qa_agent
from ep1_langgraph.agents.researcher import researcher_agent
from ep1_langgraph.agents.writer import writer_agent
from ep1_langgraph.state import PipelineState

RETRY_THRESHOLD = 0.70
MAX_RETRIES = 2  # after 2 failed QA rounds the pipeline stops regardless


def route_after_qa(state: PipelineState) -> str:
    """Conditional edge: route to END on success, back to researcher on failure.

    retry_count starts at 0 and is incremented by qa_agent on each failure.
    With MAX_RETRIES=2: the pipeline retries at most once (fails at retry_count=2).
    """
    if state["qa_score"] >= RETRY_THRESHOLD:
        return END
    if state["retry_count"] >= MAX_RETRIES:
        return END  # cap reached — return best available output
    return "researcher"


def build_graph() -> StateGraph:
    """Wire together the four-agent pipeline and return a compiled graph."""
    builder = StateGraph(PipelineState)

    builder.add_node("planner", planner_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("qa", qa_agent)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "qa")

    # The conditional edge IS the retry loop — no if/else inside any agent
    builder.add_conditional_edges(
        "qa",
        route_after_qa,
        {"researcher": "researcher", END: END},
    )

    return builder.compile()
