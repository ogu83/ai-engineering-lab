from __future__ import annotations

from ep1_langgraph.llm import call_with_tool
from ep1_langgraph.models import ResearchPlan
from ep1_langgraph.state import PipelineState

_SYSTEM_PROMPT = (
    "You are a research planning assistant. Your job is to decompose a user question "
    "into a focused set of search queries that will surface the most relevant information. "
    "Be specific. Prefer 2–4 precise queries over many vague ones."
)


def planner_agent(state: PipelineState) -> dict:
    """Decompose the user question into a structured ResearchPlan.

    This is the first node in the pipeline. It never reads from a data source —
    its only input is the question and its only output is a plan.
    """
    plan: ResearchPlan = call_with_tool(
        prompt=f"Create a research plan to answer this question:\n\n{state['question']}",
        tool_name="create_research_plan",
        tool_description=(
            "Create a structured research plan with a topic summary, "
            "specific search queries, and a scope indicator."
        ),
        model_class=ResearchPlan,
        system_prompt=_SYSTEM_PROMPT,
    )
    return {"plan": plan}
