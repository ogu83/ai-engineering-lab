from __future__ import annotations

from typing import TypedDict

from ep1_langgraph.models import Report, ResearchPlan


class PipelineState(TypedDict):
    """Shared state passed between every node in the pipeline graph.

    Design rule: design state before designing nodes — every field must
    have a clear owner and a clear lifecycle.
    """

    # Immutable input — set once by the caller, never modified
    question: str

    # PlannerAgent output
    plan: ResearchPlan | None

    # ResearchAgent output — list of relevant text passages
    research_results: list[str]

    # WriterAgent output
    report: Report | None

    # QAAgent output — drives the conditional retry edge
    qa_score: float

    # QAAgent feedback — read by ResearchAgent on retry to sharpen queries
    qa_feedback: str

    # Safety cap — prevents infinite retry loops (incremented by QAAgent on failure)
    retry_count: int
