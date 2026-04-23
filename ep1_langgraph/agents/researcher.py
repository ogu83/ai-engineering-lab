from __future__ import annotations

from ep1_langgraph.data.docs import DOC_STORE, search
from ep1_langgraph.state import PipelineState


def researcher_agent(state: PipelineState) -> dict:
    """Execute the research plan against the document store.

    On retry runs, ``qa_feedback`` is incorporated into the search to bias
    retrieval toward evidence gaps identified by the QA agent.
    """
    plan = state["plan"]
    feedback = state.get("qa_feedback", "")
    scope = plan.scope  # "broad" | "focused" — wired to search depth

    results: list[str] = []
    seen: set[str] = set()

    for query in plan.search_queries:
        for passage in search(DOC_STORE, query, feedback=feedback, scope=scope):
            if passage not in seen:
                seen.add(passage)
                results.append(passage)

    return {"research_results": results}
