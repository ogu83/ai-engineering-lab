from __future__ import annotations

from ep1_langgraph.llm import call_with_tool
from ep1_langgraph.models import Report
from ep1_langgraph.state import PipelineState

_SYSTEM_PROMPT = (
    "You are a technical report writer. Synthesize the provided research passages into a "
    "clear, factual report. Every claim must be supported by a direct citation from the "
    "research passages. Do not introduce information not present in the passages. "
    "Express your confidence honestly — if evidence is thin, score lower."
)


def writer_agent(state: PipelineState) -> dict:
    """Synthesize research results into a structured Report.

    If no research results are available the report will have a low
    confidence score and a note indicating insufficient evidence, which
    will cause the QA agent to fail the pipeline gracefully.
    """
    research = state["research_results"]
    question = state["question"]

    if not research:
        passages_text = "(No research results were found.)"
    else:
        passages_text = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(research))

    prompt = (
        f"Question: {question}\n\n"
        f"Research passages:\n{passages_text}\n\n"
        "Write a report answering the question. "
        "Cite specific passages using the exact text snippet from each passage."
    )

    report: Report = call_with_tool(
        prompt=prompt,
        tool_name="write_report",
        tool_description=(
            "Write a structured report with a body, a list of citations drawn verbatim "
            "from the research passages, and a confidence score between 0 and 1."
        ),
        model_class=Report,
        system_prompt=_SYSTEM_PROMPT,
    )
    return {"report": report}
