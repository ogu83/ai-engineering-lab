from __future__ import annotations

from ep1_langgraph.llm import call_with_tool
from ep1_langgraph.models import QAResult
from ep1_langgraph.state import PipelineState

_QA_THRESHOLD = 0.70

_SYSTEM_PROMPT = (
    "You are a quality assurance reviewer for AI-generated reports. "
    "Evaluate the report against the original question and the research passages. "
    "Score strictly — only award a high score if the report is factually grounded, "
    "directly answers the question, and all citations are traceable to the research passages."
)


def qa_agent(state: PipelineState) -> dict:
    """Evaluate the report and decide whether to accept or retry.

    The QA score drives the conditional edge in the graph:
    - score >= 0.70 → END (pipeline succeeds)
    - score < 0.70  → retry (back to ResearchAgent, with feedback)

    retry_count is incremented here to prevent infinite loops.
    """
    report = state["report"]
    question = state["question"]
    research = state["research_results"]

    research_text = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(research))

    prompt = (
        f"Original question: {question}\n\n"
        f"Research passages:\n{research_text}\n\n"
        f"Report body:\n{report.body}\n\n"
        f"Report citations:\n{chr(10).join(f'  - {c}' for c in report.citations)}\n\n"
        f"Report confidence: {report.confidence:.2f}\n\n"
        "Evaluate this report. Check:\n"
        "1. Does the report directly answer the question?\n"
        "2. Is every citation traceable to the research passages?\n"
        "3. Are there unsupported claims?\n"
        "Provide a score (0–1), actionable feedback, and note any missing evidence."
    )

    result: QAResult = call_with_tool(
        prompt=prompt,
        tool_name="qa_review",
        tool_description=(
            "Review a report against the source research and provide a quality score "
            "between 0 (completely ungrounded) and 1 (fully grounded and accurate), "
            "along with actionable feedback for improvement."
        ),
        model_class=QAResult,
        system_prompt=_SYSTEM_PROMPT,
    )

    # Increment retry_count when the report fails so route_after_qa can cap retries
    retry_count = state["retry_count"]
    if result.score < _QA_THRESHOLD:
        retry_count += 1

    return {
        "qa_score": result.score,
        "qa_feedback": result.feedback,
        "retry_count": retry_count,
    }
