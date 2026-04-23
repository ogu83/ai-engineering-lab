"""Eval harness for the Episode 1 LangGraph pipeline.

Three test categories (as shown in the episode slides):

  1. Shape tests  — deterministic assertions on report structure.
                    Always run, always pass if the graph wiring is correct.

  2. Grounding tests — verify that every citation is traceable to a research passage.
                       Catches hallucinated sources even without reading the content.

  3. Threshold tests — validate quality signals (confidence, QA score).
                       Acts as a degradation detector in CI.

All three categories run offline by mocking call_with_tool at the agent module level.
Integration tests (marked with @pytest.mark.integration) run the real Claude API.

Unit tests verify code. Eval harnesses verify LLM output. You need both.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from ep1_langgraph.graph import build_graph, RETRY_THRESHOLD
from ep1_langgraph.models import QAResult, Report, ResearchPlan
from ep1_langgraph.state import PipelineState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "Which product categories had the highest return rates last quarter?"

MOCK_PLAN = ResearchPlan(
    topic="product return rates by category",
    search_queries=["return rate by category Q4", "highest return rates electronics apparel"],
    scope="focused",
)

MOCK_REPORT = Report(
    body=(
        "Electronics had the highest return rate at 12.4%, followed by Apparel at 10.8%. "
        "Home & Garden had the lowest return rate at 3.2%."
    ),
    citations=[
        "Q4 returns summary: Overall return rate across all categories was 8.3%",
        "Electronics category detail: The 12.4% Q4 return rate was driven primarily by smartphones",
        "Apparel category detail: Return rate was 10.8% in Q4",
    ],
    confidence=0.85,
)

MOCK_QA_PASS = QAResult(score=0.90, feedback="Report is well-grounded and directly answers the question.")

MOCK_QA_FAIL = QAResult(score=0.45, feedback="Citations are present but the body lacks specific numbers.")

MOCK_QA_PASS_AFTER_RETRY = QAResult(
    score=0.80, feedback="Improved report after retry — now well-grounded."
)


def _initial_state(question: str = SAMPLE_QUESTION) -> PipelineState:
    return PipelineState(
        question=question,
        plan=None,
        research_results=[],
        report=None,
        qa_score=0.0,
        qa_feedback="",
        retry_count=0,
    )


# ---------------------------------------------------------------------------
# Patch helpers — patch at the agent module level (where the name is bound)
# ---------------------------------------------------------------------------

PLANNER_PATCH = "ep1_langgraph.agents.planner.call_with_tool"
WRITER_PATCH = "ep1_langgraph.agents.writer.call_with_tool"
QA_PATCH = "ep1_langgraph.agents.qa.call_with_tool"


# ---------------------------------------------------------------------------
# 1. Shape tests — structure is always deterministic
# ---------------------------------------------------------------------------

class TestReportShape:
    """The report must have the correct shape regardless of content."""

    def _run(self, qa_result=MOCK_QA_PASS):
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=qa_result),
        ):
            graph = build_graph()
            return graph.invoke(_initial_state())

    def test_report_is_not_none(self):
        result = self._run()
        assert result["report"] is not None

    def test_report_body_is_not_empty(self):
        result = self._run()
        assert result["report"].body.strip() != ""

    def test_report_has_citations(self):
        result = self._run()
        assert len(result["report"].citations) > 0

    def test_report_confidence_in_range(self):
        result = self._run()
        confidence = result["report"].confidence
        assert 0.0 <= confidence <= 1.0

    def test_qa_score_in_range(self):
        result = self._run()
        assert 0.0 <= result["qa_score"] <= 1.0

    def test_plan_has_queries(self):
        result = self._run()
        assert result["plan"] is not None
        assert len(result["plan"].search_queries) > 0

    def test_research_results_populated(self):
        result = self._run()
        # ResearchAgent uses the real mock doc store — should always find results
        assert len(result["research_results"]) > 0


# ---------------------------------------------------------------------------
# 2. Grounding tests — citations must be traceable to research passages
# ---------------------------------------------------------------------------

class TestCitationGrounding:
    """Every citation in the report must be a substring of at least one research passage.

    This test catches hallucinated sources — if Claude invents a citation that
    doesn't appear in the retrieved passages, this test fails.
    """

    def test_all_citations_grounded(self):
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=MOCK_QA_PASS),
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        report = result["report"]
        research = result["research_results"]

        ungrounded = [
            cite
            for cite in report.citations
            if not any(cite.lower() in passage.lower() for passage in research)
        ]

        assert ungrounded == [], (
            f"The following citations are NOT grounded in research_results:\n"
            + "\n".join(f"  - {c}" for c in ungrounded)
        )


# ---------------------------------------------------------------------------
# 3. Threshold tests — quality signals must exceed minimum baselines
# ---------------------------------------------------------------------------

class TestQualityThresholds:
    """Quality signal tests — these are degradation detectors, not content validators."""

    def test_confidence_above_minimum(self):
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=MOCK_QA_PASS),
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        assert result["report"].confidence >= 0.65, (
            f"Report confidence {result['report'].confidence:.2f} is below the minimum 0.65 threshold"
        )

    def test_qa_score_above_retry_threshold_on_passing_run(self):
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=MOCK_QA_PASS),
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        assert result["qa_score"] >= RETRY_THRESHOLD


# ---------------------------------------------------------------------------
# 4. Retry loop tests
# ---------------------------------------------------------------------------

class TestRetryLoop:
    """Verify the conditional edge retry logic behaves correctly."""

    def test_pipeline_retries_on_low_qa_score(self):
        """On a low QA score the pipeline should route back to researcher and retry."""
        qa_call_count = {"n": 0}

        def qa_side_effect(*args, **kwargs):
            qa_call_count["n"] += 1
            # Fail first call, pass second
            if qa_call_count["n"] == 1:
                return MOCK_QA_FAIL
            return MOCK_QA_PASS_AFTER_RETRY

        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, side_effect=qa_side_effect),
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        assert qa_call_count["n"] == 2, "QA agent should have been called twice (1 fail + 1 pass)"
        assert result["qa_score"] >= RETRY_THRESHOLD
        assert result["retry_count"] == 1

    def test_pipeline_caps_retries_at_max(self):
        """When all QA rounds fail the pipeline stops and returns the best available report."""
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=MOCK_QA_FAIL),  # always fail
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        # Pipeline should cap at MAX_RETRIES=2 and stop
        assert result["retry_count"] == 2
        # Report is still returned — pipeline degrades gracefully
        assert result["report"] is not None

    def test_no_retry_on_passing_score(self):
        """A passing QA score should result in zero retries."""
        with (
            patch(PLANNER_PATCH, return_value=MOCK_PLAN),
            patch(WRITER_PATCH, return_value=MOCK_REPORT),
            patch(QA_PATCH, return_value=MOCK_QA_PASS),
        ):
            graph = build_graph()
            result = graph.invoke(_initial_state())

        assert result["retry_count"] == 0


# ---------------------------------------------------------------------------
# 5. Integration tests — call the real Claude API
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end tests that call the real Claude API.

    Skipped automatically when ANTHROPIC_API_KEY is not set.
    Run with: pytest ep1_langgraph/eval/ -v -m integration
    """

    def test_full_pipeline_returns_report(self):
        graph = build_graph()
        result = graph.invoke(_initial_state())

        assert result["report"] is not None
        assert result["report"].body.strip() != ""
        assert len(result["report"].citations) > 0
        assert 0.0 <= result["report"].confidence <= 1.0
        assert 0.0 <= result["qa_score"] <= 1.0

    def test_full_pipeline_citations_grounded(self):
        graph = build_graph()
        result = graph.invoke(_initial_state())

        report = result["report"]
        research = result["research_results"]

        ungrounded = [
            cite
            for cite in report.citations
            if not any(cite.lower() in passage.lower() for passage in research)
        ]
        assert ungrounded == [], (
            "Real pipeline produced ungrounded citations:\n"
            + "\n".join(f"  - {c}" for c in ungrounded)
        )
