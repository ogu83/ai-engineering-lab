"""CLI entrypoint for the Episode 1 LangGraph pipeline.

Usage:
    python -m ep1_langgraph.run "Which product categories had the highest return rates?"
    python -m ep1_langgraph.run "Which product categories had the highest return rates?" --verbose
"""

from __future__ import annotations

import argparse
import sys

from ep1_langgraph.graph import RETRY_THRESHOLD, build_graph
from ep1_langgraph.state import PipelineState


def _initial_state(question: str) -> PipelineState:
    return PipelineState(
        question=question,
        plan=None,
        research_results=[],
        report=None,
        qa_score=0.0,
        qa_feedback="",
        retry_count=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph multi-agent report pipeline."
    )
    parser.add_argument("question", help="The question to research and answer.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print intermediate state details."
    )
    args = parser.parse_args()

    graph = build_graph()

    print(f"\nQuestion: {args.question}\n")
    print("Running pipeline…\n")

    result = graph.invoke(_initial_state(args.question))

    if args.verbose:
        print("=== RESEARCH PLAN ===")
        plan = result.get("plan")
        if plan:
            print(f"  Topic: {plan.topic}")
            print(f"  Scope: {plan.scope}")
            for q in plan.search_queries:
                print(f"  Query: {q}")
        print(f"\n  Passages retrieved: {len(result['research_results'])}")
        print()

    report = result.get("report")
    if report is None:
        print("ERROR: Pipeline produced no report.", file=sys.stderr)
        sys.exit(1)

    qa_score = result["qa_score"]
    retry_count = result["retry_count"]
    outcome = "PASSED" if qa_score >= RETRY_THRESHOLD else "CAPPED (retries exhausted)"

    print("=== REPORT ===")
    print(report.body)

    print("\n=== CITATIONS ===")
    for cite in report.citations:
        print(f"  • {cite}")

    print(f"\nWriter confidence : {report.confidence:.0%}")
    print(f"QA score          : {qa_score:.0%}")
    print(f"Retries used      : {retry_count}")
    print(f"Outcome           : {outcome}")


if __name__ == "__main__":
    main()
