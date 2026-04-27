"""Claude summarizer: returns a plain-language summary and follow-up question."""
from __future__ import annotations

import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
_client = anthropic.AsyncAnthropic()

_SUMMARY_TOOL: dict = {
    "name": "report_summary",
    "description": "Report a plain-language summary of query results and suggest a follow-up question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 sentences summarising what the data shows in plain English.",
            },
            "follow_up": {
                "type": "string",
                "description": "One follow-up question the user might want to ask next.",
            },
        },
        "required": ["summary", "follow_up"],
    },
}

_SYSTEM = (
    "You are a data analyst assistant. "
    "Summarise query results clearly and concisely in plain language. "
    "Base your summary only on the data provided — do not extrapolate beyond the sample."
)


async def summarize(question: str, sql: str, df: pd.DataFrame) -> tuple[str, str]:
    """Return (summary, follow_up) for the given query result.

    Sends a bounded sample (≤50 rows) to Claude via forced tool_use to get a
    structured summary and a follow-up question suggestion.

    Returns:
        Tuple of (summary, follow_up) strings.

    Raises:
        RuntimeError: If Claude does not return the expected tool block.
    """
    if df.empty:
        return (
            "No results were returned for this query.",
            "Could you try rephrasing the question?",
        )

    total_rows = len(df)
    sample     = df.head(50).to_json(orient="records")

    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL executed:\n{sql}\n\n"
                f"Total rows returned: {total_rows}\n"
                f"Sample (first {min(total_rows, 50)} rows):\n{sample}"
            ),
        }
    ]

    response = await _client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_SYSTEM,
        tools=[_SUMMARY_TOOL],
        tool_choice={"type": "tool", "name": "report_summary"},
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "report_summary":
            return block.input["summary"], block.input["follow_up"]

    raise RuntimeError("Claude did not return a report_summary tool block")
