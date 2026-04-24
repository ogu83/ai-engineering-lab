"""FastAPI route: POST /chat — NL-to-SQL pipeline."""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ep4_nlsql.api.models import ChatRequest, ChatResponse
from ep4_nlsql.pipeline.chart_builder import build_chart
from ep4_nlsql.pipeline.query_engine import run_query
from ep4_nlsql.pipeline.summarizer import summarize
from ep4_nlsql.pipeline.vanna_setup import CatalogVanna, get_vanna

router = APIRouter()


def get_vanna_dep() -> CatalogVanna:
    """FastAPI dependency that returns the Vanna singleton."""
    return get_vanna()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    vn: Annotated[CatalogVanna, Depends(get_vanna_dep)],
) -> ChatResponse:
    """Convert a natural language question to SQL, execute it, and return results.

    Pipeline:
    1. Vanna generates SQL from the question (sync → run in thread executor).
    2. DuckDB executes the SQL (read-only).
    3. Plotly builds a chart JSON if the data is chart-able.
    4. Claude summarises the results in plain English.
    """
    # Step 1 — NL-to-SQL
    try:
        loop = asyncio.get_running_loop()
        sql: str | None = await loop.run_in_executor(None, vn.generate_sql, body.question)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SQL generation failed: {exc}") from exc

    if not sql or not sql.strip():
        return ChatResponse(
            question=body.question, sql="", result=[], chart_json="", summary="",
            follow_up="", error="Could not generate SQL for this question.",
        )

    # Step 2 — Execute
    try:
        df = run_query(sql)
    except ValueError as exc:
        return ChatResponse(
            question=body.question, sql=sql, result=[], chart_json="", summary="",
            follow_up="", error=str(exc),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query execution failed: {exc}") from exc

    # Step 3 — Chart (pure, never raises)
    chart_json = build_chart(df)

    # Step 4 — Summarise
    try:
        summary, follow_up = await summarize(body.question, sql, df)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}") from exc

    return ChatResponse(
        question=body.question,
        sql=sql,
        result=df.to_dict(orient="records"),
        chart_json=chart_json,
        summary=summary,
        follow_up=follow_up,
    )
