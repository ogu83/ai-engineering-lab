"""FastAPI Title Enrichment Service — Episode 2

Run:
    uvicorn ep2_structured_api.main:app --reload

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from ep2_structured_api.api.routes import router

app = FastAPI(
    title="Title Enrichment API",
    description=(
        "Wraps Claude with strict Pydantic I/O. "
        "Every response is typed, validated, and structured — no raw string parsing."
    ),
    version="1.0.0",
)

app.include_router(router)
