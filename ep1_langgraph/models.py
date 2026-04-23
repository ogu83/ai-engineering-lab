from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Structured research plan produced by the PlannerAgent."""

    topic: str
    search_queries: list[str]
    scope: Literal["broad", "focused"]


class Report(BaseModel):
    """Final report produced by the WriterAgent."""

    body: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class QAResult(BaseModel):
    """Quality assessment produced by the QAAgent."""

    score: float = Field(ge=0.0, le=1.0)
    feedback: str
