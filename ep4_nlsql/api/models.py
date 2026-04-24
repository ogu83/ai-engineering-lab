"""Pydantic request and response models for the /chat endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the catalog")

    model_config = {"extra": "forbid"}


class ChatResponse(BaseModel):
    question:   str
    sql:        str
    result:     list[dict]
    chart_json: str
    summary:    str
    follow_up:  str
    error:      str | None = None

    model_config = {"extra": "ignore"}
