from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnrichmentRequest(BaseModel):
    """Validated inbound request — design the contract before implementing the LLM call."""

    model_config = ConfigDict(extra="forbid")  # reject unexpected caller fields with 422

    title: str = Field(..., min_length=1, max_length=200)
    context: str = Field(..., min_length=1, max_length=100)

    @field_validator("title", "context", mode="before")
    @classmethod
    def strip_and_reject_blank(cls, v: object) -> object:
        """Strip leading/trailing whitespace and reject strings that are blank after stripping."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank or whitespace-only")
        return v


class EnrichmentResponse(BaseModel):
    """Validated outbound response — three validation gates fire before this leaves the service.

    Gate 1: Claude tool_use schema enforces types at the LLM boundary.
    Gate 2: field validators enforce business rules (non-empty genres, clean items).
    Gate 3: model validator adds cross-field warnings (low confidence flag).
    """

    model_config = ConfigDict(extra="ignore")  # strip unexpected LLM fields silently

    title: str = Field(..., min_length=1)
    genres: list[str] = Field(min_length=1)
    summary: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("genres")
    @classmethod
    def genres_not_empty_and_clean(cls, v: list[str]) -> list[str]:
        """Reject empty genre lists and blank individual genre strings."""
        if not v:
            raise ValueError("genres must not be empty")
        cleaned = [g.strip() for g in v]
        if any(not g for g in cleaned):
            raise ValueError("genres must not contain blank entries")
        return cleaned

    @model_validator(mode="after")
    def add_low_confidence_warning(self) -> EnrichmentResponse:
        """Automatically flag low-confidence responses — caller gets the signal to act on."""
        if self.confidence < 0.5 and "Confidence below threshold" not in self.warnings:
            self.warnings.append("Confidence below threshold")
        return self
