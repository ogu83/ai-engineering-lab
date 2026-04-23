from __future__ import annotations

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ep2_structured_api.api.models import EnrichmentRequest, EnrichmentResponse
from ep2_structured_api.llm.client import ClaudeClient, get_claude_client

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post(
    "/api/enrich",
    response_model=EnrichmentResponse,
    status_code=200,
    summary="Enrich a media title with structured metadata",
    responses={
        422: {"description": "Invalid request — missing or malformed fields"},
        502: {"description": "LLM returned invalid output"},
        503: {"description": "Upstream LLM service unavailable"},
    },
)
async def enrich_title(
    request: EnrichmentRequest,
    client: ClaudeClient = Depends(get_claude_client),
) -> EnrichmentResponse:
    """Three validation gates:

    1. FastAPI validates the inbound ``EnrichmentRequest`` (422 if malformed).
    2. Claude's tool_use schema enforces types at the LLM boundary.
    3. ``EnrichmentResponse(**result)`` triggers Pydantic field and model validators
       before anything leaves the service (502 if Claude's output fails validation).
    """
    try:
        result = await client.enrich(request)
    except anthropic.APIError as exc:
        raise HTTPException(status_code=503, detail=f"LLM service error: {exc.message}")
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    try:
        return EnrichmentResponse(**result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "LLM response failed schema validation",
                "errors": exc.errors(),
            },
        )
