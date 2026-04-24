from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from ep2_structured_api.api.models import EnrichmentRequest, EnrichmentResponse
from ep2_structured_api.llm.prompts import ENRICH_TOOL_DESCRIPTION, SYSTEM_PROMPT, build_enrich_prompt

load_dotenv()

_DEFAULT_MODEL = "claude-3-5-sonnet-latest"

# Module-level singleton — one HTTP client shared across all requests
_async_client = anthropic.AsyncAnthropic()


class ClaudeClient:
    """Wraps the Anthropic API with a structured output contract.

    Uses Claude's tool_use feature to force a typed JSON response matching the
    EnrichmentResponse schema. The raw dict is returned to the caller; the route
    layer is responsible for constructing and validating the Pydantic model.
    """

    def __init__(self, model: str | None = None) -> None:
        self._client = _async_client
        self._model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    async def enrich(self, request: EnrichmentRequest) -> dict:
        """Call Claude with a required tool and return the raw tool_use input dict.

        Raises:
            anthropic.APIError: On upstream API failures (auth, rate limit, etc.).
            RuntimeError: If Claude returns no tool_use block despite forced tool_choice.
        """
        schema = EnrichmentResponse.model_json_schema()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": "enrich_title",
                    "description": ENRICH_TOOL_DESCRIPTION,
                    "input_schema": schema,
                }
            ],
            # Force exactly this tool — no free-text fallback allowed
            tool_choice={"type": "tool", "name": "enrich_title"},
            messages=[{"role": "user", "content": build_enrich_prompt(request)}],
        )

        tool_blocks = [
            b for b in response.content if b.type == "tool_use" and b.name == "enrich_title"
        ]
        if not tool_blocks:
            raise RuntimeError(
                f"Claude returned no tool_use block for 'enrich_title'. "
                f"Stop reason: {response.stop_reason}. Content: {response.content}"
            )

        return tool_blocks[0].input


def get_claude_client() -> ClaudeClient:
    """FastAPI dependency factory — override in tests via app.dependency_overrides."""
    return ClaudeClient()
