from __future__ import annotations

import os
from typing import Type

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

_DEFAULT_MODEL = "claude-sonnet-4-5"


def call_with_tool(
    prompt: str,
    tool_name: str,
    tool_description: str,
    model_class: Type[BaseModel],
    system_prompt: str = "",
) -> BaseModel:
    """Call Claude with a single required tool and return a parsed Pydantic model.

    Uses Claude's tool_use feature to enforce structured output instead of
    relying on prompt engineering to produce parseable text.
    """
    client = anthropic.Anthropic()
    model = os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    kwargs: dict = {
        "model": model,
        "max_tokens": 2048,
        "tools": [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": model_class.model_json_schema(),
            }
        ],
        # Force the model to call exactly this tool — no free-text fallback
        "tool_choice": {"type": "tool", "name": tool_name},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    tool_blocks = [b for b in response.content if b.type == "tool_use" and b.name == tool_name]
    if not tool_blocks:
        raise RuntimeError(
            f"Claude returned no tool_use block for '{tool_name}'. "
            f"Stop reason: {response.stop_reason}. Content: {response.content}"
        )

    return model_class(**tool_blocks[0].input)
