"""Decide step: Claude as the reasoning layer — returns the next Action."""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from ep3_playwright_agent.agent.actions import Action

load_dotenv()

_client = anthropic.AsyncAnthropic()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

SYSTEM_PROMPT = """You are a browser automation agent. Complete the goal by operating a web browser one step at a time.

You will receive:
- The current goal
- The current page state (visible text, interactive elements, action history)

Return the single next action using the "act" tool. Available actions:
- click:     click an element (target = visible text or label of the element)
- type:      type text into an input (target = label or placeholder of the field, value = text to type)
- scroll:    scroll down the page
- scroll_up: scroll up the page
- done:      goal is complete (summarise what was accomplished in reason)

For click and type, set target to the element's label or visible text as it appears on the page.
Always explain your reasoning in the reason field."""


def _build_messages(goal: str, page_state: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nCurrent page state:\n{page_state}",
        }
    ]


async def decide(goal: str, page_state: str) -> Action:
    """Ask Claude for the next action given the current goal and page state.

    Forces tool_use so the response is always a structured Action — no prose fallback.
    Raises RuntimeError if Claude does not return the expected tool block.
    """
    response = await _client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[{"name": "act", "input_schema": Action.model_json_schema()}],
        tool_choice={"type": "tool", "name": "act"},
        messages=_build_messages(goal, page_state),
    )

    tool_block = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "act"),
        None,
    )
    if tool_block is None:
        raise RuntimeError("Claude did not return a tool_use block for 'act'")

    return Action(**tool_block.input)
