"""Observe step: extract cheap, bounded page state for the LLM."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ep3_playwright_agent.agent.actions import Action

if TYPE_CHECKING:
    from playwright.async_api import Page

MAX_TEXT_CHARS = 2000


async def get_page_state(page: Page, history: list[Action]) -> str:
    """Return a compact string describing the current page for Claude.

    Extracts visible text (truncated), interactive element descriptors
    (role:label format), and the action history — all bounded to keep
    token costs predictable.
    """
    text = await page.inner_text("body")

    # Build "role: label" descriptors for elements Claude can act on
    elements: list[str] = await page.eval_on_selector_all(
        "input, button, a, select, textarea",
        """els => els.map(e => {
            const label = (
                e.innerText?.trim() ||
                e.getAttribute('aria-label') ||
                e.getAttribute('placeholder') ||
                e.getAttribute('name') ||
                ''
            );
            const role = e.tagName.toLowerCase();
            return label ? `${role}: ${label}` : null;
        }).filter(Boolean)""",
    )

    history_lines = "\n".join(
        f"- {a.action}: {a.target}" + (f" = {a.value!r}" if a.value else "") + f" ({a.reason})"
        for a in history
    ) or "(none yet)"

    return (
        f"Page text:\n{text[:MAX_TEXT_CHARS]}\n\n"
        f"Interactive elements:\n" + "\n".join(f"  {e}" for e in elements) + "\n\n"
        f"Actions taken:\n{history_lines}"
    )
