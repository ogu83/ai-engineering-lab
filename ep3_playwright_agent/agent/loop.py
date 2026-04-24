"""Act step and agent loop.

Separation of concerns:
- run_loop(page, goal) — pure loop logic, injectable for tests (no browser lifecycle)
- run_agent(url, goal) — browser lifecycle wrapper (lazy Playwright import)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ep3_playwright_agent.agent.actions import Action
from ep3_playwright_agent.agent.browser import get_page_state
from ep3_playwright_agent.agent.llm import decide

if TYPE_CHECKING:
    from playwright.async_api import Page

MAX_ITERATIONS_DEFAULT = 10


async def act(page: Page, action: Action) -> None:
    """Execute a single action using semantic Playwright locators.

    Uses locator.or_() fallback chains so targets survive UI refactors:
    - click: button[name] | text match
    - type:  label | placeholder
    - scroll / scroll_up: mouse wheel
    """
    match action.action:
        case "click":
            locator = (
                page.get_by_role("link", name=action.target).or_(
                    page.get_by_role("button", name=action.target)
                ).or_(
                    page.get_by_text(action.target, exact=False)
                ).first
            )
            await locator.click()
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # pages with analytics/polling never reach networkidle

        case "type":
            locator = page.get_by_label(action.target).or_(
                page.get_by_placeholder(action.target)
            )
            await locator.fill(action.value or "")

        case "scroll":
            await page.mouse.wheel(0, 300)

        case "scroll_up":
            await page.mouse.wheel(0, -300)


async def run_loop(
    page: Page,
    goal: str,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    on_action: Callable[[int, Action], None] | None = None,
) -> tuple[str, list[Action]]:
    """Run the observe → decide → act loop on an already-open page.

    Returns (result, history) where result is either the agent's done reason
    or "Max iterations reached" if the cap was hit.

    on_action: optional callback called with (step_number, action) after each
    decision — useful for live progress output without blocking the loop.
    """
    history: list[Action] = []

    for _ in range(max_iterations):
        state = await get_page_state(page, history)
        action = await decide(goal, state)
        history.append(action)

        if on_action is not None:
            on_action(len(history), action)

        if action.action == "done":
            return action.reason, history

        await act(page, action)

    return "Max iterations reached", history


async def run_agent(
    url: str,
    goal: str,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    on_action: Callable[[int, Action], None] | None = None,
) -> tuple[str, list[Action]]:
    """Launch a browser, navigate to url, run the agent loop, close the browser.

    Playwright is imported lazily here so that importing this module does not
    require Playwright to be installed (offline tests use run_loop directly).
    """
    from playwright.async_api import async_playwright  # lazy import

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        try:
            return await run_loop(page, goal, max_iterations, on_action=on_action)
        finally:
            await browser.close()
