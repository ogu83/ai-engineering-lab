"""ep3 test suite — five categories mirroring the slide design plan.

Offline tests (no API key, no browser binaries):
  TestActionModel    — Pydantic validation logic
  TestGetPageState   — observe step with mocked Playwright page
  TestDecide         — decide step with mocked Anthropic client
  TestAgentLoop      — run_loop with all I/O mocked
  TestTraceStructure — structural invariants on action history
  TestCapBehavior    — max_iterations cap enforcement

Integration tests (require ANTHROPIC_API_KEY + playwright install chromium):
  TestAgentIntegration — real Claude + real browser against local form.html
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ep3_playwright_agent.agent.actions import Action
from ep3_playwright_agent.agent.browser import get_page_state
from ep3_playwright_agent.agent.llm import decide
from ep3_playwright_agent.agent.loop import run_loop

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_TYPES = {"click", "type", "scroll", "scroll_up", "done"}

# A minimal two-step sequence: one click then done
DONE_SEQUENCE = [
    Action(action="click", target="Submit button", reason="Form is filled — submitting"),
    Action(action="done", reason="Confirmation message is visible"),
]

# A multi-step sequence to verify trace structure across longer runs
MULTI_STEP_SEQUENCE = [
    Action(action="type", target="Name", value="Test User", reason="Filling name field"),
    Action(action="type", target="Email", value="test@example.com", reason="Filling email field"),
    Action(action="click", target="Submit", reason="All fields filled"),
    Action(action="done", reason="Form submitted successfully"),
]


# ---------------------------------------------------------------------------
# 1. Action model validation
# ---------------------------------------------------------------------------


class TestActionModel:
    def test_valid_click(self):
        a = Action(action="click", target="Submit button", reason="Submitting")
        assert a.action == "click"
        assert a.value is None

    def test_valid_type(self):
        a = Action(action="type", target="Email", value="x@y.com", reason="Entering email")
        assert a.value == "x@y.com"

    def test_valid_scroll(self):
        a = Action(action="scroll", reason="Scrolling to see more")
        assert a.target == ""

    def test_valid_done_empty_target(self):
        a = Action(action="done", reason="Goal complete")
        assert a.target == ""

    def test_reason_is_stripped(self):
        a = Action(action="done", reason="  done  ")
        assert a.reason == "done"

    def test_target_is_stripped(self):
        a = Action(action="click", target="  Submit  ", reason="test")
        assert a.target == "Submit"

    def test_empty_reason_rejected(self):
        with pytest.raises(ValidationError, match="reason"):
            Action(action="click", target="button", reason="")

    def test_whitespace_reason_rejected(self):
        with pytest.raises(ValidationError, match="reason"):
            Action(action="click", target="button", reason="   ")

    def test_click_requires_target(self):
        with pytest.raises(ValidationError, match="target"):
            Action(action="click", target="", reason="test")

    def test_type_requires_target(self):
        with pytest.raises(ValidationError, match="target"):
            Action(action="type", target="", value="hello", reason="test")

    def test_type_requires_value(self):
        with pytest.raises(ValidationError, match="value"):
            Action(action="type", target="Name", reason="test")  # no value

    def test_invalid_action_literal(self):
        with pytest.raises(ValidationError):
            Action(action="hover", target="button", reason="test")  # type: ignore[arg-type]

    def test_scroll_up_allows_empty_target(self):
        a = Action(action="scroll_up", reason="Scrolling back up")
        assert a.target == ""


# ---------------------------------------------------------------------------
# 2. Observe step — get_page_state
# ---------------------------------------------------------------------------


class TestGetPageState:
    async def test_includes_page_text(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "Welcome to Test Site"
        mock_page.eval_on_selector_all.return_value = ["button: Submit"]

        state = await get_page_state(mock_page, [])
        assert "Welcome to Test Site" in state

    async def test_includes_element_descriptors(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "page"
        mock_page.eval_on_selector_all.return_value = ["input: Email", "button: Submit"]

        state = await get_page_state(mock_page, [])
        assert "input: Email" in state
        assert "button: Submit" in state

    async def test_truncates_long_body_text(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "x" * 5000
        mock_page.eval_on_selector_all.return_value = []

        state = await get_page_state(mock_page, [])
        # Only the first 2000 chars of body text should appear.
        # "Page text:\n..." adds one extra 'x' from the literal word "text".
        x_count = state.count("x")
        assert x_count <= 2001

    async def test_includes_action_history(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "page"
        mock_page.eval_on_selector_all.return_value = []

        history = [Action(action="click", target="Login", reason="Starting authentication")]
        state = await get_page_state(mock_page, history)
        assert "click: Login" in state
        assert "Starting authentication" in state

    async def test_empty_history_shows_none_yet(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "page"
        mock_page.eval_on_selector_all.return_value = []

        state = await get_page_state(mock_page, [])
        assert "(none yet)" in state

    async def test_type_action_shows_value_in_history(self):
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "page"
        mock_page.eval_on_selector_all.return_value = []

        history = [Action(action="type", target="Email", value="x@y.com", reason="entering email")]
        state = await get_page_state(mock_page, history)
        assert "'x@y.com'" in state


# ---------------------------------------------------------------------------
# 3. Decide step — Claude tool_use
# ---------------------------------------------------------------------------


class TestDecide:
    async def test_returns_action_from_tool_block(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "act"
        tool_block.input = {"action": "click", "target": "Submit", "reason": "Form ready"}

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        with patch("ep3_playwright_agent.agent.llm._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            action = await decide("Submit form", "Page state text")

        assert action.action == "click"
        assert action.target == "Submit"
        assert action.reason == "Form ready"

    async def test_raises_on_missing_tool_block(self):
        mock_response = MagicMock()
        mock_response.content = []  # no tool_use block

        with patch("ep3_playwright_agent.agent.llm._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with pytest.raises(RuntimeError, match="tool_use"):
                await decide("goal", "state")

    async def test_raises_on_wrong_tool_name(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "other_tool"  # not "act"
        tool_block.input = {}

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        with patch("ep3_playwright_agent.agent.llm._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with pytest.raises(RuntimeError, match="tool_use"):
                await decide("goal", "state")

    async def test_invalid_tool_payload_raises_validation_error(self):
        """Claude returns a tool block with an invalid action — Pydantic should raise."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "act"
        tool_block.input = {"action": "hover", "target": "button", "reason": "bad"}

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        with patch("ep3_playwright_agent.agent.llm._client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            with pytest.raises(ValidationError):
                await decide("goal", "state")


# ---------------------------------------------------------------------------
# 4. Agent loop — run_loop with mocked I/O
# ---------------------------------------------------------------------------


class TestAgentLoop:
    async def test_returns_done_reason(self):
        mock_page = AsyncMock()
        actions = iter(DONE_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            result, history = await run_loop(mock_page, "Submit the form")

        assert result == "Confirmation message is visible"
        assert len(history) == 2

    async def test_respects_max_iterations(self):
        mock_page = AsyncMock()
        stuck = Action(action="click", target="Button", reason="Retrying")

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(return_value=stuck)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            result, history = await run_loop(mock_page, "goal", max_iterations=3)

        assert result == "Max iterations reached"
        assert len(history) == 3

    async def test_done_stops_loop_immediately(self):
        """A done action as the very first step stops after one iteration."""
        mock_page = AsyncMock()
        done_now = Action(action="done", reason="Already there")

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(return_value=done_now)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()) as mock_act,
        ):
            result, history = await run_loop(mock_page, "goal")

        assert result == "Already there"
        assert len(history) == 1
        mock_act.assert_not_called()  # act is not called for done

    async def test_multi_step_sequence(self):
        mock_page = AsyncMock()
        actions = iter(MULTI_STEP_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            result, history = await run_loop(mock_page, "Fill and submit form")

        assert result == "Form submitted successfully"
        assert len(history) == 4


# ---------------------------------------------------------------------------
# 5. Trace structure invariants
# ---------------------------------------------------------------------------


class TestTraceStructure:
    async def test_all_action_types_valid(self):
        mock_page = AsyncMock()
        actions = iter(MULTI_STEP_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            _, history = await run_loop(mock_page, "goal")

        for action in history:
            assert action.action in VALID_TYPES

    async def test_all_reasons_non_empty(self):
        mock_page = AsyncMock()
        actions = iter(MULTI_STEP_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            _, history = await run_loop(mock_page, "goal")

        for action in history:
            assert len(action.reason.strip()) > 0

    async def test_last_action_is_done_on_success(self):
        mock_page = AsyncMock()
        actions = iter(DONE_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            _, history = await run_loop(mock_page, "goal")

        assert history[-1].action == "done"

    async def test_act_not_called_for_done(self):
        """act() must never be called when the action is done."""
        mock_page = AsyncMock()
        actions = iter(DONE_SEQUENCE)

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()) as mock_act,
        ):
            await run_loop(mock_page, "goal")

        # Only the click step should have triggered act — not the done step
        assert mock_act.call_count == 1


# ---------------------------------------------------------------------------
# 6. Cap behaviour — stuck detector
# ---------------------------------------------------------------------------


class TestCapBehavior:
    async def test_cap_hit_message(self):
        mock_page = AsyncMock()
        stuck = Action(action="click", target="Button", reason="Still trying")

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(return_value=stuck)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            result, history = await run_loop(mock_page, "goal", max_iterations=5)

        assert "Max iterations" in result
        assert len(history) == 5

    async def test_successful_task_does_not_hit_cap(self):
        """A short successful sequence must finish well before the cap."""
        mock_page = AsyncMock()
        actions = iter(DONE_SEQUENCE)  # 2 actions

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(side_effect=actions)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            _, history = await run_loop(mock_page, "goal", max_iterations=10)

        assert len(history) < 10

    async def test_custom_cap_respected(self):
        """Verify the cap is the exact parameter value, not a hardcoded constant."""
        mock_page = AsyncMock()
        stuck = Action(action="scroll", reason="Looking around")

        with (
            patch("ep3_playwright_agent.agent.loop.get_page_state", new=AsyncMock(return_value="state")),
            patch("ep3_playwright_agent.agent.loop.decide", new=AsyncMock(return_value=stuck)),
            patch("ep3_playwright_agent.agent.loop.act", new=AsyncMock()),
        ):
            _, history = await run_loop(mock_page, "goal", max_iterations=7)

        assert len(history) == 7


# ---------------------------------------------------------------------------
# 7. Integration tests — real Claude + real Playwright
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgentIntegration:
    async def test_agent_completes_contact_form(self, test_server: str):
        from ep3_playwright_agent.agent.loop import run_agent

        url = f"{test_server}/form.html"
        result, history = await run_agent(
            url,
            "Fill in the contact form with name 'Test User' and email 'test@example.com', then submit it",
            max_iterations=10,
        )

        assert result != "Max iterations reached", f"Agent got stuck. Trace:\n{history}"
        assert len(history) > 0
        assert history[-1].action == "done"
        assert len(history) < 10  # should not require the cap
