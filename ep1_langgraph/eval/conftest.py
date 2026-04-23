"""Pytest configuration for the Episode 1 eval harness.

Marks:
  integration — tests that call the real Claude API.
               Skipped by default unless ANTHROPIC_API_KEY is set in the environment.

Run offline (mocked):
    pytest ep1_langgraph/eval/ -v

Run integration (requires API key):
    pytest ep1_langgraph/eval/ -v -m integration
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call the real Claude API (requires ANTHROPIC_API_KEY)",
    )


def pytest_collection_modifyitems(config, items):
    api_key_present = bool(os.getenv("ANTHROPIC_API_KEY"))
    skip_integration = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set — skipping integration test")

    for item in items:
        if "integration" in item.keywords and not api_key_present:
            item.add_marker(skip_integration)
