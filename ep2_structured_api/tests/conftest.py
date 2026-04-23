"""Pytest configuration for the Episode 2 test suite.

Marks:
  integration — tests that call the real Claude API.
               Skipped automatically unless ANTHROPIC_API_KEY is set.

Run offline (all mocked):
    pytest ep2_structured_api/tests/ -v

Run integration (requires API key):
    pytest ep2_structured_api/tests/ -v -m integration
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from ep2_structured_api.llm.client import get_claude_client
from ep2_structured_api.main import app


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call the real Claude API (requires ANTHROPIC_API_KEY)",
    )


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set — skipping integration test")
    for item in items:
        if "integration" in item.keywords and not os.getenv("ANTHROPIC_API_KEY"):
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Mock response fixtures
# ---------------------------------------------------------------------------

MOCK_HIGH_CONFIDENCE = {
    "title": "Inception",
    "genres": ["sci-fi", "thriller"],
    "summary": "A skilled thief enters the dreams of others to steal their secrets.",
    "confidence": 0.91,
    "warnings": [],
}

MOCK_LOW_CONFIDENCE = {
    "title": "The Entity",
    "genres": ["horror"],
    "summary": "Partial match found — title may refer to multiple works.",
    "confidence": 0.43,
    "warnings": ["Title matched multiple entries"],
}

MOCK_INVALID_EMPTY_GENRES = {
    "title": "Unknown",
    "genres": [],
    "summary": "No metadata found.",
    "confidence": 0.80,
    "warnings": [],
}


def make_mock_client(response: dict):
    """Create a mock ClaudeClient that returns a fixed dict from enrich()."""

    class _MockClaudeClient:
        async def enrich(self, request):
            return response

    return _MockClaudeClient()


# ---------------------------------------------------------------------------
# Client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client_high_confidence():
    """TestClient with a mock that returns a high-confidence response."""
    app.dependency_overrides[get_claude_client] = lambda: make_mock_client(MOCK_HIGH_CONFIDENCE)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_low_confidence():
    """TestClient with a mock that returns a low-confidence response."""
    app.dependency_overrides[get_claude_client] = lambda: make_mock_client(MOCK_LOW_CONFIDENCE)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_invalid_genres():
    """TestClient with a mock that returns an empty genres list (LLM confabulation)."""
    app.dependency_overrides[get_claude_client] = lambda: make_mock_client(MOCK_INVALID_EMPTY_GENRES)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def integration_client():
    """TestClient that uses the real Claude API — only used in integration tests."""
    with TestClient(app) as c:
        yield c
