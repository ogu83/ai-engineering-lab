"""Tests for the ep2 Title Enrichment API.

Three test categories:

  1. Request validation  — FastAPI/Pydantic rejects malformed input with 422.
                           Always deterministic. No LLM involved.

  2. Response shape      — Mocked LLM, asserts on response structure.
                           Verifies the three validation gates work end-to-end.

  3. Contract signals    — Mocked LLM, asserts on quality thresholds and warning
                           flag behaviour.
                           NOTE: These are contract tests, not quality evals.
                           They verify the service's post-processing logic fires
                           correctly, not that Claude returns good content.

  4. Integration         — Real Claude API. Skipped without ANTHROPIC_API_KEY.

Unit tests verify code. These tests verify the service contract. Both matter.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. Request validation — 422 on malformed input (no LLM involved)
# ---------------------------------------------------------------------------

class TestRequestValidation:
    """FastAPI + Pydantic reject bad input before it ever reaches the LLM."""

    def test_empty_title_returns_422(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "", "context": "film"}
        )
        assert resp.status_code == 422

    def test_blank_title_returns_422(self, client_high_confidence: TestClient):
        """Whitespace-only strings must be rejected after stripping."""
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "   ", "context": "film"}
        )
        assert resp.status_code == 422

    def test_missing_context_returns_422(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post("/api/enrich", json={"title": "Inception"})
        assert resp.status_code == 422

    def test_title_too_long_returns_422(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "x" * 201, "context": "film"}
        )
        assert resp.status_code == 422

    def test_extra_field_returns_422(self, client_high_confidence: TestClient):
        """extra='forbid' on EnrichmentRequest rejects unexpected fields."""
        resp = client_high_confidence.post(
            "/api/enrich",
            json={"title": "Inception", "context": "film", "unexpected": "value"},
        )
        assert resp.status_code == 422

    def test_valid_request_is_accepted(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film distribution"}
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Response shape — structure is deterministic when LLM is mocked
# ---------------------------------------------------------------------------

class TestResponseShape:
    """The response must have the correct shape regardless of LLM content."""

    def test_title_is_string(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        assert isinstance(resp.json()["title"], str)
        assert resp.json()["title"] != ""

    def test_genres_is_non_empty_list(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        genres = resp.json()["genres"]
        assert isinstance(genres, list)
        assert len(genres) > 0

    def test_summary_is_string(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        assert isinstance(resp.json()["summary"], str)
        assert resp.json()["summary"] != ""

    def test_confidence_is_float_in_range(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        confidence = resp.json()["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_warnings_is_list(self, client_high_confidence: TestClient):
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        assert isinstance(resp.json()["warnings"], list)

    def test_health_endpoint(self, client_high_confidence: TestClient):
        resp = client_high_confidence.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 3. Contract signal tests — verifies post-processing logic, not LLM quality
# ---------------------------------------------------------------------------

class TestContractSignals:
    """Verifies that the service's validators and warning logic fire correctly.

    These are NOT quality evaluations of Claude's output.
    They test that the service correctly handles known mock inputs and that
    the Pydantic model validators behave as specified.
    To run quality evals against the real model, use the integration tests.
    """

    def test_high_confidence_response_has_no_automatic_warning(
        self, client_high_confidence: TestClient
    ):
        """confidence=0.91 must NOT trigger the low-confidence model_validator."""
        resp = client_high_confidence.post(
            "/api/enrich", json={"title": "Inception", "context": "film"}
        )
        assert resp.status_code == 200
        assert "Confidence below threshold" not in resp.json()["warnings"]

    def test_low_confidence_triggers_automatic_warning(
        self, client_low_confidence: TestClient
    ):
        """confidence=0.43 must trigger the model_validator to append the warning."""
        resp = client_low_confidence.post(
            "/api/enrich", json={"title": "The Entity", "context": "film"}
        )
        assert resp.status_code == 200
        assert "Confidence below threshold" in resp.json()["warnings"]

    def test_low_confidence_preserves_llm_warnings(
        self, client_low_confidence: TestClient
    ):
        """The model_validator must append, not replace, existing LLM warnings."""
        resp = client_low_confidence.post(
            "/api/enrich", json={"title": "The Entity", "context": "film"}
        )
        warnings = resp.json()["warnings"]
        assert "Title matched multiple entries" in warnings
        assert "Confidence below threshold" in warnings

    def test_llm_empty_genres_returns_502(self, client_invalid_genres: TestClient):
        """If the LLM returns an empty genres list, the Pydantic validator must reject it
        with a 502 (bad gateway — upstream produced invalid output)."""
        resp = client_invalid_genres.post(
            "/api/enrich", json={"title": "Unknown", "context": "film"}
        )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 4. Integration tests — real Claude API
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """End-to-end tests against the real Claude API.

    Skipped automatically when ANTHROPIC_API_KEY is not set.
    Run with: pytest ep2_structured_api/tests/ -v -m integration
    """

    def test_known_title_returns_200(self, integration_client: TestClient):
        resp = integration_client.post(
            "/api/enrich", json={"title": "Inception", "context": "film distribution"}
        )
        assert resp.status_code == 200

    def test_known_title_response_shape(self, integration_client: TestClient):
        resp = integration_client.post(
            "/api/enrich", json={"title": "Inception", "context": "film distribution"}
        )
        data = resp.json()
        assert isinstance(data["title"], str) and data["title"] != ""
        assert isinstance(data["genres"], list) and len(data["genres"]) > 0
        assert isinstance(data["summary"], str) and data["summary"] != ""
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["warnings"], list)

    def test_ambiguous_title_has_warnings(self, integration_client: TestClient):
        """An intentionally obscure title should return warnings or low confidence."""
        resp = integration_client.post(
            "/api/enrich",
            json={"title": "ZZZ_UNKNOWN_TITLE_XYZ_99999", "context": "film"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Either confidence is low (triggering automatic warning) or model added its own
        assert data["confidence"] < 0.7 or len(data["warnings"]) > 0
