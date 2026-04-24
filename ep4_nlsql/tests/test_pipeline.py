"""Offline test suite for ep4 NL-to-SQL pipeline."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ep4_nlsql.api.models import ChatRequest, ChatResponse
from ep4_nlsql.api.routes import get_vanna_dep
from ep4_nlsql.main import app
from ep4_nlsql.pipeline.chart_builder import build_chart
from ep4_nlsql.pipeline.query_engine import run_query


# ---------------------------------------------------------------------------
# TestQueryEngine
# ---------------------------------------------------------------------------

class TestQueryEngine:
    def test_select_returns_dataframe(self, test_db_path):
        df = run_query("SELECT * FROM titles", db_path=test_db_path)
        assert not df.empty
        assert "name" in df.columns

    def test_with_cte_is_allowed(self, test_db_path):
        sql = (
            "WITH top AS (SELECT * FROM titles LIMIT 2) "
            "SELECT * FROM top"
        )
        df = run_query(sql, db_path=test_db_path)
        assert len(df) == 2

    def test_non_select_raises_value_error(self, test_db_path):
        with pytest.raises(ValueError, match="Only SELECT"):
            run_query("DROP TABLE titles", db_path=test_db_path)

    def test_insert_raises_value_error(self, test_db_path):
        with pytest.raises(ValueError):
            run_query("INSERT INTO titles VALUES (99,'X','Y',2025)", db_path=test_db_path)

    def test_missing_db_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="create_db"):
            run_query("SELECT 1", db_path=str(tmp_path / "nope.duckdb"))

    def test_join_query(self, test_db_path):
        sql = (
            "SELECT t.name, p.revenue "
            "FROM performance p JOIN titles t ON t.id = p.title_id "
            "ORDER BY p.revenue DESC LIMIT 3"
        )
        df = run_query(sql, db_path=test_db_path)
        assert len(df) == 3
        assert "name" in df.columns

    def test_max_rows_cap(self, test_db_path, monkeypatch):
        import ep4_nlsql.pipeline.query_engine as qe
        monkeypatch.setattr(qe, "MAX_ROWS", 3)
        df = run_query("SELECT * FROM performance", db_path=test_db_path)
        assert len(df) <= 3


# ---------------------------------------------------------------------------
# TestChartBuilder
# ---------------------------------------------------------------------------

class TestChartBuilder:
    def test_categorical_plus_numeric_gives_bar(self, sample_df):
        result = build_chart(sample_df)
        assert result
        chart = json.loads(result)
        assert chart["data"][0]["type"] == "bar"

    def test_two_numeric_gives_scatter(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0], "return_rate": [0.05, 0.07, 0.09]})
        result = build_chart(df)
        assert result
        chart = json.loads(result)
        assert chart["data"][0]["type"] == "scatter"

    def test_empty_df_returns_empty_string(self):
        assert build_chart(pd.DataFrame()) == ""

    def test_single_column_returns_empty_string(self):
        df = pd.DataFrame({"genre": ["Action", "Drama"]})
        assert build_chart(df) == ""

    def test_all_categorical_returns_empty_string(self):
        df = pd.DataFrame({"genre": ["Action"], "region": ["AMER"]})
        assert build_chart(df) == ""


# ---------------------------------------------------------------------------
# TestModels
# ---------------------------------------------------------------------------

class TestModels:
    def test_chat_request_valid(self):
        req = ChatRequest(question="What is the top genre?")
        assert req.question == "What is the top genre?"

    def test_chat_request_empty_raises(self):
        with pytest.raises(Exception):
            ChatRequest(question="")

    def test_chat_response_no_error(self):
        resp = ChatResponse(
            question="q", sql="SELECT 1", result=[{"a": 1}],
            chart_json="", summary="All good", follow_up="Next?"
        )
        assert resp.error is None

    def test_chat_response_with_error(self):
        resp = ChatResponse(
            question="q", sql="", result=[], chart_json="",
            summary="", follow_up="", error="bad"
        )
        assert resp.error == "bad"

    def test_chat_request_rejects_extra_fields(self):
        with pytest.raises(Exception):
            ChatRequest(question="hi", extra_field="x")


# ---------------------------------------------------------------------------
# TestRoutes (mocked Vanna + pipeline)
# ---------------------------------------------------------------------------

_MOCK_SQL = "SELECT genre, AVG(return_rate) FROM performance GROUP BY genre"
_MOCK_DF  = pd.DataFrame({"genre": ["Action", "Horror"], "avg_return_rate": [0.07, 0.14]})


def _make_mock_vanna(sql=_MOCK_SQL):
    vn = MagicMock()
    vn.generate_sql.return_value = sql
    return vn


class TestRoutes:
    @pytest.fixture(autouse=True)
    def _client(self):
        self._app = app
        yield

    from contextlib import contextmanager

    @contextmanager
    def _patched_client(self, vn, *, run_query_return=_MOCK_DF, summary=("Good summary", "Follow-up?")):
        app.dependency_overrides[get_vanna_dep] = lambda: vn
        try:
            with (
                patch("ep4_nlsql.api.routes.run_query", return_value=run_query_return),
                patch("ep4_nlsql.api.routes.summarize", new=AsyncMock(return_value=summary)),
            ):
                yield TestClient(app, raise_server_exceptions=False)
        finally:
            app.dependency_overrides.clear()

    def test_happy_path(self):
        with self._patched_client(_make_mock_vanna()) as client:
            resp = client.post("/api/v1/chat", json={"question": "Return rate by genre?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sql"] == _MOCK_SQL
        assert data["summary"] == "Good summary"
        assert data["follow_up"] == "Follow-up?"
        assert data["error"] is None

    def test_vanna_raises_returns_503(self):
        vn = MagicMock()
        vn.generate_sql.side_effect = RuntimeError("vanna down")
        with self._patched_client(vn) as client:
            resp = client.post("/api/v1/chat", json={"question": "anything"})
        assert resp.status_code == 503

    def test_empty_sql_returns_error_field(self):
        with self._patched_client(_make_mock_vanna(sql="")) as client:
            resp = client.post("/api/v1/chat", json={"question": "anything"})
        assert resp.status_code == 200
        assert resp.json()["error"]

    def test_non_select_sql_returns_error_field(self):
        vn = _make_mock_vanna(sql="DROP TABLE titles")
        app.dependency_overrides[get_vanna_dep] = lambda: vn
        try:
            with patch("ep4_nlsql.api.routes.run_query", side_effect=ValueError("Only SELECT")):
                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.post("/api/v1/chat", json={"question": "drop it"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert "Only SELECT" in resp.json()["error"]

    def test_query_exception_returns_502(self):
        vn = _make_mock_vanna()
        app.dependency_overrides[get_vanna_dep] = lambda: vn
        try:
            with patch("ep4_nlsql.api.routes.run_query", side_effect=Exception("db dead")):
                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.post("/api/v1/chat", json={"question": "q"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 502

    def test_result_includes_table_rows(self):
        with self._patched_client(_make_mock_vanna()) as client:
            resp = client.post("/api/v1/chat", json={"question": "genre return rate"})
        data = resp.json()
        assert isinstance(data["result"], list)
        assert len(data["result"]) == 2


# ---------------------------------------------------------------------------
# TestSummarizer
# ---------------------------------------------------------------------------

class TestSummarizer:
    def test_empty_df_skips_llm(self):
        import asyncio
        from ep4_nlsql.pipeline.summarizer import summarize

        summary, follow_up = asyncio.run(summarize("q", "SELECT 1", pd.DataFrame()))
        assert "No results" in summary
        assert follow_up

    def test_summarize_calls_claude(self, sample_df):
        import asyncio
        from ep4_nlsql.pipeline.summarizer import _client, summarize

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "report_summary"
        tool_use_block.input = {"summary": "Test summary", "follow_up": "Test follow-up"}

        mock_response = MagicMock()
        mock_response.content = [tool_use_block]

        with patch.object(_client.messages, "create", new=AsyncMock(return_value=mock_response)):
            summary, follow_up = asyncio.run(summarize("question?", "SELECT 1", sample_df))

        assert summary == "Test summary"
        assert follow_up == "Test follow-up"

    def test_summarize_raises_when_no_tool_block(self, sample_df):
        import asyncio
        from ep4_nlsql.pipeline.summarizer import _client, summarize

        text_block = MagicMock()
        text_block.type = "text"

        mock_response = MagicMock()
        mock_response.content = [text_block]

        with patch.object(_client.messages, "create", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(RuntimeError, match="report_summary"):
                asyncio.run(summarize("q", "SELECT 1", sample_df))


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineIntegration:
    def test_full_pipeline_live(self):
        """End-to-end test with real Claude + real DuckDB. Requires ANTHROPIC_API_KEY."""
        from ep4_nlsql.pipeline.query_engine import run_query
        from ep4_nlsql.pipeline.vanna_setup import get_vanna
        import asyncio
        from ep4_nlsql.pipeline.summarizer import summarize
        from ep4_nlsql.pipeline.chart_builder import build_chart

        vn = get_vanna()
        sql = vn.generate_sql("Which genre has the highest total revenue?")
        assert sql and sql.strip()

        df = run_query(sql)
        assert not df.empty

        chart = build_chart(df)
        # chart may be empty or non-empty — just verify it's a string
        assert isinstance(chart, str)

        summary, follow_up = asyncio.run(
            summarize("Which genre has the highest total revenue?", sql, df)
        )
        assert summary
        assert follow_up
