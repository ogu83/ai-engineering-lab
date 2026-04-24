"""Shared fixtures for ep4 offline tests."""
import random
from pathlib import Path

import duckdb
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires ANTHROPIC_API_KEY")


def pytest_collection_modifyitems(config, items):
    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory) -> str:
    """Temp DuckDB file with 5 titles + 10 performance rows for query tests."""
    path = str(tmp_path_factory.mktemp("db") / "test.duckdb")
    random.seed(0)
    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE titles (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            genre TEXT NOT NULL,
            release_year INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE performance (
            id INTEGER PRIMARY KEY,
            title_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            quarter TEXT NOT NULL,
            year INTEGER NOT NULL,
            revenue REAL NOT NULL,
            return_rate REAL NOT NULL
        )
    """)
    conn.executemany("INSERT INTO titles VALUES (?, ?, ?, ?)", [
        (1, "Alpha",   "Action",  2022),
        (2, "Beta",    "Drama",   2023),
        (3, "Gamma",   "Comedy",  2023),
        (4, "Delta",   "Horror",  2022),
        (5, "Epsilon", "Sci-Fi",  2024),
    ])
    rows = []
    for i, title_id in enumerate([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], start=1):
        rows.append((i, title_id, "AMER", "Q1", 2023,
                     round(100_000 + i * 10_000, 2),
                     round(0.05 + i * 0.01, 4)))
    conn.executemany("INSERT INTO performance VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()
    return path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Simple genre / avg_return_rate DataFrame for chart and summarizer tests."""
    return pd.DataFrame({
        "genre":           ["Action", "Drama", "Horror", "Comedy", "Sci-Fi"],
        "avg_return_rate": [0.07,     0.05,    0.14,     0.09,     0.08],
    })
