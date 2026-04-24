"""DuckDB query execution with SELECT guard and row limit."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "catalog.duckdb"
MAX_ROWS = 500


def run_query(sql: str, db_path: str | None = None) -> pd.DataFrame:
    """Execute a SELECT query against the catalog DuckDB database.

    Guards applied in order:
    1. SQL type check — only SELECT or WITH…SELECT allowed.
    2. DB existence check — raises FileNotFoundError with setup hint.
    3. DuckDB read_only=True — writes physically rejected at the driver level.
    4. Row cap — results truncated to MAX_ROWS rows.

    Args:
        sql: The SQL query to execute.
        db_path: Path to the DuckDB file. Defaults to ep4_nlsql/data/catalog.duckdb.

    Returns:
        DataFrame of results (at most MAX_ROWS rows).

    Raises:
        ValueError: If the SQL is not a SELECT or WITH query.
        FileNotFoundError: If the database file does not exist.
    """
    path = db_path or str(_DEFAULT_DB)

    stripped = sql.strip().upper()
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        raise ValueError(f"Only SELECT queries are allowed; got: {sql[:60]!r}")

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Database not found at {path!r}. "
            "Run: python -m ep4_nlsql.data.create_db"
        )

    with duckdb.connect(path, read_only=True) as conn:
        df = conn.execute(sql).df()

    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)

    return df
