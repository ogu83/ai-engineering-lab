"""Bootstrap: creates ep4_nlsql/data/catalog.duckdb with synthetic movie catalog data.

Run once before starting the server:
    python -m ep4_nlsql.data.create_db
"""
import random
from pathlib import Path

import duckdb

from ep4_nlsql.data.schema import DDL

DB_PATH = Path(__file__).resolve().parent / "catalog.duckdb"

_TITLES = [
    (1,  "Shadow Protocol",    "Action",      2023),
    (2,  "The Last Light",     "Drama",        2022),
    (3,  "Midnight Bloom",     "Romance",      2023),
    (4,  "Iron Circuit",       "Sci-Fi",       2021),
    (5,  "The Reckoning",      "Thriller",     2022),
    (6,  "Ghost Quarter",      "Horror",       2023),
    (7,  "Laughing at Nothing","Comedy",       2022),
    (8,  "Deep Current",       "Documentary",  2021),
    (9,  "Signal Lost",        "Action",       2024),
    (10, "Second Chance",      "Drama",        2023),
    (11, "Night Protocol",     "Thriller",     2024),
    (12, "Red Planet Rising",  "Sci-Fi",       2022),
    (13, "Still Waters",       "Documentary",  2023),
    (14, "The Hollow",         "Horror",       2024),
    (15, "Perfect Score",      "Comedy",       2023),
    (16, "Final Hour",         "Action",       2022),
    (17, "Summer Lies",        "Romance",      2024),
    (18, "Override",           "Sci-Fi",       2023),
    (19, "Whisper Network",    "Thriller",     2022),
    (20, "Dark Frequency",     "Horror",       2021),
]

_GENRE_RETURN_BASE = {
    "Action": 0.07, "Drama": 0.05, "Romance": 0.06,
    "Sci-Fi": 0.08, "Thriller": 0.07, "Horror": 0.14,
    "Comedy": 0.09, "Documentary": 0.04,
}
_REGIONS  = ["AMER", "EMEA", "APAC"]
_REGION_MULT = {"AMER": 1.0, "EMEA": 0.8, "APAC": 0.6}
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
_YEAR = 2023


def create_db(path: str = str(DB_PATH)) -> None:
    """Create and populate the catalog DuckDB database."""
    random.seed(42)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    try:
        for stmt in DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

        conn.executemany("INSERT INTO titles VALUES (?, ?, ?, ?)", _TITLES)

        perf_rows = []
        perf_id = 1
        for title_id, _, genre, _ in _TITLES:
            base_rev  = random.uniform(100_000, 400_000)
            base_rate = _GENRE_RETURN_BASE[genre]
            for region in _REGIONS:
                for quarter in _QUARTERS:
                    revenue     = base_rev * _REGION_MULT[region] * random.uniform(0.8, 1.2)
                    return_rate = base_rate * random.uniform(0.7, 1.3)
                    perf_rows.append((
                        perf_id, title_id, region, quarter, _YEAR,
                        round(revenue, 2), round(return_rate, 4),
                    ))
                    perf_id += 1

        conn.executemany(
            "INSERT INTO performance VALUES (?, ?, ?, ?, ?, ?, ?)", perf_rows
        )
    finally:
        conn.close()


if __name__ == "__main__":
    create_db()
    n_perf = len(_TITLES) * len(_REGIONS) * len(_QUARTERS)
    print(f"Created {DB_PATH}")
    print(f"  {len(_TITLES)} titles, {n_perf} performance rows")
