"""Schema DDL and Vanna training Q&A pairs for the movie catalog."""

DDL = """
CREATE TABLE titles (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    genre        TEXT    NOT NULL,
    release_year INTEGER NOT NULL
);

CREATE TABLE performance (
    id          INTEGER PRIMARY KEY,
    title_id    INTEGER NOT NULL REFERENCES titles(id),
    region      TEXT    NOT NULL,
    quarter     TEXT    NOT NULL,
    year        INTEGER NOT NULL,
    revenue     REAL    NOT NULL,
    return_rate REAL    NOT NULL
);
"""

# Ground-truth Q&A pairs used to train Vanna's vector index.
QA_PAIRS = [
    {
        "question": "Which genres have the highest average return rate?",
        "sql": (
            "SELECT t.genre, AVG(p.return_rate) AS avg_return_rate "
            "FROM performance p "
            "JOIN titles t ON t.id = p.title_id "
            "GROUP BY t.genre "
            "ORDER BY avg_return_rate DESC"
        ),
    },
    {
        "question": "Show me the top 10 titles by total revenue",
        "sql": (
            "SELECT t.name, SUM(p.revenue) AS total_revenue "
            "FROM performance p "
            "JOIN titles t ON t.id = p.title_id "
            "GROUP BY t.name "
            "ORDER BY total_revenue DESC "
            "LIMIT 10"
        ),
    },
    {
        "question": "What is the total revenue by region?",
        "sql": (
            "SELECT region, SUM(revenue) AS total_revenue "
            "FROM performance "
            "GROUP BY region "
            "ORDER BY total_revenue DESC"
        ),
    },
    {
        "question": "Show quarterly revenue trends",
        "sql": (
            "SELECT year, quarter, SUM(revenue) AS total_revenue "
            "FROM performance "
            "GROUP BY year, quarter "
            "ORDER BY year, quarter"
        ),
    },
    {
        "question": "Which titles have the highest average return rate?",
        "sql": (
            "SELECT t.name, AVG(p.return_rate) AS avg_return_rate "
            "FROM performance p "
            "JOIN titles t ON t.id = p.title_id "
            "GROUP BY t.name "
            "ORDER BY avg_return_rate DESC "
            "LIMIT 10"
        ),
    },
    {
        "question": "How many titles are in each genre?",
        "sql": (
            "SELECT genre, COUNT(*) AS title_count "
            "FROM titles "
            "GROUP BY genre "
            "ORDER BY title_count DESC"
        ),
    },
    {
        "question": "Show performance in EMEA by genre",
        "sql": (
            "SELECT t.genre, SUM(p.revenue) AS total_revenue, "
            "AVG(p.return_rate) AS avg_return_rate "
            "FROM performance p "
            "JOIN titles t ON t.id = p.title_id "
            "WHERE p.region = 'EMEA' "
            "GROUP BY t.genre "
            "ORDER BY total_revenue DESC"
        ),
    },
    {
        "question": "What is the average revenue per title by genre?",
        "sql": (
            "SELECT t.genre, AVG(p.revenue) AS avg_revenue "
            "FROM performance p "
            "JOIN titles t ON t.id = p.title_id "
            "GROUP BY t.genre "
            "ORDER BY avg_revenue DESC"
        ),
    },
    {
        "question": "Show me Q1 revenue by region",
        "sql": (
            "SELECT region, SUM(revenue) AS q1_revenue "
            "FROM performance "
            "WHERE quarter = 'Q1' "
            "GROUP BY region "
            "ORDER BY q1_revenue DESC"
        ),
    },
    {
        "question": "Which titles were released in 2023?",
        "sql": (
            "SELECT name, genre, release_year "
            "FROM titles "
            "WHERE release_year = 2023 "
            "ORDER BY name"
        ),
    },
]
