"""Mock document store for the Episode 1 demo pipeline.

In a real system this would be a vector database, search index, or SQL store.
Here it's a list of text passages about Q4 product performance and return rates.
"""

from __future__ import annotations

DOC_STORE: list[dict] = [
    {
        "id": "q4-returns-summary",
        "text": (
            "Q4 returns summary: Overall return rate across all categories was 8.3%, up 1.1 pp "
            "year-over-year. Electronics led all categories with a 12.4% return rate, followed by "
            "Apparel at 10.8%. Home & Garden had the lowest return rate at 3.2%."
        ),
    },
    {
        "id": "electronics-detail",
        "text": (
            "Electronics category detail: The 12.4% Q4 return rate was driven primarily by "
            "smartphones (18.1%) and laptops (14.7%). Accessories had a much lower return rate "
            "of 5.9%. Top return reasons: 'changed mind' (34%), 'defective product' (28%), "
            "'not as described' (22%)."
        ),
    },
    {
        "id": "apparel-detail",
        "text": (
            "Apparel category detail: Return rate was 10.8% in Q4, consistent with the seasonal "
            "average of 10–11%. Women's clothing accounted for 61% of returns. Sizing issues "
            "were cited in 47% of apparel returns, followed by color mismatch (19%)."
        ),
    },
    {
        "id": "home-garden-detail",
        "text": (
            "Home & Garden Q4 performance: Category return rate was 3.2%, the lowest across all "
            "tracked categories. Furniture returns (5.1%) were the highest subcategory, while "
            "garden tools had a near-zero return rate of 0.8%. High satisfaction scores "
            "correlated with detailed product descriptions."
        ),
    },
    {
        "id": "sports-outdoor-detail",
        "text": (
            "Sports & Outdoors Q4 returns: Return rate was 6.7%, slightly above Q3 (5.9%). "
            "Fitness equipment accounted for 40% of category returns. Return spike in December "
            "attributed to holiday gift mismatches. No quality defect trend identified."
        ),
    },
    {
        "id": "books-media-detail",
        "text": (
            "Books & Media Q4 returns: Return rate was 4.1%. Digital media had a 0% return rate "
            "by policy. Physical books returned at 5.3%, mostly citing 'duplicate purchase' (41%) "
            "and 'wrong edition' (29%). Audiobooks returned at 2.1%."
        ),
    },
    {
        "id": "yoy-comparison",
        "text": (
            "Year-over-year return rate comparison: Electronics +2.1 pp, Apparel -0.3 pp, "
            "Home & Garden -0.5 pp, Sports & Outdoors +0.8 pp, Books & Media +0.1 pp. "
            "Electronics was the only category with a return rate increase above 1 pp."
        ),
    },
    {
        "id": "defect-analysis",
        "text": (
            "Defect-driven returns analysis: Products returned due to defects averaged a 28% "
            "defect rate across all categories. Electronics defect returns cost the business "
            "an estimated $2.3M in Q4 restocking and refurbishment fees. "
            "Apparel defect returns were 8% of total apparel returns."
        ),
    },
    {
        "id": "regional-breakdown",
        "text": (
            "Regional return rate breakdown Q4: West region had the highest overall return rate "
            "at 9.7%, North-East at 8.9%, South at 7.4%, Midwest at 6.8%. Electronics return "
            "rates were consistently high across all regions, ranging from 11.1% to 13.8%."
        ),
    },
    {
        "id": "recommendations",
        "text": (
            "Q4 return reduction recommendations: (1) Improve Electronics product descriptions "
            "and compatibility checkers — could reduce 'not as described' returns by est. 30%. "
            "(2) Introduce Apparel virtual try-on — industry data suggests 20–25% reduction in "
            "sizing-related returns. (3) Extend return window for Home & Garden — low current "
            "return rate suggests low risk, potential loyalty upside."
        ),
    },
]


_STOPWORDS = {"the", "a", "an", "is", "are", "was", "in", "of", "and", "or", "to", "for", "by"}


def search(doc_store: list[dict], query: str, feedback: str = "", scope: str = "focused") -> list[str]:
    """Keyword search over a list of document dicts.

    Args:
        doc_store: List of ``{"id": str, "text": str}`` dicts.
        query: Search query string.
        feedback: Optional QA feedback from a previous round — enriches the keyword set.
        scope: ``"focused"`` returns top 3 results; ``"broad"`` returns top 5.

    Returns:
        List of matching passage texts, ranked by keyword overlap.
    """
    top_k = 5 if scope == "broad" else 3

    combined = f"{query} {feedback}".lower()
    keywords = {w for w in combined.split() if len(w) > 3 and w not in _STOPWORDS}

    scored: list[tuple[int, str]] = []
    for doc in doc_store:
        text_lower = doc["text"].lower()
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scored.append((score, doc["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]
