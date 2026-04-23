from __future__ import annotations

from ep2_structured_api.api.models import EnrichmentRequest

SYSTEM_PROMPT = (
    "You are a film and media metadata expert with deep knowledge of movies, TV series, "
    "documentaries, and distribution contexts. "
    "When given a title and a context, you enrich it with accurate genres, a concise summary, "
    "and an honest confidence score. "
    "If the title is ambiguous, obscure, or matches multiple works, lower the confidence score "
    "and populate the warnings field accordingly. "
    "Do not invent information you are not confident about."
)

ENRICH_TOOL_DESCRIPTION = (
    "Enrich a film or media title with structured metadata. "
    "Provide the canonical title, a list of genres (at least one), a concise one-to-two sentence "
    "summary, a confidence score between 0 (no match) and 1 (certain match), "
    "and any warnings if the result is uncertain or ambiguous."
)


def build_enrich_prompt(request: EnrichmentRequest) -> str:
    return (
        f"Enrich the following title with structured metadata.\n\n"
        f"Title: {request.title}\n"
        f"Context: {request.context}\n\n"
        "Return the canonical title, genres, a brief summary, a confidence score, "
        "and any warnings if you are uncertain or if the title matches multiple works."
    )
