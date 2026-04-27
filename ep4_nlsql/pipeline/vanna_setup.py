"""Vanna NL-to-SQL setup: CatalogVanna class, singleton, and training bootstrap.

Run once after creating the database:
    python -m ep4_nlsql.pipeline.vanna_setup
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

from dotenv import load_dotenv
from vanna.legacy.anthropic.anthropic_chat import Anthropic_Chat
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore

from ep4_nlsql.data.schema import DDL, QA_PAIRS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL     = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
_CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


class CatalogVanna(ChromaDB_VectorStore, Anthropic_Chat):
    """Vanna instance backed by ChromaDB (vector store) + Anthropic Claude (LLM)."""

    def __init__(self, config: dict) -> None:
        ChromaDB_VectorStore.__init__(self, config=config)
        Anthropic_Chat.__init__(self, config=config)


@functools.lru_cache(maxsize=1)
def get_vanna() -> CatalogVanna:
    """Return the shared CatalogVanna singleton (created once, then cached)."""
    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    config = {
        "api_key": API_KEY,
        "model":   MODEL,
        "path":    str(_CHROMA_PATH),
    }
    return CatalogVanna(config=config)


def train() -> None:
    """Train Vanna on schema DDL and Q&A pairs. Run once."""
    vn = get_vanna()

    vn.train(ddl=DDL)
    for pair in QA_PAIRS:
        vn.train(question=pair["question"], sql=pair["sql"])

    print(f"Trained Vanna on {len(QA_PAIRS)} Q&A pairs")
    print(f"ChromaDB stored at: {_CHROMA_PATH}")


if __name__ == "__main__":
    train()
