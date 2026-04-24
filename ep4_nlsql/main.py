"""FastAPI application entry point for ep4 NL-to-SQL."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ep4_nlsql.api.routes import router

_UI_PATH = Path(__file__).resolve().parent / "ui" / "index.html"

app = FastAPI(
    title="Movie Catalog NL-to-SQL API",
    description=(
        "Ask natural language questions about a synthetic movie performance catalog. "
        "Vanna translates them to SQL, DuckDB executes the query, and Claude summarises the results."
    ),
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/ui", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(_UI_PATH, media_type="text/html")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_UI_PATH, media_type="text/html")
