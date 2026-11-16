from fastapi import FastAPI

from asof import __version__
from asof.config import get_settings
from asof.db import connect

app = FastAPI(title="AsOf", version=__version__,
              description="Time-aware RAG over Indian GST law")


@app.get("/health")
def health() -> dict:
    """Liveness for Render. Kept dependency-free so a DB blip doesn't restart the service."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    """Readiness: checks DB connectivity and reports the configured models."""
    s = get_settings()
    db_ok, db_error = True, None
    try:
        with connect(s, register=False) as conn:
            conn.execute("SELECT 1")
    except Exception as err:  # noqa: BLE001 - surfaced to the caller
        db_ok, db_error = False, type(err).__name__
    return {
        "db": "ok" if db_ok else f"error: {db_error}",
        "models": {
            "llm": s.llm_model,
            "llm_fast": s.llm_fast_model,
            "embed": f"{s.embed_model} ({s.embed_dim}d)",
            "rerank": s.rerank_model,
        },
    }
