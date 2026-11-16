from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from asof import __version__
from asof.api import site
from asof.config import get_settings
from asof.db import connect

app = FastAPI(
    title=f"{site.NAME}: {site.TAGLINE}",
    version=__version__,
    summary="Time-aware (point-in-time) retrieval-augmented generation over Indian GST law.",
    description=site.DESCRIPTION,
    openapi_tags=[
        {"name": "site", "description": "Public landing page and crawler files."},
        {"name": "ops", "description": "Health and readiness checks."},
    ],
)


def _base_url(request: Request) -> str:
    configured = get_settings().public_base_url
    base = configured or str(request.base_url)
    return base if base.endswith("/") else base + "/"


@app.get("/", response_class=HTMLResponse, tags=["site"], include_in_schema=False)
def landing(request: Request) -> HTMLResponse:
    return HTMLResponse(site.landing_html(_base_url(request)))


@app.get("/robots.txt", response_class=PlainTextResponse, tags=["site"], include_in_schema=False)
def robots(request: Request) -> str:
    return site.robots_txt(_base_url(request))


@app.get("/sitemap.xml", tags=["site"], include_in_schema=False)
def sitemap(request: Request) -> Response:
    return Response(site.sitemap_xml(_base_url(request)), media_type="application/xml")


@app.get("/llms.txt", response_class=PlainTextResponse, tags=["site"], include_in_schema=False)
def llms(request: Request) -> str:
    return site.llms_txt(_base_url(request))


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness for Render. Kept dependency-free so a DB blip doesn't restart the service."""
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["ops"])
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
            "rerank": s.openrouter_rerank_model if s.rerank_backend == "openrouter"
            else s.rerank_backend,
        },
    }
