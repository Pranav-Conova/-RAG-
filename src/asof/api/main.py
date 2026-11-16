from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from asof import __version__, pipeline
from asof.api import site
from asof.config import get_settings
from asof.db import connect
from asof.models import AskResult, Mode

STATIC_DIR = Path(__file__).parent / "static"
GST_START = date(2017, 7, 1)

app = FastAPI(
    title=f"{site.NAME}: {site.TAGLINE}",
    version=__version__,
    summary="Time-aware (point-in-time) retrieval-augmented generation over Indian GST law.",
    description=site.DESCRIPTION,
    openapi_tags=[
        {"name": "site", "description": "Public landing page and crawler files."},
        {"name": "ops", "description": "Health and readiness checks."},
        {"name": "ask", "description": "Point-in-time question answering with citations."},
    ],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


# ---------------------------------------------------------------- question answering

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["What was the ITC cap under rule 36(4) in February 2020?"])
    as_of: date | None = Field(
        default=None, description="Answer as the law stood on this date (default: the date in "
        "the question, else today). Must be between 2017-07-01 and today.")
    mode: Mode = Field(default="asof", description='"asof" (time-aware) or "baseline" '
                       "(naive RAG control group)")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question must have at least 3 non-space characters")
        return v


@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed input is a 400 (not FastAPI's default 422)."""
    errors = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=400, content={"detail": errors})


@app.post(
    "/api/ask",
    response_model=AskResult,
    tags=["ask"],
    responses={400: {"description": "Invalid input"},
               503: {"description": "Database or model provider unavailable"}},
)
def api_ask(req: AskRequest) -> AskResult:
    """Answer a GST question as the law stood on a date, with verified citations."""
    today = date.today()
    if req.as_of is not None and not GST_START <= req.as_of <= today:
        raise HTTPException(
            400, f"as_of must be between {GST_START.isoformat()} and {today.isoformat()}")
    try:
        return pipeline.ask(req.question, as_of=req.as_of, mode=req.mode, today=today)
    except pipeline.ServiceUnavailable as err:
        what = "The database" if err.component == "database" else "The language model service"
        raise HTTPException(
            503, f"{what} is unavailable right now; please try again shortly.") from err
