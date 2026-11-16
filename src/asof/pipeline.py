"""End-to-end question answering: `ask(question, as_of=..., mode=...) -> AskResult`.

mode "asof" (the system):
    understand -> embed best query -> hybrid_search(as_of, refs) -> rerank top 30 -> top 8
    -> generate_answer
mode "baseline" (naive RAG control group for the evaluation):
    no query understanding, embed the raw question, vector_search top 8 with no date filter,
    no rerank -> the same generate_answer. Its plan uses the caller's as_of if given (only to tell
    the model which date is asked about), else today.

Clients (LLM, embedder, reranker) are created lazily once per Settings object and reused;
database connections come from a small lazily-opened pool. Tests inject fakes instead.
Infrastructure failures surface as `ServiceUnavailable` (the API turns them into HTTP 503).
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, TypeVar

from asof.config import Settings, get_settings
from asof.generation.answer import generate_answer
from asof.models import AskResult, Hit, Mode, QueryPlan
from asof.retrieval.search import hybrid_search, vector_search
from asof.retrieval.understand import understand

log = logging.getLogger(__name__)

RETRIEVE_K = 30  # candidates fused by hybrid search and sent to the reranker
FINAL_K = 8  # sources shown to the answer model

T = TypeVar("T")


class ServiceUnavailable(RuntimeError):
    """A dependency (database or model provider) is down, misconfigured or rate-limited."""

    def __init__(self, component: str, message: str) -> None:
        super().__init__(f"{component} unavailable: {message}")
        self.component = component  # "database" | "model"


# ---------------------------------------------------------------- clients

@dataclass
class Clients:
    llm: Any
    embedder: Any
    reranker: Any


_clients: dict[int, tuple[Settings, Clients]] = {}
_pools: dict[str, Any] = {}
_lock = threading.Lock()


def get_clients(settings: Settings) -> Clients:
    """Build (once per Settings instance) and reuse the model clients."""
    with _lock:
        cached = _clients.get(id(settings))
        if cached and cached[0] is settings:
            return cached[1]
        from asof.providers import LLM, Embedder, make_reranker

        llm = LLM(settings)
        clients = Clients(llm=llm, embedder=Embedder(settings), reranker=make_reranker(settings))
        _clients[id(settings)] = (settings, clients)
        return clients


def _get_pool(settings: Settings):
    with _lock:
        pool = _pools.get(settings.database_url)
        if pool is None:
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                settings.database_url,
                min_size=1,
                max_size=4,
                timeout=15,
                kwargs={"autocommit": True, "prepare_threshold": None},
                configure=register_vector,
                check=ConnectionPool.check_connection,  # Neon drops idle connections
                open=True,
                name="asof-query",
            )
            _pools[settings.database_url] = pool
            atexit.register(pool.close)  # stop worker threads cleanly at interpreter exit
        return pool


@contextmanager
def _connection(settings: Settings, conn) -> Iterator[Any]:
    if conn is not None:
        yield conn
        return
    try:
        cm = _get_pool(settings).connection()
        c = cm.__enter__()
    except Exception as err:  # noqa: BLE001 - pool timeout, auth, DNS ...
        raise ServiceUnavailable("database", f"{type(err).__name__}: {err}") from err
    try:
        yield c
    except BaseException as err:
        cm.__exit__(type(err), err, err.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def _is_db_error(err: BaseException) -> bool:
    try:
        import psycopg
        from psycopg_pool import PoolTimeout
    except ImportError:  # pragma: no cover
        return False
    return isinstance(err, psycopg.OperationalError | psycopg.InterfaceError | PoolTimeout)


def _is_model_error(err: BaseException) -> bool:
    import httpx
    import openai

    # RuntimeError covers LLMError (empty/invalid output after retries) and a missing API key.
    return isinstance(err, openai.OpenAIError | httpx.HTTPError | RuntimeError)


def _guard(component: str, fn: Callable[[], T]) -> T:
    """Run a step; map infrastructure failures to ServiceUnavailable(component).

    Anything else (SQL errors, bugs) propagates unchanged and becomes a 500.
    """
    try:
        return fn()
    except ServiceUnavailable:
        raise
    except Exception as err:
        check = _is_db_error if component == "database" else _is_model_error
        if not check(err):
            raise
        raise ServiceUnavailable(component, f"{type(err).__name__}: {err}") from err


# ---------------------------------------------------------------- helpers

def rerank_passage(hit: Hit) -> str:
    head = " ".join(
        x for x in [hit.kind, hit.number or "", hit.section_ref or "",
                    f"({hit.issue_date.isoformat()})" if hit.issue_date else ""] if x
    )
    return f"{head}\n{hit.text}"


def apply_rerank(reranker, question: str, hits: list[Hit], top_n: int) -> list[Hit]:
    """Reorder by reranker score; on any reranker failure keep retrieval order."""
    if not hits:
        return []
    try:
        results = reranker.rerank(question, [rerank_passage(h) for h in hits], top_n=top_n)
    except Exception as err:  # noqa: BLE001 - degrade, don't fail the query
        log.warning("rerank failed (%s: %s); keeping retrieval order", type(err).__name__, err)
        return hits[:top_n]
    out: list[Hit] = []
    seen: set[int] = set()
    for r in results:
        if 0 <= r.index < len(hits) and r.index not in seen:
            seen.add(r.index)
            out.append(hits[r.index].model_copy(update={"score": float(r.score)}))
    return out[:top_n]


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def step(self, name: str, fn: Callable[[], T]) -> T:
        t = time.perf_counter()
        try:
            return fn()
        finally:
            self.timings[name] = round((time.perf_counter() - t) * 1000, 1)

    def total(self) -> dict[str, float]:
        return {**self.timings, "total": round((time.perf_counter() - self._t0) * 1000, 1)}


# ---------------------------------------------------------------- entry point

def ask(
    question: str,
    *,
    as_of: date | None = None,
    mode: Mode = "asof",
    settings: Settings | None = None,
    conn=None,
    today: date | None = None,
    llm=None,
    embedder=None,
    reranker=None,
    thinking: bool = False,
) -> AskResult:
    """Answer `question` as of `as_of` (default: the date in the question, else today).

    `thinking=True` lets the answer model reason first (slower, see generate_answer).
    """
    if mode not in ("asof", "baseline"):
        raise ValueError(f"unknown mode {mode!r}")
    question = question.strip()
    today = today or date.today()
    settings = settings or get_settings()
    if llm is None or embedder is None or (reranker is None and mode == "asof"):
        clients = _guard("model", lambda: get_clients(settings))
        llm = llm or clients.llm
        embedder = embedder or clients.embedder
        reranker = reranker or clients.reranker

    timer = _Timer()
    if mode == "asof":
        plan = timer.step("understand", lambda: understand(llm, question, today, as_of))
        best_query = plan.search_queries[0] if plan.search_queries else question
        vec = timer.step("embed", lambda: _guard("model", lambda: embedder.embed_query(best_query)))
        fts_text = " ".join([question, *plan.search_queries[:3]])
        with _connection(settings, conn) as c:
            candidates = timer.step("search", lambda: _guard("database", lambda: hybrid_search(
                c, vec, fts_text, as_of=plan.as_of, refs=plan.refs, k=RETRIEVE_K)))
        hits = timer.step("rerank", lambda: apply_rerank(reranker, question, candidates, FINAL_K))
    else:
        plan = QueryPlan(
            as_of=as_of or today,
            as_of_explicit=as_of is not None,
            question_type="other",
            refs=[],
            search_queries=[question],
        )
        vec = timer.step("embed", lambda: _guard("model", lambda: embedder.embed_query(question)))
        with _connection(settings, conn) as c:
            hits = timer.step("search", lambda: _guard("database", lambda: vector_search(
                c, vec, k=FINAL_K)))

    answer = timer.step("generate", lambda: _guard("model", lambda: generate_answer(
        llm, question, plan.as_of, hits, mode=mode, today=today, thinking=thinking)))
    return AskResult(question=question, mode=mode, plan=plan, hits=hits, answer=answer,
                     timings_ms=timer.total())
