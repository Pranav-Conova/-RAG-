"""Rerankers. Pick one with RERANK_BACKEND:

- "openrouter": hosted cross-encoder via OpenRouter (default model is NVIDIA's free
                nemotron-3.5-rerank-8b). Falls back to "llm" on errors / daily-limit hits.
- "llm":        the fast LLM scores passages 0-10 in one call.
- "nim":        a hosted NVIDIA cross-encoder (/reranking endpoint), when one is available.
- "none":       keep retrieval order (useful as an ablation baseline).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from asof.config import Settings
from asof.providers.llm import LLM
from asof.providers.ratelimit import RateLimiter, shared_limiter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    index: int  # position in the passages list you passed in
    score: float  # higher = more relevant; scale depends on the backend


class Reranker(Protocol):
    def rerank(
        self, query: str, passages: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]: ...


def _finish(results: list[RerankResult], top_n: int | None) -> list[RerankResult]:
    # sort() is stable, so ties keep retrieval order
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n] if top_n else results


class NoopReranker:
    def rerank(self, query, passages, top_n=None):  # noqa: ANN001
        return _finish([RerankResult(i, 0.0) for i in range(len(passages))], top_n)


# ---------------------------------------------------------------- LLM reranker

class _Score(BaseModel):
    id: int
    score: float = Field(ge=0, le=10)


class _Scores(BaseModel):
    scores: list[_Score]


_RERANK_PROMPT = """You are ranking search results for a question about Indian GST law.

Question: {query}

Score every passage from 0 to 10 for how directly it helps answer the question:
10 = states the answer, 6 = clearly relevant context, 3 = same topic only, 0 = unrelated.
Judge relevance only; do not penalise a passage for being old or amended.
Return one entry per passage id.

Passages:
{passages}"""


class LLMReranker:
    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        max_chars: int = 1200,
        batch_size: int = 25,
    ) -> None:
        self.settings = settings
        self.llm = llm or LLM(settings)
        self.max_chars = max_chars
        self.batch_size = batch_size

    def rerank(
        self, query: str, passages: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        results: list[RerankResult] = []
        for offset in range(0, len(passages), self.batch_size):
            batch = passages[offset : offset + self.batch_size]
            listing = "\n\n".join(
                f"[{i}] {p[: self.max_chars]}" for i, p in enumerate(batch)
            )
            out = self.llm.structured(
                [{"role": "user", "content": _RERANK_PROMPT.format(query=query, passages=listing)}],
                _Scores,
                model=self.settings.llm_fast_model,
            )
            scored = {s.id: s.score for s in out.scores if 0 <= s.id < len(batch)}
            # Passages the model skipped sink to the bottom instead of failing the query.
            results += [RerankResult(offset + i, scored.get(i, -1.0)) for i in range(len(batch))]
        return _finish(results, top_n)


# ---------------------------------------------------------------- NVIDIA hosted reranker

MAX_PASSAGES_PER_CALL = 512


def _is_retryable(err: BaseException) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code == 429 or err.response.status_code >= 500
    return isinstance(err, httpx.TransportError)


class NimReranker:
    """Cross-encoder via the NVIDIA API catalog retrieval endpoint (scores are raw logits)."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.settings = settings
        self.url = settings.resolved_rerank_url
        self.http = http or httpx.Client(timeout=settings.request_timeout_s)
        self.headers = {
            "Authorization": f"Bearer {settings.require_api_key()}",
            "Accept": "application/json",
        }
        self.limiter = limiter or shared_limiter(settings.nim_requests_per_minute)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _call(self, query: str, passages: Sequence[str]) -> list[dict]:
        self.limiter.acquire()
        resp = self.http.post(
            self.url,
            headers=self.headers,
            json={
                "model": self.settings.rerank_model,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
                "truncate": "END",
            },
        )
        resp.raise_for_status()
        return resp.json()["rankings"]

    def rerank(
        self, query: str, passages: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        results: list[RerankResult] = []
        for offset in range(0, len(passages), MAX_PASSAGES_PER_CALL):
            batch = passages[offset : offset + MAX_PASSAGES_PER_CALL]
            for r in self._call(query, batch):
                results.append(RerankResult(index=offset + r["index"], score=float(r["logit"])))
        return _finish(results, top_n)


# ---------------------------------------------------------------- OpenRouter reranker

def _is_transient(err: BaseException) -> bool:
    # Not 429: on OpenRouter's free tier that is usually the *daily* cap, so retrying just
    # burns time. Let the fallback reranker take over instead.
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code >= 500
    return isinstance(err, httpx.TransportError)


class OpenRouterReranker:
    """POST {base}/rerank with Cohere-style results: [{index, relevance_score}]."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        max_chars: int = 4000,  # Nemotron rerank allows ~10K tokens per query+document pair
    ) -> None:
        key = settings.openrouter_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (needed for RERANK_BACKEND=openrouter)"
            )
        self.settings = settings
        self.url = f"{settings.openrouter_base_url.rstrip('/')}/rerank"
        self.http = http or httpx.Client(timeout=settings.request_timeout_s)
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        # Separate provider, separate budget (free models: 20 req/min).
        self.limiter = limiter or RateLimiter(settings.openrouter_requests_per_minute)
        self.max_chars = max_chars

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _call(self, query: str, passages: Sequence[str]) -> list[dict]:
        self.limiter.acquire()
        resp = self.http.post(
            self.url,
            headers=self.headers,
            json={
                "model": self.settings.openrouter_rerank_model,
                "query": query,
                "documents": [p[: self.max_chars] for p in passages],
                "top_n": len(passages),
            },
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def rerank(
        self, query: str, passages: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not passages:
            return []
        scored = {
            r["index"]: float(r["relevance_score"])
            for r in self._call(query, passages)
            if 0 <= r["index"] < len(passages)
        }
        # Anything the API didn't score sinks below every scored passage.
        floor = min(scored.values(), default=0.0) - 1.0
        return _finish([RerankResult(i, scored.get(i, floor)) for i in range(len(passages))],
                       top_n)


class FallbackReranker:
    """Try `primary`; on any error use `fallback`. `last_used` records which one answered."""

    def __init__(self, primary: Reranker, fallback: Reranker) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_used: str = ""

    def rerank(
        self, query: str, passages: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        try:
            out = self.primary.rerank(query, passages, top_n)
            self.last_used = type(self.primary).__name__
            return out
        except Exception as err:  # noqa: BLE001 - any failure -> degrade, don't fail the query
            log.warning("primary reranker failed (%s: %s); using fallback",
                        type(err).__name__, err)
            self.last_used = type(self.fallback).__name__
            return self.fallback.rerank(query, passages, top_n)


def make_reranker(settings: Settings, *, fallback: bool = True) -> Reranker:
    match settings.rerank_backend:
        case "openrouter":
            primary = OpenRouterReranker(settings)
            return FallbackReranker(primary, LLMReranker(settings)) if fallback else primary
        case "llm":
            return LLMReranker(settings)
        case "nim":
            return NimReranker(settings)
        case "none":
            return NoopReranker()
    raise ValueError(f"unknown RERANK_BACKEND {settings.rerank_backend!r}")
