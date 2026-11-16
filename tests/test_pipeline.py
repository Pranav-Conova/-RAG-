"""Pipeline orchestration with fakes for the DB connection, embedder, reranker and LLM."""

import json
from contextlib import contextmanager
from datetime import date

import httpx
import psycopg
import pytest
from openai import OpenAI

from asof import pipeline
from asof.providers import LLM, RerankResult
from asof.retrieval import search

TODAY = date(2026, 11, 17)


def openai_client(handler) -> OpenAI:
    return OpenAI(
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    })


def row(i: int, text: str = "", valid_to=None):
    return (i, i, "notification", f"{i}/2019-Central Tax", date(2019, 10, 9), f"Doc {i}", None,
            "rule 36(4)", 1, text or f"text of chunk {i}", date(2019, 10, 9), valid_to, 0.01)


class FakeConn:
    def __init__(self, rows, fail: Exception | None = None):
        self.rows, self.fail = rows, fail
        self.sql: list[str] = []
        self.params: list = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql, params=None):
        if self.fail:
            raise self.fail
        self.sql.append(sql)
        self.params.append(params)
        return self

    def fetchall(self):
        return self.rows


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3, 0.4]


class ReverseReranker:
    def __init__(self):
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query, passages, top_n=None):
        self.calls.append((query, len(passages), top_n))
        n = len(passages)
        return [RerankResult(i, float(i)) for i in reversed(range(n))][:top_n]


class BrokenReranker:
    def rerank(self, query, passages, top_n=None):
        raise httpx.ConnectError("down")


def scripted_llm(settings, no_limit, replies: list, bodies: list):
    it = iter(replies)

    def handler(req):
        bodies.append(json.loads(req.content))
        reply = next(it)
        return reply if isinstance(reply, httpx.Response) else chat_response(json.dumps(reply))
    return LLM(settings, client=openai_client(handler), limiter=no_limit)


PLAN = {"question_type": "point_in_time",
        "search_queries": ["provisional input tax credit limit rule 36(4)"]}


def answer(source: int = 1, quote: str = "text of chunk") -> dict:
    return {"answer": "As of 2020-02-29 ... [S1]", "abstain": False,
            "citations": [{"source": source, "quote": quote}]}


def test_asof_mode_orchestration(settings, no_limit):
    bodies: list[dict] = []
    llm = scripted_llm(settings, no_limit, [PLAN, answer(1, "text of chunk 30")], bodies)
    conn = FakeConn([row(i) for i in range(1, 31)])
    emb, rr = FakeEmbedder(), ReverseReranker()

    res = pipeline.ask("What was the ITC cap under rule 36(4) in February 2020?", conn=conn,
                       settings=settings, today=TODAY, llm=llm, embedder=emb, reranker=rr)

    assert res.mode == "asof"
    assert res.plan.as_of == date(2020, 2, 29) and res.plan.refs == ["rule 36(4)"]
    assert emb.queries == ["provisional input tax credit limit rule 36(4)"]  # best query
    params = conn.params[-1]
    assert params["as_of"] == date(2020, 2, 29) and params["k"] == 30
    assert params["ref_section"] == ["rule 36(4)%"]
    assert "validity @>" in conn.sql[-1]
    assert rr.calls == [("What was the ITC cap under rule 36(4) in February 2020?", 30, 8)]
    assert [h.chunk_id for h in res.hits] == [30, 29, 28, 27, 26, 25, 24, 23]
    assert res.hits[0].score == 29.0  # rerank score replaces the fusion score
    assert res.answer.citations[0].chunk_id == 30 and res.answer.citations[0].verified
    assert set(res.timings_ms) == {"understand", "embed", "search", "rerank", "generate",
                                   "total"}
    assert len(bodies) == 2 and bodies[1]["model"] == settings.llm_model
    assert "Answer as of: 2020-02-29" in bodies[1]["messages"][-1]["content"]


def test_asof_mode_survives_llm_planning_and_rerank_failures(settings, no_limit):
    bodies: list[dict] = []
    llm = scripted_llm(settings, no_limit, [httpx.Response(500), answer()], bodies)
    conn = FakeConn([row(i) for i in range(1, 11)])

    res = pipeline.ask("ITC cap under rule 36(4) as on 15 Feb 2020?", conn=conn,
                       settings=settings, today=TODAY, llm=llm, embedder=FakeEmbedder(),
                       reranker=BrokenReranker())

    assert res.plan.as_of == date(2020, 2, 15)
    assert res.plan.search_queries == ["ITC cap under rule 36(4) as on 15 Feb 2020?"]
    assert [h.chunk_id for h in res.hits] == list(range(1, 9))  # retrieval order kept
    assert not res.answer.abstained


def test_as_of_override_wins(settings, no_limit):
    llm = scripted_llm(settings, no_limit, [PLAN, answer()], [])
    conn = FakeConn([row(1)])
    res = pipeline.ask("ITC cap in February 2020?", as_of=date(2021, 6, 1), conn=conn,
                       settings=settings, today=TODAY, llm=llm, embedder=FakeEmbedder(),
                       reranker=ReverseReranker())
    assert res.plan.as_of == date(2021, 6, 1) and conn.params[-1]["as_of"] == date(2021, 6, 1)


def test_baseline_mode_is_naive(settings, no_limit):
    bodies: list[dict] = []
    llm = scripted_llm(settings, no_limit, [answer()], bodies)  # only the answer call
    conn = FakeConn([row(i) for i in range(1, 9)])
    emb = FakeEmbedder()

    class NoRerank:
        def rerank(self, *a, **k):
            raise AssertionError("baseline must not rerank")

    res = pipeline.ask("What was the ITC cap under rule 36(4) in February 2020?",
                       mode="baseline", conn=conn, settings=settings, today=TODAY, llm=llm,
                       embedder=emb, reranker=NoRerank())

    assert res.plan.as_of == TODAY and not res.plan.as_of_explicit
    assert res.plan.question_type == "other" and res.plan.refs == []
    assert emb.queries == ["What was the ITC cap under rule 36(4) in February 2020?"]
    assert conn.sql[-1] == search.VECTOR_SQL and conn.params[-1]["k"] == 8
    assert len(res.hits) == 8 and len(bodies) == 1
    assert set(res.timings_ms) == {"embed", "search", "generate", "total"}
    assert "valid 2019" not in bodies[0]["messages"][-1]["content"]


def test_no_hits_abstains_without_answer_call(settings, no_limit):
    bodies: list[dict] = []
    llm = scripted_llm(settings, no_limit, [PLAN], bodies)
    res = pipeline.ask("ITC cap on 1 Jan 2018?", conn=FakeConn([]), settings=settings,
                       today=TODAY, llm=llm, embedder=FakeEmbedder(), reranker=ReverseReranker())
    assert res.hits == [] and res.answer.abstained and len(bodies) == 1


def test_database_outage_is_service_unavailable(settings, no_limit):
    llm = scripted_llm(settings, no_limit, [PLAN], [])
    conn = FakeConn([], fail=psycopg.OperationalError("connection refused"))
    with pytest.raises(pipeline.ServiceUnavailable) as exc:
        pipeline.ask("ITC cap?", conn=conn, settings=settings, today=TODAY, llm=llm,
                     embedder=FakeEmbedder(), reranker=ReverseReranker())
    assert exc.value.component == "database"


def test_sql_bug_is_not_masked_as_outage(settings, no_limit):
    llm = scripted_llm(settings, no_limit, [PLAN], [])
    conn = FakeConn([], fail=psycopg.errors.UndefinedColumn("no such column"))
    with pytest.raises(psycopg.errors.UndefinedColumn):
        pipeline.ask("ITC cap?", conn=conn, settings=settings, today=TODAY, llm=llm,
                     embedder=FakeEmbedder(), reranker=ReverseReranker())


def test_model_outage_is_service_unavailable(settings, no_limit):
    class DeadEmbedder:
        def embed_query(self, text):
            raise httpx.ConnectError("nvidia down")

    llm = scripted_llm(settings, no_limit, [PLAN], [])
    with pytest.raises(pipeline.ServiceUnavailable) as exc:
        pipeline.ask("ITC cap?", conn=FakeConn([]), settings=settings, today=TODAY, llm=llm,
                     embedder=DeadEmbedder(), reranker=ReverseReranker())
    assert exc.value.component == "model"


def test_clients_are_cached_per_settings(settings, monkeypatch):
    pipeline._clients.clear()
    a = pipeline.get_clients(settings)
    assert pipeline.get_clients(settings) is a
    assert a.llm.settings is settings
    pipeline._clients.clear()


def test_unknown_mode_rejected(settings):
    with pytest.raises(ValueError):
        pipeline.ask("q?", mode="nope", settings=settings)  # type: ignore[arg-type]
