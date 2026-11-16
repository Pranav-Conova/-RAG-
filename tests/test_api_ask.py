"""POST /api/ask with the pipeline monkeypatched (no DB, no network)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from asof import pipeline
from asof.api.main import app
from asof.models import Answer, AskResult, QueryPlan

client = TestClient(app)


@pytest.fixture
def calls(monkeypatch):
    seen: list[dict] = []

    def fake_ask(question, *, as_of=None, mode="asof", today=None, **kw):
        seen.append({"question": question, "as_of": as_of, "mode": mode})
        d = as_of or date(2020, 2, 29)
        return AskResult(
            question=question, mode=mode,
            plan=QueryPlan(as_of=d, as_of_explicit=True, question_type="point_in_time",
                           refs=["rule 36(4)"], search_queries=[question]),
            hits=[], answer=Answer(text="As of ...", as_of=d, abstained=True),
            timings_ms={"total": 1.0},
        )

    monkeypatch.setattr(pipeline, "ask", fake_ask)
    return seen


def test_ask_ok(calls):
    resp = client.post("/api/ask", json={"question": "  ITC cap under   rule 36(4)?  ",
                                         "as_of": "2020-02-15", "mode": "baseline"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "baseline" and body["answer"]["as_of"] == "2020-02-15"
    assert body["plan"]["refs"] == ["rule 36(4)"]
    assert calls == [{"question": "ITC cap under rule 36(4)?", "as_of": date(2020, 2, 15),
                      "mode": "baseline"}]


def test_ask_defaults(calls):
    resp = client.post("/api/ask", json={"question": "ITC cap in February 2020?"})
    assert resp.status_code == 200
    assert calls[0]["mode"] == "asof" and calls[0]["as_of"] is None


@pytest.mark.parametrize("body", [
    {},
    {"question": "hi"},
    {"question": "   a   "},
    {"question": "x" * 501},
    {"question": "valid question", "mode": "turbo"},
    {"question": "valid question", "as_of": "not-a-date"},
    {"question": "valid question", "as_of": "2016-01-01"},  # before GST
    {"question": "valid question", "as_of": "2999-01-01"},  # future
])
def test_ask_bad_input_is_400(calls, body):
    resp = client.post("/api/ask", json=body)
    assert resp.status_code == 400, resp.text
    assert calls == []


def test_ask_dependency_down_is_503(monkeypatch):
    def down(*a, **k):
        raise pipeline.ServiceUnavailable("database", "PoolTimeout")

    monkeypatch.setattr(pipeline, "ask", down)
    resp = client.post("/api/ask", json={"question": "ITC cap?"})
    assert resp.status_code == 503
    assert "database is unavailable" in resp.json()["detail"]


def test_ask_model_down_is_503(monkeypatch):
    def down(*a, **k):
        raise pipeline.ServiceUnavailable("model", "RateLimitError")

    monkeypatch.setattr(pipeline, "ask", down)
    resp = client.post("/api/ask", json={"question": "ITC cap?"})
    assert resp.status_code == 503 and "language model" in resp.json()["detail"]


def test_ask_in_openapi_and_static_mounted():
    assert "/api/ask" in client.get("/openapi.json").json()["paths"]
    assert client.get("/static/does-not-exist.js").status_code == 404
    assert any(getattr(r, "name", None) == "static" for r in app.routes)
