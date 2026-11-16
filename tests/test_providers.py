"""Provider tests against a mocked HTTP transport: no network, no API key needed."""

import json

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel, SecretStr

from asof.config import Settings
from asof.providers import (
    LLM,
    Embedder,
    FallbackReranker,
    LLMError,
    LLMReranker,
    NimReranker,
    NoopReranker,
    OpenRouterReranker,
    make_reranker,
)
from asof.providers.llm import extract_json


def openai_client(handler) -> OpenAI:
    return OpenAI(
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def chat_response(content: str | None, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
    })


# ---------- LLM ----------

class Cap(BaseModel):
    rule: str
    percent: int


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_retries_after_invalid_output(settings: Settings, no_limit):
    replies = iter(['{"rule": "36(4)"}', '{"rule": "36(4)", "percent": 10}'])
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return chat_response(next(replies))

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    out = llm.structured([{"role": "user", "content": "cap?"}], Cap)

    assert out == Cap(rule="36(4)", percent=10)
    assert len(seen) == 2
    assert seen[0]["model"] == settings.llm_fast_model
    assert "JSON Schema" in seen[0]["messages"][0]["content"]
    assert "invalid" in seen[1]["messages"][-1]["content"]


def test_structured_sends_guided_json_when_enabled(settings: Settings, no_limit):
    settings.llm_guided_json = True
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return chat_response('{"rule": "36(4)", "percent": 5}')

    LLM(settings, client=openai_client(handler), limiter=no_limit).structured(
        [{"role": "user", "content": "cap?"}], Cap)
    assert bodies[0]["nvext"]["guided_json"]["title"] == "Cap"


def test_chat_raises_on_empty_content(settings: Settings, no_limit):
    llm = LLM(settings, client=openai_client(lambda r: chat_response(None, "length")),
              limiter=no_limit)
    with pytest.raises(LLMError, match="finish_reason=length"):
        llm.chat([{"role": "user", "content": "hi"}])


# ---------- Embeddings ----------

def embed_handler(calls: list[dict], dim: int = 4):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        data = [{"object": "embedding", "index": i, "embedding": [float(i)] * dim}
                for i in range(len(body["input"]))]
        return httpx.Response(200, json={"object": "list", "data": list(reversed(data)),
                                         "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})
    return handler


def test_embed_documents_batches_and_uses_passage_type(settings: Settings, no_limit):
    settings.embed_batch_size = 2
    calls: list[dict] = []
    emb = Embedder(settings, client=openai_client(embed_handler(calls)), limiter=no_limit)

    vecs = emb.embed_documents(["a", "b", "c"])

    assert len(vecs) == 3
    assert vecs[0] == [0.0] * 4 and vecs[1] == [1.0] * 4  # re-sorted by index
    assert [len(c["input"]) for c in calls] == [2, 1]
    assert all(c["input_type"] == "passage" and c["dimensions"] == 4 for c in calls)


def test_embed_query_uses_query_type_and_can_omit_dimensions(settings: Settings, no_limit):
    settings.embed_send_dimensions = False
    calls: list[dict] = []
    emb = Embedder(settings, client=openai_client(embed_handler(calls)), limiter=no_limit)
    emb.embed_query("q")
    assert calls[0]["input_type"] == "query"
    assert "dimensions" not in calls[0]


def test_embed_rejects_wrong_dimension(settings: Settings, no_limit):
    emb = Embedder(settings, client=openai_client(embed_handler([], dim=8)), limiter=no_limit)
    with pytest.raises(ValueError, match="EMBED_DIM"):
        emb.embed_query("q")


# ---------- Reranker ----------

def test_rerank_url_derivation():
    s = Settings(_env_file=None, rerank_model="nvidia/llama-3.2-nv-rerankqa-1b-v2")  # type: ignore[call-arg]
    assert s.resolved_rerank_url == (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking")


def test_rerank_sorts_by_logit_and_sends_expected_body(settings: Settings, no_limit):
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        assert req.headers["Authorization"] == "Bearer nvapi-test"
        return httpx.Response(200, json={"rankings": [
            {"index": 0, "logit": -2.0}, {"index": 2, "logit": 5.5}, {"index": 1, "logit": 0.1}]})

    rr = NimReranker(settings, http=httpx.Client(transport=httpx.MockTransport(handler)),
                  limiter=no_limit)
    out = rr.rerank("q", ["a", "b", "c"], top_n=2)

    assert [r.index for r in out] == [2, 1]
    assert bodies[0]["query"] == {"text": "q"}
    assert bodies[0]["passages"] == [{"text": "a"}, {"text": "b"}, {"text": "c"}]


def test_rerank_retries_on_429(settings: Settings, no_limit, monkeypatch):
    monkeypatch.setattr(NimReranker._call.retry, "sleep", lambda s: None)  # skip real backoff
    responses = iter([httpx.Response(429), httpx.Response(200, json={"rankings": [
        {"index": 0, "logit": 1.0}]})])
    rr = NimReranker(settings, http=httpx.Client(
        transport=httpx.MockTransport(lambda r: next(responses))), limiter=no_limit)
    assert rr.rerank("q", ["a"])[0].index == 0


def test_rerank_empty_passages_skips_call(settings: Settings, no_limit):
    def boom(req):
        raise AssertionError("should not be called")
    rr = NimReranker(settings, http=httpx.Client(transport=httpx.MockTransport(boom)),
                  limiter=no_limit)
    assert rr.rerank("q", []) == []


def test_llm_reranker_orders_by_score_and_sinks_missing(settings: Settings, no_limit):
    prompts: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        prompts.append(body["messages"][-1]["content"])
        assert body["model"] == settings.llm_fast_model
        # passage 1 is the best, passage 2 is omitted by the model, id 9 is out of range
        return chat_response('{"scores": [{"id": 0, "score": 3}, {"id": 1, "score": 9},'
                             ' {"id": 9, "score": 10}]}')

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    out = LLMReranker(settings, llm=llm).rerank("ITC cap?", ["a", "b", "c"])

    assert [r.index for r in out] == [1, 0, 2]
    assert out[-1].score == -1.0
    assert "[1] b" in prompts[0] and "ITC cap?" in prompts[0]


def test_llm_reranker_batches_and_keeps_global_indexes(settings: Settings, no_limit):
    def handler(req: httpx.Request) -> httpx.Response:
        return chat_response('{"scores": [{"id": 0, "score": 1}, {"id": 1, "score": 8}]}')

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    out = LLMReranker(settings, llm=llm, batch_size=2).rerank("q", ["a", "b", "c", "d"], top_n=2)
    assert [r.index for r in out] == [1, 3]  # index 3 = second batch, local id 1


def test_noop_reranker_keeps_order():
    assert [r.index for r in NoopReranker().rerank("q", ["a", "b", "c"], top_n=2)] == [0, 1]


def test_make_reranker_backends(settings: Settings):
    settings.rerank_backend = "none"
    assert isinstance(make_reranker(settings), NoopReranker)
    settings.rerank_backend = "nim"
    assert isinstance(make_reranker(settings), NimReranker)


# ---------- OpenRouter reranker + fallback ----------

def or_settings(settings: Settings) -> Settings:
    settings.openrouter_api_key = SecretStr("sk-or-test")
    settings.rerank_backend = "openrouter"
    return settings


def test_openrouter_reranker_request_and_ordering(settings: Settings, no_limit):
    s = or_settings(settings)
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://openrouter.ai/api/v1/rerank"
        assert req.headers["Authorization"] == "Bearer sk-or-test"
        bodies.append(json.loads(req.content))
        # index 2 missing from the response; index 7 out of range
        return httpx.Response(200, json={"id": "x", "model": "m", "results": [
            {"index": 1, "relevance_score": 0.91, "document": {"text": "b"}},
            {"index": 0, "relevance_score": 0.12, "document": {"text": "a"}},
            {"index": 7, "relevance_score": 0.99, "document": {"text": "?"}}]})

    rr = OpenRouterReranker(s, http=httpx.Client(transport=httpx.MockTransport(handler)),
                            limiter=no_limit, max_chars=5)
    out = rr.rerank("ITC cap?", ["a", "b", "c" * 20])

    assert [r.index for r in out] == [1, 0, 2]
    assert out[-1].score < out[-2].score
    assert bodies[0] == {"model": "nvidia/nemotron-3.5-rerank-8b:free", "query": "ITC cap?",
                         "documents": ["a", "b", "ccccc"], "top_n": 3}


def test_openrouter_does_not_retry_429(settings: Settings, no_limit):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    rr = OpenRouterReranker(or_settings(settings),
                            http=httpx.Client(transport=httpx.MockTransport(handler)),
                            limiter=no_limit)
    with pytest.raises(httpx.HTTPStatusError):
        rr.rerank("q", ["a"])
    assert len(calls) == 1


def test_openrouter_requires_key(settings: Settings):
    settings.rerank_backend = "openrouter"
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_reranker(settings)


def test_fallback_used_when_primary_fails():
    class Boom:
        def rerank(self, query, passages, top_n=None):
            raise httpx.ConnectError("down")

    fb = FallbackReranker(Boom(), NoopReranker())
    assert [r.index for r in fb.rerank("q", ["a", "b"])] == [0, 1]
    assert fb.last_used == "NoopReranker"


def test_make_reranker_openrouter_wraps_fallback(settings: Settings):
    s = or_settings(settings)
    assert isinstance(make_reranker(s), FallbackReranker)
    assert isinstance(make_reranker(s, fallback=False), OpenRouterReranker)


def test_chat_disables_thinking_by_default_and_drops_switch_if_rejected(settings: Settings,
                                                                      no_limit):
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"error": {"message": "unknown field"}})
        return chat_response("ok")

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in bodies[1]
    llm.chat([{"role": "user", "content": "again"}])
    assert len(bodies) == 3 and "chat_template_kwargs" not in bodies[2]  # remembered


def test_chat_thinking_true_sends_no_switch(settings: Settings, no_limit):
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return chat_response("ok")

    LLM(settings, client=openai_client(handler), limiter=no_limit).chat(
        [{"role": "user", "content": "hi"}], thinking=True)
    assert "chat_template_kwargs" not in bodies[0]


def test_extract_json_repairs_unescaped_inner_quotes():
    out = extract_json('{"answer": "the words "ten per cent" apply", "abstain": false}')
    assert out == {"answer": 'the words "ten per cent" apply', "abstain": False}


def test_unrelated_400_does_not_disable_thinking_switch(settings: Settings, no_limit):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    with pytest.raises(Exception, match="context length"):
        llm.chat([{"role": "user", "content": "hi"}])
    assert llm._thinking_switch_unsupported is False


def test_truncated_output_is_retried_with_bigger_budget(settings: Settings, no_limit):
    budgets: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        budgets.append(body["max_tokens"])
        if len(budgets) == 1:
            return chat_response('{"rule": "36(4)", "perc', finish_reason="length")
        return chat_response('{"rule": "36(4)", "percent": 10}')

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    assert llm.structured([{"role": "user", "content": "cap?"}], Cap, max_tokens=100) == Cap(
        rule="36(4)", percent=10)
    assert budgets == [100, 200]


def test_truncated_chat_raises(settings: Settings, no_limit):
    from asof.providers.llm import LLMTruncated

    llm = LLM(settings, client=openai_client(lambda r: chat_response("partial", "length")),
              limiter=no_limit)
    with pytest.raises(LLMTruncated):
        llm.chat([{"role": "user", "content": "hi"}])
