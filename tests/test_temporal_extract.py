"""Relation extraction: prefilter, evidence validation, normalisation, caching (mocked LLM)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from asof.models import ExtractedRelation, ExtractionResult, ParsedDoc
from asof.providers.llm import LLM
from asof.temporal.extract import (
    ExtractStats,
    doc_text,
    evidence_in_text,
    extract_relations,
    passes_prefilter,
    postprocess,
    windows,
)

FIXTURE = Path(__file__).parent / "fixtures" / "temporal" / "rule36_chain.json"


def fixture_docs() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["docs"]


def mock_llm(settings, replies: list[str], calls: list[dict]) -> LLM:
    it = iter(replies)

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": next(it)}}],
        })

    client = OpenAI(api_key="nvapi-test", base_url="https://integrate.api.nvidia.com/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                    max_retries=0)
    from asof.providers.ratelimit import RateLimiter
    return LLM(settings, client=client, limiter=RateLimiter(10_000))


# ------------------------------------------------------------------ prefilter / windows

@pytest.mark.parametrize("text,expected", [
    ("the following rules further to amend the said rules", True),
    ("hereby rescinds notification No. 13/2017-Central Tax", True),
    ("in supersession of notification No. 1/2017", True),
    ("sub-rule (4) shall be omitted", True),
    ("the words shall be Substituted", True),
    ("Circular No. 12 is hereby withdrawn", True),
    ("it is clarified that", True),
    ("The Central Government hereby notifies the rate of tax on the goods", False),
    ("", False),
])
def test_prefilter(text, expected):
    assert passes_prefilter(text) is expected


def test_windows_keep_beginning_and_only_relevant_tails():
    head = "notifies the rate\n" * 1000          # ~18k chars, no trigger words
    tail = "\nin rule 36, sub-rule (4) shall be omitted.\n" * 10
    wins = windows(head + "x" * 20_000 + tail, size=15_000, max_windows=5)
    assert wins[0].startswith("notifies the rate")  # first window always kept
    assert all(passes_prefilter(w) for w in wins[1:])
    assert any("omitted" in w for w in wins)
    assert len(windows("short text")) == 1


# ------------------------------------------------------------------ evidence

TEXT = ("2. In the Central Goods and Services Tax Rules, 2017, in rule 36, in sub-\n"
        "rule (4), for the words “twenty per cent.”, the words “ten per\n"
        "cent.” shall be substituted.")


@pytest.mark.parametrize("evidence,ok", [
    ('in rule 36, in sub-rule (4), for the words "twenty per cent.", the words "ten per '
     'cent." shall be substituted', True),
    ("IN RULE 36,   IN SUB-RULE (4)", True),                       # case / whitespace
    ("in rule 36 ... shall be substituted", True),                  # ellipsis quote
    ("in rule 36, in sub-rule (4), for the words twenty per cent, the words ten percent "
     "shall be substitutd", True),                                  # small typos
    ("in rule 37, sub-rule (1) shall be omitted", False),            # not in text
    ("the rate of input tax credit is reduced to ten per cent", False),  # paraphrase
    ("rule", False),                                               # too short
])
def test_evidence_in_text(evidence, ok):
    assert evidence_in_text(evidence, TEXT) is ok


def test_postprocess_drops_unsupported_and_normalises():
    doc = ParsedDoc.model_validate(fixture_docs()[1]["parsed"])  # 75/2019
    raw = ExtractionResult(effective_date=date(2020, 1, 1), relations=[
        ExtractedRelation(kind="substitutes", target_kind="rule",
                          target_ref="sub-rule (4) of rule 36", effective_date=date(2020, 1, 1),
                          evidence="in rule 36, in sub-rule (4), for the words",
                          confidence=0.9),
        # duplicate after normalisation
        ExtractedRelation(kind="substitutes", target_kind="rule", target_ref="Rule 36(4)",
                          effective_date=date(2020, 1, 1),
                          evidence="the words “ten per cent.” shall be substituted",
                          confidence=0.9),
        # hallucinated evidence
        ExtractedRelation(kind="rescinds", target_kind="notification",
                          target_ref="No. 99/2018 - Central Tax",
                          evidence="hereby rescinds notification No. 99/2018-Central Tax",
                          confidence=0.9),
        # self reference
        ExtractedRelation(kind="amends", target_kind="notification",
                          target_ref="75/2019-CT", evidence="Central Goods and Services Tax "
                          "(Eighth Amendment) Rules, 2019", confidence=0.5),
    ])
    stats = ExtractStats()
    out = postprocess(raw, doc, doc_text(doc), stats)
    assert [(r.kind, r.target_kind, r.target_ref) for r in out.relations] == [
        ("substitutes", "rule", "rule 36(4)")]
    assert stats.dropped_evidence == 1 and stats.dropped_other == 1
    assert out.effective_date == date(2020, 1, 1)


# ------------------------------------------------------------------ extract_relations

def test_extract_relations_calls_llm_once_then_uses_cache(settings, tmp_path):
    case = fixture_docs()[3]  # 40/2021: one real omission + one hallucinated relation
    doc = ParsedDoc.model_validate(case["parsed"])
    calls: list[dict] = []
    llm = mock_llm(settings, [json.dumps(case["llm_output"])], calls)

    stats = ExtractStats()
    out = extract_relations(llm, doc, cache_dir=tmp_path, stats=stats)
    assert [(r.kind, r.target_ref, r.effective_date) for r in out.relations] == [
        ("omits", "rule 36(4)", date(2022, 1, 1))]
    assert out.effective_date == date(2022, 1, 1)
    assert stats.llm_calls == 1 and stats.dropped_evidence == 1
    prompt = calls[0]["messages"][-1]["content"]
    assert "40/2021-Central Tax" in prompt and "2021-12-29" in prompt
    assert "sub-rule (4) shall be omitted" in prompt
    assert calls[0]["model"] == settings.llm_fast_model
    assert (tmp_path / f"{doc.meta.key}.json").exists()

    again = extract_relations(llm, doc, cache_dir=tmp_path, stats=stats)  # no reply left
    assert again == out
    assert stats.llm_calls == 1 and stats.cache_hits == 1

    # a different model invalidates the cache
    settings.llm_fast_model = "other/model"
    llm2 = mock_llm(settings, [json.dumps(case["llm_output"])], calls)
    extract_relations(llm2, doc, cache_dir=tmp_path, stats=stats)
    assert stats.llm_calls == 2


def test_extract_relations_skips_llm_without_trigger_words(settings, tmp_path):
    doc = ParsedDoc.model_validate({
        "meta": {"kind": "notification", "number": "1/2021-Central Tax (Rate)",
                 "issue_date": "2021-01-01", "title": "rates", "url": "https://x.test/a.pdf"},
        "pages": [{"page": 1, "text": "The Government hereby notifies the rate of tax."}],
    })
    stats = ExtractStats()
    out = extract_relations(None, doc, cache_dir=tmp_path, stats=stats)
    assert out.relations == [] and stats.skipped_prefilter == 1 and stats.llm_calls == 0


def test_llm_call_disables_thinking_and_repairs_bad_json(settings, tmp_path):
    case = fixture_docs()[3]
    doc = ParsedDoc.model_validate(case["parsed"])
    calls: list[dict] = []
    llm = mock_llm(settings, ['{"relations": "oops"}', json.dumps(case["llm_output"])], calls)
    out = extract_relations(llm, doc, cache_dir=tmp_path)
    assert len(out.relations) == 1
    assert len(calls) == 2
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "JSON Schema" in calls[0]["messages"][0]["content"]
    assert "invalid" in calls[1]["messages"][-1]["content"]


def test_llm_call_falls_back_when_switch_rejected(settings, tmp_path):
    case = fixture_docs()[3]
    doc = ParsedDoc.model_validate(case["parsed"])
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"error": {"message": "unknown field"}})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(case["llm_output"])}}],
        })

    client = OpenAI(api_key="nvapi-test", base_url="https://integrate.api.nvidia.com/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                    max_retries=0)
    from asof.providers.ratelimit import RateLimiter
    llm = LLM(settings, client=client, limiter=RateLimiter(10_000))
    out = extract_relations(llm, doc, cache_dir=tmp_path)
    # LLM sends the switch once, gets a 400, retries once without it (no double fallback)
    assert len(out.relations) == 1 and len(bodies) == 2
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in bodies[1]
    # ...and remembers: the next extraction (cache bypassed) sends a single plain request
    extract_relations(llm, doc, cache_dir=tmp_path, use_cache=False)
    assert len(bodies) == 3 and "chat_template_kwargs" not in bodies[2]


def test_circulars_cannot_amend_rules():
    doc = ParsedDoc.model_validate({
        "meta": {"kind": "circular", "number": "188/20/2022-GST", "issue_date": "2022-12-27",
                 "title": "refunds", "url": "https://x.test/c.pdf"},
        "pages": [{"page": 1, "text": "sub-rule (2) of rule 89 has been amended vide "
                   "Notification No. 26/2022-Central Tax. Circular No. 125/44/2019-GST is "
                   "hereby modified as under."}],
    })
    raw = ExtractionResult(relations=[
        ExtractedRelation(kind="amends", target_kind="rule", target_ref="rule 89(2)",
                          evidence="sub-rule (2) of rule 89 has been amended", confidence=0.9),
        ExtractedRelation(kind="amends", target_kind="circular",
                          target_ref="Circular No. 125/44/2019-GST",
                          evidence="Circular No. 125/44/2019-GST is hereby modified",
                          confidence=0.9),
    ])
    out = postprocess(raw, doc, doc_text(doc))
    assert [(r.kind, r.target_ref) for r in out.relations] == [
        ("clarifies", "rule 89(2)"), ("amends", "125/44/2019-GST")]


def test_window_split_when_output_stays_invalid(settings, tmp_path):
    case = fixture_docs()[3]
    doc = ParsedDoc.model_validate(case["parsed"])
    doc.pages[0].text = ("Preamble line.\n" * 300) + doc.pages[0].text  # > 3000 chars
    calls: list[dict] = []
    replies = ["{bad", "{bad", '{"relations": [], "effective_date": null}',
               json.dumps(case["llm_output"])]
    llm = mock_llm(settings, replies, calls)
    out = extract_relations(llm, doc, cache_dir=tmp_path)
    assert len(calls) == 4  # 2 failed attempts on the full window, then one per half
    assert [r.target_ref for r in out.relations] == ["rule 36(4)"]
