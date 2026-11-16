"""Answer generation: prompt formatting, quote verification and the grounding policy."""

import json
from datetime import date

import httpx
import pytest
from openai import OpenAI

from asof.generation.answer import (
    UNVERIFIED_NOTE,
    GenCitation,
    build_messages,
    format_source,
    generate_answer,
    is_grounded,
    normalize,
    verify_quote,
    verify_quotes,
)
from asof.models import Hit
from asof.providers import LLM

TODAY = date(2026, 11, 17)
SOURCE = (
    "In rule 36 of the said rules, after sub-rule (3), the following sub-rule shall be\n"
    "inserted, namely:- “(4) Input tax credit to be availed by a registered person in "
    "respect of invoices or debit notes, the details of which have not been uploaded by the "
    "suppliers under sub-section (1) of section 37, shall not exceed 20 per cent. of the "
    "eligible credit available in respect of invoices or debit notes the details of which "
    "have been uploaded by the suppliers.”"
)


def make_hit(i: int = 1, text: str = SOURCE, **kw) -> Hit:
    base = dict(chunk_id=100 + i, doc_id=i, kind="notification", number="49/2019-Central Tax",
                issue_date=date(2019, 10, 9), title="Seeks to amend the CGST Rules",
                url="https://cbic/49.pdf", section_ref="rule 36(4)", page=2, text=text,
                valid_from=date(2019, 10, 9), valid_to=date(2020, 1, 1))
    base.update(kw)
    return Hit(**base)


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


def fake_llm(settings, no_limit, payload: dict, bodies: list | None = None) -> LLM:
    def handler(req):
        if bodies is not None:
            bodies.append(json.loads(req.content))
        return chat_response(json.dumps(payload))
    return LLM(settings, client=openai_client(handler), limiter=no_limit)


# ---------- quote verification ----------

def test_normalize_unifies_quotes_dashes_case_and_space():
    assert normalize("  “Sub–Rule”\n\n(4)  ") == '"sub-rule" (4)'


def test_verify_exact_and_variants():
    assert verify_quote("shall not exceed 20 per cent. of the eligible credit", SOURCE)
    assert verify_quote("SHALL   NOT exceed\n20 per cent.", SOURCE)  # case + whitespace
    assert verify_quote("“shall not exceed 20 per cent.”", SOURCE)  # wrapping quotes
    assert verify_quote("namely:— “(4) Input tax credit", SOURCE)  # dash/quote variants


def test_verify_fuzzy_tolerates_small_differences():
    # one word changed and one dropped in a ~110 char quote
    quote = ("Input tax credit availed by a registered person in respect of invoices or debit "
             "notes, the details of which have not been uploaded")
    assert quote not in SOURCE
    assert verify_quote(quote, SOURCE)


def test_verify_ellipsis_parts_must_each_match():
    assert verify_quote("Input tax credit to be availed ... shall not exceed 20 per cent.", SOURCE)
    assert not verify_quote("Input tax credit to be availed ... shall not exceed 10 per cent "
                            "of the eligible credit available", SOURCE.replace("20", "xx"))


def test_verify_negatives():
    assert not verify_quote("shall not exceed 10 per cent. of the eligible credit", SOURCE)
    assert not verify_quote("interest at eighteen per cent per annum shall be payable", SOURCE)
    assert not verify_quote("", SOURCE)
    assert not verify_quote("20 per cant", SOURCE)  # short quotes must match exactly


def test_verify_quotes_maps_sources_and_drops_out_of_range():
    hits = [make_hit(1), make_hit(2, text="unrelated text")]
    cits = verify_quotes([GenCitation(source=1, quote="shall not exceed 20 per cent."),
                          GenCitation(source=2, quote="shall not exceed 20 per cent."),
                          GenCitation(source=9, quote="x")], hits)
    assert [(c.source, c.chunk_id, c.verified) for c in cits] == [(1, 101, True), (2, 102, False)]


# ---------- prompt ----------

def test_format_source_shows_metadata_and_amendment_flag():
    s = format_source(3, make_hit(), today=TODAY, mode="asof")
    first = s.splitlines()[0]
    assert first.startswith("[S3] Notification 49/2019-Central Tax | issued 2019-10-09")
    assert "valid 2019-10-09 until 2020-01-01 (exclusive)" in first
    assert "LATER AMENDED/SUPERSEDED" in first and "rule 36(4)" in first and "page 2" in first
    assert "Title: Seeks to amend the CGST Rules" in s and "20 per cent." in s

    in_force = format_source(1, make_hit(valid_to=None), today=TODAY, mode="asof")
    assert "onwards (in force)" in in_force and "LATER AMENDED" not in in_force

    baseline = format_source(1, make_hit(), today=TODAY, mode="baseline")
    assert "valid" not in baseline.splitlines()[0] and "issued 2019-10-09" in baseline


def test_build_messages_numbers_sources_and_states_dates():
    msgs = build_messages("ITC cap?", date(2020, 2, 15), [make_hit(1), make_hit(2)],
                          mode="asof", today=TODAY)
    assert msgs[0]["role"] == "system" and "verbatim quote" in msgs[0]["content"]
    assert "abstain" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Answer as of: 2020-02-15" in user and "Today's date: 2026-11-17" in user
    assert "[S1] " in user and "[S2] " in user and user.endswith("Question: ITC cap?")


# ---------- generate_answer + grounding policy ----------

def test_generate_answer_verified(settings, no_limit):
    bodies: list[dict] = []
    llm = fake_llm(settings, no_limit, {
        "answer": "As of 2020-02-15 the cap was 10 per cent [S1].",
        "citations": [{"source": 1, "quote": "shall not exceed 20 per cent."}],
        "abstain": False}, bodies)
    ans = generate_answer(llm, "ITC cap?", date(2020, 2, 15), [make_hit()], mode="asof",
                          today=TODAY)
    assert ans.citations[0].verified and ans.citations[0].chunk_id == 101
    assert not ans.abstained and UNVERIFIED_NOTE not in ans.text
    assert ans.as_of == date(2020, 2, 15) and is_grounded(ans)
    assert bodies[0]["model"] == settings.llm_model and bodies[0]["max_tokens"] == 1500
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_generate_answer_with_thinking_gets_full_budget(settings, no_limit):
    bodies: list[dict] = []
    llm = fake_llm(settings, no_limit, {"answer": "a", "citations": [], "abstain": True}, bodies)
    generate_answer(llm, "q?", date(2020, 2, 15), [make_hit()], mode="asof", today=TODAY,
                    thinking=True)
    assert "chat_template_kwargs" not in bodies[0]
    assert bodies[0]["max_tokens"] == settings.llm_max_tokens


def test_generate_answer_unverified_keeps_text_with_note(settings, no_limit):
    llm = fake_llm(settings, no_limit, {
        "answer": "The cap was 5 per cent [S1].",
        "citations": [{"source": 1, "quote": "shall not exceed 5 per cent of anything at all"}],
        "abstain": False})
    ans = generate_answer(llm, "ITC cap?", date(2020, 2, 15), [make_hit()], mode="asof",
                          today=TODAY)
    assert ans.text.startswith("The cap was 5 per cent [S1].")
    assert ans.text.endswith(UNVERIFIED_NOTE)
    assert not ans.abstained and not ans.citations[0].verified and not is_grounded(ans)


def test_generate_answer_abstain(settings, no_limit):
    llm = fake_llm(settings, no_limit, {"answer": "The sources do not state the cap.",
                                        "citations": [], "abstain": True})
    ans = generate_answer(llm, "ITC cap?", date(2020, 2, 15), [make_hit()], mode="asof",
                          today=TODAY)
    assert ans.abstained and ans.text == "The sources do not state the cap." and is_grounded(ans)


def test_generate_answer_without_hits_skips_llm(settings, no_limit):
    def boom(req):
        raise AssertionError("LLM must not be called")
    llm = LLM(settings, client=openai_client(boom), limiter=no_limit)
    ans = generate_answer(llm, "ITC cap?", date(2020, 2, 15), [], mode="asof", today=TODAY)
    assert ans.abstained and "2020-02-15" in ans.text


@pytest.mark.parametrize("mode", ["asof", "baseline"])
def test_generate_answer_modes_share_prompt_shape(settings, no_limit, mode):
    bodies: list[dict] = []
    llm = fake_llm(settings, no_limit, {"answer": "a", "citations": [], "abstain": True}, bodies)
    generate_answer(llm, "q?", date(2020, 2, 15), [make_hit()], mode=mode, today=TODAY)
    user = bodies[0]["messages"][-1]["content"]
    assert "[S1] Notification" in user
    assert ("valid 2019-10-09" in user) == (mode == "asof")
