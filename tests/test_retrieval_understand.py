"""Query understanding: regexes for dates/refs, LLM refinement and regex-only fallback."""

import json
from datetime import date

import httpx
import pytest
from openai import OpenAI

from asof.providers import LLM
from asof.retrieval.understand import extract_dates, extract_refs, regex_plan, understand

TODAY = date(2026, 11, 17)


def openai_client(handler) -> OpenAI:
    return OpenAI(
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def chat_response(content: str | None) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    })


# ---------- dates ----------

@pytest.mark.parametrize("question, expected", [
    ("ITC cap as on 15 Feb 2020?", date(2020, 2, 15)),
    ("ITC cap on 15th February, 2020?", date(2020, 2, 15)),
    ("ITC cap on February 15, 2020?", date(2020, 2, 15)),
    ("ITC cap on 2020-02-15?", date(2020, 2, 15)),
    ("ITC cap on 15.02.2020?", date(2020, 2, 15)),
    ("ITC cap on 15/02/2020?", date(2020, 2, 15)),
    ("ITC cap in February 2020?", date(2020, 2, 29)),  # month -> last day of month
    ("ITC cap in Sept 2019?", date(2019, 9, 30)),
    ("ITC cap in 2020?", date(2020, 12, 31)),  # cued year -> 31 Dec
    ("ITC cap during FY 2019-20?", date(2020, 3, 31)),  # FY -> 31 March of last year
    ("ITC cap during financial year 2019-2020?", date(2020, 3, 31)),
])
def test_date_rules(question, expected):
    plan = regex_plan(question, TODAY)
    assert plan.as_of == expected
    assert plan.as_of_explicit
    assert plan.question_type == "point_in_time"


def test_bare_years_and_reference_numbers_are_not_dates():
    plan = regex_plan("What does 49/2019-Central Tax say under the CGST Rules, 2017?", TODAY)
    assert plan.as_of == TODAY and not plan.as_of_explicit
    # circular numbers look like dd/mm/yyyy but sit inside a reference
    assert not extract_dates("Circular 10/10/2017-GST", extract_refs("Circular 10/10/2017-GST")[1])


def test_dated_document_dates_are_skipped():
    plan = regex_plan("Explain notification 49/2019-Central Tax dated 9th October, 2019", TODAY)
    assert not plan.as_of_explicit


@pytest.mark.parametrize("question", [
    "Notification 49/2019-Central Tax was issued on 9 October 2019; what changed in it?",
    "Notification 49/2019-Central Tax, issued 9 October 2019 - explain it",
    "The amendment notified on 9.10.2019 - explain it",
    "A circular published on 2019-10-09 - explain it",
    "Rule 36(4) notified in October 2019 - explain it",
])
def test_issue_dates_are_skipped(question):
    assert not regex_plan(question, TODAY).as_of_explicit


@pytest.mark.parametrize("question", [
    "Notification 49/2019-Central Tax was issued on 9 October 2019; "
    "what is the ITC cap under rule 36(4) now?",
    "Rule 36(4) was inserted w.e.f. 9 October 2019. What is the current ITC cap?",
    "Rule 36(4) changed on 1 Jan 2020; what applies today?",
    "What is the ITC cap at present, given the 1 January 2020 amendment?",
    "Currently, what does rule 36(4) say after the amendment of 9 Oct 2019?",
])
def test_now_questions_use_today(question):
    plan = regex_plan(question, TODAY)
    assert plan.as_of == TODAY and not plan.as_of_explicit
    assert plan.question_type in ("current", "history")


def test_now_question_example_is_current():
    plan = regex_plan("Notification 49/2019-Central Tax was issued on 9 October 2019; "
                      "what is the ITC cap under rule 36(4) now?", TODAY)
    assert plan.question_type == "current"
    assert plan.refs == ["49/2019-Central Tax", "rule 36(4)"]


def test_now_with_explicit_as_of_cue_keeps_the_date():
    plan = regex_plan("Was rule 36(4) in force now, i.e. as on 1 Jan 2020?", TODAY)
    assert plan.as_of == date(2020, 1, 1) and plan.question_type == "point_in_time"
    plan = regex_plan("Current cap as of February 2020?", TODAY)
    assert plan.as_of == date(2020, 2, 29)


def test_now_question_override_still_wins():
    plan = regex_plan("What is the ITC cap now?", TODAY, as_of_override=date(2020, 1, 1))
    assert plan.as_of == date(2020, 1, 1) and plan.as_of_explicit


def test_now_question_ignores_llm_background_date(settings, no_limit):
    llm = LLM(settings, client=openai_client(lambda r: chat_response(
        '{"question_type": "point_in_time", "search_queries": ["q"], "as_of_date": "2019-10-09"}'
    )), limiter=no_limit)
    plan = understand(llm, "Notification 49/2019-Central Tax was issued on 9 October 2019; "
                           "what is the ITC cap under rule 36(4) now?", TODAY)
    assert plan.as_of == TODAY and not plan.as_of_explicit
    assert plan.question_type == "current"


def test_cued_date_wins_else_latest():
    assert regex_plan("Notified 1 Jan 2019; what applied as on 5 March 2019?",
                      TODAY).as_of == date(2019, 3, 5)
    assert regex_plan("Compare 1 Jan 2019 and 1 Jan 2021", TODAY).as_of == date(2021, 1, 1)


def test_future_dates_clamped_and_invalid_dates_ignored():
    assert regex_plan("ITC cap on 1 Jan 2099?", TODAY).as_of == TODAY
    assert not regex_plan("ITC cap on 31/02/2020?", TODAY).as_of_explicit


def test_override_wins():
    plan = regex_plan("ITC cap in February 2020?", TODAY, as_of_override=date(2021, 1, 1))
    assert plan.as_of == date(2021, 1, 1) and plan.as_of_explicit


# ---------- references ----------

@pytest.mark.parametrize("text, refs", [
    ("cap under rule 36(4)?", ["rule 36(4)"]),
    ("Rule 86A and RULE 36 (4)", ["rule 86A", "rule 36(4)"]),
    ("time limit in section 16(4)", ["section 16(4)"]),
    ("u/s 16(2)(c)", ["section 16(2)(c)"]),
    ("sub-rule (4) of rule 36", ["rule 36(4)"]),
    ("sub-section (4) of section 16", ["section 16(4)"]),
    ("Notification No. 49/2019-Central Tax", ["49/2019-Central Tax"]),
    ("notfn 2/2019 – Central Tax (Rate)", ["2/2019-Central Tax (Rate)"]),
    ("01/2020-Integrated Tax", ["1/2020-Integrated Tax"]),
    ("circular 123/42/2019-GST", ["123/42/2019-GST"]),
    ("Circular No. 183/15/2022", ["183/15/2022-GST"]),
    ("notification no. 10/2020", ["10/2020"]),
    ("CGST Rules, 2017", []),
])
def test_reference_extraction(text, refs):
    assert extract_refs(text)[0] == refs


# ---------- LLM refinement / fallback ----------

def test_understand_uses_llm_for_type_and_queries(settings, no_limit):
    bodies: list[dict] = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return chat_response(json.dumps({
            "question_type": "point_in_time",
            "search_queries": ["provisional input tax credit cap rule 36(4)", "  ", "x"],
            "as_of_date": "2021-05-01",  # ignored: the regex already found a date
        }))

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    plan = understand(llm, "ITC cap under rule 36(4) in February 2020?", TODAY)

    assert plan.as_of == date(2020, 2, 29)
    assert plan.refs == ["rule 36(4)"]
    assert plan.search_queries == ["provisional input tax credit cap rule 36(4)", "x"]
    assert bodies[0]["model"] == settings.llm_fast_model
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    prompt = bodies[0]["messages"][-1]["content"]
    assert "2020-02-29" in prompt and "rule 36(4)" in prompt and "2026-11-17" in prompt


def test_understand_accepts_llm_date_only_when_regex_found_none(settings, no_limit):
    def handler(req):
        return chat_response('{"question_type": "point_in_time", "search_queries": ["q"],'
                             ' "as_of_date": "2019-10-15"}')

    llm = LLM(settings, client=openai_client(handler), limiter=no_limit)
    plan = understand(llm, "What applied just after the October 2019 rule change?", TODAY)
    assert plan.as_of == date(2019, 10, 31)  # regex month rule wins

    plan = understand(llm, "What applied right after the rule 36(4) insertion?", TODAY)
    assert plan.as_of == date(2019, 10, 15) and plan.as_of_explicit


def test_understand_falls_back_to_regex_plan_when_llm_fails(settings, no_limit):
    llm = LLM(settings, client=openai_client(lambda r: httpx.Response(500)), limiter=no_limit)
    plan = understand(llm, "How has rule 36(4) changed since 2019?", TODAY)
    assert plan.question_type == "history"
    assert plan.refs == ["rule 36(4)"]
    assert plan.search_queries == ["How has rule 36(4) changed since 2019?"]


def test_understand_without_llm_is_regex_plan():
    assert understand(None, "current rule 86B limit?", TODAY).question_type == "current"


def test_explicit_date_overrides_llm_current(settings, no_limit):
    llm = LLM(settings, client=openai_client(
        lambda r: chat_response('{"question_type": "current", "search_queries": ["q"]}')),
        limiter=no_limit)
    assert understand(llm, "ITC cap on 1 Jan 2020?", TODAY).question_type == "point_in_time"


def test_point_in_time_without_any_date_becomes_current(settings, no_limit):
    llm = LLM(settings, client=openai_client(
        lambda r: chat_response('{"question_type": "point_in_time", "search_queries": ["q"]}')),
        limiter=no_limit)
    plan = understand(llm, "Minimum refund amount under circular 188/20/2022-GST?", TODAY)
    assert plan.question_type == "current" and plan.as_of == TODAY
