"""Question generation: verbatim fact checks, document selection and caching, with a fake LLM."""

from datetime import date

from asof.evaluation.generate import (
    GeneratedQuestion,
    date_in_text,
    generate,
    priority,
    select_docs,
    to_candidate,
    verify_facts,
)
from asof.models import DocMeta, Page, ParsedDoc

TEXT_75 = (
    "Notification No. 75/2019-Central Tax. In exercise of the powers conferred by section 164, "
    "the Central Government hereby makes the following rules further to amend the Central Goods "
    "and Services Tax Rules, 2017. In rule 36, in sub-rule (4), for the words \"20 per cent.\", "
    "the words \"10 per cent.\" shall be substituted. These rules shall come into force on the "
    "1st day of January, 2020. " + "Filler text. " * 30
)


def doc(number: str, kind: str = "notification", text: str = TEXT_75,
        issued: date = date(2019, 12, 26), title: str = "Seeks to amend CGST Rules") -> ParsedDoc:
    return ParsedDoc(
        meta=DocMeta(kind=kind, number=number, issue_date=issued, title=title,  # type: ignore[arg-type]
                     url="https://example.org/x.pdf"),
        pages=[Page(page=1, text=text)],
    )


class FakeLLM:
    def __init__(self, replies: list[GeneratedQuestion | Exception]):
        self.replies = list(replies)
        self.calls = 0

    def structured(self, messages, schema, **kw):
        assert schema is GeneratedQuestion
        assert "document text" in messages[-1]["content"]
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def proposal(**kw) -> GeneratedQuestion:
    base = dict(usable=True, question="What was the rule 36(4) cap from January 2020?",
                answer="10 per cent", must_contain=["10 per cent"],
                evidence="the words \"10 per cent.\" shall be substituted",
                effective_date=date(2020, 1, 1), about_current_law=False)
    base.update(kw)
    return GeneratedQuestion(**base)


def test_verify_facts_normalises_but_requires_presence():
    found, missing = verify_facts(TEXT_75, ["ten per cent", "10%", "rule 36", "15 per cent"])
    assert found == ["ten per cent", "10%", "rule 36"]
    assert missing == ["15 per cent"]


def test_candidate_kept_when_facts_are_verbatim():
    q, reason = to_candidate(doc("75/2019-Central Tax"), proposal())
    assert reason == "ok" and q is not None
    assert q.as_of == date(2020, 1, 1)  # stated effective date found in the text
    assert q.type == "point_in_time"
    assert q.gold_doc_numbers == ["75/2019-Central Tax"]
    assert "evidence verified" in q.notes


def test_candidate_dropped_when_a_fact_is_invented():
    q, reason = to_candidate(doc("75/2019-Central Tax"),
                             proposal(must_contain=["10 per cent", "GSTR-2B"]))
    assert q is None and "not found verbatim" in reason
    q, reason = to_candidate(doc("75/2019-Central Tax"), proposal(usable=False))
    assert q is None
    q, reason = to_candidate(doc("75/2019-Central Tax"),
                             proposal(question="What does Notification 75/2019 change?"))
    assert q is None and "document number" in reason


def test_unverifiable_effective_date_falls_back_to_issue_date():
    q, _ = to_candidate(doc("75/2019-Central Tax"), proposal(effective_date=date(2020, 3, 1)))
    assert q is not None and q.as_of == date(2019, 12, 26)


def test_current_law_question_has_no_as_of():
    q, _ = to_candidate(doc("75/2019-Central Tax"), proposal(about_current_law=True))
    assert q is not None and q.type == "current" and q.as_of is None


def test_date_in_text_formats():
    assert date_in_text(date(2020, 1, 1), "come into force on the 1st day of January, 2020")
    assert date_in_text(date(2022, 4, 1), "with effect from 01.04.2022")
    assert date_in_text(date(2022, 4, 1), "from April 1, 2022")
    assert not date_in_text(date(2022, 4, 2), "from 01.04.2022")


def test_select_docs_prefers_amending_notifications_then_circulars():
    docs = [
        doc("1/2021-Central Tax", text="Exempts a class of persons. " * 30, title="Exemption",
            issued=date(2021, 1, 1)),
        doc("150/06/2021-GST", kind="circular", text="Clarification on refunds. " * 30,
            title="Clarification", issued=date(2021, 3, 1)),
        doc("75/2019-Central Tax"),
        doc("tiny/2020-Central Tax", text="too short"),
    ]
    assert [priority(d) for d in docs[:3]] == [2, 1, 0]
    chosen = select_docs(docs, 2)
    assert [d.meta.number for d in chosen] == ["75/2019-Central Tax", "150/06/2021-GST"]
    assert len(select_docs(docs, 10)) == 3  # the too-short document is never chosen


def test_generate_uses_cache_and_stops_on_rate_limit(tmp_path):
    docs = [doc("75/2019-Central Tax"), doc("76/2019-Central Tax")]
    llm = FakeLLM([proposal(), proposal(must_contain=["invented fact"])])
    out, stats = generate(docs, llm, cache_dir=tmp_path, log=lambda _: None)
    assert [q.gold_doc_numbers for q in out] == [["75/2019-Central Tax"]]
    assert stats["kept"] == 1 and stats["dropped"] == 1 and stats["llm_calls"] == 2

    # Second run: served from cache, no LLM calls.
    llm2 = FakeLLM([])
    out2, stats2 = generate(docs, llm2, cache_dir=tmp_path, log=lambda _: None)
    assert llm2.calls == 0 and stats2["cached"] == 2 and len(out2) == 1

    class RateLimitError(Exception):
        pass

    llm3 = FakeLLM([RateLimitError("Error code: 429"), proposal()])
    out3, stats3 = generate(docs, llm3, cache_dir=tmp_path / "fresh", log=lambda _: None)
    assert out3 == [] and stats3["stopped"].startswith("rate limited") and llm3.calls == 1


def test_generate_respects_max_calls(tmp_path):
    docs = [doc("75/2019-Central Tax"), doc("76/2019-Central Tax")]
    llm = FakeLLM([proposal(), proposal()])
    _, stats = generate(docs, llm, cache_dir=tmp_path, max_calls=1, log=lambda _: None)
    assert llm.calls == 1 and "max-calls" in stats["stopped"]
