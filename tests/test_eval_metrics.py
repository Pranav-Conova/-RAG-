"""Evaluation metrics: pure functions, no network."""

from datetime import date

import pytest

from asof.evaluation.metrics import (
    abstention_correct,
    aggregate,
    answer_correct,
    canon_doc,
    citation_hit,
    citation_precision,
    contains_fact,
    in_force,
    normalise,
    per_type,
    percentile,
    recall_at_k,
    score_result,
    stale_citations,
)
from asof.models import Answer, AskResult, Citation, Hit, QueryPlan


def hit(chunk_id: int, number: str, *, issue: str = "2020-01-01", vf: str | None = None,
        vt: str | None = None, text: str = "text") -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id=chunk_id, kind="notification", number=number,
        issue_date=date.fromisoformat(issue), title=f"Notification {number}", url=None,
        section_ref=None, page=1, text=text,
        valid_from=date.fromisoformat(vf) if vf else None,
        valid_to=date.fromisoformat(vt) if vt else None,
    )


def result(hits: list[Hit], citations: list[tuple[int, bool]], *, text: str = "answer",
           abstained: bool = False, as_of: str = "2020-02-15") -> AskResult:
    d = date.fromisoformat(as_of)
    return AskResult(
        question="q", mode="asof",
        plan=QueryPlan(as_of=d, as_of_explicit=True, question_type="point_in_time"),
        hits=hits,
        answer=Answer(
            text=text, abstained=abstained, as_of=d,
            citations=[Citation(source=s, chunk_id=hits[s - 1].chunk_id, quote="x", verified=v)
                       for s, v in citations],
        ),
    )


# ---------------------------------------------------------------- normalisation

@pytest.mark.parametrize("raw, expected", [
    ("Ten per cent", "10 per cent"),
    ("10%", "10 per cent"),
    ("10 percent", "10 per cent"),
    ("ninety-nine per cent", "99 per cent"),
    ("twenty five", "25"),
    ("one hundred crore rupees", "100 crore"),
    ("Rs. 1,00,000", "100000"),
    ("₹20 crore", "20 crore"),
    ("  Rule   36(4) ", "rule 36(4)"),
    ("FORM GSTR-2B", "form gstr 2b"),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_contains_fact_numbers_words_and_percent():
    assert contains_fact("limited to ten percent of eligible credit", "10 per cent")
    assert contains_fact("limited to 10% of eligible credit", "ten per cent")
    assert contains_fact("twenty crore rupees", "Rs 20 crore")


def test_contains_fact_respects_token_boundaries():
    assert not contains_fact("the cap was 25 per cent", "5 per cent")
    assert not contains_fact("the cap was 2.5%", "5 per cent")
    assert not contains_fact("limit of 1.5 crore", "1 crore")
    assert not contains_fact("fifty crore", "5 crore")
    assert contains_fact("limit of 5 crore.", "5 crore")


def test_answer_correct():
    text = "From 1 January 2020 the cap was 10% of eligible ITC [S1]."
    assert answer_correct(text, ["ten per cent"], ["5 per cent"])
    assert not answer_correct(text, ["20 per cent"])
    assert not answer_correct(text + " Later it became 5 per cent.", ["10 per cent"], ["5%"])
    assert answer_correct(text, [], [])


# ---------------------------------------------------------------- documents / validity

def test_canon_doc():
    assert canon_doc("Notification No. 01/2022-Central Tax") == canon_doc("1/2022-Central Tax")
    assert canon_doc("94/2020-Central Tax") == canon_doc("94/2020 - central tax")
    assert canon_doc("94/2020-Central Tax") != canon_doc("94/2020-Central Tax (Rate)")
    assert canon_doc(None) == ""


def test_in_force_uses_half_open_window_and_issue_date_fallback():
    h = hit(1, "75/2019-Central Tax", issue="2019-12-26", vf="2020-01-01", vt="2021-01-01")
    assert not in_force(h, date(2019, 12, 31))  # issued but not yet effective
    assert in_force(h, date(2020, 1, 1))
    assert in_force(h, date(2020, 12, 31))
    assert not in_force(h, date(2021, 1, 1))  # valid_to is exclusive
    no_validity = hit(2, "1/2022-Central Tax", issue="2022-02-24")
    assert not in_force(no_validity, date(2021, 6, 1))  # did not exist yet
    assert in_force(no_validity, date(2023, 1, 1))  # unknown end = still in force


def test_stale_citations_counts_only_cited_hits():
    hits = [
        hit(1, "94/2020-Central Tax", issue="2020-12-22", vf="2021-01-01"),  # future on as_of
        hit(2, "75/2019-Central Tax", issue="2019-12-26", vf="2020-01-01", vt="2021-01-01"),
        hit(3, "49/2019-Central Tax", issue="2019-10-09", vf="2019-10-09", vt="2020-01-01"),
    ]
    r = result(hits, [(1, True), (2, True)])
    assert stale_citations(r, date(2020, 2, 15)) == 1  # hit 3 is stale but not cited
    assert stale_citations(result(hits, [(3, True)]), date(2020, 2, 15)) == 1
    assert stale_citations(result(hits, [(2, False)]), date(2020, 2, 15)) == 0


# ---------------------------------------------------------------- retrieval / citations

def test_recall_at_k():
    hits = [hit(1, "a/2020-X"), hit(2, "b/2020-X"), hit(3, "75/2019-Central Tax")]
    assert recall_at_k(hits, ["75/2019-Central Tax"], 3) is True
    assert recall_at_k(hits, ["75/2019-Central Tax"], 2) is False
    assert recall_at_k(hits, [], 5) is None


def test_citation_precision_and_hit():
    hits = [hit(1, "75/2019-Central Tax"), hit(2, "other/2020-X"), hit(3, "75/2019-Central Tax")]
    gold = ["75/2019-Central Tax"]
    r = result(hits, [(1, False), (2, True), (3, True)])
    assert citation_precision(r, gold) == pytest.approx(0.5)  # 2 distinct docs, 1 gold
    assert citation_hit(r, gold) is True  # source 3 is gold and verified
    only_unverified = result(hits, [(1, False), (2, True)])
    assert citation_hit(only_unverified, gold) is False
    assert citation_precision(result(hits, []), gold) is None
    assert citation_hit(r, []) is None


def test_citation_resolves_by_chunk_id_when_index_is_off():
    hits = [hit(10, "x/2020-X"), hit(20, "75/2019-Central Tax")]
    r = result(hits, [])
    r.answer.citations = [Citation(source=1, chunk_id=20, quote="q", verified=True)]
    assert citation_hit(r, ["75/2019-Central Tax"]) is True


def test_abstention_correct():
    assert abstention_correct(abstained=True, answerable=False)
    assert abstention_correct(abstained=False, answerable=True)
    assert not abstention_correct(abstained=True, answerable=True)
    assert not abstention_correct(abstained=False, answerable=False)


# ---------------------------------------------------------------- percentiles / aggregation

def test_percentile_linear():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_score_and_aggregate():
    hits = [hit(1, "75/2019-Central Tax", issue="2019-12-26", vf="2020-01-01"),
            hit(2, "94/2020-Central Tax", issue="2020-12-22", vf="2021-01-01")]
    good = score_result(result(hits, [(1, True)], text="10 per cent [S1]"), answerable=True,
                        as_of=date(2020, 2, 15), gold_doc_numbers=["75/2019-Central Tax"],
                        must_contain=["ten per cent"], must_not_contain=["5 per cent"])
    assert good["answer_correct"] is True and good["gold_rank"] == 1
    assert good["stale_source"] is False and good["abstention_correct"] is True

    stale = score_result(result(hits, [(2, True)], text="5 per cent [S2]"), answerable=True,
                         as_of=date(2020, 2, 15), gold_doc_numbers=["75/2019-Central Tax"],
                         must_contain=["10 per cent"], must_not_contain=["5 per cent"])
    assert stale["answer_correct"] is False and stale["stale_source"] is True
    assert stale["citation_precision"] == 0.0

    abstain = score_result(result(hits, [], text="", abstained=True), answerable=False,
                           as_of=date(2019, 8, 1))
    assert abstain["answer_correct"] is None and abstain["abstention_correct"] is True
    assert abstain["stale_source"] is None and abstain["citation_hit"] is None

    records = [
        {"type": "point_in_time", "error": None, "latency_ms": 100.0, "scores": good},
        {"type": "point_in_time", "error": None, "latency_ms": 300.0, "scores": stale},
        {"type": "unanswerable", "error": None, "latency_ms": 200.0, "scores": abstain},
        {"type": "unanswerable", "error": "RuntimeError: boom", "latency_ms": None,
         "scores": None},
    ]
    m = aggregate(records)
    assert m["n_questions"] == 4 and m["n_errors"] == 1
    assert m["answer_accuracy"] == {"value": 0.5, "num": 1, "den": 2}
    assert m["recall@1"] == {"value": 1.0, "num": 2, "den": 2}  # unanswerable has no gold
    assert m["stale_source_rate"] == {"value": 0.5, "num": 1, "den": 2}
    assert m["abstention_accuracy"]["value"] == 1.0
    assert m["latency_p50_ms"] == pytest.approx(200.0)
    assert m["citation_precision"]["value"] == pytest.approx(0.5)

    by_type = per_type(records)
    assert set(by_type) == {"point_in_time", "unanswerable"}
    assert by_type["unanswerable"]["n_errors"] == 1
    assert by_type["unanswerable"]["answer_accuracy"]["value"] is None
