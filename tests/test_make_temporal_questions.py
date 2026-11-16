import importlib.util
from datetime import date
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mtq", Path(__file__).parents[1] / "scripts" / "make_temporal_questions.py")
mtq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtq)

CHAIN = {
    "instrument": "cgst_rules",
    "ref": "rule 36(4)",
    "versions": [
        {"text": "shall not exceed twenty per cent of the eligible credit",
         "from": date(2019, 10, 9), "to": date(2020, 1, 1), "doc": "A/2019-Central Tax"},
        {"text": "shall not exceed ten per cent of the eligible credit",
         "from": date(2020, 1, 1), "to": date(2021, 1, 1), "doc": "B/2019-Central Tax"},
        {"text": "", "from": date(2022, 1, 1), "to": None, "doc": "C/2021-Central Tax"},
    ],
}


def test_build_keeps_only_verbatim_distinguishing_facts():
    prop = mtq.Proposal(
        question_template="What did rule 36(4) provide on {date}?",
        facts=[
            mtq.VersionFact(version=0, key_fact="twenty per cent"),
            mtq.VersionFact(version=1, key_fact="ten per cent"),
            mtq.VersionFact(version=2, key_fact="omitted"),
        ],
    )
    qs = mtq.build(CHAIN, prop, today=date(2026, 11, 16))
    by_id = {q["id"]: q for q in qs}

    v1 = by_id["t-rule-36-4-1"]
    assert v1["must_contain"] == ["ten per cent"]
    assert v1["must_not_contain"] == ["twenty per cent"]
    assert date(2020, 1, 1) <= date.fromisoformat(v1["as_of"]) < date(2021, 1, 1)
    assert v1["gold_doc_numbers"] == ["B/2019-Central Tax"]

    v2 = by_id["t-rule-36-4-2"]
    assert v2["type"] == "current" and v2["must_contain"] == []
    assert set(v2["must_not_contain"]) == {"twenty per cent", "ten per cent"}


def test_build_drops_non_verbatim_or_shared_facts():
    prop = mtq.Proposal(
        question_template="What did rule 36(4) provide on {date}?",
        facts=[
            mtq.VersionFact(version=0, key_fact="20%"),  # not verbatim in the text
            mtq.VersionFact(version=1, key_fact="eligible credit"),  # in every version
        ],
    )
    assert mtq.build(CHAIN, prop, today=date(2026, 11, 16)) == []


def test_build_requires_date_placeholder():
    prop = mtq.Proposal(question_template="What did rule 36(4) provide?", facts=[])
    assert mtq.build(CHAIN, prop, today=date(2026, 11, 16)) == []
