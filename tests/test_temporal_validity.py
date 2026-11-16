"""Validity resolution: document rescission, provision-version chaining, chunk ranges (no DB)."""

import json
from datetime import date
from pathlib import Path

from asof.models import ExtractionResult, ParsedDoc
from asof.temporal.extract import postprocess
from asof.temporal.normalize import provision_key
from asof.temporal.validity import (
    ChunkRow,
    DocDates,
    Event,
    Termination,
    build_versions,
    chunk_matches,
    chunk_overrides,
    document_validity,
)

FIXTURE = Path(__file__).parent / "fixtures" / "temporal" / "rule36_chain.json"
D = date


def in_range(day: date, lo: date | None, hi: date | None) -> bool:
    return (lo is None or lo <= day) and (hi is None or day < hi)


# ------------------------------------------------------------------ documents

def test_document_validity_rescission_and_supersession():
    docs = [DocDates(1, D(2017, 6, 28)), DocDates(2, D(2018, 1, 23)),
            DocDates(3, D(2019, 3, 1), effective_date=D(2019, 4, 1)),
            DocDates(4, D(2020, 5, 5)), DocDates(5, D(2017, 7, 1))]
    terms = [
        Termination(src_doc_id=3, dst_doc_id=1, kind="rescinds"),          # at 3's eff date
        Termination(src_doc_id=2, dst_doc_id=1, kind="supersedes"),        # earlier: wins
        Termination(src_doc_id=4, dst_doc_id=3, kind="rescinds",
                    effective_date=D(2020, 7, 1)),                         # explicit date
        Termination(src_doc_id=1, dst_doc_id=4, kind="rescinds"),          # bogus: before start
        Termination(src_doc_id=2, dst_doc_id=99, kind="rescinds"),         # unknown target
    ]
    anomalies: list[str] = []
    v = document_validity(docs, terms, anomalies)
    assert (v[1].lo, v[1].hi, v[1].status) == (D(2017, 6, 28), D(2018, 1, 23), "superseded")
    assert (v[3].lo, v[3].hi, v[3].status) == (D(2019, 4, 1), D(2020, 7, 1), "rescinded")
    assert (v[4].lo, v[4].hi, v[4].status) == (D(2020, 5, 5), None, "in_force")
    assert (v[2].hi, v[2].status) == (None, "in_force")
    assert v[5].status == "in_force"
    assert len(anomalies) == 1


def test_document_without_dates_is_unbounded():
    v = document_validity([DocDates(1, None)], [])
    assert (v[1].lo, v[1].hi, v[1].status) == (None, None, "in_force")


# ------------------------------------------------------------------ provision versions

def chain_events() -> list[Event]:
    return [
        Event(10, 2, "substitutes", D(2020, 1, 1), 'for "twenty per cent.", "ten per cent."',
              D(2019, 12, 26)),
        Event(11, 1, "inserts", D(2019, 10, 9), "(4) ... shall not exceed twenty per cent.",
              D(2019, 10, 9)),
        Event(12, 4, "omits", D(2022, 1, 1), "sub-rule (4) shall be omitted", D(2021, 12, 29)),
        Event(13, 3, "substitutes", D(2021, 1, 1), 'for "ten per cent.", "five per cent."',
              D(2020, 12, 22)),
    ]


def test_build_versions_rule36_chain():
    vs = build_versions(chain_events())
    assert [(v.lo, v.hi, v.source_doc_id, v.kind) for v in vs] == [
        (D(2019, 10, 9), D(2020, 1, 1), 1, "inserts"),
        (D(2020, 1, 1), D(2021, 1, 1), 2, "substitutes"),
        (D(2021, 1, 1), D(2022, 1, 1), 3, "substitutes"),
        (D(2022, 1, 1), None, 4, "omits"),
    ]
    assert vs[-1].text is None  # omitted
    at = {day: next(v for v in vs if in_range(day, v.lo, v.hi))
          for day in (D(2019, 11, 1), D(2020, 2, 15), D(2021, 6, 1), D(2023, 1, 1))}
    assert "twenty" in at[D(2019, 11, 1)].text
    assert '"ten per cent."' in at[D(2020, 2, 15)].text
    assert "five" in at[D(2021, 6, 1)].text
    assert at[D(2023, 1, 1)].text is None
    assert not any(in_range(D(2019, 1, 1), v.lo, v.hi) for v in vs)  # before insertion


def test_build_versions_same_day_and_undated():
    evs = [
        Event(1, 5, "substitutes", D(2020, 1, 1), "old doc text", D(2019, 6, 1)),
        Event(2, 6, "substitutes", D(2020, 1, 1), "later doc text", D(2019, 12, 1)),
        Event(3, 6, "omits", D(2020, 1, 1), "omitted", D(2019, 12, 1)),   # replace = omit+insert
        Event(4, 7, "amends", None, "no date", None),                     # cannot be placed
    ]
    vs = build_versions(evs)
    assert len(vs) == 1
    assert (vs[0].source_doc_id, vs[0].text, vs[0].hi) == (6, "later doc text", None)
    assert "omitted" in vs[0].evidences


def test_build_versions_closed_by_ancestor_substitution():
    vs = build_versions(chain_events()[:2], closers=[D(2020, 6, 1)])
    assert [(v.lo, v.hi) for v in vs] == [(D(2019, 10, 9), D(2020, 1, 1)),
                                          (D(2020, 1, 1), D(2020, 6, 1))]


# ------------------------------------------------------------------ chunks

def test_chunk_matches_by_ref_mention_or_evidence():
    ref = "rule 36(4)"
    assert chunk_matches(ChunkRow(1, 1, "rule 36(4)", "anything"), ref)
    assert chunk_matches(ChunkRow(2, 1, None, "in rule 36, in sub-rule (4), for the words"), ref)
    assert chunk_matches(ChunkRow(3, 1, None, "in rule 36, for the words"), ref)  # ancestor
    assert not chunk_matches(ChunkRow(4, 1, None, "in rule 37, sub-rule (4) is omitted"), ref)
    quote = 'for the words "ten per cent.", the words "five per cent." shall be substituted'
    assert chunk_matches(ChunkRow(5, 1, None, "... For the words “ten per cent.”, "
                                              "the words “five per cent.” shall be "
                                              "substituted."), ref, [quote])
    assert not chunk_matches(ChunkRow(6, 1, None, "These rules may be called ..."), ref, [quote])


def test_end_to_end_fixture_asof_retrieves_the_right_chunk():
    """Fixture docs -> (mocked) extraction -> versions -> chunk ranges -> as-of filter."""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["docs"]
    docs: dict[int, DocDates] = {}
    events: list[Event] = []
    chunks: dict[int, list[ChunkRow]] = {}
    texts: dict[int, str] = {}
    rid = 0
    for case in cases:
        doc = ParsedDoc.model_validate(case["parsed"])
        result = postprocess(ExtractionResult.model_validate(case["llm_output"]), doc,
                             "\n".join(p.text for p in doc.pages))
        did = case["id"]
        docs[did] = DocDates(did, doc.meta.issue_date, result.effective_date)
        for rel in result.relations:
            # hallucinated relations (rescinds 99/2018, rule 37) were dropped by postprocess
            assert provision_key(rel.target_kind, rel.target_ref) == ("cgst_rules", "rule 36(4)")
            rid += 1
            events.append(Event(rid, did, rel.kind, rel.effective_date or docs[did].start,
                                rel.evidence, doc.meta.issue_date))
        chunks[did] = [ChunkRow(c["id"], did, c["section_ref"], c["text"]) for c in case["chunks"]]
        texts.update({c["id"]: c["text"] for c in case["chunks"]})

    versions = build_versions(events)
    assert len(versions) == 4
    ov = chunk_overrides(((i, "rule 36(4)", v) for i, v in enumerate(versions)), chunks)
    # preamble chunks are not about the provision: they keep the document range
    assert 101 not in ov and 201 not in ov
    assert set(ov) == {102, 202, 301, 401}

    doc_v = document_validity(docs.values(), [])

    def chunk_range(cid: int, did: int):
        if cid in ov:
            return ov[cid][:2]
        return doc_v[did].lo, doc_v[did].hi

    def asof(day: date) -> set[int]:
        return {c.id for did, cs in chunks.items() for c in cs
                if c.id in ov and in_range(day, *chunk_range(c.id, did))}

    assert asof(D(2019, 11, 1)) == {102}
    hit = asof(D(2020, 2, 15))
    assert hit == {202} and "ten per cent" in texts[202]
    assert asof(D(2021, 6, 1)) == {301}
    assert asof(D(2022, 3, 1)) == {401}
    # 75/2019 was issued 2019-12-26 but its rule-36(4) text only applies from 2020-01-01
    assert asof(D(2019, 12, 28)) == {102}


def test_insertion_does_not_end_earlier_wording():
    """A new proviso leaves "five per cent." in force; only a change of that wording ends it."""
    evs = [
        Event(1, 3, "substitutes", D(2021, 1, 1),
              'for the words "ten per cent.", the words "five per cent." shall be substituted',
              D(2020, 12, 22)),
        Event(2, 5, "inserts", D(2021, 5, 1), 'after the first proviso, the following proviso '
              'shall be inserted, namely:- "Provided further that the condition applies '
              'cumulatively"', D(2021, 5, 1)),
        Event(3, 6, "substitutes", D(2021, 6, 1), 'for the second proviso, the following proviso '
              'shall be substituted, namely:- "Provided further that the condition applies '
              'cumulatively for April, May and June"', D(2021, 6, 1)),
        Event(4, 7, "substitutes", D(2021, 9, 1), 'for the words "Provided further that the '
              'condition applies cumulatively for April, May and June", the words "Provided '
              'further that it applies for July" shall be substituted', D(2021, 9, 1)),
    ]
    vs = build_versions(evs)
    by_doc = {v.source_doc_id: (v.lo, v.hi) for v in vs}
    # 6 substitutes only "the second proviso" (a sub-unit), so it must not end the
    # "five per cent." wording; the replaced proviso's version may overlap (conservative:
    # an extra in-force passage is better than losing the one that answers the question).
    assert by_doc[3] == (D(2021, 1, 1), None)
    assert by_doc[5] == (D(2021, 5, 1), None)
    # 7 replaces exactly the wording 6 introduced
    assert by_doc[6] == (D(2021, 6, 1), D(2021, 9, 1))
    assert by_doc[7] == (D(2021, 9, 1), None)
    live_may = [v.source_doc_id for v in vs if in_range(D(2021, 5, 15), v.lo, v.hi)]
    assert live_may == [3, 5]  # "five per cent." is still the law alongside the new proviso


def test_partial_omission_and_reinsertion():
    evs = [
        Event(1, 1, "inserts", D(2019, 1, 1), 'the following clause shall be inserted, namely:- '
              '"(g) details in FORM GSTR-2"', D(2019, 1, 1)),
        Event(2, 2, "inserts", D(2019, 6, 1), 'the following clause shall be inserted, namely:- '
              '"(h) other details"', D(2019, 6, 1)),
        Event(3, 3, "omits", D(2020, 1, 1), 'the words "FORM GSTR-2" shall be omitted',
              D(2020, 1, 1)),
        Event(4, 4, "omits", D(2021, 1, 1), "sub-rule (1) shall be omitted", D(2021, 1, 1)),
        Event(5, 5, "inserts", D(2022, 1, 1), 'the following sub-rule shall be inserted, '
              'namely:- "(1) new text"', D(2022, 1, 1)),
    ]
    vs = {v.source_doc_id: v for v in build_versions(evs)}
    assert vs[1].hi == D(2020, 1, 1)          # its wording was omitted
    assert vs[2].hi == D(2021, 1, 1)          # untouched until the whole unit went
    assert vs[3].text is not None             # a partial omission still has text
    assert (vs[4].text, vs[4].lo, vs[4].hi) == (None, D(2021, 1, 1), D(2022, 1, 1))
    assert vs[5].hi is None


def test_partial_substitution_of_a_proviso_keeps_earlier_versions():
    """27/2021-style: replacing a proviso must not end the 'five per cent' version."""
    from asof.temporal.validity import Event, build_versions, is_partial

    D = date
    events = [
        Event(1, 1, "substitutes", D(2021, 1, 1),
              'in sub-rule (4), for the words "ten per cent.", the words "five per cent." '
              "shall be substituted", D(2020, 12, 22)),
        Event(2, 2, "inserts", D(2021, 5, 1),
              'after the first proviso, the following proviso shall be inserted, namely:- '
              '"Provided further that ..."', D(2021, 5, 1)),
        Event(3, 3, "substitutes", D(2021, 6, 1),
              'in sub-rule (4) of rule 36, for the second proviso, the following proviso shall '
              'be substituted, namely: - "Provided also that ..."', D(2021, 6, 1)),
        Event(4, 4, "substitutes", D(2022, 1, 1),
              'in rule 36, for sub-rule (4), the following sub-rule shall be substituted, '
              'namely:- "(4) No input tax credit shall be availed ... clause (aa) ..."',
              D(2021, 12, 29)),
    ]
    assert is_partial(events[2]) and not is_partial(events[3])  # quote text is ignored
    v = {x.source_doc_id: x for x in build_versions(events)}
    assert v[1].lo == D(2021, 1, 1) and v[1].hi == D(2022, 1, 1)  # 5% survives the proviso swap
    assert v[3].hi == D(2022, 1, 1) and v[4].hi is None


def test_partial_omission_of_explanation_is_not_whole_omission():
    from asof.temporal.validity import Event, build_versions

    D = date
    events = [
        Event(1, 1, "inserts", D(2020, 1, 1), 'the following sub-rule shall be inserted, '
              'namely:- "(5) a new rule"', D(2020, 1, 1)),
        Event(2, 2, "omits", D(2021, 1, 1), "in sub-rule (5), the Explanation shall be omitted",
              D(2021, 1, 1)),
    ]
    v = {x.source_doc_id: x for x in build_versions(events)}
    assert v[1].hi is None  # not ended by removing an Explanation
