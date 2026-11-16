"""Hybrid search pieces that can be tested without a database."""

import re
from contextlib import contextmanager
from datetime import date

import pytest
from pgvector import HalfVector

from asof.retrieval.search import (
    REF_BOOST,
    RRF_K,
    build_hybrid_sql,
    fts_query_text,
    hybrid_search,
    ref_patterns,
    row_to_hit,
    rrf_scores,
    vector_search,
)


def test_rrf_math():
    scores = rrf_scores([[10, 20, 30], [20, 40]])
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[10] == pytest.approx(1 / 61)
    assert scores[40] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == 20  # in both lists beats rank 1 in one
    assert rrf_scores([[1]], k=0)[1] == 1.0


def test_ref_boost_outweighs_a_single_top_rank_but_not_two_lists():
    assert REF_BOOST > 1 / (RRF_K + 1)
    assert REF_BOOST < 2 / (RRF_K + 1)


def test_fts_query_text_ors_content_words():
    q = fts_query_text("What was the ITC cap under rule 36(4)?", "ITC cap", "rule 36(4)")
    assert q == "itc or cap or rule or 36 or 4"
    assert fts_query_text("the of and") == ""


def test_ref_patterns_escape_like_and_split_by_kind():
    pats = ref_patterns(["rule 36(4)", "49/2019-Central Tax", "a_b%"])
    assert pats["section"] == ["rule 36(4)", "rule 36(4)(%"]
    assert pats["number"] == ["49/2019-Central Tax", "a\\_b\\%"]  # no wildcard: equality
    assert len(pats["text"]) == 2  # numbers only; provisions use chunks.refs
    assert not any("36" in t for t in pats["text"])


def _ilike(pattern: str, value: str) -> bool:
    """Python model of Postgres ILIKE with backslash escapes."""
    rx = ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            rx += re.escape(pattern[i + 1])
            i += 2
            continue
        rx += ".*" if ch == "%" else "." if ch == "_" else re.escape(ch)
        i += 1
    return re.fullmatch(rx, value, re.I | re.S) is not None


@pytest.mark.parametrize("ref, section_ref, hit", [
    ("rule 3", "rule 3", True),
    ("rule 3", "rule 3(1)", True),
    ("rule 3", "rule 36(4)", False),
    ("rule 3", "rule 31A", False),
    ("rule 36(4)", "rule 36(4)(a)", True),
    ("rule 36(4)", "Rule 36(4)", True),
    ("rule 36(4)", "rule 36(40)", False),
    ("section 1", "section 129", False),
    ("section 1", "section 1(2)", True),
    ("rule 86A", "rule 86", False),
])
def test_section_patterns_do_not_overmatch(ref, section_ref, hit):
    pats = ref_patterns([ref])["section"]
    assert any(_ilike(p, section_ref) for p in pats) is hit


@pytest.mark.parametrize("ref, number, hit", [
    ("9/2019-Central Tax", "9/2019-Central Tax", True),
    ("9/2019-Central Tax", "9/2019-central tax", True),
    ("9/2019-Central Tax", "49/2019-Central Tax", False),
    ("2/2019-Central Tax", "2/2019-Central Tax (Rate)", False),
    ("2/2019-Central Tax (Rate)", "2/2019-Central Tax (Rate)", True),
    ("2/2019-Central Tax (Rate)", "2/2019-Central Tax", False),
    ("49/2019", "49/2019-Central Tax", True),  # bare number: any series
    ("49/2019", "149/2019-Central Tax", False),
    ("123/42/2019-GST", "123/42/2019-GST", True),
    ("23/42/2019-GST", "123/42/2019-GST", False),
])
def test_number_patterns_match_document_number_exactly(ref, number, hit):
    pats = ref_patterns([ref])["number"]
    assert any(_ilike(p, number) for p in pats) is hit


@pytest.mark.parametrize("ref, text, hit", [
    ("9/2019-Central Tax", "vide notification No. 9/2019-Central Tax, dated", True),
    ("9/2019-Central Tax", "notification 9/2019 – Central  Tax", True),  # flexible separators
    ("9/2019-Central Tax", "No. 49/2019-Central Tax", False),
    ("9/2019-Central Tax", "No. 9/2019-Central Tax (Rate)", False),
    ("9/2019-Central Tax", "9/2019-Central Tax", True),  # at start of text
    ("2/2019-Central Tax", "notification No. 2/2019-Central Tax ( Rate ) dated", False),
    ("2/2019-Central Tax (Rate)", "notification No. 2/2019-Central Tax (Rate) dated", True),
    ("2/2019-Central Tax (Rate)", "notification No. 2/2019-Central Tax dated", False),
    ("2/2019-Central Tax (Rate)", "No. 12/2019-Central Tax (Rate)", False),
    ("123/42/2019-GST", "Circular No. 123/42/2019-GST dated", True),
    ("123/42/2019-GST", "Circular No. 1123/42/2019-GST", False),
    ("123/42/2019-GST", "Circular No. 123/42/2019-GSTR", False),
    ("49/2019", "notification 49/2019-Central Tax", True),
    ("49/2019", "notification 149/2019-Central Tax", False),
    ("1.5/2019-Central Tax", "1x5/2019-Central Tax", False),  # metacharacters escaped
])
def test_number_text_regex_is_word_bounded(ref, text, hit):
    (rx,) = ref_patterns([ref])["text"]
    # Python's re agrees with Postgres ARE on every construct used here (~* = case-insensitive)
    assert (re.search(rx, text, re.I) is not None) is hit


def test_sql_date_filter_only_when_dated():
    dated = build_hybrid_sql(dated=True, with_refs=False)
    assert dated.count("c.validity @> %(as_of)s::date") == 2  # vec + fts CTEs
    assert "d.issue_date, '-infinity'::date) <= %(as_of)s::date" in dated
    assert "websearch_to_tsquery('english', %(qtext)s)" in dated
    assert "ts_rank_cd" in dated and "<=>" in dated
    assert f"1.0 / ({RRF_K} + rnk)" in dated
    assert "refs AS" not in dated and str(REF_BOOST) not in dated

    undated = build_hybrid_sql(dated=False, with_refs=False)
    assert "validity @>" not in undated and "%(as_of)s" not in undated


def test_sql_ref_cte_and_boost():
    sql = build_hybrid_sql(dated=True, with_refs=True)
    assert "refs AS" in sql and "UNION ALL SELECT id, rnk FROM refs" in sql
    assert f"THEN {REF_BOOST}" in sql
    assert sql.count("c.validity @> %(as_of)s::date") == 3  # refs are date-filtered too
    assert "c.text ~* ANY(%(ref_text)s::text[])" in sql  # regex, not ILIKE, for text
    assert "c.text ILIKE" not in sql


def test_sql_history_mode_keeps_issue_date_but_not_validity():
    sql = build_hybrid_sql(dated=True, with_refs=True, in_force=False)
    assert "validity @>" not in sql
    assert sql.count("d.issue_date, '-infinity'::date) <= %(as_of)s::date") == 3


def test_hybrid_search_in_force_false_drops_validity_filter():
    conn = FakeConn([])
    hybrid_search(conn, [0.1], "q", as_of=date(2020, 2, 15), refs=["rule 36(4)"],
                  in_force=False)
    sql, params = conn.calls[-1]
    assert "validity @>" not in sql and "<= %(as_of)s::date" in sql
    assert params["as_of"] == date(2020, 2, 15)


def test_row_to_hit():
    row = (1, 2, "notification", "49/2019-Central Tax", date(2019, 10, 9), "Seeks to amend",
           "https://x/y.pdf", "rule 36(4)", 3, "text", date(2019, 10, 9), None, 0.03)
    hit = row_to_hit(row)
    assert hit.chunk_id == 1 and hit.valid_to is None and hit.score == 0.03
    assert hit.kind == "notification" and hit.page == 3


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, object]] = []
        self.in_tx = False

    @contextmanager
    def transaction(self):
        self.in_tx = True
        yield
        self.in_tx = False

    def execute(self, sql, params=None):
        assert self.in_tx, "search must run inside a transaction (SET LOCAL)"
        self.calls.append((sql, params))
        return self

    def fetchall(self):
        return self.rows


ROW = (7, 1, "circular", "123/42/2019-GST", date(2019, 11, 11), "Clarification", None, None,
       None, "body", date(2019, 11, 11), date(2020, 1, 1), 0.5)


def test_hybrid_search_params_and_hnsw_tuning():
    conn = FakeConn([ROW])
    hits = hybrid_search(conn, [0.1, 0.2], "ITC cap rule 36(4)", as_of=date(2020, 2, 15),
                         refs=["rule 36(4)"], k=5)
    assert [h.chunk_id for h in hits] == [7]
    assert hits[0].valid_to == date(2020, 1, 1)
    sets = [sql for sql, _ in conn.calls[:2]]
    assert sets == ["SET LOCAL hnsw.ef_search = 100",
                    "SET LOCAL hnsw.iterative_scan = relaxed_order"]
    sql, params = conn.calls[2]
    assert "refs AS" in sql and "validity @>" in sql
    assert isinstance(params["qvec"], HalfVector)
    assert params["as_of"] == date(2020, 2, 15) and params["k"] == 5
    assert params["ref_section"] == ["rule 36(4)", "rule 36(4)(%"]
    assert params["ref_exact"] == ["rule 36(4)", "rule 36"]
    assert "itc or cap" in params["qtext"]


def test_hybrid_search_without_date_has_no_filter():
    conn = FakeConn([])
    assert hybrid_search(conn, [0.1], "q", as_of=None) == []
    sql, params = conn.calls[-1]
    assert "validity @>" not in sql and "refs AS" not in sql


def test_vector_search_has_no_filters():
    conn = FakeConn([ROW])
    hits = vector_search(conn, [0.1, 0.2], k=8)
    sql, params = conn.calls[-1]
    assert "validity" not in sql.split("FROM")[-1] and "@@" not in sql
    assert params["k"] == 8 and hits[0].number == "123/42/2019-GST"


def test_ref_with_ancestors():
    from asof.retrieval.search import ref_with_ancestors

    assert ref_with_ancestors(["rule 9(2)(a)", "49/2019-Central Tax", "Section 16(4)"]) == [
        "rule 9(2)(a)", "rule 9(2)", "rule 9", "section 16(4)", "section 16"]
