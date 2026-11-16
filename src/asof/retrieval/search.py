"""Hybrid, date-filtered retrieval over `chunks` (pgvector + Postgres full-text search).

`hybrid_search` runs ONE SQL statement:

- ``vec``  CTE: top 50 chunks by cosine distance (``embedding <=> q``),
- ``fts``  CTE: top 50 chunks by ``ts_rank_cd`` against an OR-ed ``websearch_to_tsquery``,
- ``refs`` CTE (only when the question names a rule/section/notification/circular): up to 50
  chunks whose ``section_ref`` or document number matches, so a named instrument is always a
  candidate even if neither ranker found it,

fused with reciprocal rank fusion: ``score = sum(1 / (RRF_K + rank))`` over the vector and FTS
lists, plus ``REF_BOOST`` when the chunk's ``section_ref``, text or document number matches an
explicit reference.

As-of filter (asof mode): ``chunks.validity @> as_of AND documents.issue_date <= as_of``
(documents without an issue date, e.g. the Acts, pass the issue-date test). Baseline mode
passes ``as_of=None``: no date filter at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date

from pgvector import HalfVector

from asof.models import Hit

RRF_K = 60
CANDIDATES = 50
# Slightly more than a rank-1 hit in one list (1/61 ~= 0.0164): an explicitly named
# rule/notification beats an equally ranked unnamed one, but two strong ranks still matter.
REF_BOOST = 0.02
HNSW_EF_SEARCH = 100

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "is", "it", "its", "may", "not", "of", "on",
    "or", "per", "the", "to", "under", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "should", "there", "this", "that", "these", "those", "any",
}


def rrf_scores(rankings: Iterable[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion of ranked id lists (rank 1 = first). Mirrors the SQL."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def fts_query_text(*texts: str, max_terms: int = 40) -> str:
    """OR together the distinct content words, for ``websearch_to_tsquery``.

    websearch_to_tsquery ANDs plain words, which on a full question almost never matches a
    single chunk. ``a or b or c`` keeps recall; ts_rank_cd still rewards chunks with more
    (and denser) matches.
    """
    words: list[str] = []
    for text in texts:
        for w in re.findall(r"[A-Za-z0-9]+", text or ""):
            lw = w.lower()
            if lw in _STOP or (len(lw) < 2 and not lw.isdigit()):
                continue
            words.append(lw)
    return " or ".join(list(dict.fromkeys(words))[:max_terms])


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def ref_patterns(refs: Sequence[str]) -> dict[str, list[str]]:
    """ILIKE patterns for explicit references.

    - ``section``: section_ref starts with the ref ("rule 36(4)" matches "rule 36(4)(a)")
    - ``text``:    the chunk text contains the ref
    - ``number``:  documents.number starts with the ref ("49/2019" -> "49/2019-Central Tax")
    """
    section, text, number = [], [], []
    for ref in refs:
        esc = _like_escape(ref.strip())
        if not esc:
            continue
        if re.match(r"(rule|section)\s", ref, re.I):
            section.append(f"{esc}%")
            text.append(f"%{esc}%")
        else:
            number.append(f"{esc}%")
            text.append(f"%{esc}%")
    return {"section": section, "text": text, "number": number}


_DATE_FILTER = (
    "c.validity @> %(as_of)s::date "
    "AND coalesce(d.issue_date, '-infinity'::date) <= %(as_of)s::date"
)

_HIT_COLUMNS = """
  c.id, c.doc_id, d.kind, d.number,
  CASE WHEN isfinite(d.issue_date) THEN d.issue_date END AS issue_date,
  d.title, d.url, c.section_ref, c.page, c.text,
  CASE WHEN isfinite(lower(c.validity)) THEN lower(c.validity) END AS valid_from,
  CASE WHEN isfinite(upper(c.validity)) THEN upper(c.validity) END AS valid_to"""

_REF_MATCH = (
    "(c.section_ref ILIKE ANY(%(ref_section)s::text[]) "
    "OR c.text ILIKE ANY(%(ref_text)s::text[]) "
    "OR d.number ILIKE ANY(%(ref_number)s::text[]))"
)


def build_hybrid_sql(*, dated: bool, with_refs: bool) -> str:
    where = f"AND {_DATE_FILTER}" if dated else ""
    refs_cte = ""
    refs_union = ""
    if with_refs:
        refs_cte = f""",
refs AS (
  SELECT c.id, NULL::bigint AS rnk
  FROM chunks c JOIN documents d ON d.id = c.doc_id
  WHERE (c.section_ref ILIKE ANY(%(ref_section)s::text[])
         OR d.number ILIKE ANY(%(ref_number)s::text[])) {where}
  ORDER BY d.issue_date DESC NULLS LAST, c.ord
  LIMIT {CANDIDATES}
)"""
        refs_union = "UNION ALL SELECT id, rnk FROM refs"
    boost = f"CASE WHEN {_REF_MATCH} THEN {REF_BOOST} ELSE 0 END" if with_refs else "0"
    return f"""
WITH q AS (
  SELECT %(qvec)s::halfvec AS v, websearch_to_tsquery('english', %(qtext)s) AS tsq
),
vec_raw AS MATERIALIZED (
  SELECT c.id, c.embedding <=> q.v AS dist
  FROM chunks c JOIN documents d ON d.id = c.doc_id CROSS JOIN q
  WHERE c.embedding IS NOT NULL {where}
  ORDER BY c.embedding <=> q.v
  LIMIT {CANDIDATES}
),
vec AS (
  SELECT id, row_number() OVER (ORDER BY dist, id) AS rnk FROM vec_raw
),
fts AS (
  SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.id) AS rnk
  FROM chunks c JOIN documents d ON d.id = c.doc_id CROSS JOIN q
  WHERE c.tsv @@ q.tsq {where}
  ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.id
  LIMIT {CANDIDATES}
){refs_cte},
fused AS (
  SELECT id, sum(CASE WHEN rnk IS NULL THEN 0 ELSE 1.0 / ({RRF_K} + rnk) END) AS rrf
  FROM (SELECT id, rnk FROM vec UNION ALL SELECT id, rnk FROM fts {refs_union}) u
  GROUP BY id
)
SELECT {_HIT_COLUMNS},
  (f.rrf + {boost})::float8 AS score
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.doc_id
ORDER BY score DESC, c.id
LIMIT %(k)s
"""


VECTOR_SQL = f"""
SELECT {_HIT_COLUMNS},
  (1 - (c.embedding <=> %(qvec)s::halfvec))::float8 AS score
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(qvec)s::halfvec
LIMIT %(k)s
"""


def row_to_hit(row: Sequence) -> Hit:
    (chunk_id, doc_id, kind, number, issue_date, title, url, section_ref, page, text,
     valid_from, valid_to, score) = row
    return Hit(
        chunk_id=chunk_id, doc_id=doc_id, kind=kind, number=number, issue_date=issue_date,
        title=title or "", url=url, section_ref=section_ref, page=page, text=text,
        valid_from=valid_from, valid_to=valid_to, score=float(score or 0.0),
    )


def _tune_hnsw(conn) -> None:
    """Session-local HNSW settings so filtered queries still return enough rows."""
    conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")  # pgvector >= 0.8


def hybrid_search(
    conn,
    query_vec: Sequence[float],
    query_text: str,
    *,
    as_of: date | None,
    refs: Sequence[str] = (),
    k: int = 30,
) -> list[Hit]:
    """Vector + full-text candidates fused by RRF, filtered to chunks in force on `as_of`."""
    pats = ref_patterns(refs)
    with_refs = any(pats.values())
    params = {
        "qvec": HalfVector(list(query_vec)),
        "qtext": fts_query_text(query_text, *refs),
        "as_of": as_of,
        "k": k,
        "ref_section": pats["section"],
        "ref_text": pats["text"],
        "ref_number": pats["number"],
    }
    sql = build_hybrid_sql(dated=as_of is not None, with_refs=with_refs)
    with conn.transaction():
        _tune_hnsw(conn)
        rows = conn.execute(sql, params).fetchall()
    return [row_to_hit(r) for r in rows]


def vector_search(conn, query_vec: Sequence[float], *, k: int = 8) -> list[Hit]:
    """Plain cosine top-k, no filters: the naive-RAG baseline."""
    with conn.transaction():
        _tune_hnsw(conn)
        rows = conn.execute(VECTOR_SQL, {"qvec": HalfVector(list(query_vec)), "k": k}).fetchall()
    return [row_to_hit(r) for r in rows]
