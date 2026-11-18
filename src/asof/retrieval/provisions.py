"""Structured lookups in the amendment history, used next to hybrid search.

Hybrid search finds chunks that *talk about* a rule. That is not the same as finding the
amendment that *set its wording* on a date: one omnibus notification (e.g. 94/2020) changes
twenty rules in a few lines, and long explanatory circulars about the same rule outrank it.
The temporal step already worked out, for each provision, which amendment text was in force
when (`provision_versions`), and which notifications amend which (`relations`). These two
lookups read those tables directly:

- `provision_hits`: the version(s) of a named rule/section in force on `as_of` (or, for a
  history question, every version up to `as_of`). Overlapping versions are normal: a proviso
  inserted later does not end the main text, so several can be in force at once.
- `chain_hits`: for history questions about a notification, the other notifications that
  amend the same base notification (e.g. the e-invoicing threshold: 13/2020 amended by
  88/2020, 5/2021, 1/2022, 17/2022).

Each result is a `Hit` whose text is the verbatim amendment wording (the stored evidence
quote) and whose chunk_id is the source document's chunk that contains it, so citations and
quote verification work exactly as for search hits.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from asof.models import Hit
from asof.temporal.validity import text_contains_quote

PIT_LIMIT = 4  # versions shown for a point-in-time question
HISTORY_LIMIT = 8  # versions / chain documents shown for a history question
CHAIN_KINDS = ("amends", "substitutes", "inserts", "omits", "rescinds", "supersedes")

_PROVISION_REF = re.compile(r"^(rule|section)\s+\S")


def provision_refs(refs: Sequence[str]) -> list[str]:
    """Lower-cased rule/section refs from a query plan ('Rule 21A(4)' -> 'rule 21a(4)')."""
    out: list[str] = []
    for r in refs:
        r = " ".join(r.strip().lower().split())
        if _PROVISION_REF.match(r) and r not in out:
            out.append(r)
    return out


_PROVISION_SQL = """
SELECT pv.id, p.ref, pv.text, lower(pv.validity), upper(pv.validity),
       d.id, d.kind, d.number, d.issue_date, d.title, d.url
  FROM provisions p
  JOIN provision_versions pv ON pv.provision_id = p.id
  JOIN documents d ON d.id = pv.source_doc_id
 WHERE (lower(p.ref) = ANY(%(refs)s) OR lower(p.ref) LIKE ANY(%(children)s))
   AND pv.text IS NOT NULL
   AND {date_test}
 ORDER BY lower(pv.validity) {order}, pv.id
 LIMIT 40
"""


def _chunks_by_doc(conn, doc_ids: Sequence[int]) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    if not doc_ids:
        return out
    for cid, did, page, text in conn.execute(
        "SELECT id, doc_id, page, text FROM chunks WHERE doc_id = ANY(%s) ORDER BY doc_id, ord",
        (list(doc_ids),),
    ).fetchall():
        out.setdefault(did, []).append((cid, page, text))
    return out


def _letters(s: str) -> str:
    """Lower-case letters and digits only: compares wording, not spacing or punctuation."""
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def _find_chunk(chunks: list[tuple], evidence: str) -> tuple[tuple, bool] | None:
    """(source chunk containing the evidence quote, whether it contains all of it).

    A chunk holding the whole quote wins over one that only shares a long run with it (the
    neighbouring chunk often repeats "in the said rules, in rule 9, ..."). None if no chunk
    matches at all.
    """
    want = _letters(evidence)
    for ch in chunks:
        if want and want in _letters(ch[2]):
            return ch, True
    for ch in chunks:
        if text_contains_quote(ch[2], evidence):
            return ch, False
    return None


def provision_hits(
    conn, refs: Sequence[str], as_of: date, *, history: bool = False
) -> list[Hit]:
    """Versions of the named provisions (and their sub-provisions) from `provision_versions`.

    Point in time: versions in force on `as_of`, newest first. History: every version that
    started on or before `as_of`, oldest first. Omitted versions (no text) are skipped: they
    have nothing to quote.
    """
    prefs = provision_refs(refs)
    if not prefs:
        return []
    date_test = "lower(pv.validity) <= %(as_of)s" if history else "pv.validity @> %(as_of)s"
    sql = _PROVISION_SQL.format(date_test=date_test, order="ASC" if history else "DESC")
    rows = conn.execute(sql, {
        "refs": prefs, "children": [r + "(%" for r in prefs], "as_of": as_of,
    }).fetchall()
    chunks = _chunks_by_doc(conn, sorted({r[5] for r in rows}))
    limit = HISTORY_LIMIT if history else PIT_LIMIT
    hits: list[Hit] = []
    seen: set[str] = set()
    for _vid, ref, text, lo, hi, did, kind, number, issued, title, url in rows:
        key = " ".join(text.split())
        if key in seen:
            continue
        found = _find_chunk(chunks.get(did, []), text)
        if found is None:
            continue
        ch, whole = found
        seen.add(key)
        # The evidence alone when the chunk has all of its words (focused, and quotes from it
        # are source wording); otherwise the whole chunk, so every quote is real text.
        shown = text if whole else ch[2]
        hits.append(Hit(
            chunk_id=ch[0], doc_id=did, kind=kind, number=number, issue_date=issued,
            title=title, url=url, section_ref=f"{ref} (amendment text)", page=ch[1],
            text=shown, valid_from=lo, valid_to=hi, score=1.0,
        ))
        if len(hits) >= limit:
            break
    return hits


_CHAIN_SQL = """
WITH seed AS (SELECT unnest(%(docs)s::bigint[]) AS id),
base AS (                       -- the seeds and the notifications they amend
  SELECT id FROM seed
  UNION
  SELECT r.dst_doc_id FROM relations r JOIN seed s ON s.id = r.src_doc_id
   WHERE r.dst_doc_id IS NOT NULL AND r.kind = ANY(%(kinds)s)
)
SELECT r.id, r.evidence_span, r.effective_date,
       d.id, d.kind, d.number, d.issue_date, d.title, d.url, b.number
  FROM relations r
  JOIN base ON base.id = r.dst_doc_id
  JOIN documents d ON d.id = r.src_doc_id
  JOIN documents b ON b.id = r.dst_doc_id
 WHERE r.kind = ANY(%(kinds)s)
   AND r.evidence_span IS NOT NULL
   AND (d.issue_date IS NULL OR d.issue_date <= %(as_of)s)
 ORDER BY coalesce(r.effective_date, d.issue_date), r.id
 LIMIT 60
"""


def chain_hits(conn, seed_doc_ids: Sequence[int], as_of: date) -> list[Hit]:
    """Amendments to the notifications behind `seed_doc_ids`, oldest first (history questions).

    One hit per amending document (its first evidence quote that can be located in a chunk).
    """
    seeds = list(dict.fromkeys(seed_doc_ids))
    if not seeds:
        return []
    rows = conn.execute(_CHAIN_SQL, {
        "docs": seeds, "kinds": list(CHAIN_KINDS), "as_of": as_of,
    }).fetchall()
    chunks = _chunks_by_doc(conn, sorted({r[3] for r in rows}))
    hits: list[Hit] = []
    done: set[int] = set()
    for _rid, evidence, eff, did, kind, number, issued, title, url, base in rows:
        if did in done:
            continue
        found = _find_chunk(chunks.get(did, []), evidence)
        if found is None:
            continue
        ch = found[0]
        done.add(did)
        hits.append(Hit(
            chunk_id=ch[0], doc_id=did, kind=kind, number=number, issue_date=issued,
            title=title, url=url, section_ref=f"amends {base}", page=ch[1],
            text=ch[2], valid_from=eff or issued, valid_to=None, score=1.0,
        ))
        if len(hits) >= HISTORY_LIMIT:
            break
    return hits


def merge_hits(first: Sequence[Hit], rest: Sequence[Hit], k: int) -> list[Hit]:
    """`first` (structured lookups) then `rest` (search results), no chunk twice, at most k."""
    out: list[Hit] = []
    ids: set[int] = set()
    for h in [*first, *rest]:
        if len(out) >= k:
            break
        if h.chunk_id not in ids:
            ids.add(h.chunk_id)
            out.append(h)
    return out
