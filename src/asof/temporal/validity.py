"""Recompute validity ranges for documents, provision versions and chunks from `relations`.

Validity ranges are half-open [lo, hi); hi = None means still in force.

Pure-Python core (testable without a DB):
- `document_validity`: a document is valid from coalesce(effective_date, issue_date) until the
  earliest date at which any other document rescinds / supersedes it.
- `build_versions`: for one provision, order the amending relations by effective date; version
  i is valid [d_i, d_{i+1}); an "omits" version has text None. A substitution/omission of an
  ancestor provision (e.g. all of rule 36) closes the chain of its descendants (rule 36(4)).
- `chunk_overrides`: chunks of an amending document that are about a provision get that
  provision version's range instead of the document's range.

`resolve_validity(conn)` loads the inputs, runs the core and writes the results back with
set-based SQL, touching only rows that change (idempotent).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache

from asof.temporal.normalize import (
    clean,
    find_provision_refs,
    is_ancestor,
    normalize_provision_ref,
    related_refs,
)

AMENDING_KINDS = ("amends", "substitutes", "inserts", "omits")
TERMINATING_KINDS = ("rescinds", "supersedes")
CLOSING_KINDS = ("substitutes", "omits")  # ancestor changes that end a descendant's text
_KIND_PRIORITY = {"substitutes": 0, "inserts": 1, "amends": 2, "omits": 3}


# ------------------------------------------------------------------ documents

@dataclass(frozen=True)
class DocDates:
    id: int
    issue_date: date | None
    effective_date: date | None = None

    @property
    def start(self) -> date | None:
        return self.effective_date or self.issue_date


@dataclass(frozen=True)
class Termination:
    src_doc_id: int
    dst_doc_id: int
    kind: str  # rescinds | supersedes
    effective_date: date | None = None


@dataclass(frozen=True)
class DocValidity:
    lo: date | None
    hi: date | None
    status: str  # in_force | rescinded | superseded


def document_validity(docs: Iterable[DocDates], terminations: Iterable[Termination],
                      anomalies: list[str] | None = None) -> dict[int, DocValidity]:
    by_id = {d.id: d for d in docs}
    end: dict[int, tuple[date, str]] = {}
    for t in terminations:
        dst, src = by_id.get(t.dst_doc_id), by_id.get(t.src_doc_id)
        if dst is None or src is None or t.dst_doc_id == t.src_doc_id:
            continue
        when = t.effective_date or src.start
        if when is None:
            continue
        if dst.start is not None and when <= dst.start:
            # Killer takes effect before the target starts: almost always an extraction or
            # resolution error (or a same-day replacement); never produce an empty range.
            if anomalies is not None:
                anomalies.append(f"doc {src.id} {t.kind} doc {dst.id} on {when} <= start "
                                 f"{dst.start}")
            continue
        status = "rescinded" if t.kind == "rescinds" else "superseded"
        if t.dst_doc_id not in end or when < end[t.dst_doc_id][0]:
            end[t.dst_doc_id] = (when, status)
    out: dict[int, DocValidity] = {}
    for d in by_id.values():
        if d.id in end:
            hi, status = end[d.id]
            out[d.id] = DocValidity(d.start, hi, status)
        else:
            out[d.id] = DocValidity(d.start, None, "in_force")
    return out


# ------------------------------------------------------------------ provisions

@dataclass(frozen=True)
class Event:
    """One amending relation on a provision, with its effective date already resolved."""

    relation_id: int
    src_doc_id: int
    kind: str  # amends | substitutes | inserts | omits
    date: date | None
    evidence: str | None
    src_issue_date: date | None = None


@dataclass
class Version:
    lo: date
    hi: date | None
    text: str | None  # None = omitted
    source_doc_id: int
    kind: str
    evidences: list[str] = field(default_factory=list)  # all supporting quotes (for matching)


def _order_key(e: Event) -> tuple:
    return (e.src_issue_date or date.min, e.src_doc_id)


_WORDING = (r"(?:words?|figures?|letters?|brackets?|signs?|symbols?|expressions?|entr(?:y|ies)"
            r"|portion|numerals?)")
# 'for the words "twenty per cent."' -> the wording being replaced
_OLD_SUBST_RE = re.compile(rf"\bfor\s+the\s+{_WORDING}[^\"]{{0,80}}\"([^\"]+)\"", re.IGNORECASE)
# 'the words "in FORM GSTR-2" shall be omitted' -> the wording being removed
_OLD_OMIT_RE = re.compile(
    rf"\bthe\s+{_WORDING}[^\"]{{0,80}}\"([^\"]+)\"[^\"]{{0,160}}?\bomitted", re.IGNORECASE)
_QUOTE_RE = re.compile(r"\"([^\"]{2,})\"")


# A substitution/omission whose operative part (before "namely" or the first quote) targets a
# sub-unit, e.g. 'for the second proviso, the following proviso shall be substituted' or
# 'the Explanation shall be omitted', changes only that part: it must not end every earlier
# version of the provision (27/2021 replacing a proviso of rule 36(4) left the 5% cap in force).
_PARTIAL_UNIT_RE = re.compile(
    r"\b(?:provisos?|explanations?|illustrations?|(?:sub-)?clauses?|items?|entr(?:y|ies)"
    r"|tables?|serial\s+(?:number|no)s?|columns?|headings?|annexures?)\b",
    re.IGNORECASE)


def operative_part(evidence: str) -> str:
    """The instruction part of an amendment, without the new text it inserts."""
    ev = clean(evidence or "")
    cut = len(ev)
    for marker in ('"', "“", "namely"):
        i = ev.find(marker)
        if i != -1:
            cut = min(cut, i)
    return ev[:cut]


def is_partial(e: Event) -> bool:
    """True when the amendment targets only a sub-unit (proviso, clause, Explanation...)."""
    return bool(_PARTIAL_UNIT_RE.search(operative_part(e.evidence or "")))


def old_wording(e: Event) -> list[str]:
    """Wording an amendment replaces/removes ([] = it targets the provision as a whole)."""
    ev = clean(e.evidence or "")
    if e.kind == "substitutes":
        return _OLD_SUBST_RE.findall(ev)
    if e.kind == "omits":
        return _OLD_OMIT_RE.findall(ev)
    return []


def new_wording(v: Version) -> str:
    """Wording a version introduced: the last quote of a substitution, quotes of an insertion."""
    text = v.text or ""
    quotes = _QUOTE_RE.findall(text)
    if v.kind == "substitutes" and quotes:
        return quotes[-1]
    if v.kind == "inserts" and quotes:
        return " ".join(quotes)
    return text


def build_versions(events: Iterable[Event], closers: Iterable[date] = ()) -> list[Version]:
    """Turn a provision's amendments into versions with validity ranges.

    In the simple case (each amendment rewrites the provision or the same wording, e.g.
    "twenty per cent." -> "ten per cent." -> "five per cent." -> omitted) version i is valid
    [d_i, d_{i+1}). An amendment only ends an earlier version when it replaces that version's
    wording ('for the words "X"' where X is wording the version introduced), substitutes or
    omits the whole provision, or an ancestor provision is substituted/omitted (`closers`).
    Pure insertions (a new proviso) and generic amendments leave earlier versions in force, so
    versions may overlap: together they describe the provision at that date.

    Events on the same date: the latest-issued source document wins; within one document an
    insertion/substitution beats a whole omission (a "replace"). An omission of the whole
    provision is a version with text None; any later amendment ends it.
    """
    by_date: dict[date, list[Event]] = defaultdict(list)
    for e in events:
        if e.date is not None:
            by_date[e.date].append(e)
    closer_set = set(closers)
    out: list[Version] = []
    live: list[Version] = []

    def close(d: date, which: list[Version]) -> None:
        for v in which:
            v.hi = d
        live[:] = [v for v in live if v not in which]

    for d in sorted(set(by_date) | closer_set):
        if d in closer_set:
            close(d, list(live))
        group = by_date.get(d)
        if not group:
            continue
        winner_doc = max(group, key=_order_key).src_doc_id
        mine = sorted((e for e in group if e.src_doc_id == winner_doc),
                      key=lambda e: (_KIND_PRIORITY.get(e.kind, 9), e.relation_id))
        evidences = list(dict.fromkeys(clean(e.evidence) for e in mine if e.evidence))
        olds = {e.relation_id: old_wording(e) for e in mine}
        whole_omit = [e for e in mine
                      if e.kind == "omits" and not olds[e.relation_id] and not is_partial(e)]
        texted = [e for e in mine if e not in whole_omit]
        if texted:
            text_parts = list(dict.fromkeys(clean(e.evidence) for e in texted if e.evidence))
            text: str | None = "\n".join(text_parts) or None
            kind = texted[0].kind
        else:
            text, kind = None, "omits"
        replaces_all = bool(whole_omit) or any(
            e.kind == "substitutes" and not olds[e.relation_id] and not is_partial(e)
            for e in mine)
        if replaces_all:
            close(d, list(live))
        else:
            targets = [_alnum(w) for ws in olds.values() for w in ws if len(_alnum(w)) >= 2]
            close(d, [v for v in live if v.text is None or any(
                t in _alnum(new_wording(v)) for t in targets)])
        v = Version(lo=d, hi=None, text=text, source_doc_id=winner_doc, kind=kind,
                    evidences=evidences)
        out.append(v)
        live.append(v)
    return out


# ------------------------------------------------------------------ chunks

@dataclass(frozen=True)
class ChunkRow:
    id: int
    doc_id: int
    section_ref: str | None
    text: str


@lru_cache(maxsize=8192)
def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", clean(s).lower())


@lru_cache(maxsize=8192)
def _refs_in(text: str) -> tuple[str, ...]:
    return tuple(find_provision_refs(text))


@lru_cache(maxsize=8192)
def _section_ref(section_ref: str) -> str | None:
    sr = normalize_provision_ref(section_ref)
    return sr[1] if sr else None


def _contains_quote(chunk_norm: str, evidence: str) -> bool:
    ev = _alnum(evidence)
    if len(ev) < 12:
        return False
    if ev in chunk_norm:
        return True
    m = SequenceMatcher(None, chunk_norm, ev, autojunk=False).find_longest_match(
        0, len(chunk_norm), 0, len(ev))
    return m.size >= min(60, int(0.6 * len(ev)))


def chunk_matches(chunk: ChunkRow, ref: str, evidences: Iterable[str] = ()) -> bool:
    """Is this chunk about provision `ref` (by section_ref, a mention, or the evidence quote)?"""
    if chunk.section_ref:
        sr = _section_ref(chunk.section_ref)
        if sr and related_refs(sr, ref):
            return True
    if any(related_refs(r, ref) for r in _refs_in(chunk.text)):
        return True
    norm = _alnum(chunk.text)
    return any(_contains_quote(norm, ev) for ev in evidences if ev)


def _union(a: tuple[date, date | None], b: tuple[date, date | None]) -> tuple[date, date | None]:
    lo = min(a[0], b[0])
    hi = None if a[1] is None or b[1] is None else max(a[1], b[1])
    return lo, hi


def chunk_overrides(
    versions: Iterable[tuple[object, str, Version]],
    chunks_by_doc: dict[int, list[ChunkRow]],
) -> dict[int, tuple[date, date | None, object]]:
    """chunk_id -> (lo, hi, version_key) for chunks of amending docs about a provision.

    `versions` yields (version_key, provision_ref, Version). When a chunk is about several
    provisions its range is the union (it stays retrievable while any of them applies) and
    it points at the first matching version.
    """
    out: dict[int, tuple[date, date | None, object]] = {}
    for vkey, ref, v in versions:
        for ch in chunks_by_doc.get(v.source_doc_id, []):
            if not chunk_matches(ch, ref, v.evidences):
                continue
            if ch.id in out:
                lo, hi, first = out[ch.id]
                lo, hi = _union((lo, hi), (v.lo, v.hi))
                out[ch.id] = (lo, hi, first)
            else:
                out[ch.id] = (v.lo, v.hi, vkey)
    return out


# ------------------------------------------------------------------ DB driver

def _load_docs(conn) -> list[DocDates]:
    rows = conn.execute("SELECT id, issue_date, effective_date FROM documents").fetchall()
    return [DocDates(*r) for r in rows]


def update_documents(conn, validity: dict[int, DocValidity]) -> int:
    ids = list(validity)
    cur = conn.execute(
        """
        UPDATE documents d
           SET validity = daterange(v.lo, v.hi), status = v.st
          FROM unnest(%s::bigint[], %s::date[], %s::date[], %s::text[]) AS v(id, lo, hi, st)
         WHERE d.id = v.id
           AND (d.validity IS DISTINCT FROM daterange(v.lo, v.hi) OR d.status IS DISTINCT FROM v.st)
        """,
        (ids, [validity[i].lo for i in ids], [validity[i].hi for i in ids],
         [validity[i].status for i in ids]),
    )
    return cur.rowcount


def resolve_validity(conn) -> dict:
    """Recompute documents.validity/status, provision_versions and chunks.validity. Idempotent."""
    from asof.temporal.store import resolve_dangling  # local: keep the pure core import-light

    stats: dict = {"anomalies": []}
    stats["relinked_relations"] = resolve_dangling(conn)

    # (a) documents
    docs = _load_docs(conn)
    by_id = {d.id: d for d in docs}
    terms = [Termination(*r) for r in conn.execute(
        "SELECT src_doc_id, dst_doc_id, kind, effective_date FROM relations "
        "WHERE kind IN ('rescinds', 'supersedes') AND dst_doc_id IS NOT NULL").fetchall()]
    dv = document_validity(docs, terms, stats["anomalies"])
    stats["documents_updated"] = update_documents(conn, dv)
    stats["documents_by_status"] = dict(
        conn.execute("SELECT status, count(*) FROM documents GROUP BY status").fetchall())

    # (b) provision versions
    provisions = {pid: (inst, ref) for pid, inst, ref in
                  conn.execute("SELECT id, instrument, ref FROM provisions").fetchall()}
    gold = {r[0] for r in conn.execute(
        "SELECT DISTINCT provision_id FROM provision_versions WHERE verified").fetchall()}
    events: dict[int, list[Event]] = defaultdict(list)
    for rid, pid, src, kind, eff, evidence in conn.execute(
        "SELECT id, dst_provision_id, src_doc_id, kind, effective_date, evidence_span "
        "FROM relations WHERE dst_provision_id IS NOT NULL AND kind = ANY(%s)",
        (list(AMENDING_KINDS),),
    ).fetchall():
        src_doc = by_id.get(src)
        when = eff or (src_doc.start if src_doc else None)
        events[pid].append(Event(rid, src, kind, when, evidence,
                                 src_doc.issue_date if src_doc else None))

    by_instrument: dict[str, list[int]] = defaultdict(list)
    for pid, (inst, _ref) in provisions.items():
        by_instrument[inst].append(pid)

    computed: dict[tuple, Version] = {}  # (pid, lo, hi, text, src) -> Version
    computed_ref: dict[tuple, str] = {}
    for pid, evs in events.items():
        if pid in gold or pid not in provisions:
            continue
        inst, ref = provisions[pid]
        closers = [e.date for a in by_instrument[inst] if is_ancestor(provisions[a][1], ref)
                   for e in events.get(a, []) if e.kind in CLOSING_KINDS and e.date]
        for v in build_versions(evs, closers):
            key = (pid, v.lo, v.hi, v.text, v.source_doc_id)
            computed[key] = v
            computed_ref[key] = ref

    existing = conn.execute(
        "SELECT id, provision_id, lower(validity), upper(validity), text, source_doc_id "
        "FROM provision_versions WHERE NOT verified").fetchall()
    existing_ids: dict[tuple, int] = {}
    stale: list[int] = []
    for vid, pid, lo, hi, text, src in existing:
        key = (pid, lo, hi, text, src)
        if key in computed and key not in existing_ids:
            existing_ids[key] = vid
        else:
            stale.append(vid)
    if stale:
        conn.execute("DELETE FROM provision_versions WHERE id = ANY(%s)", (stale,))
    inserted = 0
    for key, v in computed.items():
        if key in existing_ids:
            continue
        row = conn.execute(
            "INSERT INTO provision_versions (provision_id, text, validity, source_doc_id) "
            "VALUES (%s, %s, daterange(%s, %s), %s) RETURNING id",
            (key[0], v.text, v.lo, v.hi, v.source_doc_id),
        ).fetchone()
        existing_ids[key] = row[0]
        inserted += 1
    stats.update(versions_total=len(computed), versions_inserted=inserted,
                 versions_deleted=len(stale))
    per_prov = defaultdict(int)
    for key in computed:
        per_prov[key[0]] += 1
    stats["provisions_with_versions"] = len(per_prov)
    stats["provisions_multi_version"] = sum(1 for n in per_prov.values() if n > 1)

    # chunk overrides from (b)
    src_docs = sorted({v.source_doc_id for v in computed.values()})
    chunks_by_doc: dict[int, list[ChunkRow]] = defaultdict(list)
    if src_docs:
        for cid, did, sref, text in conn.execute(
            "SELECT id, doc_id, section_ref, text FROM chunks WHERE doc_id = ANY(%s)",
            (src_docs,),
        ).fetchall():
            chunks_by_doc[did].append(ChunkRow(cid, did, sref, text))
    ov = chunk_overrides(((k, computed_ref[k], v) for k, v in computed.items()), chunks_by_doc)
    stats["chunks_with_provision_range"] = len(ov)

    # (c) all chunks: provision range if set by (b), else the document's range
    cids = list(ov)
    cur = conn.execute(
        """
        WITH o AS (
          SELECT * FROM unnest(%s::bigint[], %s::date[], %s::date[], %s::bigint[])
                 AS o(id, lo, hi, pv)
        ), t AS (
          SELECT c2.id,
                 CASE WHEN o.id IS NULL THEN d.validity ELSE daterange(o.lo, o.hi) END AS rng,
                 o.pv
            FROM chunks c2
            JOIN documents d ON d.id = c2.doc_id
            LEFT JOIN o ON o.id = c2.id
        )
        UPDATE chunks c
           SET validity = t.rng, provision_version_id = t.pv
          FROM t
         WHERE c.id = t.id
           AND (c.validity IS DISTINCT FROM t.rng
                OR c.provision_version_id IS DISTINCT FROM t.pv)
        """,
        (cids, [ov[c][0] for c in cids], [ov[c][1] for c in cids],
         [existing_ids[ov[c][2]] for c in cids]),
    )
    stats["chunks_updated"] = cur.rowcount
    return stats
