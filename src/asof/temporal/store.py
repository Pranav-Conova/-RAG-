"""Persist extracted relations into the `relations` table (idempotent per source document)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import psycopg

from asof.models import ExtractionResult
from asof.temporal.normalize import (
    normalize_circular_ref,
    normalize_notification_ref,
    provision_key,
)


@dataclass
class StoreStats:
    docs: int = 0
    relations: int = 0
    resolved_doc: int = 0
    resolved_provision: int = 0
    unresolved: int = 0
    by_kind: Counter = field(default_factory=Counter)


def canonical_number(number: str | None) -> str | None:
    """Canonical form of a documents.number, for matching against extracted refs."""
    if not number:
        return None
    return (normalize_circular_ref(number) or normalize_notification_ref(number)
            or number.strip()).lower()


class DocIndex:
    """Lookup of document ids by key and by canonical number (loaded once)."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.by_key: dict[str, int] = {}
        self.by_number: dict[str, int] = {}
        self.number_of: dict[int, str] = {}
        rows = conn.execute(
            "SELECT id, key, number, kind FROM documents ORDER BY issue_date NULLS LAST, id"
        ).fetchall()
        for doc_id, key, number, _kind in rows:
            if key:
                self.by_key[key] = doc_id
            if number:
                self.number_of[doc_id] = number
            canon = canonical_number(number)
            if canon:
                self.by_number.setdefault(canon, doc_id)  # earliest-issued wins on duplicates

    def doc_for_ref(self, ref: str) -> int | None:
        return self.by_number.get(canonical_number(ref) or "")


def upsert_provision(conn: psycopg.Connection, instrument: str, ref: str) -> int:
    row = conn.execute(
        "INSERT INTO provisions (instrument, ref) VALUES (%s, %s) "
        "ON CONFLICT (instrument, ref) DO UPDATE SET ref = EXCLUDED.ref RETURNING id",
        (instrument, ref),
    ).fetchone()
    assert row is not None
    return int(row[0])


def store_relations(conn: psycopg.Connection, src_doc_id: int, result: ExtractionResult,
                    index: DocIndex, stats: StoreStats | None = None) -> StoreStats:
    """Replace all relations of `src_doc_id` with `result`; set documents.effective_date."""
    stats = stats if stats is not None else StoreStats()
    stats.docs += 1
    rows = []
    prov_cache: dict[tuple[str, str], int] = {}
    for rel in result.relations:
        dst_doc_id = dst_prov_id = None
        dst_ref = rel.target_ref
        if rel.target_kind in ("notification", "circular"):
            dst_doc_id = index.doc_for_ref(rel.target_ref)
            if dst_doc_id == src_doc_id:
                continue
            if dst_doc_id:  # spell it exactly as the corpus does (e.g. '170/02/2022-GST')
                dst_ref = index.number_of.get(dst_doc_id, dst_ref)
        pk = provision_key(rel.target_kind, rel.target_ref)
        if pk:
            if pk not in prov_cache:
                prov_cache[pk] = upsert_provision(conn, *pk)
            dst_prov_id = prov_cache[pk]
        if dst_doc_id:
            stats.resolved_doc += 1
        elif dst_prov_id:
            stats.resolved_provision += 1
        else:
            stats.unresolved += 1
        stats.by_kind[rel.kind] += 1
        stats.relations += 1
        rows.append((src_doc_id, dst_doc_id, dst_prov_id, dst_ref, rel.target_kind,
                     rel.kind, rel.effective_date, rel.evidence, rel.confidence))
    with conn.transaction():
        conn.execute("DELETE FROM relations WHERE src_doc_id = %s", (src_doc_id,))
        if rows:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO relations (src_doc_id, dst_doc_id, dst_provision_id, dst_ref, "
                    "dst_kind, kind, effective_date, evidence_span, confidence) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
        if result.effective_date:
            conn.execute("UPDATE documents SET effective_date = %s WHERE id = %s",
                         (result.effective_date, src_doc_id))
    return stats


def resolve_dangling(conn: psycopg.Connection, index: DocIndex | None = None) -> int:
    """Point relations at documents loaded after the relation was stored. Returns #fixed."""
    index = index or DocIndex(conn)
    rows = conn.execute(
        "SELECT id, src_doc_id, dst_ref FROM relations "
        "WHERE dst_doc_id IS NULL AND dst_kind IN ('notification', 'circular')"
    ).fetchall()
    fixes = [(d, index.number_of.get(d, ref), rid) for rid, src, ref in rows
             if (d := index.doc_for_ref(ref or "")) and d != src]
    if fixes:
        with conn.cursor() as cur:
            cur.executemany("UPDATE relations SET dst_doc_id = %s, dst_ref = %s WHERE id = %s",
                            fixes)
    return len(fixes)
