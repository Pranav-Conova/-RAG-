"""store_relations against a fake connection (no DB): target resolution and idempotent writes."""

from contextlib import contextmanager
from datetime import date

from asof.models import ExtractedRelation, ExtractionResult
from asof.temporal.store import DocIndex, canonical_number, store_relations


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, docs):
        self.docs = docs  # (id, key, number, kind)
        self.calls: list[tuple[str, tuple]] = []
        self.provisions: dict[tuple[str, str], int] = {}

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if sql.startswith("SELECT id, key, number, kind FROM documents"):
            return _Result(self.docs)
        if sql.startswith("INSERT INTO provisions"):
            pid = self.provisions.setdefault(tuple(params), len(self.provisions) + 1)
            return _Result([(pid,)])
        return _Result([])

    @contextmanager
    def transaction(self):
        yield

    @contextmanager
    def cursor(self):
        conn = self

        class Cur:
            def executemany(self, sql, rows):
                conn.calls.append((sql, list(rows)))

        yield Cur()


def rel(kind, tkind, ref, **kw):
    return ExtractedRelation(kind=kind, target_kind=tkind, target_ref=ref,
                             evidence=kw.pop("evidence", "quote"), confidence=0.9, **kw)


def test_canonical_number():
    assert canonical_number("170/02/2022-GST") == canonical_number("170/2/2022-GST")
    assert canonical_number("49/2019-Central Tax") == "49/2019-central tax"
    assert canonical_number(None) is None


def test_store_relations_resolves_and_replaces():
    conn = FakeConn([(1, "notification_75-2019-central-tax", "75/2019-Central Tax", "notification"),
                     (2, "notification_13-2017-central-tax", "13/2017-Central Tax", "notification"),
                     (3, "circular_170-02-2022-gst", "170/02/2022-GST", "circular")])
    index = DocIndex(conn)
    result = ExtractionResult(effective_date=date(2020, 1, 1), relations=[
        rel("substitutes", "rule", "rule 36(4)", effective_date=date(2020, 1, 1)),
        rel("rescinds", "notification", "13/2017-Central Tax"),
        rel("supersedes", "circular", "170/2/2022-GST"),
        rel("rescinds", "notification", "99/2016-Central Tax"),  # not in corpus: ref kept
        rel("amends", "notification", "75/2019-Central Tax"),    # self: skipped
    ])
    stats = store_relations(conn, 1, result, index)
    sqls = [c[0] for c in conn.calls]
    delete_at = next(i for i, s in enumerate(sqls) if s.startswith("DELETE FROM relations"))
    insert_at = next(i for i, s in enumerate(sqls) if s.startswith("INSERT INTO relations"))
    assert delete_at < insert_at
    rows = conn.calls[insert_at][1]
    # (src, dst_doc, dst_prov, dst_ref, dst_kind, kind, eff, evidence, conf)
    assert [(r[1], r[2], r[3], r[5]) for r in rows] == [
        (None, 1, "rule 36(4)", "substitutes"),
        (2, None, "13/2017-Central Tax", "rescinds"),
        (3, None, "170/02/2022-GST", "supersedes"),
        (None, None, "99/2016-Central Tax", "rescinds"),
    ]
    assert conn.provisions == {("cgst_rules", "rule 36(4)"): 1}
    assert (stats.resolved_doc, stats.resolved_provision, stats.unresolved) == (2, 1, 1)
    assert any(s.startswith("UPDATE documents SET effective_date") for s in sqls)
