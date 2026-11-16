"""Temporal pipeline: extract relations from data/parsed/*.json, store them, resolve validity.

    python -m asof.temporal.run [--limit N] [--only-extract] [--only-validity] [--keys K ...]

--limit N counts documents that pass the prefilter (i.e. that may need an LLM call).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from asof.config import get_settings
from asof.models import ParsedDoc
from asof.temporal.extract import ExtractStats, doc_text, extract_relations, passes_prefilter


def _is_quota_error(err: Exception) -> bool:
    name = type(err).__name__
    msg = str(err).lower()
    return name in ("RateLimitError", "AuthenticationError", "PermissionDeniedError") or (
        "429" in msg or "quota" in msg or "rate limit" in msg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m asof.temporal.run", description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="max documents passing the prefilter to process")
    ap.add_argument("--only-extract", action="store_true", help="extract (and cache) only")
    ap.add_argument("--only-validity", action="store_true", help="only run resolve_validity")
    ap.add_argument("--keys", nargs="*", default=None, help="only these document keys")
    ap.add_argument("--in-db-only", action="store_true",
                    help="skip extraction for parsed docs not yet loaded in the DB")
    ap.add_argument("--show", type=int, default=0, help="print N sample relations")
    args = ap.parse_args(argv)

    settings = get_settings()
    parsed_dir = Path(settings.data_dir) / "parsed"
    t0 = time.perf_counter()
    conn = None
    if not args.only_extract:
        from asof.db.connection import connect
        conn = connect(settings, register=False)

    xstats = ExtractStats()
    by_kind: Counter = Counter()
    samples: list[tuple[str, dict]] = []
    store_summary = None
    if not args.only_validity:
        from asof.providers.llm import LLM
        from asof.temporal.store import DocIndex, StoreStats, store_relations

        index = DocIndex(conn) if conn is not None else None
        sstats = StoreStats()
        llm = None
        candidates = 0
        files = sorted(parsed_dir.glob("*.json"))
        if args.keys:
            wanted = set(args.keys)
            files = [f for f in files if f.stem in wanted]
        for path in files:
            try:
                doc = ParsedDoc.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as err:
                xstats.errors.append(f"{path.name}: unreadable ({err})")
                continue
            doc_id = index.by_key.get(doc.meta.key) if index else None
            if args.in_db_only and index is not None and doc_id is None:
                continue
            if passes_prefilter(doc_text(doc)):
                if args.limit is not None and candidates >= args.limit:
                    continue
                candidates += 1
                if llm is None:
                    llm = LLM(settings)
            try:
                result = extract_relations(llm, doc, stats=xstats)
            except Exception as err:  # noqa: BLE001 - keep going on per-document failures
                xstats.failures += 1
                xstats.errors.append(f"{doc.meta.key}: {type(err).__name__}: {err}"[:300])
                if _is_quota_error(err):
                    print(f"STOP: provider limit hit on {doc.meta.key}: {err}", file=sys.stderr)
                    break
                continue
            for rel in result.relations:
                by_kind[rel.kind] += 1
                if len(samples) < args.show:
                    samples.append((doc.meta.number, rel.model_dump(mode="json")))
            if conn is not None and doc_id is not None and index is not None:
                store_relations(conn, doc_id, result, index, sstats)
        store_summary = sstats

    vstats = None
    if conn is not None and not args.only_extract:
        from asof.temporal.validity import resolve_validity
        vstats = resolve_validity(conn)

    print("== temporal run ==")
    print(f"docs scanned: {xstats.scanned}  prefilter-skipped: {xstats.skipped_prefilter}  "
          f"LLM calls: {xstats.llm_calls}  cache hits: {xstats.cache_hits}  "
          f"failures: {xstats.failures}")
    print(f"relations extracted by kind: {dict(by_kind)}  "
          f"dropped (evidence not in text): {xstats.dropped_evidence}  "
          f"dropped (self/empty): {xstats.dropped_other}")
    if store_summary is not None and store_summary.docs:
        s = store_summary
        print(f"stored for {s.docs} docs: {s.relations} relations; resolved to doc: "
              f"{s.resolved_doc}, to provision: {s.resolved_provision}, "
              f"unresolved (ref kept): {s.unresolved}")
    if conn is not None:
        row = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE dst_doc_id IS NOT NULL "
            "OR dst_provision_id IS NOT NULL) FROM relations").fetchone()
        print(f"relations in DB: {row[0]} (resolved targets: {row[1]}, "
              f"unresolved: {row[0] - row[1]})")
    if vstats is not None:
        anomalies = vstats.pop("anomalies")
        print("validity:", json.dumps(vstats, default=str))
        if anomalies:
            print(f"anomalies ({len(anomalies)}):", *anomalies[:10], sep="\n  ")
    for number, rel in samples:
        print(f"- [{number}] {rel['kind']} {rel['target_kind']} {rel['target_ref']!r} "
              f"eff={rel['effective_date']} conf={rel['confidence']}\n    "
              f"evidence: {rel['evidence'][:220]!r}")
    for e in xstats.errors[:10]:
        print("error:", e)
    print(f"elapsed: {time.perf_counter() - t0:.1f}s")
    if conn is not None:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
