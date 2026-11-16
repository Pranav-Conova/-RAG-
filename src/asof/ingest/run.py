"""Ingestion pipeline: crawl -> download -> parse -> load (chunk + embed + DB).

    python -m asof.ingest.run [--years 2019-2022] [--kinds notification,circular] [--limit N]
                              [--stages crawl,download,parse,load] [--force] [--no-ocr]

Every stage is resumable: PDFs and parsed JSON already on disk are reused, and documents that
already have chunks in the DB are skipped unless --force.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

from asof.config import get_settings
from asof.ingest import crawl as crawl_mod
from asof.ingest.chunk import chunk_document
from asof.ingest.parse import is_empty, parse_pdf, read_parsed, write_parsed
from asof.models import DocMeta, ParsedDoc

log = logging.getLogger("asof.ingest")

ALL_STAGES = ("crawl", "download", "parse", "load")
ALL_KINDS = ("notification", "circular")


def parse_years(s: str) -> tuple[int, int]:
    a, _, b = s.partition("-")
    return int(a), int(b or a)


def _filter(docs: list[DocMeta], kinds: set[str], years: tuple[int, int]) -> list[DocMeta]:
    return [d for d in docs if d.kind in kinds and years[0] <= d.issue_date.year <= years[1]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m asof.ingest.run", description=__doc__)
    ap.add_argument("--years", default="2019-2022", type=parse_years)
    ap.add_argument("--kinds", default=",".join(ALL_KINDS))
    ap.add_argument("--limit", type=int, default=None, help="max documents (after filtering)")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--force", action="store_true", help="re-parse and re-load everything")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    unknown = kinds - set(ALL_KINDS)
    if unknown:
        ap.error(f"unknown kinds: {sorted(unknown)}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - set(ALL_STAGES)
    if bad:
        ap.error(f"unknown stages: {sorted(bad)}")

    settings = get_settings()
    data = Path(settings.data_dir)
    root = data.resolve().parent
    manifest = data / "raw" / "manifest.jsonl"
    pdf_dir = data / "raw" / "pdf"
    parsed_dir = data / "parsed"
    summary: Counter[str] = Counter()
    t0 = time.monotonic()

    client: crawl_mod.PoliteClient | None = None
    try:
        # ---------------------------------------------------------------- crawl
        if "crawl" in stages:
            client = client or crawl_mod.PoliteClient()
            res = crawl_mod.crawl(client, sorted(kinds), args.years)
            docs = res.docs
            # keep pdf_path of anything downloaded before
            if manifest.exists():
                old = {d.key: d.pdf_path for d in crawl_mod.read_manifest(manifest)}
                for d in docs:
                    d.pdf_path = old.get(d.key)
            crawl_mod.write_manifest(docs, manifest)
            for k, v in res.counts.items():
                summary[f"crawl.{k}"] = v
            log.info("crawl: %d documents -> %s", len(docs), manifest)
        if not manifest.exists():
            log.error("no manifest at %s; run the crawl stage first", manifest)
            return 1
        all_docs = crawl_mod.read_manifest(manifest)
        docs = _filter(all_docs, kinds, args.years)
        if args.limit:
            docs = docs[: args.limit]
        summary["docs_listed"] = len(docs)
        for d in docs:
            summary[f"listed.{d.kind}.{d.issue_date.year}"] += 1

        # ---------------------------------------------------------------- download
        if "download" in stages:
            client = client or crawl_mod.PoliteClient()
            counts = crawl_mod.download(
                client, docs, pdf_dir, rel_to=root, fallback=crawl_mod.TaxPortal(client)
            )
            for k, v in counts.items():
                summary[f"download.{k}"] = v
            by_key = {d.key: d for d in docs}
            crawl_mod.write_manifest([by_key.get(d.key, d) for d in all_docs], manifest)
        for d in docs:  # pick up PDFs present on disk even if the manifest predates them
            p = pdf_dir / f"{d.key}.pdf"
            if p.exists():
                d.pdf_path = p.resolve().relative_to(root).as_posix()
        have_pdf = [d for d in docs if d.pdf_path]
        summary["pdfs_available"] = len(have_pdf)

        # ---------------------------------------------------------------- parse
        parsed_docs: list[ParsedDoc] = []
        if "parse" in stages or "load" in stages:
            for i, d in enumerate(have_pdf, 1):
                out = parsed_dir / f"{d.key}.json"
                if out.exists() and not (args.force and "parse" in stages):
                    pd = read_parsed(out)
                    pd.meta = d  # manifest is the source of truth for metadata
                    summary["parse.cached"] += 1
                elif "parse" in stages:
                    try:
                        pd = parse_pdf(d, root / d.pdf_path, use_ocr=not args.no_ocr)
                    except Exception as exc:
                        summary["parse.failed"] += 1
                        log.error("[%d/%d] parse failed %s: %s", i, len(have_pdf), d.key, exc)
                        continue
                    write_parsed(pd, parsed_dir)
                    summary["parse.parsed"] += 1
                    log.info(
                        "[%d/%d] parsed %s: %d pages%s",
                        i, len(have_pdf), d.key, len(pd.pages), " (OCR)" if pd.ocr_used else "",
                    )
                else:
                    continue
                parsed_docs.append(pd)
                summary["docs_parsed"] += 1
                if pd.ocr_used:
                    summary["ocr_used"] += 1
                if is_empty(pd):
                    summary["empty_docs"] += 1
                summary["empty_pages"] += sum(1 for p in pd.pages if not p.text.strip())

        # ---------------------------------------------------------------- chunk + load
        if "load" in stages:
            from openai import OpenAI

            from asof.db.connection import connect
            from asof.ingest.load import (
                AdaptiveEmbedder,
                PreparedDoc,
                load_documents,
                loaded_keys,
            )
            from asof.providers.embeddings import Embedder

            prepared = [PreparedDoc(pd, chunk_document(pd)) for pd in parsed_docs]
            summary["chunks_total"] = sum(len(p.chunks) for p in prepared)
            with connect(settings) as conn:
                if not args.force:
                    done = loaded_keys(conn)
                    before = len(prepared)
                    prepared = [p for p in prepared if p.parsed.meta.key not in done]
                    summary["load.skipped_existing"] = before - len(prepared)
                todo_chunks = sum(len(p.chunks) for p in prepared)
                log.info(
                    "load: %d documents, %d chunks to embed (NIM rpm=%d)",
                    len(prepared), todo_chunks, settings.nim_requests_per_minute,
                )
                # no client-side retries in the OpenAI SDK: they would bypass the rate limiter.
                # AdaptiveEmbedder retries through Embedder (and so through the limiter).
                embedder = Embedder(
                    settings,
                    client=OpenAI(
                        api_key=settings.require_api_key(),
                        base_url=settings.nim_base_url,
                        timeout=settings.request_timeout_s,
                        max_retries=0,
                    ),
                )
                embed = AdaptiveEmbedder(embedder.embed_documents)
                progress = {"docs": 0, "chunks": 0}

                def on_loaded(p: PreparedDoc, doc_id: int) -> None:
                    progress["docs"] += 1
                    progress["chunks"] += len(p.chunks)
                    summary["docs_loaded"] += 1
                    summary["chunks_loaded"] += len(p.chunks)
                    if progress["docs"] % 10 == 0 or progress["chunks"] >= todo_chunks:
                        log.info(
                            "loaded %d/%d docs, %d/%d chunks (%.0fs)",
                            progress["docs"], len(prepared), progress["chunks"], todo_chunks,
                            time.monotonic() - t0,
                        )

                try:
                    load_documents(conn, prepared, embed, on_loaded=on_loaded)
                except Exception as exc:
                    log.error("load stopped: %s: %s (re-run to resume)", type(exc).__name__, exc)
                    summary["load.error"] = 1
                finally:
                    summary["embed.calls"] = embed.calls
                    summary["embed.failures"] = embed.failures
                    summary["embed.rejected_422"] = embed.rejected
                if summary["docs_loaded"]:
                    # Loading (re)sets document/chunk validity to [issue_date, open); recompute
                    # rescissions, provision versions and chunk refs from stored relations so
                    # superseded text never looks in force between ingest and `asof temporal`.
                    from asof.temporal.refs import backfill_chunk_refs
                    from asof.temporal.validity import resolve_validity
                    summary["refs.updated"] = backfill_chunk_refs(conn)
                    vstats = resolve_validity(conn)
                    summary["validity.chunks_updated"] = vstats.get("chunks_updated", 0)
    finally:
        if client:
            client.close()
        summary["seconds"] = int(time.monotonic() - t0)
        log.info("summary:")
        for k in sorted(summary):
            log.info("  %-36s %s", k, summary[k])
    return 1 if summary.get("load.error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
