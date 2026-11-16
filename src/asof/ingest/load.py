"""Load parsed + chunked documents into Postgres: upsert `documents`, replace `chunks`.

Embeddings are computed in batches spanning several documents (fewer API calls), then each
document is written in its own transaction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psycopg
from pgvector import HalfVector

from asof.models import ChunkIn, ParsedDoc

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float] | None]]


@dataclass
class PreparedDoc:
    parsed: ParsedDoc
    chunks: list[ChunkIn]


class EmbeddingUnavailable(RuntimeError):
    """The embedding service keeps failing (quota, 429s or persistent 5xx): stop the run."""


class AdaptiveEmbedder:
    """Wraps a batch embed function with adaptive batch sizes and bounded retries.

    The NVIDIA embedding endpoint intermittently returns HTTP 500 for multi-text batches while
    single texts mostly succeed, so: on a 5xx the batch is split in half (down to one text),
    a run of successes grows the batch again, and a single text that keeps failing (or
    repeated 429s) raises EmbeddingUnavailable so the caller can stop cleanly and resume
    later. Every attempt goes through the wrapped function, i.e. through the rate limiter.
    """

    def __init__(
        self,
        embed: EmbedFn,
        start_batch: int = 8,
        max_batch: int = 32,
        max_failures: int = 8,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self._embed = embed
        self.batch = start_batch
        self.max_batch = max_batch
        self.max_failures = max_failures
        self._sleep = sleep or time.sleep
        self.ceiling = max_batch
        self._streak = 0
        self.calls = 0
        self.failures = 0
        self.rejected = 0  # texts refused by the provider's content screening (HTTP 422)

    def _status(self, exc: Exception) -> int | None:
        return getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )

    def __call__(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Vectors in input order; None for a text the provider refuses to embed (HTTP 422).

        Such chunks are still stored (without embedding) so full-text search can find them.
        """
        out: list[list[float] | None] = []
        i = 0
        consecutive = 0
        while i < len(texts):
            part = texts[i : i + self.batch]
            self.calls += 1
            try:
                vecs = self._embed(part)
            except Exception as exc:
                status = self._status(exc)
                self.failures += 1
                self._streak = 0
                if status == 422:
                    if len(part) == 1:
                        self.rejected += 1
                        log.warning("embedding rejected (422) for one text; stored without "
                                    "embedding: %r", part[0][:80])
                        out.append(None)
                        i += 1
                    else:
                        self.batch = max(1, len(part) // 2)  # find the rejected text
                    continue
                consecutive += 1
                if status is not None and status < 500 and status != 429:
                    raise EmbeddingUnavailable(f"HTTP {status}: {exc}") from exc
                if consecutive >= self.max_failures:
                    raise EmbeddingUnavailable(
                        f"{consecutive} consecutive embedding failures, last: {exc}"
                    ) from exc
                if len(part) > 1:
                    self.batch = max(1, len(part) // 2)
                    self.ceiling = self.batch  # don't grow straight back into failures
                    log.info("embedding batch of %d failed (%s); batch -> %d",
                             len(part), status, self.batch)
                else:
                    wait = min(60.0, 3.0 * 2 ** (consecutive - 1))
                    log.warning("embedding failed (%s); retry in %.0fs", status, wait)
                    self._sleep(wait)
                continue
            out.extend(vecs)
            i += len(part)
            consecutive = 0
            self._streak += 1
            if self._streak >= 4 and self.batch < self.ceiling:
                self.batch = min(self.ceiling, self.batch * 2)
                self._streak = 0
            elif self._streak >= 50 and self.ceiling < self.max_batch:
                self.ceiling = min(self.max_batch, self.ceiling * 2)  # occasionally probe up
                self._streak = 0
        return out


def embedding_input(parsed: ParsedDoc, chunk: ChunkIn) -> str:
    m = parsed.meta
    return f"{m.number} dated {m.issue_date:%d.%m.%Y} — {m.title}\n{chunk.text}"


def loaded_keys(conn: psycopg.Connection) -> set[str]:
    """Keys of documents that already have at least one chunk."""
    rows = conn.execute(
        "SELECT d.key FROM documents d WHERE d.key IS NOT NULL "
        "AND EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.id)"
    ).fetchall()
    return {r[0] for r in rows}


_UPSERT_DOC = """
INSERT INTO documents (kind, number, issue_date, title, url, pdf_path, lang, status, validity, key)
VALUES (%(kind)s, %(number)s, %(issue_date)s, %(title)s, %(url)s, %(pdf_path)s, 'en', 'unknown',
        daterange(%(issue_date)s, NULL), %(key)s)
ON CONFLICT (key) DO UPDATE SET
  kind = EXCLUDED.kind, number = EXCLUDED.number, issue_date = EXCLUDED.issue_date,
  title = EXCLUDED.title, url = EXCLUDED.url, pdf_path = EXCLUDED.pdf_path,
  status = EXCLUDED.status, validity = EXCLUDED.validity
RETURNING id
"""

_INSERT_CHUNK = """
INSERT INTO chunks (doc_id, section_ref, page, ord, text, embedding, validity)
VALUES (%s, %s, %s, %s, %s, %s, daterange(%s, NULL))
"""


def write_document(
    conn: psycopg.Connection, doc: PreparedDoc, vectors: Sequence[Sequence[float] | None]
) -> int:
    """Upsert one document and replace its chunks, in a single transaction. Returns doc id."""
    if len(vectors) != len(doc.chunks):
        raise ValueError("one embedding per chunk expected")
    m = doc.parsed.meta
    params = m.model_dump() | {"key": m.key}
    with conn.transaction():
        doc_id = conn.execute(_UPSERT_DOC, params).fetchone()[0]  # type: ignore[index]
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        if not doc.chunks:
            return doc_id
        with conn.cursor() as cur:
            cur.executemany(
                _INSERT_CHUNK,
                [
                    (
                        doc_id,
                        c.section_ref,
                        c.page,
                        c.ord,
                        c.text,
                        HalfVector(list(v)) if v is not None else None,
                        m.issue_date,
                    )
                    for c, v in zip(doc.chunks, vectors, strict=True)
                ],
            )
    return doc_id


def load_documents(
    conn: psycopg.Connection,
    docs: Sequence[PreparedDoc],
    embed: EmbedFn,
    batch_chunks: int = 16,
    on_loaded: Callable[[PreparedDoc, int], None] | None = None,
) -> int:
    """Embed and write documents, grouping several documents per embedding call.

    Returns the number of chunks written. Raises on embedding/DB errors after writing all
    documents that were complete before the failure (so a re-run resumes cleanly).
    """
    written = 0
    pending: list[PreparedDoc] = []

    def flush() -> None:
        nonlocal written
        if not pending:
            return
        texts = [embedding_input(d.parsed, c) for d in pending for c in d.chunks]
        vectors = embed(texts)
        pos = 0
        for d in pending:
            n = len(d.chunks)
            doc_id = write_document(conn, d, vectors[pos : pos + n])
            pos += n
            written += n
            if on_loaded:
                on_loaded(d, doc_id)
        pending.clear()

    for d in docs:
        if not d.chunks:
            doc_id = write_document(conn, d, [])  # keep the document row, nothing to embed
            if on_loaded:
                on_loaded(d, doc_id)
            continue
        pending.append(d)
        if sum(len(p.chunks) for p in pending) >= batch_chunks:
            flush()
    flush()
    return written
