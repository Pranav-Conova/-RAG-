"""Every provision a chunk mentions, stored in `chunks.refs` for exact reference matching.

A chunk carries one `section_ref`, but amending notifications touch several rules in one
passage ("in rule 9, in sub-rule (2), ...; in rule 36, ..."). Without this, a question about
rule 9(2) cannot find a chunk labelled "rule 8". No LLM calls: regex only.
"""

from asof.temporal.normalize import find_provision_refs


def chunk_refs(text: str, section_ref: str | None) -> list[str]:
    refs = find_provision_refs(text or "")
    if section_ref:
        refs += [r for r in find_provision_refs(section_ref) if r not in refs]
    return sorted({r.lower() for r in refs})


def backfill_chunk_refs(conn) -> int:
    """Recompute `chunks.refs` for all chunks; returns how many rows changed."""
    rows = conn.execute("SELECT id, text, section_ref, refs FROM chunks").fetchall()
    changed = [(chunk_refs(text, sref), cid) for cid, text, sref, old in rows
               if chunk_refs(text, sref) != sorted(old or [])]
    if changed:
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany("UPDATE chunks SET refs = %s WHERE id = %s", changed)
    return len(changed)
