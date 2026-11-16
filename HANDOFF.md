# Overnight build log (2026-11-16)

Branch: `overnight-build` (local commits, **not pushed**). Session window: 19:25 → ~23:20 IST.
If work stopped early (usage limit), this file says exactly where. Nothing restarts on its own.

## Status
| Step | Status | Notes |
|---|---|---|
| Schema v2/v3 (documents.key/effective_date, relations.dst_ref/dst_kind, chunks.refs) | done, applied to Neon | `asof db-init` is idempotent |
| 1. Ingestion | done | 337/339 docs (237 notifications, 100 circulars, 2019–2022), 833 chunks, 832 embedded |
| 2. Temporal layer | done | 895 relations, 279 provision versions (40 provisions with 2+ versions), 15 docs rescinded/superseded |
| 3. Query pipeline + API | done | `asof ask`, `POST /api/ask`; LLM thinking off by default (~1 s/call) |
| 4. Web UI + eval harness | done | `/` has the ask panel + AsOf-vs-baseline compare view |
| 5. Eval questions | done | `eval/benchmark.jsonl` = 20 hand-written seed (checked against source text) + 29 generated (23 point-in-time + 6 unanswerable) |
| 6. Full evaluation run | see `eval/results/summary.md` | reranker = LLM for the eval (OpenRouter's 50/day cap would mix rerankers) |
| 7. README results + screenshot | see README | |

## Fixes made during integration
- Provision validity: substituting/omitting only a proviso/clause/Explanation no longer ends every
  earlier version (rule 36(4) "5 per cent" now valid 2021-01-01 → 2022-01-01, as in the law).
- `chunks.refs`: all provisions a chunk mentions; search also matches parent rules
  (question "rule 9(2)" finds "in rule 9, … in sub-rule (2)").
- Eval questions: only verbatim, substantive, unambiguous key facts; as-of dates kept inside the
  corpus period (≤ 2022-12-31); `a || b` alternative phrasings in metrics.

## Known issues / next steps
1. **Dead PDF links (fixed in DB)**: CBIC moved 164 of the 2019–2020 notification PDFs (404);
   ingestion recovered them from taxinformation.cbic.gov.in's API. `scripts/fix_dead_links.py`
   repointed those 164 to the portal's notification search page and the UI labels them
   "Find <number> on the CBIC portal". Re-run it after `asof ingest --force`. A per-document
   deep link would need the portal id (`crawl.py` `notification_ids`) or self-hosted PDFs (82 MB).
2. NVIDIA embeddings endpoint 500s on multi-text batches; the loader falls back to 1 text/request
   (full re-embed ≈ 45 min at 20 RPM). `providers/embeddings.py` could do this itself.
3. One chunk (circular 131/01/2020) was refused by NVIDIA's safety filter → no embedding (FTS only).
4. Notification 18/2021's "English" PDF is Hindi → almost no English text.
5. Rule-level chains start at the first amendment in 2019 (no base text of the 2017 Rules).
6. Retrieval ranking: for some provision questions the reranker prefers a related circular over
   the amending notification (e.g. rule 9(2)); ablations (reranker on/off, ref boost) are next.
7. Deploy to Render not done (needs your env vars in the dashboard). `PUBLIC_BASE_URL` optional.
8. Temporal agent briefly wrote test rows into Neon's public tables during development; it removed
   them and I verified the DB (only `public` schema, halfvec(1024), counts consistent).

## How to resume
```bash
.venv/Scripts/asof smoke                                  # models alive? OpenRouter quota?
.venv/Scripts/asof ingest --years 2019-2022               # resumable
.venv/Scripts/asof temporal                               # cached LLM answers → cheap
RERANK_BACKEND=llm .venv/Scripts/asof eval --systems asof,baseline --questions eval/benchmark.jsonl
python scripts/make_temporal_questions.py --max-provisions 40   # regenerate grounded questions
```
