# Overnight build log (2026-11-16)

Branch: `overnight-build` (not pushed). Session window: 19:25 → ~23:20 IST.
If work stopped early (usage limit), this file says exactly where. Nothing restarts on its own.

## Plan
1. Ingestion: CBIC Central Tax notifications + CGST circulars, 2019–2022 → parse → chunk → embed → Neon.
2. Temporal layer: LLM-extracted relations (amends/rescinds/supersedes…) → document + provision validity.
3. Query pipeline: understanding → hybrid date-filtered search → rerank → grounded answer with verified quotes → `POST /api/ask`.
4. Web UI (ask box, as-of date, AsOf vs baseline compare) + evaluation harness.
5. Integration, full runs, evaluation (baseline vs AsOf), README results.

## Status
| Step | Status | Notes |
|---|---|---|
| Schema v2 (documents.key/effective_date, relations.dst_ref/dst_kind) | done, applied to Neon | |
| Shared contracts `src/asof/models.py` | done | |
| 1. Ingestion | in progress (agent) | |
| 2. Temporal layer | code done (agent); full extraction running | `asof temporal` (LLM answers cached in data/cache/relations) |
| 3. Query pipeline + API | done (agent) | `asof.pipeline.ask`, `POST /api/ask`; thinking off by default (~1 s/LLM call) |
| 4. UI + eval harness | done (agent) | compare view (AsOf vs baseline) at `/`; `python -m asof.evaluation.run`; 22 seed questions need checking against corpus |
| 5. Integration / eval / README | not started | |

## How to resume
- `.venv/Scripts/python -m asof.ingest.run` (resumable) → `python -m asof.temporal.run` → `python -m asof.evaluation.run --systems asof,baseline`.
- Check limits first: `asof smoke` (shows OpenRouter daily quota).
