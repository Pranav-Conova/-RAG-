# AsOf — Architecture & Build Plan

> A time-aware RAG system for Indian GST law that answers **"what was the rule on date X?"**,
> follows amendment / supersession chains, and refuses to answer from repealed text.

---

## 1. The problem this project solves

Regulatory corpora are **versioned**, but standard RAG treats every chunk as equally "true".

Example (verify exact notification numbers/dates against CBIC before using as gold data):
Rule 36(4) of the CGST Rules (provisional ITC cap) was introduced at **20%** (Oct 2019),
cut to **10%** (Jan 2020), then **5%** (Jan 2021), and finally **omitted** (Jan 2022).
Ask a naive RAG *"What was the ITC cap under Rule 36(4) in February 2020?"* and it will
happily return whichever version is most semantically similar — often the wrong one.

AsOf fixes this by making **time a first-class retrieval dimension**.

### What makes it "advanced" (the portfolio story)
| Capability | Why naive RAG fails | AsOf approach |
|---|---|---|
| Point-in-time answers | No notion of validity | Every chunk carries a `[valid_from, valid_to)` interval; retrieval filters on `as_of` |
| Amendment / supersession awareness | Retrieves superseded circulars | LLM-extracted **amendment graph** (amends / substitutes / omits / rescinds / supersedes / clarifies) |
| "What changed between X and Y?" | Can't diff | Provision version history + diff view |
| Conflicting sources | Picks one at random | Conflict detection with precedence rules (Act > Rules > Notification > Circular; later > earlier) |
| Hallucinated citations | Cites loosely | Numbered-source citations with verbatim quotes checked against the source text + claim verification + abstention |
| "Does it actually work?" | No numbers | Custom temporal eval set, baseline vs. full-system results table |

---

## 2. Scope (keep it shippable)

**Corpus (MVP)** — all public, from the CBIC GST portal:
- CGST Act, 2017 and CGST Rules, 2017
- Central Tax notifications (2017 → present)
- CBIC GST circulars (2017 → present)

**Gold set** — hand-verify the full version history of **~25 high-traffic provisions**
(e.g. Rule 36(4), Rule 86B, Rule 89, Section 16(4), Section 50 interest, e-invoicing thresholds).
These anchor the eval and the provision-level timeline UI. Everything else gets
document-level temporal handling (validity + supersession), which is far easier to automate.

**Out of scope for v1:** state GST (SGST) variations, case law, full automatic consolidation of
every provision's text.

---

## 3. Tech stack (Python)

| Layer | Choice | Why |
|---|---|---|
| Packaging | `uv`, Python 3.12, `pyproject.toml` | Fast, modern |
| Scraping | `httpx` + `selectolax` | Simple, polite rate-limited crawler |
| PDF parsing | `PyMuPDF` + `docling` (tables); OCR fallback (`rapidocr-onnxruntime`) for scanned PDFs | Older notifications are often scans |
| Database | **Postgres 16 + pgvector** (Docker locally; Neon free tier hosted, Singapore) | Vector search, full-text search, temporal range filters and graph edges (recursive CTEs) in **one SQL query**. Vectors stored as `halfvec(1024)` to fit free-tier storage |
| Embeddings | NVIDIA API catalog: `nvidia/nemotron-3.5-embed-1b`, 1024-d (`input_type` passage/query) | Free, hosted, multilingual |
| Reranker | OpenRouter `nvidia/nemotron-3.5-rerank-8b:free` (cross-encoder); automatic fallback to LLM scoring on errors or when the free daily cap (50/day, 1000/day after a one-time $10 top-up) is hit | Free; NVIDIA doesn't serve a reranker to free build.nvidia.com keys |
| LLM | NVIDIA API catalog (OpenAI-compatible): `nvidia/nemotron-3.5-super-vl-120b-a12b` for both answers and fast steps (about 1 s per call; `mistral-large-4` took 7–300 s on the free tier) | Free. Structured output = JSON schema in prompt + Pydantic validation + one repair retry (optional NIM `guided_json`). Catalog models get retired, so `asof smoke` checks they are live |
| Rate limits | Shared client-side limiter (35 req/min) + retries on 429 | Free tier is about 40 req/min; bulk extraction is cached on disk so it only runs once |
| Schemas | `pydantic` v2 | Typed extraction + API models |
| API | `FastAPI` | |
| UI | `Streamlit` (+ `pyvis` for graph view) | As-of date picker, timeline, amendment graph |
| Eval | `pytest` + custom harness | Domain-specific metrics beat generic ones |
| Tracing | structured JSON logs per pipeline step (Langfuse optional) | Debuggability is a portfolio signal |
| Deploy | Render free web service (API only, about 100 MB RAM since all models are remote); ingestion runs locally | |

---

## 4. Architecture

```mermaid
flowchart LR
  subgraph Ingest
    A[CBIC crawler] --> B[PDF parse + OCR]
    B --> C[Section-aware chunker]
    B --> D[Relation extractor<br/>LLM, structured output]
    D --> E[(Amendment graph)]
    C --> F[Validity resolver<br/>assign valid_from/valid_to]
    E --> F
    F --> G[(Postgres + pgvector<br/>chunks, versions, edges)]
  end
  subgraph Query
    Q[User question] --> U[Query understanding<br/>as_of date, type, refs, sub-questions]
    U --> R1[Hybrid retrieval<br/>vector + FTS, filtered by as_of]
    U --> R2[Direct ref lookup<br/>'rule 36(4)']
    R1 --> X[RRF fusion]
    R2 --> X
    X --> GE[Graph expansion<br/>later amendments, clarifying circulars]
    GE --> RR[Rerank]
    RR --> SC[Supersession + conflict check]
    SC --> GEN[Answer generation<br/>Citations API]
    GEN --> V[Claim verification<br/>+ abstention]
    V --> OUT[Answer + citations + timeline]
  end
  G --> R1
  G --> R2
  E --> GE
```

### Data model (core tables)
```
documents(id, kind[act|rules|notification|circular|order], number, issue_date,
          title, url, pdf_path, lang, status[in_force|rescinded|superseded])
provisions(id, instrument[cgst_act|cgst_rules], ref)            -- e.g. "rule 36(4)"
provision_versions(id, provision_id, text, valid_from, valid_to, source_doc_id)
chunks(id, doc_id, provision_version_id NULL, section_ref, page, text,
       embedding halfvec(1024), tsv tsvector, validity daterange)
relations(id, src_doc_id, dst_doc_id NULL, dst_provision_id NULL,
          kind[amends|substitutes|inserts|omits|rescinds|supersedes|clarifies],
          effective_date, evidence_span, confidence, verified bool)
```

### Key query: temporal hybrid retrieval (sketch)
```sql
WITH vec AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> :qvec) AS r
  FROM chunks WHERE validity @> :as_of::date
  ORDER BY embedding <=> :qvec LIMIT 50
), fts AS (
  SELECT id, row_number() OVER (ORDER BY ts_rank(tsv, q) DESC) AS r
  FROM chunks, plainto_tsquery('english', :qtext) q
  WHERE tsv @@ q AND validity @> :as_of::date
  LIMIT 50
)
SELECT id, SUM(1.0 / (60 + r)) AS rrf
FROM (SELECT * FROM vec UNION ALL SELECT * FROM fts) s
GROUP BY id ORDER BY rrf DESC LIMIT 30;
```

---

## 5. Build plan (≈ 8 weeks part-time)

Rule: **build the baseline and the eval set before the clever parts**, so every improvement
has a measured number attached. That results table is the centrepiece of the portfolio.

### Phase 0 — Setup (2–3 days) ✅ scaffolded
- `pyproject.toml`, `docker-compose.yml` (Postgres + pgvector), `.env.example`, `render.yaml`,
  schema, NVIDIA providers (LLM / embeddings / reranker + rate limiter), `asof db-init`,
  `asof smoke`, FastAPI `/health` + `/ready`, GitHub Actions with a pgvector service.
- **Done when:** `asof smoke` passes with your key and CI is green.

### Phase 1 — Corpus + naive baseline (week 1–2)
- Crawler for notifications, circulars, Act, Rules → `data/raw/` + metadata JSON.
- Parsing (PyMuPDF → docling for tables → OCR fallback), section-aware chunking
  (split on Section / Rule / para numbering, keep `section_ref`).
- **Baseline RAG:** vector-only retrieval, no dates, simple prompt. Keep it in the repo
  as `baseline/` — it is the control group.
- **Done when:** baseline answers questions end to end; corpus stats logged (docs, pages, OCR %).

### Phase 2 — Evaluation set (week 2) ← do this early
~150 questions in `eval/questions.jsonl`, each with gold answer, gold citations and `as_of`:
| Type | Example | Count |
|---|---|---|
| Point-in-time | "ITC cap under Rule 36(4) in Feb 2020?" | 40 |
| Current law | "What is the current time limit under Section 16(4)?" | 30 |
| Diff / history | "How did Rule 36(4) change between 2019 and 2022?" | 25 |
| Supersession trap | Question whose most similar chunk is a superseded circular | 30 |
| Unanswerable | Outside corpus or before the provision existed | 25 |

Metrics (`eval/metrics.py`): answer correctness (LLM-graded against gold + exact-match on
numbers/dates), **temporal accuracy**, citation precision/recall, **stale-source rate**
(% answers citing a document not in force at `as_of`), correct-abstention rate, latency, cost/query.
- **Done when:** `uv run python -m eval.run --system baseline` prints the first results row.

### Phase 3 — Temporal layer (week 3–4) ← the novel core
- **Relation extraction:** for each notification/circular, extract
  `{target, action, effective_date, new_text?, evidence_span}` with `LLM.structured(...)`
  into Pydantic models. Cache each result to `data/cache/` keyed by document hash, so the
  rate-limited run (a few thousand docs at 35 req/min ≈ 1–2 h) happens once and can resume.
- **Validation:** regex cross-check on notification numbers and dates; low-confidence edges go
  to a review CSV; gold provisions are hand-verified.
- **Validity resolver:** compute `valid_from/valid_to` per document and per gold provision version
  by walking the graph (recursive CTE or `networkx` in Python, then persist).
- **As-of retrieval:** add the `validity @> as_of` filter.
- **Done when:** eval shows a clear drop in stale-source rate vs. baseline; Rule 36(4) timeline is correct.

### Phase 4 — Retrieval upgrades (week 5)
- Query understanding (structured output): `as_of` (default today), query type, explicit refs,
  sub-questions for multi-part queries.
- Hybrid search (vector + Postgres FTS) with RRF; direct ref lookup; graph expansion
  (pull later amending notifications and clarifying circulars); bge reranker.
- Ablation runs: each component on/off.
- **Done when:** ablation table shows each component's contribution.

### Phase 5 — Grounded generation, verification, abstention (week 6)
- Generate with retrieved chunks numbered `[S1]…[Sn]`; the model must cite `[S#]` and include a
  short verbatim quote per claim. Code checks each quote actually appears in that source
  (fuzzy match), which gives exact character spans for UI highlighting.
- Verification pass: for each claim, check it is entailed by its cited text (structured output
  `{claim, supported, reason}`); drop unsupported claims; abstain if the core claim is unsupported.
- Answer always states its `as_of` date and flags "this was later amended by …" when relevant.
- **Done when:** citation precision ≥ 0.9 on eval and unanswerable questions are abstained on.

### Phase 6 — Conflict detection (week 7, stretch)
- When retrieved sources disagree, resolve by precedence (statute > rules > notification >
  circular; later > earlier; specific > general) and surface the conflict to the user.
- **Done when:** a dedicated conflict subset in eval passes.

### Phase 7 — Product + portfolio polish (week 8)
- FastAPI endpoints: `/ask`, `/provision/{ref}/history`, `/document/{id}`.
- Streamlit UI: question box + **as-of date picker**, answer with clickable citations,
  provision **timeline** view, amendment **graph** view.
- README: problem, architecture diagram, **results table (baseline vs. AsOf)**, ablations,
  demo GIF, honest limitations. Optional blog post: *"Why your RAG gives outdated legal answers."*
- Deploy demo.

---

## 6. Repo layout

```
-RAG-/
  pyproject.toml
  docker-compose.yml
  .env.example                  # NVIDIA_API_KEY, DATABASE_URL, model names
  src/asof/
    config.py
    db/        schema.sql, repo.py
    ingest/    crawl.py, parse.py, chunk.py
    temporal/  extract_relations.py, models.py, graph.py, validity.py
    retrieval/ understand.py, hybrid.py, expand.py, rerank.py
    generation/answer.py, verify.py
    pipeline.py
    api/       main.py
  baseline/    naive_rag.py
  ui/          app.py
  eval/        questions.jsonl, run.py, metrics.py, results/
  tests/
  data/        raw/, parsed/   (gitignored)
```

---

## 7. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Scanned / messy PDFs | OCR fallback; log OCR share; exclude unparseable docs explicitly |
| Wrong amendment edges | Confidence scores, regex cross-checks, review queue, hand-verified gold provisions |
| Full text consolidation is hard | Only consolidate the ~25 gold provisions; document-level validity elsewhere |
| LLM cost | Batch API for extraction, cache parsed results on disk, local embeddings + reranker |
| Scope creep | SGST, case law, multilingual UI are explicitly v2 |

## 8. v2 ideas
- Extend to RBI master directions / SEBI circulars (same architecture, new crawler).
- "Watch mode": new CBIC notification → auto-extract edges → alert which provisions changed.
- Hindi query support (the Nemotron embedder is multilingual).
