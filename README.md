# AsOf: Time-Aware RAG for Indian GST Law

[![CI](https://github.com/Pranav-Conova/-RAG-/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranav-Conova/-RAG-/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![PostgreSQL + pgvector](https://img.shields.io/badge/DB-PostgreSQL%20%2B%20pgvector-336791)
![Status: working prototype](https://img.shields.io/badge/status-working%20prototype-yellowgreen)

> **AsOf is an open-source, time-aware retrieval-augmented generation (RAG) system for Indian
> GST law.** It answers *"what was the rule on date X?"* by storing the period each
> notification, circular and provision was in force, following amendment and supersession
> chains, and citing the exact source text.

**Contents:** [Key facts](#key-facts) · [The problem](#the-problem-standard-rag-answers-from-outdated-law) ·
[How it works](#how-asof-works) · [Features](#features-and-status) · [Tech stack](#tech-stack) ·
[Quick start](#quick-start) · [Configuration](#configuration) · [Deployment](#deploying-to-render) ·
[Project structure](#project-structure) · [Roadmap](#roadmap) · [FAQ](#faq) ·
[Glossary](#glossary-of-technical-terms) · [Citation](#how-to-cite)

---

## Key facts

| | |
|---|---|
| **What it is** | A time-aware (point-in-time) RAG pipeline: legal question in, dated and cited answer out |
| **Domain** | Indian Goods and Services Tax (GST): CGST Act 2017, CGST Rules 2017, CBIC Central Tax notifications and CBIC circulars |
| **Core idea** | Every chunk of text stores a **validity date range**; retrieval only uses text that was in force on the requested date |
| **Language** | Python 3.11+ |
| **Models** | NVIDIA Nemotron 3.5 (LLM and embeddings, NVIDIA API catalog) and Nemotron 3.5 Rerank 8B (via OpenRouter); all free tier |
| **Storage** | PostgreSQL + pgvector (vector search, full-text search, date ranges and an amendment graph in one database) |
| **Hosting** | FastAPI on Render, database on Neon (both free tier, Singapore region) |
| **Status** | Working end to end: ingestion, temporal layer, hybrid retrieval, grounded answers, web UI and evaluation harness |
| **Disclaimer** | Portfolio and research project. **Not legal or tax advice.** |

## The problem: standard RAG answers from outdated law

Regulations are **versioned**. A GST rule can be introduced, amended several times, and finally
omitted, and each change arrives as a separate notification. For example, the provisional
input tax credit (ITC) cap in **Rule 36(4) of the CGST Rules** changed several times between
2019 and 2022 before the sub-rule was omitted.

A standard RAG system retrieves whichever passage is most *semantically similar* to the
question. For versioned law that is often an **amended or superseded** version, so the answer
sounds right, cites a real document, and is wrong for the date that matters.

**AsOf makes time a first-class retrieval dimension**:

| Question type | Standard RAG | AsOf |
|---|---|---|
| "What was the ITC cap under Rule 36(4) in February 2020?" | Returns the most similar version, any year | Returns only the version valid on 2020-02-15 |
| "Is circular X still in force?" | Can't tell | Follows the supersession chain |
| "What changed between 2019 and 2022?" | No notion of versions | Lists each version with its validity dates |
| Conflicting sources | Picks one arbitrarily | Applies precedence: statute > rules > notification > circular; later > earlier |
| Unsupported claim | Often answers anyway | Verifies citations and abstains when unsupported |

## How AsOf works

```mermaid
flowchart LR
  subgraph Ingestion
    A[CBIC crawler] --> B[PDF parsing + OCR]
    B --> C[Section-aware chunking]
    B --> D[LLM relation extraction]
    D --> E[(Amendment graph)]
    C --> F[Validity resolver]
    E --> F
    F --> G[(PostgreSQL + pgvector)]
  end
  subgraph Query
    Q[Question + as-of date] --> U[Query understanding]
    U --> R[Hybrid search filtered by date]
    R --> X[Graph expansion]
    X --> RR[Reranking]
    RR --> GEN[Grounded answer generation]
    GEN --> V[Citation verification]
    V --> OUT[Dated answer + citations]
  end
  G --> R
  E --> X
```

1. **Ingestion.** CBIC notifications and circulars are downloaded, parsed (with OCR for scanned
   PDFs) and split into section-aware chunks such as "rule 36(4)".
2. **Amendment graph.** An LLM extracts relations between documents (*amends*, *substitutes*,
   *inserts*, *omits*, *rescinds*, *supersedes*, *clarifies*) with effective dates.
3. **Validity resolution.** Each chunk gets a validity date range `[valid_from, valid_to)`.
4. **Query understanding.** The question is parsed for its as-of date (default: today),
   question type and explicit references such as rule or notification numbers.
5. **Hybrid temporal retrieval.** Vector search and full-text search run in one SQL query,
   filtered to text valid on the as-of date, and merged with reciprocal rank fusion.
6. **Graph expansion and reranking.** Later amendments and clarifying circulars are pulled in,
   then a cross-encoder reranker orders the candidates.
7. **Grounded generation and verification.** The LLM answers from numbered sources and quotes
   them; each quote is checked against its source, and unsupported claims are removed or the
   system abstains.

The core retrieval query (simplified):

```sql
SELECT text FROM chunks
WHERE validity @> DATE '2020-02-15'          -- only text in force on the as-of date
ORDER BY embedding <=> :query_embedding      -- cosine distance (pgvector)
LIMIT 20;
```

## Features and status

| Feature | Status |
|---|---|
| PostgreSQL schema with date-range validity, amendment relations, HNSW and full-text indexes | ✅ Done |
| NVIDIA LLM + embedding providers with rate limiting, retries and structured (JSON) output | ✅ Done |
| OpenRouter reranker with automatic LLM-based fallback | ✅ Done |
| `asof smoke` live check of every model, `asof db-init` schema setup | ✅ Done |
| FastAPI service with health checks, SEO landing page, `robots.txt`, `sitemap.xml`, `llms.txt` | ✅ Done |
| CBIC crawler (notifications + circulars, 2019–2022), PDF parsing, OCR fallback, section-aware chunking, embedding loader | ✅ Done |
| Baseline (date-unaware, vector-only) RAG as a control group | ✅ Done |
| LLM extraction of amendment relations (amends / substitutes / inserts / omits / rescinds / supersedes / clarifies) with verbatim-evidence checks | ✅ Done |
| Validity resolution: document rescission/supersession and provision-level version chains | ✅ Done |
| Query understanding (as-of date, references, question type), hybrid search (vector + full-text + reference boost, RRF), date filter, reranking | ✅ Done |
| Grounded answers with numbered citations, quote verification and abstention | ✅ Done |
| `POST /api/ask` and a web UI with as-of date picker and side-by-side AsOf vs baseline comparison | ✅ Done |
| Evaluation harness (correctness, recall@k, citation precision, stale-source rate, abstention, latency) | ✅ Done |
| Larger evaluation set (~150 questions) and published ablations | 🚧 Next |
| Conflict detection between sources, provision timeline view | 🚧 Next |

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem for AI, parsing and data work |
| LLM | `nvidia/nemotron-3.5-super-vl-120b-a12b` via the NVIDIA API catalog | Free, about 1 s per call, OpenAI-compatible API |
| Embeddings | `nvidia/nemotron-3.5-embed-1b`, 1024 dimensions | Free, multilingual, asymmetric query/passage modes |
| Reranker | `nvidia/nemotron-3.5-rerank-8b:free` via OpenRouter, LLM fallback | Free cross-encoder with a graceful fallback |
| Database | PostgreSQL + pgvector (`halfvec`), hosted on Neon | Vectors, full-text search, date ranges and graph in one SQL query |
| API | FastAPI + Uvicorn | Typed, async-ready, automatic OpenAPI docs |
| Validation | Pydantic v2, pydantic-settings | Typed config and LLM output validation |
| HTTP | httpx, OpenAI Python SDK, tenacity | Clients and retries with backoff |
| Parsing (planned) | PyMuPDF, Docling, RapidOCR | PDFs, tables and scanned documents |
| CI | GitHub Actions with a pgvector service container | Lint and run all tests on every push |
| Hosting | Render (API), Neon (database) | Free tiers, same region (Singapore) |

## Quick start

```bash
# 1. Install (Python 3.11+)
git clone https://github.com/Pranav-Conova/-RAG-.git asof && cd asof
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,ingest for PDF parsing (Phase 1)

# 2. Configure: copy the template and add your keys
cp .env.example .env               # NVIDIA key: build.nvidia.com, OpenRouter key: openrouter.ai

# 3. Create the database schema (Neon connection string in .env, or: docker compose up -d)
asof db-init

# 4. Check every model responds
asof smoke

# 5. Build the corpus (resumable; downloads ~340 CBIC PDFs, embeds and loads them)
pip install -e ".[ingest]"
asof ingest --years 2019-2022
asof temporal                      # extract amendment relations + resolve validity dates

# 6. Ask questions
asof ask "What was the provisional ITC limit under rule 36(4) in February 2020?"
asof ask "..." --mode baseline     # the naive RAG, for comparison

# 7. Run the web app, the evaluation and the tests
uvicorn asof.api.main:app --reload   # web UI at http://localhost:8000, API docs at /docs
asof eval --systems asof,baseline    # writes eval/results/summary.md
pytest
```

## Configuration

All settings are environment variables (or `.env`). See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | none | NVIDIA API catalog key (`nvapi-…`) |
| `LLM_MODEL` / `LLM_FAST_MODEL` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Answer model / model for extraction, query understanding, verification |
| `EMBED_MODEL` / `EMBED_DIM` | `nvidia/nemotron-3.5-embed-1b` / `1024` | Embedding model and vector size (must match the database) |
| `RERANK_BACKEND` | `llm` (`openrouter` recommended) | `openrouter`, `llm`, `nim` or `none` |
| `OPENROUTER_API_KEY` | none | OpenRouter key (`sk-or-…`) for reranking |
| `OPENROUTER_RERANK_MODEL` | `nvidia/nemotron-3.5-rerank-8b:free` | Reranker model on OpenRouter |
| `NIM_REQUESTS_PER_MINUTE` | `35` | Client-side rate limit for NVIDIA (free tier is about 40/min) |
| `DATABASE_URL` | local Docker Postgres | PostgreSQL connection string (Neon pooled URL in production) |
| `PUBLIC_BASE_URL` | derived from request | Public site URL for canonical links, sitemap and `llms.txt` |

## Deploying to Render

1. Create a free **Neon** project in **AWS Asia Pacific (Singapore)** (1 GB storage, scales to
   zero when idle). Put its **pooled** connection string in `.env` as `DATABASE_URL` and run
   `asof db-init`; this also enables the `vector` extension.
2. In Render choose **New → Blueprint** and select this repository. [`render.yaml`](render.yaml)
   deploys the API in the Singapore region. Set `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`,
   `DATABASE_URL` and optionally `PUBLIC_BASE_URL`.
3. Run ingestion from your own machine with `DATABASE_URL` pointing at Neon. Heavy parsing and
   OCR never run on Render.

**Free-tier limits:** NVIDIA API catalog about 40 requests/min (varies by model); OpenRouter
free models 20 requests/min and 50 requests/day (1,000/day after a one-time $10 credit
purchase); Render free instances sleep when idle, so the first request takes about a minute.

## Project structure

```
src/asof/
  config.py              settings from environment / .env
  cli.py                 `asof db-init`, `asof smoke`
  db/schema.sql          tables, date ranges, vector + full-text + GiST indexes
  db/connection.py       connection, schema init, embedding-model pinning
  providers/llm.py       chat + validated JSON output (NVIDIA API catalog)
  providers/embeddings.py  query/passage embeddings
  providers/rerank.py    OpenRouter, LLM, NVIDIA and no-op rerankers + fallback
  providers/ratelimit.py sliding-window rate limiter
  api/main.py            FastAPI app: landing page, crawler files, /health, /ready
  api/site.py            landing page, FAQ, JSON-LD, robots.txt, sitemap.xml, llms.txt
tests/                   unit tests (mocked HTTP) + database tests (pgvector)
PLAN.md                  architecture, data model and phased build plan
llms.txt                 project summary for AI assistants and answer engines
CITATION.cff             citation metadata
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup: schema, model providers, CLI, API, CI | ✅ |
| 1 | Corpus ingestion and a date-unaware baseline RAG | ✅ |
| 2 | Evaluation harness and seed question set (point-in-time, current, history, supersession-trap, unanswerable) | ✅ (set growing) |
| 3 | Temporal layer: amendment graph, validity resolution, as-of retrieval | ✅ |
| 4 | Retrieval upgrades: hybrid search, query understanding, reranking | ✅ (ablations next) |
| 5 | Grounded generation, citation verification, abstention | ✅ |
| 6 | Conflict detection between sources | 🚧 |
| 7 | Web UI ✅, deployment and published results | 🚧 |

Details are in [PLAN.md](PLAN.md).

## FAQ

### What is AsOf?
AsOf is an open-source, time-aware retrieval-augmented generation (RAG) system for Indian GST
law. It answers questions as of a specific date, using only the notifications, circulars and
provisions that were in force on that date, and cites the exact source text.

### What is time-aware (temporal) RAG?
Time-aware RAG adds a date dimension to retrieval. Every piece of text stores the period it was
valid for, and a question is answered only from text that was in force on the requested date,
instead of from whichever text sounds most similar.

### How is AsOf different from a standard RAG chatbot?
A standard RAG chatbot retrieves the most semantically similar passage, which for versioned law
is often an amended or superseded rule. AsOf filters by validity dates, follows amendment and
supersession chains, resolves conflicts by legal precedence, and verifies that every cited
quote appears in its source.

### Which documents does AsOf cover?
The planned corpus is the CGST Act 2017, the CGST Rules 2017, CBIC Central Tax notifications and
CBIC GST circulars from 2017 onwards. State GST variations and case law are out of scope for
version 1.

### Which AI models does AsOf use?
NVIDIA Nemotron 3.5 Super for generation, NVIDIA Nemotron 3.5 Embed 1B for 1024-dimensional
embeddings, and NVIDIA Nemotron 3.5 Rerank 8B (via OpenRouter) for reranking. All models are
configurable, and `asof smoke` checks that the configured models are still available.

### Is AsOf free to run?
Yes. It runs entirely on free tiers: the NVIDIA API catalog, OpenRouter free models, Neon
PostgreSQL and Render. Free tiers are rate limited and intended for development and demos.

### Why PostgreSQL and pgvector instead of a dedicated vector database?
AsOf needs vector similarity, keyword search, date-range filtering and graph traversal in the
same query. PostgreSQL with pgvector does all four, so a single SQL query can return text that
is both relevant and valid on the requested date.

### How will accuracy be measured?
With a custom evaluation set of about 150 dated questions, including "supersession trap"
questions whose most similar passage is outdated, and unanswerable questions. Each phase is
compared against a date-unaware baseline RAG. Results will be published here when available.

### Can the approach be used for other regulations?
Yes. The design applies to any versioned rulebook, such as RBI master directions, SEBI
circulars, insurance policy wordings, medical guidelines or technical standards. Only the
crawler and relation extraction prompts are domain-specific.

### Is AsOf legal or tax advice?
No. AsOf is a portfolio and research project. Always confirm answers against the official CBIC
documents or with a qualified professional.

## Glossary of technical terms

**AI and retrieval**

- **Abstention**: the system declining to answer when its sources do not support an answer, instead of guessing.
- **Ablation**: turning one component off (for example reranking) to measure how much it contributes.
- **Answer engine optimization (AEO)**: structuring content as direct questions and answers so search and answer engines can quote it.
- **As-of date (point-in-time query)**: the date a question refers to; only text valid on that date may be used.
- **Asymmetric embeddings**: embedding models that encode questions (`query`) and documents (`passage`) differently for better retrieval.
- **Chunking**: splitting documents into smaller passages for indexing; AsOf splits on section and rule boundaries.
- **Citation verification**: checking that each quoted passage in an answer actually appears in the cited source.
- **Cosine similarity / distance**: a measure of how similar two embedding vectors are; pgvector's `<=>` operator returns cosine distance.
- **Cross-encoder**: a model that reads a question and a passage together and scores their relevance; more accurate than embeddings alone, used for reranking.
- **Embedding**: a list of numbers (a vector) representing the meaning of a text, so similar texts have nearby vectors.
- **Evaluation set (eval)**: a fixed set of questions with known answers and sources, used to measure accuracy objectively.
- **Fallback**: a backup component used automatically when the primary one fails, such as the LLM reranker when OpenRouter is unavailable.
- **Generative engine optimization (GEO)**: making content easy for AI systems (ChatGPT, Claude, Perplexity, Google AI Overviews) to understand and cite accurately.
- **Grounding / grounded generation**: generating answers only from retrieved sources, not from the model's memory.
- **Hallucination**: a fluent but false or unsupported statement produced by a language model.
- **Hybrid search**: combining vector (semantic) search with keyword (full-text) search.
- **Large language model (LLM)**: a neural network trained on text that generates and reasons over language.
- **Logit**: a model's raw, unnormalized score; for rerankers, higher means more relevant.
- **Matryoshka embeddings**: embeddings that can be shortened (for example 2048 to 1024 dimensions) with little quality loss.
- **Query understanding**: extracting structure from a question, such as its date, type and referenced rules.
- **Rate limiting**: capping requests per minute to stay within a provider's limits.
- **Reciprocal rank fusion (RRF)**: merging several ranked result lists by summing `1 / (k + rank)` for each item.
- **Reranking**: re-ordering retrieved candidates with a more accurate (slower) model before generation.
- **Retrieval-augmented generation (RAG)**: answering questions by first retrieving relevant documents, then having an LLM generate an answer from them.
- **Structured output**: making an LLM return JSON that matches a schema, validated in code (here with Pydantic).
- **Temporal / time-aware RAG**: RAG in which every source has a validity period and retrieval is filtered by date.

**Legal and GST**

- **Amendment**: a change to an existing provision (by substitution, insertion or omission).
- **Amendment graph**: a graph whose edges record which document amends, supersedes or clarifies which provision or document, and from when.
- **CBIC**: Central Board of Indirect Taxes and Customs, which administers GST in India and issues notifications and circulars.
- **CGST / SGST / IGST**: Central, State and Integrated Goods and Services Tax.
- **Circular**: a CBIC clarification of how the law should be applied.
- **Input tax credit (ITC)**: tax paid on purchases that a business can offset against tax on its sales.
- **Notification**: a legally binding instrument that brings provisions into force, amends rules or changes rates.
- **Omission**: removal of a provision, after which it has no text in force.
- **Precedence**: the order used to resolve conflicts: statute > rules > notification > circular; later > earlier; specific > general.
- **Provision**: a citable unit of law, such as "Section 16(4)" or "Rule 36(4)".
- **Rescission**: cancellation of a notification.
- **Supersession**: replacement of an earlier document by a newer one.
- **Validity interval**: the half-open date range `[valid_from, valid_to)` in which a text was in force.

**Databases and infrastructure**

- **Connection pooler**: a service that shares database connections between clients (Neon's pooled endpoint).
- **daterange**: a PostgreSQL type holding a date interval; `validity @> date` tests whether a date falls inside it.
- **FastAPI**: a Python web framework for typed APIs with automatic OpenAPI documentation.
- **GiST index**: a PostgreSQL index type that makes range queries such as date containment fast.
- **halfvec**: pgvector's 16-bit vector type; it halves storage compared with 32-bit vectors.
- **HNSW (Hierarchical Navigable Small World)**: an approximate nearest-neighbour index that makes vector search fast.
- **JSON-LD / schema.org**: structured data embedded in web pages so search engines understand the content (used on the landing page).
- **llms.txt**: a plain-text file that summarizes a project for AI assistants and crawlers.
- **Neon**: serverless PostgreSQL hosting with a free tier and scale-to-zero.
- **NVIDIA API catalog (NIM)**: NVIDIA's hosted model APIs at build.nvidia.com, with an OpenAI-compatible interface.
- **OCR (optical character recognition)**: extracting text from scanned documents.
- **OpenRouter**: a single API that routes requests to many model providers, including free models.
- **pgvector**: a PostgreSQL extension that adds vector types, distance operators and vector indexes.
- **Render**: a cloud platform that deploys web services from a Git repository using a `render.yaml` blueprint.
- **Row-level security (RLS)**: PostgreSQL access control applied per row; enabled here to block unintended public APIs.
- **Scale to zero**: a database or service that stops when idle and starts on the next request.
- **tsvector / full-text search**: PostgreSQL's keyword index for matching words in documents.

## How to cite

If you reference this project, use the metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows a
**"Cite this repository"** button), or:

```bibtex
@software{asof_time_aware_rag,
  title  = {AsOf: Time-Aware RAG for Indian GST Law},
  author = {Pranav-Conova},
  year   = {2026},
  url    = {https://github.com/Pranav-Conova/-RAG-},
  note   = {Open-source time-aware retrieval-augmented generation for versioned regulations}
}
```

## Disclaimer

AsOf is a portfolio and research project. It is not legal or tax advice, and its answers may be
incomplete or out of date. Always check the official [CBIC](https://cbic-gst.gov.in/) source
documents or consult a qualified professional.
