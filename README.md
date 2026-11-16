# AsOf — time-aware RAG for Indian GST law

Regular RAG returns whichever version of a rule *sounds* closest to your question, even if it was
replaced years ago. **AsOf answers "what was the rule on date X?"**, follows amendment and
supersession chains, and refuses to answer from repealed text.

> Status: Phase 0 (scaffold). See [PLAN.md](PLAN.md) for the architecture and roadmap.

## Stack
- **Models (all free, hosted on the NVIDIA API catalog):**
  LLM `nvidia/nemotron-3.5-super-vl-120b-a12b` (answers, query understanding, extraction,
  verification; about 1 s per call), embeddings `nvidia/nemotron-3.5-embed-1b` (1024-d).
- **Reranker (free, via OpenRouter):** `nvidia/nemotron-3.5-rerank-8b:free`, falling back to
  LLM-based reranking if OpenRouter errors or the daily free limit is hit. All swappable in `.env`.
  NVIDIA retires catalog models regularly; `asof smoke` checks the configured ones are still live.
- **Storage:** Postgres + pgvector on Neon (vectors, full-text search, date-range validity,
  amendment graph).
- **App:** FastAPI, deployed on Render. Ingestion runs locally.

## Quick start

```bash
# 1. Install (Python 3.11+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,ingest when you start Phase 1

# 2. Configure
cp .env.example .env               # paste your key from build.nvidia.com

# 3. Database: a Neon connection string in .env (or local: docker compose up -d)
asof db-init

# 4. Check every model responds
asof smoke

# 5. Run
uvicorn asof.api.main:app --reload   # http://localhost:8000/docs
pytest
```

## Deploying to Render
1. Create a free **Neon** project in **AWS Asia Pacific (Singapore)** (free plan: 1 GB storage,
   scales to zero when idle). Copy the **pooled** connection string into `.env` as `DATABASE_URL`
   and run `asof db-init` from your machine; that also enables the `vector` extension.
2. In Render: **New → Blueprint** and point it at this repo (`render.yaml`, region Singapore to
   match the database). Set `NVIDIA_API_KEY`, `OPENROUTER_API_KEY` and `DATABASE_URL` when prompted.
3. Ingest the corpus locally with `DATABASE_URL` pointing at the hosted DB.

Notes: the NVIDIA free tier is rate limited (about 40 requests/min, varies by model) and meant for
development and demos; OpenRouter free models allow 50 requests/day (1,000/day after a one-time
$10 credit purchase); and Render's free instances sleep when idle (the first request takes about
a minute). Both are fine for a portfolio demo.
