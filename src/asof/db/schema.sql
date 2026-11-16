-- AsOf schema. Idempotent: safe to run repeatedly.
-- __EMBED_DIM__ is replaced at init time with Settings.embed_dim.
-- Validity is a half-open DATERANGE [valid_from, valid_to); an unbounded upper end = still in force.

CREATE EXTENSION IF NOT EXISTS vector;

-- Records which embedding model/dim the stored vectors came from, so we never mix them.
CREATE TABLE IF NOT EXISTS index_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,
  kind        TEXT NOT NULL CHECK (kind IN ('act','rules','notification','circular','order','other')),
  number      TEXT,                       -- e.g. '49/2019-Central Tax'
  issue_date  DATE,
  title       TEXT NOT NULL,
  url         TEXT UNIQUE,
  pdf_path    TEXT,
  lang        TEXT NOT NULL DEFAULT 'en',
  status      TEXT NOT NULL DEFAULT 'unknown'
              CHECK (status IN ('in_force','rescinded','superseded','unknown')),
  validity    DATERANGE NOT NULL DEFAULT 'empty',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_validity_idx ON documents USING gist (validity);
CREATE INDEX IF NOT EXISTS documents_kind_number_idx ON documents (kind, number);

-- A citable unit of law, e.g. ('cgst_rules', 'rule 36(4)').
CREATE TABLE IF NOT EXISTS provisions (
  id          BIGSERIAL PRIMARY KEY,
  instrument  TEXT NOT NULL CHECK (instrument IN ('cgst_act','cgst_rules','igst_act','other')),
  ref         TEXT NOT NULL,
  UNIQUE (instrument, ref)
);

CREATE TABLE IF NOT EXISTS provision_versions (
  id             BIGSERIAL PRIMARY KEY,
  provision_id   BIGINT NOT NULL REFERENCES provisions(id) ON DELETE CASCADE,
  text           TEXT,                   -- NULL when the provision was omitted
  validity       DATERANGE NOT NULL,
  source_doc_id  BIGINT REFERENCES documents(id),
  verified       BOOLEAN NOT NULL DEFAULT false  -- hand-checked gold history
);
CREATE INDEX IF NOT EXISTS provision_versions_validity_idx
  ON provision_versions USING gist (validity);

CREATE TABLE IF NOT EXISTS chunks (
  id                    BIGSERIAL PRIMARY KEY,
  doc_id                BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  provision_version_id  BIGINT REFERENCES provision_versions(id) ON DELETE SET NULL,
  section_ref           TEXT,
  page                  INT,
  ord                   INT NOT NULL,     -- order within the document
  text                  TEXT NOT NULL,
  embedding             halfvec(__EMBED_DIM__),
  tsv                   tsvector GENERATED ALWAYS AS
                          (to_tsvector('english', coalesce(section_ref, '') || ' ' || text)) STORED,
  validity              DATERANGE NOT NULL DEFAULT 'empty',
  UNIQUE (doc_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_validity_idx ON chunks USING gist (validity);

-- Amendment graph. dst is either a document or a provision.
CREATE TABLE IF NOT EXISTS relations (
  id                BIGSERIAL PRIMARY KEY,
  src_doc_id        BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  dst_doc_id        BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  dst_provision_id  BIGINT REFERENCES provisions(id) ON DELETE CASCADE,
  kind              TEXT NOT NULL CHECK (kind IN
                      ('amends','substitutes','inserts','omits','rescinds','supersedes','clarifies')),
  effective_date    DATE,
  evidence_span     TEXT,
  confidence        REAL CHECK (confidence BETWEEN 0 AND 1),
  verified          BOOLEAN NOT NULL DEFAULT false,
  CHECK (dst_doc_id IS NOT NULL OR dst_provision_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS relations_src_idx ON relations (src_doc_id);
CREATE INDEX IF NOT EXISTS relations_dst_doc_idx ON relations (dst_doc_id);
CREATE INDEX IF NOT EXISTS relations_dst_prov_idx ON relations (dst_provision_id);

-- Supabase exposes the public schema through its auto-generated REST API. RLS with no policies
-- blocks that API; our app connects as the table owner, which RLS does not apply to.
-- Harmless on plain Postgres.
ALTER TABLE index_meta         ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents          ENABLE ROW LEVEL SECURITY;
ALTER TABLE provisions         ENABLE ROW LEVEL SECURITY;
ALTER TABLE provision_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks             ENABLE ROW LEVEL SECURITY;
ALTER TABLE relations          ENABLE ROW LEVEL SECURITY;
