"""Schema tests. Run against a real Postgres + pgvector (docker compose or CI service).

Skipped automatically when DATABASE_URL is unreachable.
"""

import functools
import os
from datetime import date

import psycopg
import pytest
from pgvector import HalfVector

from asof.config import Settings
from asof.db import connect, init_schema
from asof.db.connection import schema_sql

pytestmark = pytest.mark.db

DB_URL = os.environ.get("DATABASE_URL", "postgresql://asof:asof@localhost:5432/asof")


@functools.cache
def _reachable() -> bool:
    try:
        psycopg.connect(DB_URL, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture
def db_settings() -> Settings:
    if not _reachable():
        pytest.skip("no Postgres at DATABASE_URL")
    s = Settings(_env_file=None, database_url=DB_URL, embed_dim=4)  # type: ignore[call-arg]
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS relations, chunks, provision_versions, provisions, "
                     "documents, index_meta CASCADE")
    return s


def test_schema_sql_substitutes_dim():
    assert "halfvec(768)" in schema_sql(768)
    assert "__EMBED_DIM__" not in schema_sql(768)


def test_init_is_idempotent_and_pins_embedding(db_settings: Settings):
    init_schema(db_settings)
    init_schema(db_settings)
    db_settings.embed_dim = 8
    with pytest.raises(RuntimeError, match="different embedding settings"):
        init_schema(db_settings)


def test_as_of_filter_returns_only_version_in_force(db_settings: Settings):
    """The core idea of the project, in one query."""
    init_schema(db_settings)
    with connect(db_settings) as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (kind, title) VALUES ('rules', 'CGST Rules') RETURNING id"
        ).fetchone()[0]
        versions = [
            ("cap is 20 per cent", "[2019-10-09,2020-01-01)", [1, 0, 0, 0]),
            ("cap is 10 per cent", "[2020-01-01,2021-01-01)", [1, 0.1, 0, 0]),
            ("cap is 5 per cent", "[2021-01-01,2022-01-01)", [1, 0.2, 0, 0]),
        ]
        for ord_, (text, validity, vec) in enumerate(versions):
            conn.execute(
                "INSERT INTO chunks (doc_id, ord, section_ref, text, validity, embedding) "
                "VALUES (%s, %s, 'rule 36(4)', %s, %s::daterange, %s)",
                (doc_id, ord_, text, validity, HalfVector(vec)),
            )

        def search(as_of: date) -> list[str]:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE validity @> %s::date "
                "ORDER BY embedding <=> %s LIMIT 5",
                (as_of, HalfVector([1, 0, 0, 0])),
            ).fetchall()
            return [r[0] for r in rows]

        assert search(date(2020, 2, 15)) == ["cap is 10 per cent"]
        assert search(date(2021, 1, 1)) == ["cap is 5 per cent"]  # half-open: new version wins
        assert search(date(2022, 6, 1)) == []  # omitted: nothing in force

        hits = conn.execute(
            "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'per cent cap')"
        ).fetchone()[0]
        assert hits == 3
