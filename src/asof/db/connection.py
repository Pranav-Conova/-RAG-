from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector

from asof.config import Settings


def connect(settings: Settings, *, register: bool = True) -> psycopg.Connection:
    # prepare_threshold=None disables server-side prepared statements, which connection
    # poolers in transaction mode (e.g. Supabase port 6543) don't support. Our query volume
    # is low, so the cost is negligible and the same code works on any Postgres.
    conn = psycopg.connect(settings.database_url, autocommit=True, prepare_threshold=None)
    if register:
        register_vector(conn)  # requires the `vector` extension to exist
    return conn


def schema_sql(embed_dim: int) -> str:
    sql = files("asof.db").joinpath("schema.sql").read_text(encoding="utf-8")
    return sql.replace("__EMBED_DIM__", str(int(embed_dim)))


def init_schema(settings: Settings) -> None:
    """Create extension + tables, and pin the embedding model/dim the index was built with."""
    with connect(settings, register=False) as conn:
        conn.execute(schema_sql(settings.embed_dim))
        expected = {"embed_model": settings.embed_model, "embed_dim": str(settings.embed_dim)}
        for key, value in expected.items():
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                (key, value),
            )
        stored = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
        mismatched = {k: (stored[k], v) for k, v in expected.items() if stored.get(k) != v}
        if mismatched:
            raise RuntimeError(
                f"Database was indexed with different embedding settings {mismatched} "
                "(stored, configured). Re-embed the corpus or change your config back."
            )
