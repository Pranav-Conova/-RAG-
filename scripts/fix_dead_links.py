"""Point documents whose CBIC PDF link is dead at the CBIC tax-information portal instead.

CBIC moved most 2019-2020 notification PDFs (the old cbic-gst.gov.in links return 404).
Ingestion recovers the PDFs from taxinformation.cbic.gov.in's API, but that portal has no
per-document deep link, so the UI links to its notification search page with the number.
Re-run after `asof ingest --force` (which restores the original URLs).

Usage: python scripts/fix_dead_links.py [--dry-run]
"""

import argparse
import sys
import time
from urllib.parse import quote

import httpx

from asof.config import get_settings
from asof.db import connect

PORTAL = "https://taxinformation.cbic.gov.in/content-page/explore-notification?notification="
UA = {"User-Agent": "AsOf research crawler (+https://github.com/Pranav-Conova/-RAG-)"}


def is_live(client: httpx.Client, url: str) -> bool:
    try:
        with client.stream("GET", url) as r:  # HEAD is rejected by the CBIC server
            return r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower()
    except httpx.HTTPError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    args = ap.parse_args()

    with connect(get_settings()) as conn:
        rows = conn.execute(
            "SELECT id, number FROM documents WHERE kind = 'notification' AND url NOT LIKE %s",
            (PORTAL + "%",)).fetchall()
        urls = dict(conn.execute("SELECT id, url FROM documents").fetchall())
        dead = []
        with httpx.Client(headers=UA, timeout=20, follow_redirects=True) as client:
            for doc_id, number in rows:
                if not is_live(client, urls[doc_id]):
                    dead.append((PORTAL + quote(number), doc_id))
                time.sleep(args.delay)
        print(f"checked {len(rows)} notification links, dead: {len(dead)}")
        if dead and not args.dry_run:
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany("UPDATE documents SET url = %s WHERE id = %s", dead)
            print(f"repointed {len(dead)} to the CBIC portal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
