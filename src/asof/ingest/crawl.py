"""Crawl the CBIC GST listing pages into DocMeta records, and download the English PDFs.

Sources (static HTML tables):
- Central Tax notifications: https://cbic-gst.gov.in/hindi/central-tax-notifications.html
  rows: "<number>/<year> - Central Tax dated DD.MM.YYYY" | English PDF | Hindi PDF | subject
- CGST circulars: https://cbic-gst.gov.in/hindi/circulars.html (tab ``div#circulars``)
  rows: "<n>/<m>/<year>-GST" | English PDF | Hindi PDF | DD-MM-YYYY | subject
"""

from __future__ import annotations

import base64
import json
import logging
import re
import ssl
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import httpx
from selectolax.lexbor import LexborHTMLParser

from asof.models import DocMeta

log = logging.getLogger(__name__)

NOTIFICATIONS_URL = "https://cbic-gst.gov.in/hindi/central-tax-notifications.html"
CIRCULARS_URL = "https://cbic-gst.gov.in/hindi/circulars.html"
LISTING_URLS = {"notification": NOTIFICATIONS_URL, "circular": CIRCULARS_URL}

USER_AGENT = "AsOf research crawler (+https://github.com/Pranav-Conova/-RAG-)"

_DATE_RE = re.compile(r"(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})")
_NOTIF_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d{4})\s*[-–]?\s*Central\s+Tax\b(.*)$", re.I | re.S)
_CIRC_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d{4})\s*[-–]\s*GST\b", re.I)
_CORR_RE = re.compile(r"corr|corrigend", re.I)
_WS_RE = re.compile(r"\s+")


@dataclass
class ListingResult:
    docs: list[DocMeta] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


# ------------------------------------------------------------------ normalisation helpers

def clean_text(s: str) -> str:
    return _WS_RE.sub(" ", s.replace("\xa0", " ")).strip()


def parse_date(s: str) -> date | None:
    """Parse the first DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY in `s`."""
    m = _DATE_RE.search(s)
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def normalise_notification(cell: str) -> tuple[str, date | None] | None:
    """'27/2022 - Central Tax dated 26.12.2022' -> ('27/2022-Central Tax', date(2022,12,26))."""
    m = _NOTIF_RE.match(clean_text(cell))
    if not m:
        return None
    num, year, rest = m.groups()
    return f"{int(num)}/{year}-Central Tax", parse_date(rest)


def normalise_circular(cell: str) -> str | None:
    """'92/11/2019-GST (Central GST)' -> '92/11/2019-GST'."""
    m = _CIRC_RE.match(clean_text(cell))
    if not m:
        return None
    a, b, year = m.groups()
    return f"{int(a)}/{int(b):02d}/{year}-GST"


def _english_pdf(cell, base_url: str) -> str | None:
    """First non-corrigendum PDF link in the English column, made absolute."""
    hrefs = [a.attributes.get("href") or "" for a in cell.css("a")]
    hrefs = [h.strip() for h in hrefs if h.strip().lower().endswith(".pdf")]
    main = [h for h in hrefs if not _CORR_RE.search(h.rsplit("/", 1)[-1])]
    pick = (main or hrefs or [None])[0]
    return urljoin(base_url, pick) if pick else None


def _in_years(d: date, years: tuple[int, int] | None) -> bool:
    return years is None or years[0] <= d.year <= years[1]


# ------------------------------------------------------------------ listing parsers

def parse_notifications(
    html: str, base_url: str = NOTIFICATIONS_URL, years: tuple[int, int] | None = None
) -> ListingResult:
    res = ListingResult()
    seen: set[str] = set()
    tree = LexborHTMLParser(html)
    for tr in tree.css("tr"):
        tds = tr.css("td")
        if len(tds) < 3:
            continue  # header / year separator rows
        res.counts["rows"] += 1
        norm = normalise_notification(tds[0].text())
        if norm is None:
            res.counts["skip_unparsed_number"] += 1
            log.debug("unparsed notification row: %r", clean_text(tds[0].text())[:100])
            continue
        number, issue = norm
        if issue is None:
            res.counts["skip_no_date"] += 1
            continue
        if not _in_years(issue, years):
            res.counts["skip_out_of_years"] += 1
            continue
        url = _english_pdf(tds[1], base_url)
        if url is None:
            res.counts["skip_no_pdf"] += 1
            continue
        title = clean_text(tds[-1].text())
        meta = DocMeta(kind="notification", number=number, issue_date=issue, title=title, url=url)
        if meta.key in seen:
            res.counts["skip_duplicate"] += 1
            continue
        seen.add(meta.key)
        res.docs.append(meta)
        res.counts["ok"] += 1
    return res


def parse_circulars(
    html: str, base_url: str = CIRCULARS_URL, years: tuple[int, int] | None = None
) -> ListingResult:
    res = ListingResult()
    seen: set[str] = set()
    tree = LexborHTMLParser(html)
    root = tree.css_first("div#circulars") or tree.root
    for tr in root.css("tr"):
        tds = tr.css("td")
        if len(tds) < 3:
            continue
        res.counts["rows"] += 1
        number = normalise_circular(tds[0].text())
        if number is None:
            res.counts["skip_unparsed_number"] += 1
            log.debug("unparsed circular row: %r", clean_text(tds[0].text())[:100])
            continue
        # the date is its own cell, but rows have stray/nested <td>s: take the first cell that
        # is just a date
        issue = None
        for td in tds[1:]:
            txt = clean_text(td.text())
            if _DATE_RE.fullmatch(txt):
                issue = parse_date(txt)
                break
        if issue is None:
            res.counts["skip_no_date"] += 1
            continue
        if not _in_years(issue, years):
            res.counts["skip_out_of_years"] += 1
            continue
        url = _english_pdf(tds[1], base_url)
        if url is None:
            res.counts["skip_no_pdf"] += 1
            continue
        title = clean_text(tds[-1].text())
        meta = DocMeta(kind="circular", number=number, issue_date=issue, title=title, url=url)
        if meta.key in seen:
            res.counts["skip_duplicate"] += 1
            continue
        seen.add(meta.key)
        res.docs.append(meta)
        res.counts["ok"] += 1
    return res


PARSERS = {"notification": parse_notifications, "circular": parse_circulars}


# ------------------------------------------------------------------ polite HTTP

def _ssl_context() -> ssl.SSLContext | bool:
    """Verify TLS against the OS trust store when `truststore` is installed.

    taxinformation.cbic.gov.in serves an incomplete certificate chain; the OS verifier (which
    fetches missing intermediates) accepts it, certifi/OpenSSL alone does not.
    """
    try:
        import truststore
    except ImportError:
        return True
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class PoliteClient:
    """httpx client with a minimum interval between requests and retries with backoff."""

    def __init__(self, min_interval_s: float = 1.0, retries: int = 4, timeout_s: float = 60.0):
        self.min_interval_s = min_interval_s
        self.retries = retries
        self._last = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
            follow_redirects=True,
            verify=_ssl_context(),
        )

    def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.client.get(url, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise  # 404 etc: retrying won't help
                backoff = 2.0 * (2**attempt)
                log.warning("GET %s failed (%s); retry in %.0fs", url, exc, backoff)
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self.client.close()


# ------------------------------------------------------------------ pipeline steps

def crawl(
    client: PoliteClient, kinds: Iterable[str], years: tuple[int, int] | None
) -> ListingResult:
    out = ListingResult()
    for kind in kinds:
        url = LISTING_URLS[kind]
        log.info("fetching %s listing: %s", kind, url)
        html = client.get(url).content.decode("utf-8", errors="replace")
        res = PARSERS[kind](html, url, years)
        log.info("%s listing: %s", kind, dict(res.counts))
        out.docs.extend(res.docs)
        out.counts.update({f"{kind}.{k}": v for k, v in res.counts.items()})
    # documents.url is UNIQUE: drop a later row that reuses an already-listed PDF
    seen_urls: set[str] = set()
    deduped = []
    for d in out.docs:
        if d.url in seen_urls:
            out.counts["skip_duplicate_url"] += 1
            log.warning("duplicate PDF url %s (%s) skipped", d.url, d.number)
            continue
        seen_urls.add(d.url)
        deduped.append(d)
    out.docs = deduped
    return out


def write_manifest(docs: list[DocMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")


def read_manifest(path: Path) -> list[DocMeta]:
    with path.open(encoding="utf-8") as f:
        return [DocMeta.model_validate(json.loads(line)) for line in f if line.strip()]


class TaxPortal:
    """Fallback source: CBIC's tax information portal (taxinformation.cbic.gov.in).

    Many 2019/2020 PDFs linked from cbic-gst.gov.in now return 404; the portal still serves
    them through its public JSON API (anonymous token, PDF returned base64-encoded).
    """

    BASE = "https://taxinformation.cbic.gov.in/api"
    GST_TAX_ID = 1000001

    def __init__(self, client: PoliteClient) -> None:
        self.client = client
        self._token: str | None = None
        self._index: dict[int, dict[str, int]] = {}

    def _headers(self) -> dict[str, str]:
        if self._token is None:
            resp = self.client.client.post(f"{self.BASE}/authenticate-token", content=b"")
            resp.raise_for_status()
            self._token = resp.json()["id_token"]
        return {"Authorization1": f"homeToken {self._token}", "language": "en"}

    def _get(self, url: str) -> httpx.Response:
        return self.client.get(url, headers=self._headers())

    def notification_ids(self, year: int) -> dict[str, int]:
        """Canonical number ('49/2019-Central Tax') -> portal id, for one year."""
        if year in self._index:
            return self._index[year]
        index: dict[str, int] = {}
        page, size = 0, 50
        while True:
            url = (
                f"{self.BASE}/cbic-notification-msts/fetchNotificationByYearAndCategory"
                f"?year={year}&page={page}&size={size}&category=Central%20Tax"
                f"&taxId={self.GST_TAX_ID}"
            )
            rows = self._get(url).json()
            for r in rows:
                norm = normalise_notification(r.get("notificationNo") or "")
                if norm and r.get("id") and (r.get("contentLanguage") or "ENGLISH") == "ENGLISH":
                    index.setdefault(norm[0], int(r["id"]))
            if len(rows) < size or page > 20:
                break
            page += 1
        log.info("tax portal: %d Central Tax notifications listed for %d", len(index), year)
        self._index[year] = index
        return index

    def fetch(self, meta: DocMeta) -> bytes | None:
        if meta.kind != "notification":
            return None
        pid = self.notification_ids(meta.issue_date.year).get(meta.number)
        if pid is None:
            return None
        data = self._get(f"{self.BASE}/cbic-notification-msts/download/{pid}/ENG").json()
        return base64.b64decode(data["data"]) if data.get("data") else None


def download(
    client: PoliteClient,
    docs: list[DocMeta],
    pdf_dir: Path,
    rel_to: Path | None = None,
    fallback: TaxPortal | None = None,
) -> Counter[str]:
    """Download each English PDF to <pdf_dir>/<key>.pdf (skips existing) and set pdf_path."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for i, d in enumerate(docs, 1):
        target = pdf_dir / f"{d.key}.pdf"
        rel = target.resolve().relative_to(rel_to) if rel_to else target
        if target.exists() and target.stat().st_size > 0:
            counts["present"] += 1
            d.pdf_path = rel.as_posix()
            continue
        try:
            body = client.get(d.url).content
        except Exception as exc:
            body = None
            if fallback is not None:
                try:
                    body = fallback.fetch(d)
                except Exception as exc2:
                    log.warning("tax portal fallback failed for %s: %s", d.number, exc2)
            if body is None:
                counts["failed"] += 1
                log.error("[%d/%d] download failed %s %s: %s", i, len(docs), d.number, d.url, exc)
                continue
            counts["via_tax_portal"] += 1
        if not body.startswith(b"%PDF"):
            counts["not_pdf"] += 1
            log.error("[%d/%d] not a PDF: %s %s", i, len(docs), d.number, d.url)
            continue
        tmp = target.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(target)
        d.pdf_path = rel.as_posix()
        counts["downloaded"] += 1
        log.info("[%d/%d] downloaded %s (%d KB)", i, len(docs), d.key, len(body) // 1024)
    return counts
