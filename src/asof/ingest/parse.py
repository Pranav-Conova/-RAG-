"""PDF -> ParsedDoc: per-page English text via PyMuPDF, OCR fallback for scanned pages.

Gazette PDFs are often bilingual (Hindi + English): lines that are mostly Devanagari are
dropped so the English text remains.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from asof.models import DocMeta, Page, ParsedDoc

log = logging.getLogger(__name__)

# A page with fewer extractable letters than this is treated as scanned (image only).
MIN_PAGE_LETTERS = 40
OCR_DPI = 170
OCR_MAX_PAGES = 30  # per document; very long scanned annexures are not worth the time

_DEVANAGARI = re.compile("[\u0900-\u097F\uA8E0-\uA8FF]")
_JUNK = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\uFFFD\uE000-\uF8FF\u200b-\u200f\ufeff]"
)
_PAGE_MARK = re.compile(r"^\s*(page\s+\d+\s+of\s+\d+|-\s*\d{1,3}\s*-)\s*$", re.I)
_SPACES = re.compile("[ \t\u00a0]+")


def devanagari_ratio(line: str) -> float:
    """Share of letters in `line` that are Devanagari (0.0 when the line has no letters)."""
    letters = [c for c in line if c.isalpha() or _DEVANAGARI.match(c)]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _DEVANAGARI.match(c)) / len(letters)


def is_mostly_devanagari(line: str, threshold: float = 0.5) -> bool:
    return devanagari_ratio(line) > threshold


def clean_page_text(text: str) -> str:
    """Drop Hindi lines, page markers and junk glyphs; normalise whitespace; keep line breaks."""
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    blank = False
    for raw in text.splitlines():
        line = _SPACES.sub(" ", _JUNK.sub("", raw)).strip()
        if line and (is_mostly_devanagari(line) or _PAGE_MARK.match(line)):
            continue
        if not line:
            if not blank and out:
                out.append("")
            blank = True
            continue
        blank = False
        out.append(line)
    return "\n".join(out).strip()


def letter_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


@lru_cache(maxsize=1)
def _ocr_engine() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
    except Exception as exc:  # optional dependency
        log.warning("OCR unavailable (%s); scanned pages will be left empty", exc)
        return None
    return RapidOCR()


def ocr_page(page: Any, dpi: int = OCR_DPI) -> str:
    """OCR one PyMuPDF page with RapidOCR; returns '' if OCR is unavailable."""
    engine = _ocr_engine()
    if engine is None:
        return ""
    import numpy as np

    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    result, _ = engine(img)
    if not result:
        return ""
    # result: [[box, text, score], ...]; group boxes into lines by their vertical centre
    items = []
    for box, txt, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), txt))
    items.sort()
    lines: list[list[tuple[float, str]]] = []
    last_y = None
    for y, x, h, txt in items:
        if last_y is None or y - last_y > max(h, 8) * 0.6:
            lines.append([])
            last_y = y
        lines[-1].append((x, txt))
    return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)


def strip_running_headers(pages: list[Page], edge: int = 3) -> list[Page]:
    """Remove short lines repeated at the top/bottom of several pages (running headers)."""
    if len(pages) < 2:
        return pages
    edges: dict[str, int] = {}
    for p in pages:
        lines = [ln for ln in p.text.splitlines() if ln.strip()]
        for ln in set(lines[:edge] + lines[-edge:]):
            if len(ln) <= 80:
                edges[ln] = edges.get(ln, 0) + 1
    repeated = {ln for ln, n in edges.items() if n >= 2}
    if not repeated:
        return pages
    out = [pages[0]]
    for p in pages[1:]:
        lines = p.text.splitlines()
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(nonblank[:edge] + nonblank[-edge:])
        kept = [ln for i, ln in enumerate(lines) if not (i in edge_idx and ln in repeated)]
        out.append(Page(page=p.page, text="\n".join(kept).strip()))
    return out


def parse_pdf(meta: DocMeta, pdf_path: Path, use_ocr: bool = True) -> ParsedDoc:
    import pymupdf

    pages: list[Page] = []
    ocr_used = False
    ocr_done = 0
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            raw = page.get_text("text", sort=True)
            text = clean_page_text(raw)
            if letter_count(text) < MIN_PAGE_LETTERS and use_ocr and ocr_done < OCR_MAX_PAGES:
                # Hindi-only pages also have few English letters but plenty of raw text:
                # only OCR pages that have (almost) no text layer at all.
                if letter_count(raw) < MIN_PAGE_LETTERS:
                    ocr_text = clean_page_text(ocr_page(page))
                    ocr_done += 1
                    if letter_count(ocr_text) > letter_count(text):
                        text = ocr_text
                        ocr_used = True
            pages.append(Page(page=i, text=text))
    return ParsedDoc(meta=meta, pages=strip_running_headers(pages), ocr_used=ocr_used)


def is_empty(parsed: ParsedDoc) -> bool:
    return sum(letter_count(p.text) for p in parsed.pages) < MIN_PAGE_LETTERS


def write_parsed(parsed: ParsedDoc, parsed_dir: Path) -> Path:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    path = parsed_dir / f"{parsed.meta.key}.json"
    path.write_text(parsed.model_dump_json(), encoding="utf-8")
    return path


def read_parsed(path: Path) -> ParsedDoc:
    return ParsedDoc.model_validate_json(path.read_text(encoding="utf-8"))
