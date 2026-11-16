"""Section-aware chunking of a ParsedDoc into ChunkIn records.

Text is split into paragraphs (blank lines and list/para markers such as "2.", "(a)", "(iv)"),
long paragraphs into sentences, and these units are packed greedily into ~250-450 word chunks
with a small sentence overlap. Each chunk keeps the page it starts on and a best-effort
`section_ref` such as "rule 36(4)", "section 16(2)" or "para 3".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from asof.models import ChunkIn, ParsedDoc

MIN_WORDS = 250
TARGET_WORDS = 350
MAX_WORDS = 450
OVERLAP_WORDS = 40

_MARKER = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*\.?\s|\(\d{1,3}[A-Z]?\)\s?|\([a-z]{1,2}\)\s?|\([ivxlc]{1,6}\)\s?"
    r"|[A-Z]\.\s|Explanation|Provided|Subject|Table|Annexure|Sl\.? ?No|S\. ?No)",
)
_SENT_SPLIT = re.compile(r"(?<=[.;?!])[\"'”’)]?\s+(?=[\"“(\[A-Z0-9])")
_ABBREV = re.compile(
    r"(?:\b(?:No|Nos|Rs|dt|Dt|Sr|Sl|S|F|Govt|i\.e|e\.g|viz|etc|Ltd|Pvt|Co|Mr|Mrs|Ms|Dr|St|"
    r"vs|w\.e\.f|u/s|cl|Vol|Art|Para|para|Sec|sec|G\.S\.R|S\.O|M/s|A|B|C|D|E|I|II|III|IV|V)"
    r"\.|\b[A-Z]\.)$"
)
_HEADER = re.compile(
    r"\[?\s*TO BE PUBLISHED IN THE GAZETTE.*?\]|PART\s*[-–—,]?\s*II\W{0,5}SECTION\s*3\b"
    r"(?:\W{0,5}SUB-?SECTION\s*\(i+\))?",
    re.I | re.S,
)
_ENDS_UNIT = re.compile(r"[.:;?!\])—–-]\s*$")


@dataclass
class _Unit:
    text: str
    page: int
    para_start: bool

    @property
    def words(self) -> int:
        return len(self.text.split())


# ------------------------------------------------------------------ splitting

def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            paras.append(" ".join(cur).strip())
            cur.clear()

    lines = [ln.strip() for ln in text.splitlines()]
    for idx, line in enumerate(lines):
        if not line:
            # Some PDFs put a blank line after every wrapped line: only treat a blank line as
            # a paragraph break when the text before it ends a sentence/clause or the next
            # line starts a new unit.
            nxt = next((ln for ln in lines[idx + 1 :] if ln), "")
            if not cur or _ENDS_UNIT.search(cur[-1]) or _MARKER.match(nxt) or not nxt:
                flush()
            continue
        if cur and _MARKER.match(line):
            flush()
        if cur and cur[-1].endswith("-") and not cur[-1].endswith(" -"):
            cur[-1] = cur[-1] + line  # re-join a word hyphenated across lines ("sub-\nrule")
        else:
            cur.append(line)
    flush()
    return [p for p in paras if p]


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, but not after common legal abbreviations ("No.", "Rs.")."""
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _hard_split(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _units(parsed: ParsedDoc, max_words: int) -> list[_Unit]:
    units: list[_Unit] = []
    for page in parsed.pages:
        for para in _paragraphs(page.text):
            pieces = split_sentences(para) if len(para.split()) > max_words // 3 else [para]
            first = True
            for piece in pieces:
                for sub in _hard_split(piece, max_words) if len(piece.split()) > max_words else [
                    piece
                ]:
                    units.append(_Unit(sub, page.page, first))
                    first = False
    return units


# ------------------------------------------------------------------ section refs

_SUB = r"((?:\s?\((?:\d{1,3}[A-Z]{0,2}|[a-z]{1,3}|[ivxl]{1,5})\))*)"
_REF_DIRECT = re.compile(rf"\b(rule|section)\s+(\d{{1,3}}[A-Z]{{0,3}})\b{_SUB}", re.I)
_REF_SUBOF = re.compile(
    r"\bsub-(rule|section)\s*\((\w{1,4})\)\s+of\s+(?:the\s+said\s+)?(rule|section)\s+"
    r"(\d{1,3}[A-Z]{0,3})\b",
    re.I,
)
_REF_IN = re.compile(rf"\bin\s+(rule|section)\s+(\d{{1,3}}[A-Z]{{0,3}})\b{_SUB}", re.I)
_SUB_AFTER = re.compile(r"^[,\s\-—–]*(?:in\s+)?sub-(?:rule|section)\s*\((\w{1,4})\)", re.I)
_PARA_START = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})+|\d{1,2}(?=\.))\.?\s+\S")
_PARA_MENTION = re.compile(r"\bpara(?:graph)?\s+(\d{1,2}(?:\.\d{1,2})*)\b", re.I)


def _fmt(kind: str, num: str, subs: str) -> str:
    subs = re.sub(r"\s+", "", subs or "")
    return f"{kind.lower()} {num.upper()}{subs}"


def _ref_from_match(m: re.Match[str], text: str) -> str:
    kind, num, subs = m.group(1), m.group(2), m.group(3)
    if not subs:
        after = _SUB_AFTER.match(text[m.end() : m.end() + 60])
        if after:
            subs = f"({after.group(1)})"
    return _fmt(kind, num, subs)


def _is_shouting(m: re.Match[str]) -> bool:
    return m.group(1).isupper()  # "PART II, SECTION 3" in gazette headers


def detect_section_ref(text: str, kind: str = "notification") -> str | None:
    """Best-effort citable unit a chunk is about, e.g. 'rule 36(4)', 'section 16', 'para 3'."""
    text = _HEADER.sub(" ", text)
    if kind == "circular":
        m = _PARA_START.match(text)
        if m:
            return f"para {m.group(1)}"
    # amendment style ("in rule 36, in sub-rule (4)") wins over enabling-power mentions
    for m in _REF_IN.finditer(text):
        if not _is_shouting(m):
            return _ref_from_match(m, text)
    candidates: list[tuple[int, str]] = []
    for m in _REF_SUBOF.finditer(text):
        candidates.append((m.start(), _fmt(m.group(3), m.group(4), f"({m.group(2)})")))
    for m in _REF_DIRECT.finditer(text):
        if not _is_shouting(m):
            candidates.append((m.start(), _ref_from_match(m, text)))
            break
    if candidates:
        return min(candidates)[1]
    m = _PARA_MENTION.search(text)
    if m:
        return f"para {m.group(1)}"
    return None


# ------------------------------------------------------------------ packing

def chunk_document(
    parsed: ParsedDoc,
    min_words: int = MIN_WORDS,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[ChunkIn]:
    units = _units(parsed, max_words)
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_words = 0
    n_overlap = 0  # leading units of `cur` that are overlap from the previous chunk

    for u in units:
        fresh = cur_words - sum(x.words for x in cur[:n_overlap])
        over = cur_words + u.words > max_words
        at_boundary = u.para_start and cur_words >= target_words
        if cur and fresh > 0 and ((over and cur_words >= min_words) or at_boundary or (
            over and cur_words + u.words > max_words * 1.3
        )):
            groups.append(cur)
            # overlap: carry trailing units totalling <= overlap_words
            carry: list[_Unit] = []
            carried = 0
            for prev in reversed(cur):
                if carried + prev.words > overlap_words:
                    break
                carry.insert(0, prev)
                carried += prev.words
            cur, cur_words, n_overlap = carry, carried, len(carry)
        cur.append(u)
        cur_words += u.words
    if cur and cur_words - sum(x.words for x in cur[:n_overlap]) > 0:
        # a short tail is merged into the previous chunk when that keeps it reasonable
        tail_fresh = cur[n_overlap:]
        tail_words = sum(x.words for x in tail_fresh)
        if groups and tail_words < min_words // 3 and (
            sum(x.words for x in groups[-1]) + tail_words <= max_words * 1.2
        ):
            groups[-1].extend(tail_fresh)
        else:
            groups.append(cur)

    chunks: list[ChunkIn] = []
    for g in groups:
        # sentences of one paragraph are re-joined with spaces, paragraphs with newlines
        text = "".join(
            (("\n" if x.para_start else " ") if j else "") + x.text for j, x in enumerate(g)
        ).strip()
        if not text:
            continue
        chunks.append(
            ChunkIn(
                ord=len(chunks),
                text=text,
                page=g[0].page,
                section_ref=detect_section_ref(text, parsed.meta.kind),
            )
        )
    return chunks
