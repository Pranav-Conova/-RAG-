"""Grounded answer generation with verified citations.

The LLM sees numbered sources [S1]..[Sn] and must cite each claim with a short verbatim quote.
Every quote is then checked against its source text (`verify_quote`).

Grounding policy (tested in tests/test_generation_answer.py):

- No hits at all -> no LLM call; abstain and say nothing in force matched.
- The model abstains -> ``abstained=True``; its text explains what is missing.
- The model answers and at least one citation verifies -> normal answer.
- The model answers but NO citation verifies (or it cites nothing) -> the answer text is kept
  (it may still be right, and evaluation wants to see it) but ``UNVERIFIED_NOTE`` is appended,
  ``abstained`` stays False and every citation keeps ``verified=False`` so clients can flag it.
  `is_grounded(answer)` gives callers the same test.
- Citations pointing outside [S1..Sn] are dropped (there is no chunk to attach them to).

Mode: "baseline" (naive RAG control group) gets the same prompt, minus the validity ranges and
amendment notes, which only exist because of AsOf's temporal layer.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from datetime import date
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from asof.generation.structured import structured
from asof.models import Answer, Citation, Hit, Mode

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.85
MIN_FUZZY_CHARS = 20  # shorter quotes must match exactly (after normalisation)
MAX_SOURCE_CHARS = 3000
ANSWER_MAX_TOKENS = 1500

UNVERIFIED_NOTE = (
    "Note: none of the quotes cited for this answer could be verified against the retrieved "
    "sources, so treat it with caution."
)

KIND_LABEL = {
    "act": "Act", "rules": "Rules", "notification": "Notification", "circular": "Circular",
    "order": "Order", "other": "Document",
}


# ---------------------------------------------------------------- structured output

class GenCitation(BaseModel):
    source: int = Field(description="Source number n from [Sn]")
    quote: str = Field(description="Short verbatim quote (under 30 words) copied from that source")


class GenAnswer(BaseModel):
    answer: str = Field(description="The answer, with [S#] markers after each claim")
    citations: list[GenCitation] = Field(default_factory=list)
    abstain: bool = Field(default=False, description="True if the sources do not answer it")


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You answer questions about Indian GST law using ONLY the numbered sources \
provided. You are careful, precise and never rely on outside knowledge.

Rules:
1. State the as-of date the answer applies to in the first sentence.
2. Use only facts found in the sources. Cite every claim with [S#] right after it.
3. For each citation, give a short verbatim quote (under 30 words) copied exactly from that \
source's text, character for character. Do not paraphrase quotes.
4. If the sources do not answer the question, set "abstain" to true and say briefly what \
information is missing. Do not guess.
5. If a source you rely on was later amended or superseded (its validity ends before today), \
say so.
6. Be concise: a direct answer first, then any conditions or exceptions."""


def _fmt_date(d: date | None) -> str:
    return d.isoformat() if d else "unknown"


def format_source(n: int, hit: Hit, *, today: date, mode: Mode) -> str:
    label = KIND_LABEL.get(hit.kind, "Document")
    head = [f"[S{n}] {label} {hit.number or ''}".rstrip(), f"issued {_fmt_date(hit.issue_date)}"]
    if mode == "asof":
        start = _fmt_date(hit.valid_from)
        if hit.valid_to is None:
            head.append(f"valid {start} onwards (in force)")
        else:
            # Half-open [from, to): the last day in force is the day before `to`.
            head.append(f"valid {start} until {hit.valid_to.isoformat()} (exclusive)")
            if hit.valid_to <= today:
                head.append("LATER AMENDED/SUPERSEDED")
    if hit.section_ref:
        head.append(hit.section_ref)
    if hit.page:
        head.append(f"page {hit.page}")
    text = hit.text.strip()
    if len(text) > MAX_SOURCE_CHARS:
        text = text[:MAX_SOURCE_CHARS] + " ..."
    return f"{' | '.join(head)}\nTitle: {hit.title}\n{text}"


def build_messages(
    question: str, as_of: date, hits: Sequence[Hit], *, mode: Mode, today: date
) -> list[dict]:
    sources = "\n\n".join(
        format_source(i, h, today=today, mode=mode) for i, h in enumerate(hits, start=1)
    )
    user = (
        f"Today's date: {today.isoformat()}\n"
        f"Answer as of: {as_of.isoformat()}\n\n"
        f"Sources:\n\n{sources}\n\n"
        f"Question: {question.strip()}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- quote verification

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ",
})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTE_MAP).casefold()
    text = re.sub(r"-\s*\n\s*", "-", text)  # hyphen at a line break
    return re.sub(r"\s+", " ", text).strip()


def _strip_wrapping(q: str) -> str:
    return q.strip().strip("\"'").strip()


def _fuzzy_ratio(needle: str, hay: str) -> float:
    """Best SequenceMatcher ratio of `needle` against same-sized windows of `hay`."""
    n = len(needle)
    if n == 0 or not hay:
        return 0.0
    if len(hay) <= n:
        return SequenceMatcher(None, needle, hay, autojunk=False).ratio()
    step = max(1, n // 8)
    best, best_pos = 0.0, 0
    for size in {n, int(n * 0.9), int(n * 1.1)}:
        if size <= 0:
            continue
        for pos in range(0, max(1, len(hay) - size + 1), step):
            sm = SequenceMatcher(None, needle, hay[pos : pos + size], autojunk=False)
            if sm.real_quick_ratio() < best or sm.quick_ratio() < best:
                continue
            r = sm.ratio()
            if r > best:
                best, best_pos = r, pos
    # Coarse windows can sit up to `step` chars off the best alignment: refine around it.
    for size in {n, int(n * 0.9), int(n * 1.1)}:
        for pos in range(max(0, best_pos - step), min(len(hay) - size, best_pos + step) + 1):
            r = SequenceMatcher(None, needle, hay[pos : pos + size], autojunk=False).ratio()
            best = max(best, r)
    return best


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def verify_quote(quote: str, source_text: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True if `quote` appears in `source_text`, exactly or fuzzily, after normalisation.

    Normalisation: NFKC, casefold, curly quotes/dashes unified, whitespace collapsed.
    Fuzzy: best SequenceMatcher ratio over sliding windows >= `threshold`, only for quotes of
    at least MIN_FUZZY_CHARS, and only if every number in the quote occurs in the source.
    Quotes with an ellipsis are split and every part must verify.
    """
    # NFKC turns the ellipsis character into "...", and curly quotes become ASCII.
    parts = [_strip_wrapping(p) for p in re.split(r"\.{3,}", normalize(quote))]
    parts = [p for p in parts if p]
    if not parts:
        return False
    src = normalize(source_text)
    src_numbers = set(_NUMBER.findall(src))
    for part in parts:
        if part in src:
            continue
        if len(part) < MIN_FUZZY_CHARS:
            return False
        # Fuzzy matching tolerates OCR noise and small wording slips, never a changed number:
        # "10 per cent" must not verify against "20 per cent".
        if not set(_NUMBER.findall(part)) <= src_numbers:
            return False
        if _fuzzy_ratio(part, src) < threshold:
            return False
    return True


def verify_quotes(citations: Sequence[GenCitation], hits: Sequence[Hit]) -> list[Citation]:
    out: list[Citation] = []
    for c in citations:
        if not 1 <= c.source <= len(hits):
            log.info("dropping citation to unknown source S%s", c.source)
            continue
        hit = hits[c.source - 1]
        out.append(Citation(source=c.source, chunk_id=hit.chunk_id, quote=c.quote,
                            verified=verify_quote(c.quote, hit.text)))
    return out


def is_grounded(answer: Answer) -> bool:
    """An answer counts as grounded if it abstained or has at least one verified citation."""
    return answer.abstained or any(c.verified for c in answer.citations)


# ---------------------------------------------------------------- entry point

def no_sources_answer(as_of: date) -> Answer:
    return Answer(
        text=f"As of {as_of.isoformat()}, I found no sources in force that answer this "
        "question, so I cannot answer it from the corpus.",
        abstained=True,
        as_of=as_of,
    )


def generate_answer(
    llm,
    question: str,
    as_of: date,
    hits: Sequence[Hit],
    *,
    mode: Mode,
    today: date | None = None,
    thinking: bool = False,
) -> Answer:
    """Answer from `hits` only. `thinking=True` lets the reasoning model think first: richer
    answers but ~40 s instead of ~1-2 s per call (measured on nemotron-3.5-super), so it is off
    by default for the interactive API."""
    today = today or date.today()
    if not hits:
        return no_sources_answer(as_of)
    out = structured(
        llm,
        build_messages(question, as_of, hits, mode=mode, today=today),
        GenAnswer,
        model=llm.settings.llm_model,
        # thinking tokens count against max_tokens: give a reasoning run the full budget
        max_tokens=llm.settings.llm_max_tokens if thinking else ANSWER_MAX_TOKENS,
        thinking=thinking,
    )
    citations = verify_quotes(out.citations, hits)
    text = out.answer.strip()
    if not out.abstain and not any(c.verified for c in citations):
        text = f"{text}\n\n{UNVERIFIED_NOTE}" if text else UNVERIFIED_NOTE
    return Answer(text=text, citations=citations, abstained=out.abstain, as_of=as_of)
