"""Query understanding: question (+ optional as-of date) -> QueryPlan.

Regexes run first and are authoritative for dates and legal references. The fast LLM adds the
question type and 1-3 rewritten search queries. If the LLM call fails for any reason we return a
regex-only plan: query understanding must never fail a request.

Date rules (documented, and tested in tests/test_retrieval_understand.py):

- An explicit ``as_of_override`` (API/CLI parameter) always wins.
- Full dates: "15 Feb 2020", "15th February, 2020", "February 15, 2020", "2020-02-15",
  "15.02.2020" / "15/02/2020" / "15-02-2020" (numeric dates are read day-first, Indian style).
- A month without a day ("February 2020", "Feb, 2020") means the LAST day of that month.
- A bare year needs a cue word ("in 2020", "during 2020", "as of 2020", "year 2020") and means
  31 December of that year. Bare years without a cue are ignored, so "49/2019-Central Tax" or
  "CGST Rules, 2017" never set a date.
- A financial year ("FY 2019-20", "financial year 2019-2020") means 31 March of its last year.
  Rationale for "end of period": the user asks about the law during that period, and the end
  of the period sees every instrument issued in it.
- Dates immediately preceded by "dated" / "dt." describe a document, not the as-of point, and are
  skipped. Text inside recognised references (e.g. "123/42/2019-GST") is never read as a date.
- With several dates: the first one preceded by an as-of cue ("as on", "as of", "as at", "on",
  "w.e.f.", "with effect from") wins; otherwise the latest date mentioned.
- Dates after ``today`` are clamped to ``today`` (we cannot know future law).
- If the LLM proposes a date and the regexes found none, it is used only when it lies between
  2017-07-01 (GST start) and ``today``.
- question_type: the LLM's label, except "point_in_time" without any date becomes "current",
  and "current" with an explicit past date becomes "point_in_time". Without the LLM a keyword
  heuristic decides (history words > explicit date > "current"-type words > "other").
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from asof.generation.structured import structured
from asof.models import QueryPlan

log = logging.getLogger(__name__)

GST_START = date(2017, 7, 1)

# ---------------------------------------------------------------- references

_TAX_TYPES = {
    "central": "Central Tax",
    "integrated": "Integrated Tax",
    "union territory": "Union Territory Tax",
    "compensation cess": "Compensation Cess",
    "cess": "Compensation Cess",
}

_PAREN = r"(?:\s*\(\s*[0-9a-zA-Z]{1,5}\s*\))"

_CIRCULAR_RE = re.compile(
    r"\b(?:circular\s*(?:no\.?)?\s*)?(\d{1,4})\s*/\s*(\d{1,3})\s*/\s*((?:19|20)\d{2})"
    r"(?:\s*[-–—]\s*gst\b)",
    re.I,
)
_CIRCULAR_PREFIXED_RE = re.compile(
    r"\bcircular\s*(?:no\.?)?\s*(\d{1,4})\s*/\s*(\d{1,3})\s*/\s*((?:19|20)\d{2})\b", re.I
)
_NOTIFICATION_RE = re.compile(
    r"\b(?:notification\s*(?:no\.?)?\s*)?(\d{1,3})\s*/\s*((?:19|20)\d{2})\s*[-–—]?\s*"
    r"(central|integrated|union\s+territory|compensation\s+cess|cess)(?:\s*tax)?"
    r"(\s*\(\s*rate\s*\))?",
    re.I,
)
_NOTIFICATION_PREFIXED_RE = re.compile(
    r"\bnotification\s*(?:no\.?)?\s*(\d{1,3})\s*/\s*((?:19|20)\d{2})\b", re.I
)
_SUB_RE = re.compile(
    r"\bsub[-\s]?(rule|section)\s*(\(\s*[0-9a-zA-Z]{1,5}\s*\))\s*of\s+(?:the\s+)?"
    r"(?:rule|section)\s+(\d{1,3}[A-Za-z]{0,2})\b(" + _PAREN + r"*)",
    re.I,
)
_RULE_RE = re.compile(r"\brule\s+(\d{1,3}[A-Za-z]{0,2})\b(" + _PAREN + r"*)", re.I)
_SECTION_RE = re.compile(
    r"(?:\bsection|\bsec\.|\bu/s\.?)\s*(\d{1,3}[A-Za-z]{0,2})\b(" + _PAREN + r"*)", re.I
)


def _parens(raw: str) -> str:
    parts = re.findall(r"\(\s*([0-9a-zA-Z]{1,5})\s*\)", raw or "")
    return "".join(f"({p.lower()})" for p in parts)


def _num(raw: str) -> str:
    """'86a' -> '86A' (rule/section numbers carry upper-case letter suffixes)."""
    m = re.match(r"(\d+)([A-Za-z]*)", raw)
    assert m
    return f"{int(m.group(1))}{m.group(2).upper()}"


def extract_refs(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (canonical references, spans they occupy in `text`).

    Canonical forms: "rule 36(4)", "section 16(4)", "49/2019-Central Tax",
    "2/2019-Central Tax (Rate)", "123/42/2019-GST", and "49/2019" when only
    "notification no. 49/2019" is written.
    """
    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        return all(m.end() <= s or m.start() >= e for s, e in spans)

    def add(m: re.Match, ref: str) -> None:
        spans.append(m.span())
        found.append((m.start(), ref))

    for m in _CIRCULAR_RE.finditer(text):
        add(m, f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}-GST")
    for m in _CIRCULAR_PREFIXED_RE.finditer(text):
        if free(m):
            add(m, f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}-GST")
    for m in _NOTIFICATION_RE.finditer(text):
        if free(m):
            kind = _TAX_TYPES[re.sub(r"\s+", " ", m.group(3).lower())]
            rate = " (Rate)" if m.group(4) else ""
            add(m, f"{int(m.group(1))}/{m.group(2)}-{kind}{rate}")
    for m in _NOTIFICATION_PREFIXED_RE.finditer(text):
        if free(m):
            add(m, f"{int(m.group(1))}/{m.group(2)}")
    for m in _SUB_RE.finditer(text):
        if free(m):
            kind = m.group(1).lower()
            add(m, f"{kind} {_num(m.group(3))}{_parens(m.group(2))}{_parens(m.group(4))}")
    for m in _RULE_RE.finditer(text):
        if free(m):
            add(m, f"rule {_num(m.group(1))}{_parens(m.group(2))}")
    for m in _SECTION_RE.finditer(text):
        if free(m):
            add(m, f"section {_num(m.group(1))}{_parens(m.group(2))}")

    refs = list(dict.fromkeys(ref for _, ref in sorted(found)))
    return refs, spans


# ---------------------------------------------------------------- dates

_MONTH = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_YEAR = r"((?:19|20)\d{2})"
_ORD = r"(?:st|nd|rd|th)?"

_DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("iso", re.compile(rf"\b{_YEAR}-(\d{{1,2}})-(\d{{1,2}})\b")),
    ("dmy_num", re.compile(rf"\b(\d{{1,2}})[./-](\d{{1,2}})[./-]{_YEAR}\b")),
    ("d_month_y", re.compile(rf"\b(\d{{1,2}}){_ORD}\s*(?:day\s+of\s+|of\s+)?{_MONTH},?\s*{_YEAR}\b",
                             re.I)),
    ("month_d_y", re.compile(rf"\b{_MONTH}\s*(\d{{1,2}}){_ORD},?\s*{_YEAR}\b", re.I)),
    ("fy", re.compile(r"\b(?:fy|f\.y\.|financial\s+year)\s*((?:19|20)\d{2})\s*[-–/]\s*(\d{2,4})\b",
                      re.I)),
    ("month_y", re.compile(rf"\b{_MONTH},?\s*{_YEAR}\b", re.I)),
    ("year", re.compile(
        r"\b(?:in|during|throughout|as\s+of|as\s+on|as\s+at|(?:calendar\s+)?year|end\s+of)"
        rf"\s+(?:the\s+year\s+)?{_YEAR}\b(?!\s*[-–/]\s*\d)", re.I)),
]

_ASOF_CUE = re.compile(
    r"(?:\bas\s+(?:on|of|at)|\bon|\bw\.?\s*e\.?\s*f\.?|\bwith\s+effect\s+from)\s*(?:the\s+)?$",
    re.I,
)
_DATED_CUE = re.compile(r"(?:\bdated|\bdt\.?)\s*$", re.I)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _parse(kind: str, g: tuple[str, ...]) -> date | None:
    if kind == "iso":
        return _safe_date(int(g[0]), int(g[1]), int(g[2]))
    if kind == "dmy_num":
        return _safe_date(int(g[2]), int(g[1]), int(g[0]))
    if kind == "d_month_y":
        return _safe_date(int(g[2]), _MONTHS[g[1][:3].lower()], int(g[0]))
    if kind == "month_d_y":
        return _safe_date(int(g[2]), _MONTHS[g[0][:3].lower()], int(g[1]))
    if kind == "fy":
        start = int(g[0])
        end = int(g[1])
        end = end if end >= 100 else (start // 100) * 100 + end
        return date(end, 3, 31) if end == start + 1 else None
    if kind == "month_y":
        return _month_end(int(g[1]), _MONTHS[g[0][:3].lower()])
    if kind == "year":
        return date(int(g[0]), 12, 31)
    return None


def extract_dates(text: str, skip_spans: list[tuple[int, int]] = ()) -> list[tuple[date, bool]]:
    """All as-of candidate dates in text order, as (date, preceded_by_as_of_cue)."""
    taken: list[tuple[int, int]] = list(skip_spans)
    found: list[tuple[int, date, bool]] = []
    for kind, pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append(m.span())
            d = _parse(kind, m.groups())
            prefix = text[max(0, m.start() - 30) : m.start()]
            if d is None or _DATED_CUE.search(prefix):
                continue
            found.append((m.start(), d, bool(_ASOF_CUE.search(prefix))))
    return [(d, cue) for _, d, cue in sorted(found, key=lambda t: t[0])]


def pick_date(candidates: list[tuple[date, bool]]) -> date | None:
    if not candidates:
        return None
    cued = [d for d, cue in candidates if cue]
    return cued[0] if cued else max(d for d, _ in candidates)


# ---------------------------------------------------------------- question type (regex fallback)

_HISTORY = re.compile(
    r"\b(history|histor\w+|amend\w*|changed?|changes|evolv\w*|evolution|over\s+time|timeline|"
    r"before\s+and\s+after|since|substitut\w+|previous(?:ly)?|earlier|originally)\b",
    re.I,
)
_CURRENT = re.compile(r"\b(current(?:ly)?|now|today|latest|presently|at\s+present|still)\b", re.I)


def guess_question_type(question: str, has_date: bool) -> str:
    if _HISTORY.search(question):
        return "history"
    if has_date:
        return "point_in_time"
    if _CURRENT.search(question):
        return "current"
    return "other"


# ---------------------------------------------------------------- LLM step

QuestionType = Literal["point_in_time", "current", "history", "other"]


class LLMPlan(BaseModel):
    question_type: QuestionType = Field(
        description="point_in_time: asks about the law at a specific past date/period; "
        "current: asks about the law now; history: asks how something changed over time; "
        "other: anything else"
    )
    search_queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="1-3 standalone search queries for a corpus of CBIC GST notifications, "
        "circulars, the CGST Act and CGST Rules. Expand abbreviations (ITC -> input tax "
        "credit), use statutory wording, keep rule/section/notification numbers, omit dates.",
    )
    as_of_date: date | None = Field(
        default=None,
        description="Only if the question states or clearly implies a specific date or "
        "period; otherwise null. Never use today's date as a default.",
    )


_PROMPT = """You plan searches for a question-answering system over Indian GST law (CGST Act, \
CGST Rules, CBIC notifications and circulars).

Today's date: {today}
Question: {question}
Dates already detected: {dates}
References already detected: {refs}

Classify the question and write 1-3 search queries."""


def regex_plan(question: str, today: date, as_of_override: date | None = None) -> QueryPlan:
    """Plan from regexes only (also the fallback when the LLM is unavailable)."""
    refs, spans = extract_refs(question)
    detected = pick_date(extract_dates(question, spans))
    as_of = as_of_override or detected
    explicit = as_of is not None
    as_of = min(as_of, today) if as_of else today
    return QueryPlan(
        as_of=as_of,
        as_of_explicit=explicit,
        question_type=guess_question_type(question, explicit),  # type: ignore[arg-type]
        refs=refs,
        search_queries=[question.strip()],
    )


def understand(llm, question: str, today: date, as_of_override: date | None = None) -> QueryPlan:
    """Regex plan, refined by the fast LLM (question type, search queries) when it answers."""
    plan = regex_plan(question, today, as_of_override)
    if llm is None:
        return plan
    try:
        out = structured(
            llm,
            [{"role": "user", "content": _PROMPT.format(
                today=today.isoformat(),
                question=question.strip(),
                dates=plan.as_of.isoformat() if plan.as_of_explicit else "none",
                refs=", ".join(plan.refs) or "none",
            )}],
            LLMPlan,
            model=llm.settings.llm_fast_model,
            max_tokens=1024,
            thinking=False,  # ~1 s instead of ~15 s; classification needs no reasoning
        )
    except Exception as err:  # noqa: BLE001 - never fail the request on planning
        log.warning("query understanding LLM failed (%s: %s); using regex plan",
                    type(err).__name__, err)
        return plan

    updates: dict = {"question_type": out.question_type}
    queries = [q.strip() for q in out.search_queries if q and q.strip()][:3]
    if queries:
        updates["search_queries"] = queries
    if not plan.as_of_explicit and out.as_of_date and GST_START <= out.as_of_date <= today:
        updates["as_of"] = out.as_of_date
        updates["as_of_explicit"] = True
    final_explicit = updates.get("as_of_explicit", plan.as_of_explicit)
    if out.question_type == "point_in_time" and not final_explicit:
        # No date anywhere: the plan answers as of today, so call it what it is.
        updates["question_type"] = "current"
    if out.question_type == "current" and plan.as_of_explicit and plan.as_of < today:
        # The LLM may say "current" for "what was X on 1 Jan 2020": the explicit date wins.
        updates["question_type"] = "point_in_time"
    return plan.model_copy(update=updates)
