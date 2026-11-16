"""Evaluation metrics. Pure functions: no I/O, no network, no database.

Per question, `score_result` turns an `AskResult` into a flat dict of scores (None = the
metric does not apply to this question). `aggregate` turns a list of those into rates.
"""

import math
import re
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from asof.models import AskResult, Hit

# ---------------------------------------------------------------- text normalisation

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_TENS_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[\s-]+(one|two|three|four|five|six|seven|eight|nine))?\b"
)
_UNITS_RE = re.compile(r"\b(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b")
_HUNDRED_RE = re.compile(r"\b(\d{1,2}) hundred(?: and)?(?: (\d{1,2}))?\b")


def _tens(m: re.Match[str]) -> str:
    return str(_TENS[m.group(1)] + (_UNITS[m.group(2)] if m.group(2) else 0))


def _hundred(m: re.Match[str]) -> str:
    return str(int(m.group(1)) * 100 + (int(m.group(2)) if m.group(2) else 0))


def normalise(text: str) -> str:
    """Canonical form for fact matching.

    casefold; "%" / "percent" -> "per cent"; number words up to 999 -> digits ("ten" -> "10",
    "ninety-nine" -> "99", "one hundred" -> "100"); thousands separators dropped
    ("1,00,000" -> "100000"); currency markers dropped ("Rs. 20 crore" == "20 crore rupees");
    other punctuation (except . / ( ) needed for "rule 36(4)" and "94/2020") becomes a space;
    whitespace collapsed.
    """
    s = text.casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"')
    s = s.replace("”", '"').replace("–", "-").replace("—", "-")
    s = s.replace("%", " per cent ")
    s = re.sub(r"\bper\s*-?\s*cent(?:um)?\b|\bpercent\b", "per cent", s)
    s = s.replace("₹", " rs ")
    s = re.sub(r"\b(?:rs\.?|inr)(?=\s|\d|$)", " rs ", s)
    s = _TENS_RE.sub(_tens, s)
    s = _UNITS_RE.sub(lambda m: str(_UNITS[m.group(1)]), s)
    s = _HUNDRED_RE.sub(_hundred, s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w./()]+", " ", s)
    s = re.sub(r"\b(?:rs|rupees?)\b", " ", s)
    return " ".join(s.split())


def _fact_pattern(fact: str) -> re.Pattern[str] | None:
    f = normalise(fact).strip(" .")
    if not f:
        return None
    # Whole-token match: "5 per cent" must not match inside "2.5 per cent" or "25 per cent".
    return re.compile(r"(?<![\w.])" + re.escape(f) + r"(?![\w]|\.\d)")


def contains_fact(text: str, fact: str, *, normalised: bool = False) -> bool:
    pat = _fact_pattern(fact)
    if pat is None:
        return False
    return bool(pat.search(text if normalised else normalise(text)))


def answer_correct(
    answer_text: str, must_contain: Sequence[str], must_not_contain: Sequence[str] = ()
) -> bool:
    """All must_contain facts present and no must_not_contain fact present (normalised)."""
    norm = normalise(answer_text)
    return all(contains_fact(norm, f, normalised=True) for f in must_contain) and not any(
        contains_fact(norm, f, normalised=True) for f in must_not_contain
    )


# ---------------------------------------------------------------- documents and validity

def canon_doc(number: str | None) -> str:
    """Comparable document number: 'Notification No. 01/2022-Central Tax' -> '1/2022-centraltax'."""
    if not number:
        return ""
    s = number.casefold()
    s = re.sub(r"^\s*(?:notification|circular|order)?\s*(?:no\.?|number)?\s*", "", s)
    s = re.sub(r"^0+(?=\d)", "", s)  # "01/2022-..." == "1/2022-..."
    return re.sub(r"[\s_]+", "", s)


def in_force(hit: Hit, as_of: date) -> bool:
    """True unless the hit's validity window demonstrably excludes `as_of`.

    The window is [valid_from, valid_to). When valid_from is unknown, the issue date is used as
    the lower bound (a document cannot be in force before it was issued). Unknown = in force.
    """
    lower = hit.valid_from or hit.issue_date
    if lower is not None and lower > as_of:
        return False
    return not (hit.valid_to is not None and hit.valid_to <= as_of)


def _cited_hits(result: AskResult) -> list[tuple[Hit, bool]]:
    """(hit, verified) for each citation that resolves to a hit, in citation order."""
    by_chunk = {h.chunk_id: h for h in result.hits}
    out = []
    for c in result.answer.citations:
        hit = None
        if 1 <= c.source <= len(result.hits) and result.hits[c.source - 1].chunk_id == c.chunk_id:
            hit = result.hits[c.source - 1]
        elif c.chunk_id in by_chunk:
            hit = by_chunk[c.chunk_id]
        elif 1 <= c.source <= len(result.hits):
            hit = result.hits[c.source - 1]
        if hit is not None:
            out.append((hit, c.verified))
    return out


def gold_rank(hits: Sequence[Hit], gold_doc_numbers: Iterable[str]) -> int | None:
    """1-based rank of the first hit from a gold document, or None."""
    gold = {canon_doc(g) for g in gold_doc_numbers}
    for i, h in enumerate(hits, start=1):
        if canon_doc(h.number) in gold:
            return i
    return None


def recall_at_k(hits: Sequence[Hit], gold_doc_numbers: Sequence[str], k: int) -> bool | None:
    """Any gold document among hits[:k]. None when there is no gold document."""
    if not gold_doc_numbers:
        return None
    rank = gold_rank(hits, gold_doc_numbers)
    return rank is not None and rank <= k


def citation_precision(result: AskResult, gold_doc_numbers: Sequence[str]) -> float | None:
    """Share of distinct cited documents that are gold. None without gold or citations."""
    if not gold_doc_numbers:
        return None
    docs = {canon_doc(h.number) or f"chunk:{h.chunk_id}" for h, _ in _cited_hits(result)}
    if not docs:
        return None
    gold = {canon_doc(g) for g in gold_doc_numbers}
    return len(docs & gold) / len(docs)


def citation_hit(result: AskResult, gold_doc_numbers: Sequence[str]) -> bool | None:
    """At least one *verified* citation to a gold document. None without gold."""
    if not gold_doc_numbers:
        return None
    gold = {canon_doc(g) for g in gold_doc_numbers}
    return any(v and canon_doc(h.number) in gold for h, v in _cited_hits(result))


def stale_citations(result: AskResult, as_of: date) -> int:
    """Number of cited hits that were not in force on `as_of`."""
    return sum(1 for h, _ in _cited_hits(result) if not in_force(h, as_of))


def abstention_correct(abstained: bool, answerable: bool) -> bool:
    return abstained != answerable


# ---------------------------------------------------------------- per question / aggregate

RECALL_KS = (1, 5, 10)


def score_result(
    result: AskResult,
    *,
    answerable: bool,
    as_of: date,
    gold_doc_numbers: Sequence[str] = (),
    must_contain: Sequence[str] = (),
    must_not_contain: Sequence[str] = (),
) -> dict[str, Any]:
    """Flat per-question scores. `as_of` is the date the question is about."""
    abstained = result.answer.abstained
    correct: bool | None = None
    if answerable and (must_contain or must_not_contain):
        correct = (not abstained) and answer_correct(
            result.answer.text, must_contain, must_not_contain
        )
    n_cited = len(_cited_hits(result))
    n_stale = stale_citations(result, as_of)
    return {
        "answer_correct": correct,
        "gold_rank": gold_rank(result.hits, gold_doc_numbers) if gold_doc_numbers else None,
        "has_gold": bool(gold_doc_numbers),
        "citation_precision": citation_precision(result, gold_doc_numbers),
        "citation_hit": citation_hit(result, gold_doc_numbers),
        "n_citations": n_cited,
        "n_verified_citations": sum(1 for c in result.answer.citations if c.verified),
        "n_stale_citations": n_stale,
        "stale_source": (n_stale > 0) if n_cited else None,
        "abstained": abstained,
        "abstention_correct": abstention_correct(abstained, answerable),
    }


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default), p in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _rate(values: Iterable[bool | None]) -> dict[str, Any]:
    """{'value': share or None, 'num': successes, 'den': questions the metric applies to}."""
    vals = [bool(v) for v in values if v is not None]
    return {"value": (sum(vals) / len(vals)) if vals else None, "num": sum(vals), "den": len(vals)}


def _mean(values: Iterable[float | None]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    return {"value": (sum(vals) / len(vals)) if vals else None, "num": None, "den": len(vals)}


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question records ({'type', 'error', 'latency_ms', 'scores'}) into metrics.

    Errored questions are counted in `n_errors` and excluded from every rate.
    """
    ok = [r for r in records if not r.get("error") and r.get("scores")]
    sc = [r["scores"] for r in ok]
    metrics: dict[str, Any] = {
        "n_questions": len(records),
        "n_errors": len(records) - len(ok),
        "answer_accuracy": _rate(s["answer_correct"] for s in sc),
    }
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = _rate(
            (s["gold_rank"] is not None and s["gold_rank"] <= k) if s["has_gold"] else None
            for s in sc
        )
    metrics["citation_precision"] = _mean(s["citation_precision"] for s in sc)
    metrics["citation_hit_rate"] = _rate(s["citation_hit"] for s in sc)
    metrics["stale_source_rate"] = _rate(s["stale_source"] for s in sc)
    metrics["abstention_accuracy"] = _rate(s["abstention_correct"] for s in sc)
    lat = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    metrics["latency_p50_ms"] = percentile(lat, 50)
    metrics["latency_p95_ms"] = percentile(lat, 95)
    return metrics


def per_type(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    types: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        types.setdefault(r["type"], []).append(r)
    return {t: aggregate(rs) for t, rs in sorted(types.items())}
