"""Draft evaluation questions from parsed source documents, with machine-checked key facts.

    python -m asof.evaluation.generate --parsed-dir data/parsed --n 40 --out eval/candidates.jsonl

For each selected document the LLM proposes one factual question whose answer the document
states, plus short key facts. Every key fact must appear verbatim (after normalisation, see
`metrics.normalise`) in the document text; candidates that fail are dropped, so the generator
cannot invent facts. Output is in the `eval/questions.jsonl` format, for human review: copy the
good ones into questions.jsonl. LLM replies are cached under data/cache/evalgen/.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from asof.evaluation.metrics import contains_fact, normalise
from asof.evaluation.questions import EvalQuestion, dump_question
from asof.models import ParsedDoc

PROMPT_VERSION = "evalgen-v2"
MAX_DOC_CHARS = 14_000

SYSTEM_PROMPT = """You write evaluation questions for a question-answering system over Indian GST \
law (CBIC notifications, circulars, the CGST Act and Rules).

You are given ONE source document. Write ONE factual question that a GST practitioner might ask \
and that this document answers on its own. Rules:
- The answer must be stated explicitly in the document. Do not use outside knowledge.
- Prefer concrete facts: percentages, amounts, thresholds, time limits, dates, form names, \
which rule or section changes.
- Ask about the substance (which rule of the CGST Rules changes and how, what a circular \
clarifies), not about the document's own numbering or amendment-rule clauses.
- The question must be self-contained and must NOT mention the notification or circular \
number or its issue date (that would make finding the source trivial). Name the rule, section, \
form or topic instead, and set the time context naturally (e.g. "in March 2021") when the \
answer depends on the date.
- must_contain: 1-3 key facts that any correct answer must state, each the SHORTEST core \
phrase (1-4 words, usually a number with its unit), copied EXACTLY from the document text, \
e.g. "ten per cent", "GSTR-2B", "fifty crore". Leave out surrounding legal wording such as \
"number of" or "the words". For a date, give the month name and the year as two separate \
facts (e.g. "January", "2022"). Facts are checked mechanically against the document and \
against answers.
- evidence: one sentence copied exactly from the document that states the answer.
- effective_date: the date from which the provision takes effect, only if the document states \
it; otherwise null.
- about_current_law: true only if the question asks about the law today rather than at a date.
- If the document has no clear, checkable fact worth asking about, set usable to false."""


class GeneratedQuestion(BaseModel):
    usable: bool = Field(description="False if the document has no clear checkable fact")
    question: str = ""
    answer: str = Field(default="", description="Short answer, as stated in the document")
    must_contain: list[str] = Field(
        default=[], description="1-3 short key facts copied verbatim from the document"
    )
    evidence: str = Field(default="", description="One sentence copied verbatim from the document")
    effective_date: date | None = Field(
        default=None, description="Date the provision takes effect, if the document states it"
    )
    about_current_law: bool = False


class StructuredLLM(Protocol):
    def structured(self, messages: list[dict[str, Any]], schema: type[Any], **kw: Any) -> Any: ...


# ---------------------------------------------------------------- documents

def doc_text(doc: ParsedDoc) -> str:
    return "\n".join(p.text for p in doc.pages)


def load_parsed(parsed_dir: Path) -> list[ParsedDoc]:
    docs = []
    for path in sorted(parsed_dir.glob("*.json")):
        try:
            docs.append(ParsedDoc.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValidationError, ValueError) as err:
            print(f"skip {path.name}: not a ParsedDoc ({str(err).splitlines()[0]})",
                  file=sys.stderr)
    return docs


_AMEND_RE = re.compile(r"\bamend|\bsubstitut|\binsert|\bomit", re.I)


def priority(doc: ParsedDoc) -> int:
    """Lower is better: rule-amending notifications, then circulars, then the rest."""
    head = f"{doc.meta.title} {doc_text(doc)[:1500]}"
    if doc.meta.kind == "notification" and _AMEND_RE.search(head) and re.search(
        r"\brules?\b", head, re.I
    ):
        return 0
    if doc.meta.kind == "circular":
        return 1
    if doc.meta.kind == "notification":
        return 2
    return 3


def select_docs(docs: Sequence[ParsedDoc], n: int, *, seed: int = 0,
                min_chars: int = 400) -> list[ParsedDoc]:
    """Best-priority documents first, spread across years (round-robin), deterministic."""
    rng = random.Random(seed)
    usable = [d for d in docs if len(doc_text(d).strip()) >= min_chars]
    chosen: list[ParsedDoc] = []
    for pr in sorted({priority(d) for d in usable}):
        by_year: dict[int, list[ParsedDoc]] = {}
        for d in usable:
            if priority(d) == pr:
                by_year.setdefault(d.meta.issue_date.year, []).append(d)
        for group in by_year.values():
            group.sort(key=lambda d: d.meta.key)
            rng.shuffle(group)
        years = sorted(by_year)
        while len(chosen) < n and any(by_year.values()):
            for y in years:
                if by_year[y] and len(chosen) < n:
                    chosen.append(by_year[y].pop())
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------- verification

def verify_facts(text: str, facts: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split facts into (found verbatim in text after normalisation, missing)."""
    norm = normalise(text)
    found, missing = [], []
    for f in facts:
        (found if f.strip() and contains_fact(norm, f, normalised=True) else missing).append(f)
    return found, missing


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]


def date_in_text(d: date, text: str) -> bool:
    """Is this date written in the text in a common Indian legal format?"""
    t = " ".join(text.casefold().split())
    month = _MONTHS[d.month - 1]
    day = rf"0?{d.day}(?:st|nd|rd|th)?"
    patterns = [
        rf"\b{day}\s+(?:day\s+of\s+)?{month},?\s+{d.year}\b",
        rf"\b{month}\s+{day},?\s+{d.year}\b",
        rf"\b0?{d.day}[./-]0?{d.month}[./-]{d.year}\b",
    ]
    return any(re.search(p, t) for p in patterns)


_DOCNO_RE = re.compile(r"\b\d{1,4}/(?:\d{1,4}/)?(?:19|20)\d{2}\b")


def to_candidate(doc: ParsedDoc, gen: GeneratedQuestion) -> tuple[EvalQuestion | None, str]:
    """Turn an LLM proposal into a checked candidate, or (None, reason it was dropped)."""
    if not gen.usable or not gen.question.strip():
        return None, "model marked the document unusable"
    if _DOCNO_RE.search(gen.question):
        return None, "question names a document number (makes retrieval trivial)"
    facts = [" ".join(f.split()) for f in gen.must_contain if f.strip()][:3]
    if not facts:
        return None, "no key facts"
    text = doc_text(doc)
    _, missing = verify_facts(text, facts)
    if missing:
        return None, f"key facts not found verbatim in the document: {missing}"
    evidence_ok = bool(gen.evidence.strip()) and contains_fact(text, gen.evidence)

    as_of: date | None
    if gen.about_current_law:
        qtype, as_of, when = "current", None, "today"
    else:
        qtype = "point_in_time"
        if gen.effective_date and date_in_text(gen.effective_date, text):
            as_of, when = gen.effective_date, "stated effective date"
        else:
            as_of, when = doc.meta.issue_date, "issue date"
    notes = (
        f"auto-generated ({PROMPT_VERSION}) from {doc.meta.kind} {doc.meta.number} "
        f"({doc.meta.issue_date.isoformat()}); as_of = {when}; "
        f"evidence {'verified' if evidence_ok else 'NOT verified'}; "
        f"draft answer: {gen.answer.strip()[:300]}. Review before adding to questions.jsonl."
    )
    try:
        q = EvalQuestion(
            id=f"gen-{doc.meta.key}"[:120],
            question=gen.question,
            as_of=as_of,
            type=qtype,  # type: ignore[arg-type]
            gold_doc_numbers=[doc.meta.number],
            must_contain=facts,
            notes=notes,
        )
    except ValidationError as err:
        return None, f"invalid question: {err.errors()[0]['msg']}"
    return q, "ok"


# ---------------------------------------------------------------- LLM + cache

def build_messages(doc: ParsedDoc) -> list[dict[str, Any]]:
    m = doc.meta
    body = doc_text(doc)[:MAX_DOC_CHARS]
    user = (
        f"Document: {m.kind} {m.number}, issued {m.issue_date.isoformat()}\n"
        f"Subject: {m.title}\n\n--- document text ---\n{body}\n--- end ---"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cache_key(doc: ParsedDoc, model: str) -> str:
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, SYSTEM_PROMPT, model, doc.meta.key, doc_text(doc)[:MAX_DOC_CHARS]):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return f"{doc.meta.key}-{h.hexdigest()[:16]}"


def propose(doc: ParsedDoc, llm: StructuredLLM, cache_dir: Path | None, model: str = "",
            ) -> tuple[GeneratedQuestion, bool]:
    """LLM proposal for one document; returns (proposal, served_from_cache)."""
    path = cache_dir / f"{cache_key(doc, model)}.json" if cache_dir else None
    if path and path.exists():
        return GeneratedQuestion.model_validate_json(path.read_text(encoding="utf-8")), True
    gen = llm.structured(build_messages(doc), GeneratedQuestion, max_tokens=2048)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(gen.model_dump_json(indent=2), encoding="utf-8")
    return gen, False


def _is_rate_limit(err: BaseException) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def generate(
    docs: Sequence[ParsedDoc],
    llm: StructuredLLM,
    *,
    cache_dir: Path | None,
    model: str = "",
    sleep_s: float = 0.0,
    max_calls: int | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[EvalQuestion], dict[str, Any]]:
    """Propose + verify one question per document. Stops on a rate limit or 3 errors in a row."""
    out: list[EvalQuestion] = []
    stats: dict[str, Any] = {"docs": len(docs), "llm_calls": 0, "cached": 0, "kept": 0,
                             "dropped": 0, "errors": 0, "stopped": None}
    streak = 0
    for doc in docs:
        if max_calls is not None and stats["llm_calls"] >= max_calls:
            stats["stopped"] = f"reached --max-calls {max_calls}"
            break
        try:
            if stats["llm_calls"] and sleep_s:
                time.sleep(sleep_s)
            gen, cached = propose(doc, llm, cache_dir, model)
        except Exception as err:  # noqa: BLE001 - keep going, but not forever
            stats["errors"] += 1
            streak += 1
            log(f"{doc.meta.number}: ERROR {type(err).__name__}: {str(err)[:200]}")
            if _is_rate_limit(err):
                stats["stopped"] = "rate limited: stopped; rerun later (cache keeps progress)"
                break
            if streak >= 3:
                stats["stopped"] = "3 errors in a row: stopped"
                break
            continue
        streak = 0
        stats["cached" if cached else "llm_calls"] += 1
        q, reason = to_candidate(doc, gen)
        if q is None:
            stats["dropped"] += 1
            log(f"{doc.meta.number}: dropped ({reason})")
        else:
            stats["kept"] += 1
            out.append(q)
            log(f"{doc.meta.number}: kept: {q.question}")
    return out, stats


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m asof.evaluation.generate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parsed-dir", default="data/parsed", type=Path)
    p.add_argument("--n", type=int, default=40, help="number of documents to try")
    p.add_argument("--out", default="eval/candidates.jsonl", type=Path)
    p.add_argument("--cache-dir", default="data/cache/evalgen", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.0, help="extra seconds between LLM calls")
    p.add_argument("--max-calls", type=int, default=None, help="cap on live LLM calls")
    p.add_argument("--dry-run", action="store_true", help="list selected documents and exit")
    args = p.parse_args(argv)

    docs = load_parsed(args.parsed_dir)
    if not docs:
        print(f"no ParsedDoc files in {args.parsed_dir}; run ingestion first", file=sys.stderr)
        return 1
    chosen = select_docs(docs, args.n, seed=args.seed)
    if args.dry_run:
        for d in chosen:
            print(f"p{priority(d)}  {d.meta.issue_date}  {d.meta.kind:<12} {d.meta.number}")
        return 0

    from asof.config import get_settings
    from asof.providers.llm import LLM

    settings = get_settings()
    llm = LLM(settings)
    questions, stats = generate(chosen, llm, cache_dir=args.cache_dir,
                                model=settings.llm_fast_model, sleep_s=args.sleep,
                                max_calls=args.max_calls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(dump_question(q) + "\n" for q in questions), encoding="utf-8")
    print(json.dumps(stats))
    print(f"wrote {len(questions)} candidates to {args.out}")
    return 1 if stats["stopped"] and stats["stopped"].startswith("rate") else 0


if __name__ == "__main__":
    sys.exit(main())
