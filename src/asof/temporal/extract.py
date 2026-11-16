"""Extract amendment / rescission / clarification relations from a parsed document with the LLM.

Pipeline per document:
1. cheap regex prefilter: no amend/rescind/supersede/... wording -> no LLM call;
2. LLM structured extraction (`ExtractionResult`), one call per ~15k-char window (the first
   window always; later windows only when they themselves pass the prefilter, capped);
3. post-validation: drop relations whose `evidence` is not (fuzzily) in the text, normalise
   `target_ref`, drop self-references and duplicates.

Raw LLM output is cached in data/cache/relations/<key>.json keyed by a hash of
(model, PROMPT_VERSION, text), so reruns cost nothing and post-validation can change freely.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from asof.models import ExtractedRelation, ExtractionResult, ParsedDoc
from asof.providers.llm import LLMError
from asof.temporal.normalize import clean, normalize_target

PROMPT_VERSION = "rel-v2"
WINDOW_CHARS = 15_000
MAX_WINDOWS = 3
MAX_EVIDENCE_CHARS = 1200

PREFILTER_RE = re.compile(
    r"amend|rescind|supersession|supersede|omit|substitut|insert|clarif|withdraw",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You extract legal relations from Indian GST notifications and circulars \
(CBIC). Read the document and list EVERY relation it states to other instruments.
Only report changes that THIS document itself makes. If it merely mentions that a rule or \
notification was amended by some other instrument ("rule 89 has been amended vide \
notification No. 26/2022-Central Tax"), that is not an amendment by this document: for a \
circular report it as "clarifies" at most, otherwise skip it.

Relation kinds:
- "substitutes": text is replaced ("for the words X, the words Y shall be substituted", \
"the following rule shall be substituted").
- "inserts": new text is added ("after sub-rule (3), the following sub-rule shall be inserted").
- "omits": text is removed ("sub-rule (4) shall be omitted").
- "amends": any other amendment, or a notification amending another notification \
("in the said notification ... shall be amended").
- "rescinds": "hereby rescinds notification No. ..."; also a circular/instruction that is \
"withdrawn".
- "supersedes": "in supersession of notification No. ..."; "this circular supersedes ...".
- "clarifies": a circular clarifying a provision or notification.

Targets (target_kind / target_ref):
- Amendments to the Central Goods and Services Tax Rules: target_kind "rule", target_ref the \
most specific unit changed, written like "rule 36(4)" or "rule 89(4)(c)" or "rule 86B". \
Resolve "in the said rules, in rule 36, in sub-rule (4)" to "rule 36(4)".
- Amendments to the CGST Act: target_kind "section", e.g. "section 16(4)". For the IGST Act \
write "section 5 of the IGST Act".
- Notifications: target_kind "notification", target_ref like "49/2019-Central Tax" or \
"1/2017-Central Tax (Rate)". Circulars: "circular", like "123/42/2019-GST". \
Forms: "form", like "FORM GSTR-3B". Anything else: "other".
- Do NOT list the document itself as a target. One relation per (kind, target); if one \
paragraph changes several sub-rules, list each sub-rule separately.

effective_date (per relation): the date that specific change takes effect if the text \
states it ("shall come into force on the 1st day of January, 2020", "with effect from", \
"shall be deemed to have come into force on"), as YYYY-MM-DD; otherwise null.
Top-level effective_date: the date this document comes into force. If it says it comes into \
force "on the date of its publication in the Official Gazette", use the issue date given below.

evidence: an EXACT verbatim quote copied from the document (no paraphrase, no ellipsis), \
at most ~600 characters. For "substitutes" and "inserts" the quote MUST include the new \
wording. confidence: 0..1.
If the document states no relations, return {"relations": [], "effective_date": ...}."""


AMENDING_KINDS = ("amends", "substitutes", "inserts", "omits")

@dataclass
class ExtractStats:
    scanned: int = 0
    skipped_prefilter: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    dropped_evidence: int = 0
    dropped_other: int = 0
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ text helpers

def doc_text(doc: ParsedDoc) -> str:
    return "\n\n".join(p.text for p in doc.pages if p.text)


def passes_prefilter(text: str) -> bool:
    return PREFILTER_RE.search(text) is not None


def windows(text: str, size: int = WINDOW_CHARS, max_windows: int = MAX_WINDOWS) -> list[str]:
    """First `size` chars always; later windows only if they match the prefilter."""
    out: list[str] = []
    pos = 0
    while pos < len(text) and len(out) < max_windows:
        end = min(len(text), pos + size)
        if end < len(text):  # cut at a paragraph / line break near the end when possible
            cut = max(text.rfind("\n\n", pos + size // 2, end), text.rfind("\n", pos + size // 2,
                                                                            end))
            if cut > pos:
                end = cut
        win = text[pos:end]
        if not out or passes_prefilter(win):
            out.append(win)
        pos = end
    return out


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", clean(s).lower())


def evidence_in_text(evidence: str, text: str, *, threshold: float = 0.85,
                     _norm_text: str | None = None) -> bool:
    """Is `evidence` (fuzzily) a quote from `text`?

    Compared after case/whitespace/punctuation normalisation (which also absorbs PDF line-break
    hyphenation); falls back to the longest common block covering >= `threshold` of it, and
    accepts "a ... b" quotes when every part is present.
    """
    ev = _alnum(evidence)
    if len(ev) < 8:
        return False
    norm = _norm_text if _norm_text is not None else _alnum(text)
    if ev in norm:
        return True
    parts = [p for p in re.split(r"\.{3,}|…", evidence) if len(_alnum(p)) >= 8]
    if len(parts) > 1 and all(_alnum(p) in norm for p in parts):
        return True
    sm = SequenceMatcher(None, norm, ev, autojunk=False)
    match = sm.find_longest_match(0, len(norm), 0, len(ev))
    if match.size >= threshold * len(ev):
        return True
    # tolerate a few OCR/typo differences: compare the best-aligned window as a whole
    if match.size >= 20:
        start = max(0, match.a - match.b)
        window = norm[start: start + len(ev) + 10]
        return SequenceMatcher(None, window, ev, autojunk=False).ratio() >= threshold
    return False


# ------------------------------------------------------------------ prompt

def build_messages(doc: ParsedDoc, text: str, *, part: int = 1, parts: int = 1) -> list[dict]:
    m = doc.meta
    header = (
        f"Document: {m.kind} No. {m.number}\n"
        f"Issue date: {m.issue_date.isoformat()}\n"
        f"Subject: {m.title}\n"
    )
    if parts > 1:
        header += f"(Text part {part} of {parts}; extract the relations stated in this part.)\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}\nDocument text:\n<<<\n{text}\n>>>"},
    ]


# ------------------------------------------------------------------ post-validation

def _self_refs(doc: ParsedDoc) -> set[str]:
    _, own = normalize_target(doc.meta.kind if doc.meta.kind in ("notification", "circular")
                              else "other", doc.meta.number)
    return {own.lower(), doc.meta.number.lower()}


def postprocess(raw: ExtractionResult, doc: ParsedDoc, text: str,
                stats: ExtractStats | None = None) -> ExtractionResult:
    """Validate evidence, normalise targets, drop self-refs and duplicates."""
    norm_text = _alnum(text)
    own = _self_refs(doc)
    seen: set[tuple] = set()
    kept: list[ExtractedRelation] = []
    for rel in raw.relations:
        evidence = clean(rel.evidence)[:MAX_EVIDENCE_CHARS]
        if not evidence_in_text(evidence, text, _norm_text=norm_text):
            if stats:
                stats.dropped_evidence += 1
            continue
        kind, ref = normalize_target(rel.target_kind, rel.target_ref)
        if not ref or ref.lower() in own:
            if stats:
                stats.dropped_other += 1
            continue
        rkind = rel.kind
        if (doc.meta.kind == "circular" and rkind in AMENDING_KINDS
                and kind != "circular"):
            rkind = "clarifies"  # circulars cannot amend rules, sections or notifications
        key = (rkind, kind, ref.lower(), rel.effective_date)
        if key in seen:
            continue
        seen.add(key)
        kept.append(rel.model_copy(update={
            "kind": rkind,
            "target_kind": kind if kind in ("notification", "circular", "rule", "section",
                                            "form") else "other",
            "target_ref": ref,
            "evidence": evidence,
        }))
    return ExtractionResult(relations=kept, effective_date=raw.effective_date)


# ------------------------------------------------------------------ cache

def cache_key(model: str, text: str) -> str:
    h = hashlib.sha256()
    for part in (model, PROMPT_VERSION, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_load(path: Path, key: str) -> ExtractionResult | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("hash") == key:
            return ExtractionResult.model_validate(data["raw"])
    except (OSError, ValueError, KeyError):
        return None
    return None


def _cache_save(path: Path, key: str, model: str, raw: ExtractionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"hash": key, "model": model, "prompt_version": PROMPT_VERSION,
                               "raw": raw.model_dump(mode="json")}, ensure_ascii=False,
                              indent=1), encoding="utf-8")
    tmp.replace(path)


def _merge(results: list[ExtractionResult]) -> ExtractionResult:
    eff = next((r.effective_date for r in results if r.effective_date), None)
    return ExtractionResult(relations=[x for r in results for x in r.relations],
                            effective_date=eff)


# ------------------------------------------------------------------ main entry

def _extract_window(llm, doc: ParsedDoc, win: str, part: int, parts: int, *, model: str,
                    max_tokens: int, thinking: bool, stats: ExtractStats,
                    depth: int = 0) -> list[ExtractionResult]:
    """One LLM call; if the output stays invalid (typically a long, broken JSON answer for a
    dense amending notification), split the window in two and try each half once."""
    stats.llm_calls += 1
    try:
        return [llm.structured(build_messages(doc, win, part=part, parts=parts),
                               ExtractionResult, model=model, retries=1,
                               max_tokens=max_tokens, thinking=thinking)]
    except LLMError:
        if depth >= 1 or len(win) < 3000:
            raise
    mid = win.rfind("\n", len(win) // 3, 2 * len(win) // 3)
    mid = mid if mid > 0 else len(win) // 2
    out: list[ExtractionResult] = []
    for n, half in enumerate((win[:mid], win[mid:])):
        if n == 0 or passes_prefilter(half):
            out += _extract_window(llm, doc, half, part, parts, model=model,
                                   max_tokens=max_tokens, thinking=thinking, stats=stats,
                                   depth=depth + 1)
    return out


def extract_relations(llm, doc: ParsedDoc, *, cache_dir: Path | str | None = None,
                      stats: ExtractStats | None = None, use_cache: bool = True,
                      max_tokens: int = 8192, thinking: bool = False) -> ExtractionResult:
    """Relations stated in `doc` (validated + normalised). `llm` is an `asof.providers.LLM`.

    `llm` is only used on a cache miss for a document that passes the prefilter.
    """
    stats = stats if stats is not None else ExtractStats()
    stats.scanned += 1
    text = doc_text(doc)
    if not passes_prefilter(text):
        stats.skipped_prefilter += 1
        return ExtractionResult(relations=[], effective_date=None)

    if llm is None:
        raise ValueError("an LLM is required for documents that pass the prefilter")
    model = llm.settings.llm_fast_model
    if cache_dir is None:
        cache_dir = Path(llm.settings.data_dir) / "cache" / "relations"
    path = Path(cache_dir) / f"{doc.meta.key}.json"
    key = cache_key(model + ("+think" if thinking else ""), text)

    raw = _cache_load(path, key) if use_cache else None
    if raw is not None:
        stats.cache_hits += 1
    else:
        wins = windows(text)
        results = []
        for i, win in enumerate(wins, 1):
            results.extend(_extract_window(llm, doc, win, i, len(wins), model=model,
                                           max_tokens=max_tokens, thinking=thinking,
                                           stats=stats))
        raw = _merge(results)
        _cache_save(path, key, model, raw)
    return postprocess(raw, doc, text, stats)
