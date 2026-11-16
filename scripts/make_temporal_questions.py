"""Build point-in-time / supersession-trap / unanswerable eval questions from the database.

Grounded by construction: for every provision with 2+ extracted versions, the LLM proposes a
question and one distinguishing "key fact" per version; the key fact is kept only if it
appears verbatim (normalised) in that version's text and in no other version. The other
versions' key facts become `must_not_contain` (stale answers).

Usage: python scripts/make_temporal_questions.py [--max-provisions 25] [--out eval/temporal.jsonl]
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel

from asof.config import get_settings
from asof.db import connect
from asof.providers import LLM

CACHE = Path("data/cache/temporal_questions")
PROMPT_VERSION = "v1"


def norm(s: str) -> str:
    s = s.casefold().replace("%", " per cent")
    s = re.sub(r"[‘’“”\"']", "", s)
    return re.sub(r"\s+", " ", s).strip()


class VersionFact(BaseModel):
    version: int  # index as given in the prompt
    key_fact: str  # short verbatim phrase from that version's text


class Proposal(BaseModel):
    question_template: str  # must contain the literal placeholder {date}
    facts: list[VersionFact]


PROMPT = """You are writing evaluation questions for a legal search system about Indian GST law.

Provision: {ref} ({instrument})
Its successive versions (oldest first), each with the period it was in force:
{versions}

Write ONE natural question template about what this provision provided at a given date. It must
contain the literal placeholder {{date}} (e.g. "What did rule 36(4) of the CGST Rules provide
on {{date}}?"). Do not mention any version's content in the question.

For EACH version give one short key fact (2-8 words) copied VERBATIM from that version's text
that distinguishes it from the other versions (e.g. a percentage, an amount, a time limit).
If a version's text is empty (the provision was omitted), use key_fact "omitted"."""


def fetch_chains(conn, max_provisions: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.id, p.instrument, p.ref, pv.text, lower(pv.validity), upper(pv.validity),
               d.number
        FROM provision_versions pv
        JOIN provisions p ON p.id = pv.provision_id
        LEFT JOIN documents d ON d.id = pv.source_doc_id
        WHERE p.id IN (SELECT provision_id FROM provision_versions
                       GROUP BY provision_id HAVING count(*) >= 2)
        ORDER BY p.id, lower(pv.validity) NULLS FIRST
        """
    ).fetchall()
    chains: dict[int, dict] = {}
    for pid, instrument, ref, text, lo, hi, number in rows:
        c = chains.setdefault(pid, {"instrument": instrument, "ref": ref, "versions": []})
        c["versions"].append({"text": text or "", "from": lo, "to": hi, "doc": number})
    return list(chains.values())[:max_provisions]


def propose(llm: LLM, chain: dict) -> Proposal | None:
    versions = "\n".join(
        f"[{i}] in force {v['from']} to {v['to'] or 'present'} (source {v['doc']}): "
        f"{(v['text'] or '(omitted - no text)')[:600]}"
        for i, v in enumerate(chain["versions"])
    )
    prompt = PROMPT.format(ref=chain["ref"], instrument=chain["instrument"], versions=versions)
    key = hashlib.sha256(f"{PROMPT_VERSION}|{prompt}".encode()).hexdigest()[:24]
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return Proposal.model_validate_json(cached.read_text(encoding="utf-8"))
    try:
        out = llm.structured([{"role": "user", "content": prompt}], Proposal, max_tokens=1200)
    except Exception as err:  # noqa: BLE001
        print(f"  skip {chain['ref']}: {type(err).__name__}: {err}", file=sys.stderr)
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(out.model_dump_json(), encoding="utf-8")
    return out


# Key facts must be substance, not the amending instruction ("shall be inserted", "after rule
# 88A") or boilerplate ("Provided that"), which an answer about the rule would not contain.
_INSTRUCTION_RE = re.compile(
    r"\b(?:inserted|substituted|omitted|namely|said rules?|hereinafter)\b"
    r"|^(?:provided that|after (?:rule|sub-rule|clause)\b.*|in rule \d+.*)$", re.IGNORECASE)


def good_fact(fact: str) -> bool:
    f = fact.strip().strip(".,;:")
    multiword_or_code = len(f.split()) >= 2 or any(ch.isdigit() for ch in f)
    return multiword_or_code and not _INSTRUCTION_RE.search(f)


def midpoint(lo: date, hi: date | None, today: date) -> date:
    end = min(hi - timedelta(days=1), today) if hi else today
    return lo + (end - lo) // 2 if end > lo else lo


def build(chain: dict, prop: Proposal, today: date) -> list[dict]:
    if "{date}" not in prop.question_template:
        return []
    vs = chain["versions"]
    facts: dict[int, str] = {}
    for f in prop.facts:
        if not 0 <= f.version < len(vs):
            continue
        kf = norm(f.key_fact)
        if kf == "omitted" and not vs[f.version]["text"]:
            facts[f.version] = "omitted"
            continue
        own = norm(vs[f.version]["text"])
        others = [norm(v["text"]) for j, v in enumerate(vs) if j != f.version]
        if kf and good_fact(f.key_fact) and kf in own and not any(kf in o for o in others):
            facts[f.version] = f.key_fact
    out = []
    for i, v in enumerate(vs):
        if i not in facts or v["from"] is None:
            continue
        as_of = midpoint(v["from"], v["to"], today)
        # Stale = facts of versions NOT in force on as_of (versions may overlap, e.g. a
        # proviso inserted while the main wording stays in force).
        def in_force(j: int, day: date = as_of) -> bool:
            lo, hi = vs[j]["from"], vs[j]["to"]
            return lo is not None and lo <= day and (hi is None or day < hi)

        # Unambiguous questions only: no other version with a key fact may be in force too.
        if any(in_force(j) for j in facts if j != i):
            continue
        stale = [facts[j] for j in facts
                 if j != i and facts[j] != "omitted" and not in_force(j)]
        out.append({
            "id": f"t-{re.sub(r'[^a-z0-9]+', '-', chain['ref'].lower()).strip('-')}-{i}",
            "question": prop.question_template.replace("{date}", as_of.strftime("%d %B %Y")),
            "as_of": as_of.isoformat(),
            "type": "point_in_time",
            "gold_doc_numbers": [v["doc"]] if v["doc"] else [],
            "must_contain": [] if facts[i] == "omitted" else [facts[i]],
            "must_not_contain": stale,
            "notes": f"auto: {chain['ref']} version {i} of {len(vs)}; "
                     f"valid {v['from']}..{v['to'] or 'present'}",
        })
    return out


UNANSWERABLE = [
    ("What did notification 12/2024-Central Tax notify?", "2024 is outside the 2019-2022 corpus"),
    ("What change did the CGST (Amendment) Rules, 2025 make?", "outside corpus years"),
    ("What is the basic customs duty on imported gold bars?", "customs duty, not GST"),
    ("What did the GST Council decide at its 56th meeting?", "council minutes not in corpus"),
    ("What is the income tax rate for a domestic company?", "income tax, not GST"),
    ("Which circular in 2024 clarified GST on corporate guarantees?", "outside corpus years"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-provisions", type=int, default=25)
    ap.add_argument("--out", default="eval/temporal.jsonl")
    ap.add_argument("--corpus-end", type=date.fromisoformat, default=date(2022, 12, 31),
                    help="latest date the corpus can speak for; as-of dates stay before it")
    args = ap.parse_args()

    settings = get_settings()
    llm = LLM(settings)
    today = min(date.today(), args.corpus_end)
    with connect(settings) as conn:
        chains = fetch_chains(conn, args.max_provisions)
    print(f"{len(chains)} provisions with 2+ versions")

    rows: list[dict] = []
    for chain in chains:
        prop = propose(llm, chain)
        if prop:
            built = build(chain, prop, today)
            print(f"  {chain['ref']}: {len(chain['versions'])} versions -> {len(built)} questions")
            rows += built
    for k, (q, why) in enumerate(UNANSWERABLE):
        rows.append({"id": f"u-{k}", "question": q, "as_of": None, "type": "unanswerable",
                     "gold_doc_numbers": [], "must_contain": [], "must_not_contain": [],
                     "notes": why})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
