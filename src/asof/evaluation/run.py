"""Run the evaluation set against one or more systems and write results + a summary.

    python -m asof.evaluation.run --systems asof,baseline
    python -m asof.evaluation.run --systems asof --limit 5 --sleep 2
    python -m asof.evaluation.run --systems asof --resume eval/results/20260101-120000-asof.jsonl

Per-question records go to eval/results/<timestamp>-<system>.jsonl (appended as they finish, so
a crash loses nothing); eval/results/summary.md and summary.json are rewritten at the end.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from asof.evaluation.metrics import aggregate, canon_doc, per_type, score_result
from asof.evaluation.questions import EvalQuestion, load_questions
from asof.evaluation.report import render_markdown
from asof.models import AskResult

SYSTEMS = ("asof", "baseline")
AskFn = Callable[..., AskResult]


class _Stop(Exception):
    """Internal: abort the run early (e.g. repeated rate limits)."""


def _default_ask() -> AskFn:
    from asof.pipeline import ask  # lazy: needs the DB and model providers

    return ask


def _is_rate_limit(err: BaseException) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def evaluate_one(
    q: EvalQuestion, system: str, ask: AskFn, *, today: date
) -> dict[str, Any]:
    """Ask one question and score it. Never raises: errors are recorded on the record."""
    record: dict[str, Any] = {
        "id": q.id, "system": system, "type": q.type, "question": q.question,
        "as_of": q.as_of.isoformat() if q.as_of else None,
        "error": None, "latency_ms": None, "scores": None, "result": None,
    }
    start = time.perf_counter()
    try:
        result = ask(q.question, as_of=q.as_of, mode=system)
    except Exception as err:  # noqa: BLE001 - one bad question must not stop the run
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        record["rate_limited"] = _is_rate_limit(err)
        return record
    record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    record["result"] = result.model_dump(mode="json")
    record["scores"] = score_result(
        result,
        answerable=q.answerable,
        as_of=q.as_of or today,
        gold_doc_numbers=q.gold_doc_numbers,
        must_contain=q.must_contain,
        must_not_contain=q.must_not_contain,
    )
    return record


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def build_summary(
    by_system: dict[str, list[dict[str, Any]]],
    files: dict[str, list[str]],
    questions_file: str,
    generated_at: str,
    incomplete: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "questions_file": questions_file,
        "incomplete": incomplete,
        "systems": {
            s: {
                "results_files": files.get(s, []),
                "metrics": aggregate(recs),
                "per_type": per_type(recs),
            }
            for s, recs in by_system.items()
        },
    }


def run_eval(
    questions: Sequence[EvalQuestion],
    systems: Sequence[str],
    *,
    out_dir: Path,
    ask: AskFn | None = None,
    resume: Sequence[Path] = (),
    sleep_s: float = 0.0,
    questions_file: str = "eval/questions.jsonl",
    now: datetime | None = None,
    max_consecutive_rate_limits: int = 3,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Evaluate every question on every system; returns the summary dict (also written)."""
    ask = ask or _default_ask()
    now = now or datetime.now(UTC)
    today = now.date()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: earlier records per system, and the file to keep appending to.
    by_system: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    files: dict[str, list[str]] = {s: [] for s in systems}
    target: dict[str, Path] = {}
    for path in resume:
        for rec in read_records(path):
            if rec.get("system") in by_system:
                by_system[rec["system"]].append(rec)
                target.setdefault(rec["system"], path)
    for s in systems:
        target.setdefault(s, out_dir / f"{stamp}-{s}.jsonl")
        files[s].append(target[s].as_posix())

    wanted = {q.id for q in questions}
    for s in systems:  # drop resumed records for questions no longer in the set
        by_system[s] = [r for r in by_system[s] if r["id"] in wanted]

    incomplete = None
    streak = 0
    calls = 0
    try:
        for s in systems:
            done = {r["id"] for r in by_system[s]}
            with target[s].open("a", encoding="utf-8") as fh:
                for i, q in enumerate(questions, start=1):
                    if q.id in done:
                        continue
                    if calls and sleep_s:
                        time.sleep(sleep_s)
                    calls += 1
                    rec = evaluate_one(q, s, ask, today=today)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    by_system[s].append(rec)
                    status = "ERROR " + rec["error"] if rec["error"] else (
                        "ok" if rec["scores"]["answer_correct"] in (True, None) else "wrong")
                    log(f"[{s} {i}/{len(questions)}] {q.id}: {status}")
                    streak = streak + 1 if rec.get("rate_limited") else 0
                    if streak >= max_consecutive_rate_limits:
                        incomplete = (f"stopped after {streak} consecutive rate-limit errors; "
                                      "rerun later with --resume")
                        raise _Stop
    except _Stop:
        log(incomplete or "stopped")
    except KeyboardInterrupt:
        incomplete = "interrupted; rerun with --resume to continue"
        log(incomplete)

    summary = build_summary(by_system, files, questions_file, now.isoformat(timespec="seconds"),
                            incomplete)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def missing_gold(questions: Sequence[EvalQuestion], parsed_dir: Path) -> dict[str, list[str]]:
    """Question id -> gold document numbers with no ParsedDoc in `parsed_dir`."""
    from asof.models import ParsedDoc

    have = set()
    for path in parsed_dir.glob("*.json"):
        try:
            doc = ParsedDoc.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        have.add(canon_doc(doc.meta.number))
    out = {}
    for q in questions:
        gone = [g for g in q.gold_doc_numbers if canon_doc(g) not in have]
        if gone:
            out[q.id] = gone
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m asof.evaluation.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default="asof,baseline",
                   help="comma-separated: asof, baseline (default: both)")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--ids", default=None, help="comma-separated question ids to run")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")
    p.add_argument("--resume", action="append", default=[], type=Path,
                   help="results .jsonl to continue (skips its question ids); repeatable")
    p.add_argument("--out-dir", default="eval/results", type=Path)
    p.add_argument("--check-corpus", type=Path, default=None, metavar="PARSED_DIR",
                   help="only report gold documents missing from PARSED_DIR (no model calls)")
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    bad = [s for s in systems if s not in SYSTEMS]
    if bad or not systems:
        p.error(f"unknown system(s) {bad}; choose from {', '.join(SYSTEMS)}")
    questions = load_questions(args.questions)
    if args.ids:
        ids = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q.id in ids]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        p.error("no questions selected")
    if args.check_corpus:
        missing = missing_gold(questions, args.check_corpus)
        for qid, docs in missing.items():
            print(f"{qid}: gold not in corpus: {', '.join(docs)}")
        print(f"{len(questions) - len(missing)}/{len(questions)} questions have all gold "
              f"documents in {args.check_corpus}")
        return 1 if missing else 0

    summary = run_eval(questions, systems, out_dir=args.out_dir, resume=args.resume,
                       sleep_s=args.sleep, questions_file=args.questions)
    print()
    print(render_markdown(summary))
    return 1 if summary.get("incomplete") else 0


if __name__ == "__main__":
    sys.exit(main())
