"""The evaluation run loop and summary rendering, with a fake `ask` (no DB, no network)."""

import json
from datetime import UTC, date, datetime

from asof.evaluation.questions import parse_questions
from asof.evaluation.report import render_markdown
from asof.evaluation.run import main, missing_gold, run_eval
from asof.models import Answer, AskResult, Citation, DocMeta, Hit, ParsedDoc, QueryPlan

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

QUESTIONS = parse_questions([
    json.dumps({"id": "q1", "question": "Rule 36(4) cap in February 2020?", "as_of": "2020-02-15",
                "type": "point_in_time", "gold_doc_numbers": ["75/2019-Central Tax"],
                "must_contain": ["10 per cent"], "must_not_contain": ["5 per cent"]}),
    json.dumps({"id": "q2", "question": "Service tax rate in 2012?", "as_of": "2012-06-01",
                "type": "unanswerable"}),
    json.dumps({"id": "q3", "question": "This one crashes the pipeline", "as_of": "2020-02-15",
                "type": "point_in_time", "gold_doc_numbers": ["75/2019-Central Tax"]}),
])

HITS = [
    Hit(chunk_id=1, doc_id=1, kind="notification", number="75/2019-Central Tax",
        issue_date=date(2019, 12, 26), title="t", url=None, section_ref="rule 36(4)", page=1,
        text="ten per cent", valid_from=date(2020, 1, 1), valid_to=date(2021, 1, 1)),
    Hit(chunk_id=2, doc_id=2, kind="notification", number="94/2020-Central Tax",
        issue_date=date(2020, 12, 22), title="t", url=None, section_ref="rule 36(4)", page=1,
        text="5 per cent", valid_from=date(2021, 1, 1)),
]


def fake_ask(question, *, as_of=None, mode="asof", **_):
    if "crashes" in question:
        raise RuntimeError("database unavailable")
    d = as_of or date(2026, 1, 2)
    plan = QueryPlan(as_of=d, as_of_explicit=as_of is not None, question_type="point_in_time")
    if "2012" in question:
        answer = Answer(text="Not covered.", abstained=mode == "asof", as_of=d)
    elif mode == "asof":
        answer = Answer(text="10% [S1]", as_of=d,
                        citations=[Citation(source=1, chunk_id=1, quote="ten", verified=True)])
    else:  # the baseline cites the later, not-yet-in-force notification
        answer = Answer(text="5 per cent [S2]", as_of=d,
                        citations=[Citation(source=2, chunk_id=2, quote="5", verified=True)])
    return AskResult(question=question, mode=mode, plan=plan, hits=HITS, answer=answer,
                     timings_ms={"total": 12.0})


def test_run_eval_writes_records_and_summary(tmp_path):
    logs: list[str] = []
    summary = run_eval(QUESTIONS, ["asof", "baseline"], out_dir=tmp_path, ask=fake_ask,
                       now=NOW, log=logs.append)

    asof_file = tmp_path / "20260102-030405-asof.jsonl"
    records = [json.loads(x) for x in asof_file.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in records] == ["q1", "q2", "q3"]
    assert records[2]["error"] == "RuntimeError: database unavailable"
    assert records[0]["result"]["answer"]["text"] == "10% [S1]"

    a = summary["systems"]["asof"]["metrics"]
    b = summary["systems"]["baseline"]["metrics"]
    assert a["n_errors"] == 1 and b["n_errors"] == 1
    assert a["answer_accuracy"]["value"] == 1.0 and b["answer_accuracy"]["value"] == 0.0
    assert a["stale_source_rate"]["value"] == 0.0 and b["stale_source_rate"]["value"] == 1.0
    assert a["abstention_accuracy"]["value"] == 1.0 and b["abstention_accuracy"]["value"] == 0.5

    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert md == render_markdown(summary)
    assert "| Metric | `asof` | `baseline` |" in md
    assert "| Errors (excluded from rates) | 1 | 1 |" in md
    assert "100.0% (1/1)" in md and "## By question type" in md
    assert json.loads((tmp_path / "summary.json").read_text())["systems"].keys() == {
        "asof", "baseline"}
    assert any("ERROR" in line for line in logs)


def test_resume_skips_done_ids_and_appends(tmp_path):
    calls: list[str] = []

    def counting_ask(question, **kw):
        calls.append(question)
        return fake_ask(question, **kw)

    run_eval(QUESTIONS[:1], ["asof"], out_dir=tmp_path, ask=counting_ask, now=NOW,
             log=lambda _: None)
    first = tmp_path / "20260102-030405-asof.jsonl"
    assert len(calls) == 1

    later = datetime(2026, 1, 3, tzinfo=UTC)
    summary = run_eval(QUESTIONS, ["asof"], out_dir=tmp_path, ask=counting_ask, now=later,
                       resume=[first], log=lambda _: None)
    assert len(calls) == 3  # q1 skipped on the second run
    assert len(first.read_text(encoding="utf-8").splitlines()) == 3
    assert not (tmp_path / "20260103-000000-asof.jsonl").exists()
    assert summary["systems"]["asof"]["metrics"]["n_questions"] == 3


def test_stops_after_repeated_rate_limits(tmp_path):
    class RateLimitError(Exception):
        pass

    def limited(question, **_):
        raise RateLimitError("Error code: 429")

    summary = run_eval(QUESTIONS, ["asof"], out_dir=tmp_path, ask=limited, now=NOW,
                       max_consecutive_rate_limits=2, log=lambda _: None)
    assert "rate-limit" in summary["incomplete"]
    assert summary["systems"]["asof"]["metrics"]["n_questions"] == 2
    assert "Incomplete run" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_cli_rejects_unknown_system(tmp_path, capsys):
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(q.model_dump(mode="json")) for q in QUESTIONS))
    try:
        main(["--systems", "asof,gpt", "--questions", str(qfile)])
    except SystemExit as exit_:
        assert exit_.code == 2
    assert "unknown system" in capsys.readouterr().err


def test_missing_gold_reports_uncovered_documents(tmp_path):
    meta = DocMeta(kind="notification", number="75/2019-Central Tax",
                   issue_date=date(2019, 12, 26), title="t", url="https://example.org/x.pdf")
    (tmp_path / "n.json").write_text(ParsedDoc(meta=meta, pages=[]).model_dump_json())
    (tmp_path / "junk.json").write_text("{}")
    assert missing_gold(QUESTIONS, tmp_path) == {}
    other = parse_questions([json.dumps({
        "id": "x", "question": "Composition limit in 2019?", "as_of": "2019-06-01",
        "type": "point_in_time", "gold_doc_numbers": ["14/2019-Central Tax"]})])
    assert missing_gold(other, tmp_path) == {"x": ["14/2019-Central Tax"]}
