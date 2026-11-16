"""Question file format and validation."""

import json
from pathlib import Path

import pytest

from asof.evaluation.questions import (
    EvalQuestion,
    QuestionFileError,
    load_questions,
    parse_questions,
)

REPO = Path(__file__).resolve().parents[1]


def line(**overrides) -> str:
    q = {
        "id": "pit-1", "question": "What was the rule 36(4) cap in February 2020?",
        "as_of": "2020-02-15", "type": "point_in_time",
        "gold_doc_numbers": ["75/2019-Central Tax"], "must_contain": ["10 per cent"],
        "must_not_contain": ["5 per cent"], "notes": "",
    }
    q.update(overrides)
    return json.dumps(q)


def test_valid_question_parses_and_cleans_whitespace():
    [q] = parse_questions([line(question="  What was   the cap?  ", must_contain=[" 10  %"])])
    assert q.question == "What was the cap?"
    assert q.must_contain == ["10 %"]
    assert q.answerable


@pytest.mark.parametrize("overrides, message", [
    ({"type": "nonsense"}, "type"),
    ({"as_of": "15/02/2020"}, "as_of"),
    ({"as_of": None}, "explicit as_of"),
    ({"type": "current"}, "as_of null"),
    ({"type": "unanswerable"}, "must not have must_contain"),
    ({"gold_doc_numbers": [], "must_contain": []}, "need gold_doc_numbers"),
    ({"must_not_contain": ["ten per cent"]}, "both must_contain and must_not_contain"),
    ({"id": "has space"}, "id"),
    ({"extra_field": 1}, "extra"),
    ({"must_contain": [""]}, "non-empty"),
])
def test_invalid_questions_are_rejected(overrides, message):
    with pytest.raises(QuestionFileError) as err:
        parse_questions([line(**overrides)], source="q.jsonl")
    assert "q.jsonl:1:" in str(err.value)
    assert message in str(err.value)


def test_all_errors_reported_with_line_numbers():
    lines = [line(), "", "{not json", line(), line(id="ok-2", type="bogus")]
    with pytest.raises(QuestionFileError) as err:
        parse_questions(lines, source="q.jsonl")
    text = str(err.value)
    assert "q.jsonl:3: invalid JSON" in text
    assert "q.jsonl:4: duplicate id 'pit-1' (first on line 1)" in text
    assert "q.jsonl:5:" in text


def test_unanswerable_and_current_shapes():
    [u] = parse_questions([line(id="u", type="unanswerable", gold_doc_numbers=[],
                                must_contain=[], must_not_contain=[])])
    assert not u.answerable
    [c] = parse_questions([line(id="c", type="current", as_of=None)])
    assert c.as_of is None


def test_repo_question_set_is_valid():
    questions = load_questions(REPO / "eval" / "questions.jsonl")
    assert len(questions) >= 10
    assert {q.type for q in questions} >= {"point_in_time", "supersession_trap", "unanswerable"}
    assert all(isinstance(q, EvalQuestion) for q in questions)
