"""Evaluation questions: the `eval/questions.jsonl` format, its validation and a loader.

One JSON object per line:

    {"id": "pit-36-4-feb-2020",
     "question": "What was the provisional ITC limit under rule 36(4) in February 2020?",
     "as_of": "2020-02-15",
     "type": "point_in_time",
     "gold_doc_numbers": ["75/2019-Central Tax"],
     "must_contain": ["10 per cent"],
     "must_not_contain": ["5 per cent"],
     "notes": "..."}
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

QuestionType = Literal[
    "point_in_time", "current", "history", "supersession_trap", "unanswerable", "lookup"
]
QUESTION_TYPES: tuple[str, ...] = (
    "point_in_time", "current", "history", "supersession_trap", "unanswerable", "lookup"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class QuestionFileError(ValueError):
    """Raised with every problem found in a questions file, one per line."""


def _clean_list(values: list[str]) -> list[str]:
    out = []
    for v in values:
        v = " ".join(v.split())
        if not v:
            raise ValueError("entries must be non-empty strings")
        out.append(v)
    return out


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str = Field(min_length=5)
    as_of: date | None = None  # None = "today" (the system's own date)
    type: QuestionType
    gold_doc_numbers: list[str] = []  # canonical numbers, e.g. "94/2020-Central Tax"
    must_contain: list[str] = []  # key facts a correct answer states
    must_not_contain: list[str] = []  # stale / wrong facts a correct answer must not state
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("id may contain only letters, digits and _ . : -")
        return v

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        return " ".join(v.split())

    @field_validator("gold_doc_numbers", "must_contain", "must_not_contain")
    @classmethod
    def _lists(cls, v: list[str]) -> list[str]:
        return _clean_list(v)

    @model_validator(mode="after")
    def _consistent(self) -> "EvalQuestion":
        from asof.evaluation.metrics import normalise  # local: metrics imports nothing from here

        if self.type == "unanswerable":
            if self.must_contain:
                raise ValueError("unanswerable questions must not have must_contain facts")
        elif not (self.gold_doc_numbers or self.must_contain):
            raise ValueError(f"{self.type} questions need gold_doc_numbers or must_contain")
        if self.type in ("point_in_time", "supersession_trap") and self.as_of is None:
            raise ValueError(f"{self.type} questions need an explicit as_of date")
        if self.type == "current" and self.as_of is not None:
            raise ValueError("current questions must have as_of null (they are asked as of today)")
        overlap = {normalise(f) for f in self.must_contain} & {
            normalise(f) for f in self.must_not_contain
        }
        if overlap:
            raise ValueError(f"facts in both must_contain and must_not_contain: {sorted(overlap)}")
        return self

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def _short(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        parts.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return "; ".join(parts)


def parse_questions(lines: list[str], source: str = "<questions>") -> list[EvalQuestion]:
    """Parse JSONL lines. Blank lines are skipped. All problems are reported together."""
    questions: list[EvalQuestion] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as err:
            errors.append(f"{source}:{lineno}: invalid JSON: {err.msg}")
            continue
        try:
            q = EvalQuestion.model_validate(obj)
        except ValidationError as err:
            errors.append(f"{source}:{lineno}: {_short(err)}")
            continue
        if q.id in seen:
            errors.append(f"{source}:{lineno}: duplicate id {q.id!r} (first on line {seen[q.id]})")
            continue
        seen[q.id] = lineno
        questions.append(q)
    if errors:
        raise QuestionFileError("\n".join(errors))
    return questions


def load_questions(path: str | Path) -> list[EvalQuestion]:
    path = Path(path)
    return parse_questions(path.read_text(encoding="utf-8").splitlines(), source=str(path))


def dump_question(q: EvalQuestion) -> str:
    return json.dumps(q.model_dump(mode="json"), ensure_ascii=False)
