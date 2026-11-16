"""Shared data contracts between ingestion, temporal, retrieval, generation and evaluation.

Every module imports these types instead of defining its own, so the pieces fit together.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

DocKind = Literal["act", "rules", "notification", "circular", "order", "other"]
RelationKind = Literal[
    "amends", "substitutes", "inserts", "omits", "rescinds", "supersedes", "clarifies"
]
Mode = Literal["asof", "baseline"]


# ---------------------------------------------------------------- ingestion

class DocMeta(BaseModel):
    """One source document as listed on the CBIC site."""

    kind: DocKind
    number: str  # canonical, e.g. "27/2022-Central Tax" or "183/15/2022-GST"
    issue_date: date
    title: str  # the "Subject" column
    url: str  # absolute URL of the English PDF
    pdf_path: str | None = None  # local path once downloaded (relative to repo root)

    @property
    def key(self) -> str:
        """Stable file-name-safe id, e.g. 'notification_27-2022-central-tax'."""
        slug = "".join(c if c.isalnum() else "-" for c in self.number.lower()).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return f"{self.kind}_{slug}"


class Page(BaseModel):
    page: int  # 1-based
    text: str


class ParsedDoc(BaseModel):
    """What ingestion writes to data/parsed/<key>.json."""

    meta: DocMeta
    pages: list[Page]
    ocr_used: bool = False


class ChunkIn(BaseModel):
    ord: int
    text: str
    section_ref: str | None = None  # e.g. "rule 36(4)" or "para 3"
    page: int | None = None


# ---------------------------------------------------------------- temporal

class ExtractedRelation(BaseModel):
    """A relation stated in a source document's text, as extracted by the LLM."""

    kind: RelationKind
    target_kind: Literal["notification", "circular", "rule", "section", "form", "other"]
    target_ref: str = Field(
        description="As written, normalised: e.g. '49/2019-Central Tax', 'rule 36(4)', "
        "'Circular No. 123/42/2019-GST'"
    )
    effective_date: date | None = Field(
        default=None, description="Date the change takes effect, if stated"
    )
    evidence: str = Field(description="Short verbatim quote from the document supporting this")
    confidence: float = Field(ge=0, le=1)


class ExtractionResult(BaseModel):
    relations: list[ExtractedRelation]
    effective_date: date | None = Field(
        default=None, description="Date the document itself comes into force, if stated"
    )


# ---------------------------------------------------------------- retrieval / generation

class QueryPlan(BaseModel):
    as_of: date
    as_of_explicit: bool  # False when we defaulted to today
    question_type: Literal["point_in_time", "current", "history", "other"]
    refs: list[str] = []  # explicit references, e.g. ["rule 36(4)", "49/2019-Central Tax"]
    search_queries: list[str] = []  # 1-3 rewritten queries for retrieval


class Hit(BaseModel):
    chunk_id: int
    doc_id: int
    kind: DocKind
    number: str | None
    issue_date: date | None
    title: str
    url: str | None
    section_ref: str | None
    page: int | None
    text: str
    valid_from: date | None = None
    valid_to: date | None = None  # None = still in force / unbounded
    score: float = 0.0


class Citation(BaseModel):
    source: int  # 1-based index into the hits shown to the LLM ([S1], [S2], ...)
    chunk_id: int
    quote: str
    verified: bool  # quote found (fuzzily) in that source's text


class Answer(BaseModel):
    text: str
    citations: list[Citation] = []
    abstained: bool = False
    as_of: date


class AskResult(BaseModel):
    question: str
    mode: Mode
    plan: QueryPlan
    hits: list[Hit]
    answer: Answer
    timings_ms: dict[str, float] = {}
