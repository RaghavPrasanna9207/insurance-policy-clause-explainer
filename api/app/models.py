"""Database tables (SQLModel).

SQLModel is one class serving two roles: a SQLAlchemy table AND a Pydantic
model. That means FastAPI can validate and serialise these directly, with no
parallel set of "API schema" classes to keep in sync.

Design note - why Clause and ClauseAnalysis are separate tables:
parsing a PDF is deterministic and slow-ish; analysing it is LLM-driven and
something we will re-run many times while tuning prompts. Keeping them apart
means re-analysis never risks corrupting the parsed structure, and a document
can hold results from two prompt versions side by side for comparison.
"""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

from app.taxonomy import DocStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    stored_path: str
    status: str = Field(default=DocStatus.PENDING, index=True)
    # Human-readable note about the current stage, surfaced in the polling UI.
    # A 2-5 minute wait needs honest progress, not an indefinite spinner.
    stage_detail: str = ""
    progress: float = 0.0  # 0.0 - 1.0
    error: str | None = None

    page_count: int = 0
    clause_count: int = 0
    insurer: str | None = None
    policy_type: str | None = None
    # Full extracted text. Every clause's char offsets index into THIS string,
    # so it is the single source of truth for "what the document actually says".
    raw_text: str = ""

    created_at: datetime = Field(default_factory=_now)


class Clause(SQLModel, table=True):
    id: str = Field(primary_key=True)
    doc_id: str = Field(foreign_key="document.id", index=True)
    order_idx: int

    # e.g. "4 Exclusions > 4.2 Permanent Exclusions" - powers breadcrumbs and
    # gives the model useful context about where a clause sits.
    section_path: str = ""
    heading: str = ""
    text: str

    page_start: int
    page_end: int
    # Offsets into Document.raw_text. The M1 test asserts these slice back
    # byte-identical, which is what makes every citation verifiable.
    char_start: int
    char_end: int
    # PyMuPDF span rectangles, stored as JSON. Unused by v1's side-by-side UI,
    # captured now so the PDF highlighter is a later drop-in with no migration.
    bboxes_json: str = "[]"


class ClauseAnalysis(SQLModel, table=True):
    clause_id: str = Field(primary_key=True, foreign_key="clause.id")
    doc_id: str = Field(foreign_key="document.id", index=True)

    clause_type: str = Field(index=True)
    plain_language: str
    what_it_means: str
    triggers_json: str = "[]"
    limits_json: str = "{}"
    time_windows_json: str = "{}"

    # The model's two judgments (1-5), the only LLM input to the score.
    likelihood: int
    severity: int
    # Computed, never guessed - see pipeline/score.py.
    buriedness: float = 0.0
    impact_score: float = Field(default=0.0, index=True)

    model: str
    prompt_version: str
    created_at: datetime = Field(default_factory=_now)


class ScenarioRun(SQLModel, table=True):
    id: str = Field(primary_key=True)
    doc_id: str = Field(foreign_key="document.id", index=True)
    scenario_text: str
    facts_json: str = "{}"
    verdict: str
    rationale: str
    cited_clause_ids_json: str = "[]"
    # False when a quoted span failed the verbatim check. The UI must show such
    # an answer as unverified rather than presenting it as established fact.
    verified: bool = True
    created_at: datetime = Field(default_factory=_now)
