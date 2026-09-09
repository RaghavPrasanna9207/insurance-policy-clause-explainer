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
    # The clause's own number ("4.2"). This is the stable human-facing
    # identifier; `heading` is optional decoration many policies omit entirely.
    number: str = ""
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
    # Structured, so the scenario simulator can compare rather than re-read.
    waiting_period_days: int | None = None
    exceptions_json: str = "[]"
    # The operands of a reduction, likewise structured so stage 5 can compare
    # rather than re-read. See app/pipeline/reduction.py.
    copay_percent: int | None = None
    copay_min_age_at_inception: int | None = None
    cap_percent_of_sum_insured: int | None = None
    icu_cap_percent_of_sum_insured: int | None = None

    # The model's two judgments (1-5), the only LLM input to the score.
    likelihood: int
    severity: int
    # Computed, never guessed - see pipeline/score.py.
    buriedness: float = 0.0
    impact_score: float = Field(default=0.0, index=True)
    # Flesch-Kincaid grade of the ORIGINAL clause text. Stored so the UI can
    # justify a ranking ("reads at university level") without recomputing it.
    reading_grade: float = 0.0
    # The four buriedness components, stored separately rather than only as
    # their blended total. The UI's "Why you'd miss this" line names the actual
    # reasons a clause is easy to overlook - buried on page 34, reads at grade
    # 17, points at three other clauses - and a single combined number cannot
    # say which of those applied. Keeping the parts is what makes the score
    # explainable instead of an oracle.
    position_signal: float = 0.0
    reading_signal: float = 0.0
    crossref_signal: float = 0.0
    jargon_signal: float = 0.0

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
    # Full citations, not just clause ids: each carries the quoted text, how the
    # clause bears on the outcome, and whether the quote passed the verbatim
    # check. An id-only column would lose exactly the evidence that makes the
    # answer checkable.
    citations_json: str = "[]"
    # Facts the person never stated. Shown in the UI so an
    # insufficient_information verdict can say WHAT it would need, rather than
    # being a dead end.
    missing_facts_json: str = "[]"
    clauses_considered: int = 0
    # False when a quoted span failed the verbatim check. The UI must show such
    # an answer as unverified rather than presenting it as established fact.
    verified: bool = True
    created_at: datetime = Field(default_factory=_now)
