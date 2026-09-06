"""API response shapes.

Kept separate from the SQLModel tables in `models.py` on purpose. The tables are
storage; these are the contract with the frontend. Returning ORM rows directly
would mean any column added for internal bookkeeping silently becomes part of
the public API, and any column rename becomes a breaking change for the UI.

The split also lets a response carry things no single table holds - a clause
response merges `Clause` (source text, offsets) with `ClauseAnalysis` (plain
language, score), which is how the UI actually wants to consume it.
"""

from pydantic import BaseModel, Field


class DocumentCreated(BaseModel):
    id: str
    filename: str
    status: str


class DocumentStatus(BaseModel):
    """Polled by the UI while the pipeline runs.

    Analysis takes minutes on a local model, so this reports which stage is
    running and how far through it is. A bare "processing" flag would leave the
    user staring at a spinner with no idea whether anything is happening.
    """

    id: str
    status: str
    stage_detail: str
    progress: float = Field(ge=0.0, le=1.0)
    page_count: int
    clause_count: int
    error: str | None = None


class ClauseSummary(BaseModel):
    """A clause as it appears in a list or a risk card."""

    id: str
    order_idx: int
    number: str
    heading: str
    section_path: str
    page: int

    clause_type: str
    plain_language: str
    what_it_means: str
    impact_score: float
    buriedness: float
    # Why this clause is easy to miss, component by component (each 0-1).
    # Surfaced on the risk card, not just stored: the product's whole promise
    # is showing the reader what the document obscured, so the reasons have to
    # be visible rather than compressed into one opaque number.
    position_signal: float = 0.0
    reading_signal: float = 0.0
    crossref_signal: float = 0.0
    jargon_signal: float = 0.0
    reading_grade: float = 0.0

    triggers: list[str] = []
    monetary_limits: list[str] = []
    time_windows: list[str] = []


class ClauseDetail(ClauseSummary):
    """Everything in the summary, plus the evidence.

    `source_text` is the verbatim slice of the document, and `char_start`/
    `char_end` are its offsets. Together they are what lets the UI show the
    original wording beside the plain-language version - the grounding claim
    this whole project rests on, made visible to the reader.
    """

    source_text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int

    likelihood: int
    severity: int


class TypeCount(BaseModel):
    clause_type: str
    count: int


class DocumentSummary(BaseModel):
    """The risk dashboard's payload."""

    id: str
    filename: str
    status: str
    page_count: int
    clause_count: int
    type_counts: list[TypeCount]
    # The headline: the clauses most likely to cost the policyholder money.
    top_risks: list[ClauseSummary]
    # Surfaced rather than hidden. If a clause could not be analysed the user
    # is seeing an incomplete picture, and has a right to know that.
    unanalysed_count: int


class ScenarioRequest(BaseModel):
    scenario: str = Field(min_length=8, max_length=2000)


class CitationOut(BaseModel):
    """One clause the answer rests on."""

    clause_id: str
    clause_db_id: str | None = None
    number: str = ""
    heading: str = ""
    page: int = 0
    effect: str
    quote: str
    # False when the quoted text could not be found in the cited clause. The UI
    # must show such a citation as unverified rather than as evidence.
    verified: bool = True
    unverified_reason: str = ""


class ScenarioResponse(BaseModel):
    id: str
    scenario: str
    verdict: str
    reasoning: str
    citations: list[CitationOut] = []
    # Facts the description never supplied, so an insufficient_information
    # answer can say what it would need instead of just refusing.
    missing_facts: list[str] = []
    facts: dict = {}
    # False if ANY citation failed verification. The whole answer is presented
    # as unverified in that case: a reader cannot be expected to work out which
    # half of an explanation was the sound one.
    verified: bool = True
    clauses_considered: int = 0
