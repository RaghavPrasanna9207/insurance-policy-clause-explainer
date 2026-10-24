"""The clause list the scenario step reasons over, loaded from the database.

The scenario endpoint and the eval both build that list through
`load_clauses`, so what the eval measures is what the endpoint ships.
"""

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401  - registers the tables
from app.models import Clause, ClauseAnalysis, Document
from app.pipeline.run import load_clauses
from app.taxonomy import DocStatus


def _clause(idx: int, number: str, page_start: int) -> Clause:
    return Clause(
        id=f"doc:{idx}", doc_id="doc", order_idx=idx, number=number,
        heading=f"{number} Heading", section_path="SECTION 3", text=f"Clause {idx}.",
        page_start=page_start, page_end=page_start, char_start=0, char_end=9,
    )


def _analysis(idx: int) -> ClauseAnalysis:
    return ClauseAnalysis(
        clause_id=f"doc:{idx}", doc_id="doc", clause_type="waiting_period",
        plain_language="", what_it_means="", triggers_json="[]", limits_json="[]",
        time_windows_json="[]", waiting_periods_json="[30, 730]",
        exceptions_json='["accident"]', copay_percent=20,
        likelihood=3, severity=4, buriedness=0.4, impact_score=50.0,
        reading_grade=12.0, model="test", prompt_version="test",
    )


def test_load_clauses_gives_the_scenario_step_what_the_database_holds():
    # In memory, through the same engine seam the eval uses.
    engine = create_engine("sqlite://", poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Document(id="doc", filename="p.pdf", stored_path="p.pdf", status=DocStatus.READY))
        # Added out of document order: the loader, not the insert, sets order.
        session.add(_clause(2, "11", page_start=4))  # never analysed
        session.add(_clause(1, "10", page_start=3))
        session.add(_clause(0, "10", page_start=2))
        session.add(_analysis(1))
        session.add(_analysis(0))
        session.commit()

        clauses = load_clauses(session, "doc")

    # A repeated number is made unique; the unanalysed clause is left out.
    assert [c.clause_id for c in clauses] == ["10", "10#2"]
    first = clauses[0]
    assert first.db_id == "doc:0"
    assert first.heading == "10 Heading"
    assert first.page == 3  # stored 0-based, shown 1-based
    # JSON columns come back as lists, not strings.
    assert first.waiting_periods_days == [30, 730]
    assert first.exceptions == ["accident"]
    assert first.copay_percent == 20
    assert first.section_path == "SECTION 3"


def test_a_group_heading_is_never_given_to_the_scenario_step():
    """HDFC's "1. Waiting Periods", with the section's lead-in ("shall not make
    payment for any claim ... caused by any of the following:"), is a heading
    for the items below it, not a rule. Shown to the model, it refused typhoid
    at 45 days under the pre-existing-disease wait (Failure 116); joined to the
    first item instead, it refused five covered claims (Failure 117). It stays
    in the clause map, and is left out here. Citation ids are counted with it,
    so the clause after it keeps the id it always had."""
    engine = create_engine("sqlite://", poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Document(id="doc", filename="p.pdf", stored_path="p.pdf", status=DocStatus.READY))
        heading = _clause(0, "10", page_start=1)
        heading.is_heading = True
        session.add(heading)
        session.add(_clause(1, "10", page_start=1))
        session.add(_analysis(0))
        session.add(_analysis(1))
        session.commit()

        clauses = load_clauses(session, "doc")

    assert [c.clause_id for c in clauses] == ["10#2"]
