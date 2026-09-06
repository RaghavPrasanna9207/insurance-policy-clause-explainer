"""M3 gate: the REST API.

Split deliberately into two kinds of test:

  Fast (no model)  - routing, filtering, sorting, validation, error handling,
                     against a directly seeded database. Milliseconds.
  End-to-end (llm) - one real upload through all four stages on a 4-clause
                     policy. Marked `llm` so it can be skipped.

The split matters. The fast tests assert what the API does; making them depend
on the model would make them slow, non-deterministic, and would mean a change
in model output could fail a test about URL routing.
"""

from pathlib import Path

import pytest

from app.taxonomy import ClauseType, DocStatus

# --- upload validation ------------------------------------------------------


def test_rejects_non_pdf_extension(api_client):
    response = api_client.post(
        "/documents", files={"file": ("policy.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_rejects_a_file_that_is_not_really_a_pdf(api_client):
    """Extension is a claim; the magic bytes are evidence.

    A file renamed to .pdf must fail here with a clear message, rather than
    deep inside PyMuPDF where the error is unrecognisable to a user.
    """
    response = api_client.post(
        "/documents",
        files={"file": ("policy.pdf", b"I am not a PDF at all", "application/pdf")},
    )
    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_rejects_an_empty_file(api_client):
    response = api_client.post(
        "/documents", files={"file": ("policy.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400


# --- status -----------------------------------------------------------------


def test_status_404_for_unknown_document(api_client):
    assert api_client.get("/documents/nosuchdoc/status").status_code == 404


def test_status_reports_stage_and_progress(api_client, seeded_document):
    body = api_client.get(f"/documents/{seeded_document}/status").json()

    assert body["status"] == DocStatus.READY
    assert body["progress"] == 1.0
    assert body["clause_count"] == 4
    assert body["error"] is None
    # A human-readable stage, not just a machine status. The UI shows this
    # during a multi-minute wait.
    assert body["stage_detail"]


# --- summary ----------------------------------------------------------------


def test_summary_ranks_by_impact_not_document_order(api_client, seeded_document):
    """The core product behaviour.

    Document order is exactly what hides the dangerous clauses, so the summary
    must not reproduce it. In the seeded set, the exclusion (impact 66.4) sits
    third in the document but must come first here.
    """
    body = api_client.get(f"/documents/{seeded_document}/summary").json()
    scores = [c["impact_score"] for c in body["top_risks"]]

    assert scores == sorted(scores, reverse=True)
    assert body["top_risks"][0]["clause_type"] == ClauseType.EXCLUSION
    assert body["top_risks"][0]["order_idx"] == 2


def test_summary_counts_clause_types(api_client, seeded_document):
    body = api_client.get(f"/documents/{seeded_document}/summary").json()
    counts = {tc["clause_type"]: tc["count"] for tc in body["type_counts"]}

    assert counts == {
        ClauseType.COVERAGE: 1,
        ClauseType.WAITING_PERIOD: 1,
        ClauseType.EXCLUSION: 1,
        ClauseType.SUB_LIMIT: 1,
    }


def test_summary_respects_the_top_parameter(api_client, seeded_document):
    body = api_client.get(f"/documents/{seeded_document}/summary?top=2").json()
    assert len(body["top_risks"]) == 2


def test_summary_reports_unanalysed_clauses(api_client, seeded_document):
    """An unanalysed clause must be counted, not quietly omitted.

    If the model failed on a clause the user is looking at an incomplete
    picture of their policy, and has a right to know that.
    """
    from sqlmodel import Session

    from app.db import engine
    from app.models import Clause

    with Session(engine) as session:
        session.add(Clause(
            id=f"{seeded_document}:99", doc_id=seeded_document, order_idx=99,
            number="9.9", heading="9.9 Unread", section_path="SECTION 9",
            text="Something the model could not read.",
            page_start=1, page_end=1, char_start=0, char_end=10,
        ))
        session.commit()

    body = api_client.get(f"/documents/{seeded_document}/summary").json()
    assert body["unanalysed_count"] == 1
    # ...and it must not appear among the analysed results.
    assert all(c["order_idx"] != 99 for c in body["top_risks"])


# --- clause listing ---------------------------------------------------------


def test_clauses_default_to_impact_order(api_client, seeded_document):
    rows = api_client.get(f"/documents/{seeded_document}/clauses").json()
    scores = [c["impact_score"] for c in rows]

    assert len(rows) == 4
    assert scores == sorted(scores, reverse=True)


def test_clauses_can_be_read_in_document_order(api_client, seeded_document):
    rows = api_client.get(f"/documents/{seeded_document}/clauses?sort=document").json()
    assert [c["order_idx"] for c in rows] == [0, 1, 2, 3]


def test_clauses_filter_by_type(api_client, seeded_document):
    rows = api_client.get(
        f"/documents/{seeded_document}/clauses?clause_type={ClauseType.EXCLUSION}"
    ).json()

    assert len(rows) == 1
    assert rows[0]["clause_type"] == ClauseType.EXCLUSION


def test_invalid_sort_is_rejected(api_client, seeded_document):
    """Validated by the route signature's regex, so bad input fails at the
    boundary rather than silently falling through to a default."""
    response = api_client.get(f"/documents/{seeded_document}/clauses?sort=sideways")
    assert response.status_code == 422


def test_clauses_404_for_unknown_document(api_client):
    assert api_client.get("/documents/nosuchdoc/clauses").status_code == 404


# --- clause detail ----------------------------------------------------------


def test_clause_detail_returns_verbatim_source(api_client, seeded_document):
    """The grounding claim, exposed through the API.

    The detail endpoint returns the original wording alongside the plain
    language version, so a reader can check for themselves that nothing was
    invented. Without `source_text` the UI could only ask to be trusted.
    """
    body = api_client.get(f"/clauses/{seeded_document}:2").json()

    assert body["source_text"] == "The Company shall not be liable for cosmetic or plastic surgery."
    assert body["clause_type"] == ClauseType.EXCLUSION
    assert body["char_start"] == 0
    assert body["char_end"] == len(body["source_text"])
    assert body["likelihood"] == 3
    assert body["severity"] == 4


def test_clause_detail_404_for_unknown_clause(api_client, seeded_document):
    assert api_client.get("/clauses/nope:1").status_code == 404


def test_every_listed_clause_id_resolves(api_client, seeded_document):
    """No dangling ids.

    The UI navigates from the list to the detail view by id. An id that lists
    but does not resolve is a broken link in the middle of the product.
    """
    rows = api_client.get(f"/documents/{seeded_document}/clauses").json()
    assert rows

    for row in rows:
        assert api_client.get(f"/clauses/{row['id']}").status_code == 200


# --- end to end -------------------------------------------------------------


@pytest.mark.llm
def test_upload_runs_the_whole_pipeline(api_client, mini_pdf: Path):
    """One real document, all four stages, through the HTTP surface.

    TestClient runs background tasks synchronously once the response is
    returned, so by the time the POST completes the pipeline has finished.
    That is convenient here, and worth remembering: in production the same
    call returns immediately and the client polls.
    """
    response = api_client.post(
        "/documents",
        files={"file": ("mini.pdf", mini_pdf.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    doc_id = response.json()["id"]

    status = api_client.get(f"/documents/{doc_id}/status").json()
    assert status["status"] == DocStatus.READY, status
    assert status["error"] is None
    assert status["progress"] == 1.0
    assert status["page_count"] >= 1
    assert status["clause_count"] >= 4

    rows = api_client.get(f"/documents/{doc_id}/clauses").json()
    assert len(rows) >= 4

    for row in rows:
        assert row["clause_type"] in ClauseType.values()
        assert row["plain_language"].strip()
        assert 0.0 <= row["impact_score"] <= 100.0

    # Every clause's stored text must still be a byte-exact slice of the
    # document, all the way through the pipeline and out of the HTTP layer.
    # This is the M1 offset invariant, verified end to end.
    from sqlmodel import Session

    from app.db import engine
    from app.models import Document

    with Session(engine) as session:
        raw = session.get(Document, doc_id).raw_text

    for row in rows:
        detail = api_client.get(f"/clauses/{row['id']}").json()
        assert raw[detail["char_start"]:detail["char_end"]] == detail["source_text"]


@pytest.mark.llm
def test_a_corrupt_pdf_fails_the_document_not_the_server(api_client):
    """A file that passes the header check but is unreadable must be recorded
    as a failed document, not raise out of a background task where nobody is
    listening and leave the document stuck on 'analyzing' forever.
    """
    response = api_client.post(
        "/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4\nthen nothing but garbage", "application/pdf")},
    )
    assert response.status_code == 202
    doc_id = response.json()["id"]

    status = api_client.get(f"/documents/{doc_id}/status").json()
    assert status["status"] == DocStatus.FAILED
    assert status["error"]


# --- schema drift -----------------------------------------------------------


def test_schema_drift_is_detected(tmp_path, monkeypatch):
    """Regression test for a bug the rest of this file could not have caught.

    `create_all` only creates tables that do not exist; it never alters one. So
    adding a column to a model leaves an older database silently stale, and the
    mismatch surfaces much later as an OperationalError thrown from inside a
    background task, where the message reaches nobody useful.

    Every other test here starts from `drop_all`, so they always run against a
    freshly built schema and never against one that has aged - which is exactly
    why this went unnoticed until a live server hit it. A test suite that
    rebuilds the world every time is blind to a whole class of bug.
    """
    import sqlite3

    from sqlalchemy import create_engine

    import app.db as db
    from app.db import SchemaOutOfDate, check_schema

    # A database built the way an older version of the app would have built it:
    # the clause table, minus a column added later.
    path = tmp_path / "stale.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE clause (id TEXT PRIMARY KEY, doc_id TEXT, order_idx INTEGER, "
        "section_path TEXT, heading TEXT, text TEXT, page_start INTEGER, "
        "page_end INTEGER, char_start INTEGER, char_end INTEGER, bboxes_json TEXT)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "engine", create_engine(f"sqlite:///{path}"))

    with pytest.raises(SchemaOutOfDate) as excinfo:
        check_schema()

    message = str(excinfo.value)
    assert "number" in message, "the missing column must be named"
    assert "clause" in message
    # The message has to tell the reader what to actually do about it.
    assert "Delete" in message


# --- scenario endpoint ------------------------------------------------------


def test_scenario_404_for_unknown_document(api_client):
    response = api_client.post(
        "/documents/nosuchdoc/scenarios", json={"scenario": "I had knee surgery last year."}
    )
    assert response.status_code == 404


def test_scenario_rejects_a_document_still_being_analysed(api_client, seeded_document):
    """409, not a partial answer.

    Reasoning over a half-analysed policy would consider a subset of the
    clauses while appearing to consider all of them - the same silent-omission
    failure the whole shortlist design exists to avoid.
    """
    from sqlmodel import Session

    from app.db import engine
    from app.models import Document
    from app.taxonomy import DocStatus

    with Session(engine) as session:
        doc = session.get(Document, seeded_document)
        doc.status = DocStatus.ANALYZING
        session.add(doc)
        session.commit()

    response = api_client.post(
        f"/documents/{seeded_document}/scenarios",
        json={"scenario": "I had knee surgery last year."},
    )
    assert response.status_code == 409


def test_scenario_rejects_an_empty_question(api_client, seeded_document):
    """Validated at the boundary by the request model's min_length."""
    response = api_client.post(
        f"/documents/{seeded_document}/scenarios", json={"scenario": "hi"}
    )
    assert response.status_code == 422


@pytest.mark.llm
def test_scenario_end_to_end_is_grounded(api_client, mini_pdf: Path):
    """One real question, all the way through, on the 4-clause policy.

    The assertion that matters is the last one: every quotation the answer
    relies on must be findable in the clause it was attributed to. That is a
    correctness property, not a quality one - an answer quoting text that is
    not in the policy is fabricated, however sensible it reads.
    """
    from app.grounding import verify_quote
    from app.taxonomy import Verdict

    upload = api_client.post(
        "/documents",
        files={"file": ("mini.pdf", mini_pdf.read_bytes(), "application/pdf")},
    )
    doc_id = upload.json()["id"]
    assert api_client.get(f"/documents/{doc_id}/status").json()["status"] == "ready"

    response = api_client.post(
        f"/documents/{doc_id}/scenarios",
        json={
            "scenario": "I had a nose job because I did not like how it looked. "
            "I have held the policy for six years."
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["verdict"] in Verdict.values()
    assert body["reasoning"].strip()
    assert body["clauses_considered"] >= 4

    # Every citation must point at a clause of this document and quote it
    # accurately. Verified independently here rather than trusting the
    # `verified` flag the API set.
    clauses = api_client.get(f"/documents/{doc_id}/clauses").json()
    text_by_number = {c["number"]: c for c in clauses if c["number"]}

    for citation in body["citations"]:
        assert citation["clause_id"] in text_by_number, "cited a clause not in this policy"
        detail = api_client.get(f"/clauses/{text_by_number[citation['clause_id']]['id']}").json()
        verified, reason = verify_quote(citation["quote"], detail["source_text"])
        assert verified == citation["verified"], (
            f"API said verified={citation['verified']} but re-checking says {verified}: {reason}"
        )
