"""Document endpoints: upload, poll, read results."""

import json
import logging
import uuid
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session
from app.models import Clause, ClauseAnalysis, Document
from app.pipeline.run import clause_rows, process_document
from app.schemas import (
    ClauseDetail,
    ClauseSummary,
    DocumentCreated,
    DocumentStatus,
    DocumentSummary,
    TypeCount,
)
from app.taxonomy import DocStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

# Policy wordings run to a few hundred KB. 25MB is far above any legitimate
# document and well below anything that would exhaust memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _summary(clause: Clause, analysis: ClauseAnalysis) -> ClauseSummary:
    return ClauseSummary(
        id=clause.id,
        order_idx=clause.order_idx,
        number=clause.number,
        heading=clause.heading,
        section_path=clause.section_path,
        page=clause.page_start + 1,  # 1-indexed for humans
        clause_type=analysis.clause_type,
        plain_language=analysis.plain_language,
        what_it_means=analysis.what_it_means,
        impact_score=analysis.impact_score,
        buriedness=analysis.buriedness,
        position_signal=analysis.position_signal,
        reading_signal=analysis.reading_signal,
        crossref_signal=analysis.crossref_signal,
        jargon_signal=analysis.jargon_signal,
        reading_grade=analysis.reading_grade,
        triggers=json.loads(analysis.triggers_json),
        monetary_limits=json.loads(analysis.limits_json),
        time_windows=json.loads(analysis.time_windows_json),
    )


@router.post("/documents", response_model=DocumentCreated, status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile,
    session: Session = Depends(get_session),
):
    """Accept a policy PDF and start analysing it.

    Returns 202 Accepted, not 200: the work has been queued, not completed.
    Analysis takes minutes on a local model, so the client polls
    `/documents/{id}/status` and the response body deliberately contains no
    results yet.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds the 25MB limit")
    # Checked rather than trusting the extension: a mislabelled file should
    # fail here with a clear message, not deep inside PyMuPDF.
    if not payload.startswith(b"%PDF"):
        raise HTTPException(400, "That file is not a valid PDF")

    doc_id = uuid.uuid4().hex[:12]
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Stored under the generated id, never the user's filename, which could
    # contain path separators or collide with another upload.
    stored = upload_dir / f"{doc_id}.pdf"
    stored.write_bytes(payload)

    document = Document(
        id=doc_id,
        filename=file.filename,
        stored_path=str(stored),
        status=DocStatus.PENDING,
        stage_detail="Queued",
    )
    session.add(document)
    session.commit()

    background.add_task(process_document, doc_id, str(stored))
    return DocumentCreated(id=doc_id, filename=document.filename, status=document.status)


@router.get("/documents/{doc_id}/status", response_model=DocumentStatus)
def get_status(doc_id: str, session: Session = Depends(get_session)):
    document = session.get(Document, doc_id)
    if document is None:
        raise HTTPException(404, "No such document")
    return DocumentStatus(
        id=document.id,
        status=document.status,
        stage_detail=document.stage_detail,
        progress=round(document.progress, 3),
        page_count=document.page_count,
        clause_count=document.clause_count,
        error=document.error,
    )


@router.get("/documents/{doc_id}/summary", response_model=DocumentSummary)
def get_summary(
    doc_id: str,
    top: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_session),
):
    """The risk dashboard: what could get this policyholder's claim denied."""
    document = session.get(Document, doc_id)
    if document is None:
        raise HTTPException(404, "No such document")

    rows = clause_rows(session, doc_id)
    analysed = [(c, a) for c, a in rows if a is not None]
    counts = Counter(a.clause_type for _, a in analysed)

    top_risks = sorted(analysed, key=lambda r: r[1].impact_score, reverse=True)[:top]

    return DocumentSummary(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        clause_count=document.clause_count,
        type_counts=[
            TypeCount(clause_type=t, count=n) for t, n in counts.most_common()
        ],
        top_risks=[_summary(c, a) for c, a in top_risks],
        unanalysed_count=len(rows) - len(analysed),
    )


@router.get("/documents/{doc_id}/clauses", response_model=list[ClauseSummary])
def list_clauses(
    doc_id: str,
    clause_type: str | None = Query(None, description="Filter by clause type"),
    sort: str = Query("impact", pattern="^(impact|document)$"),
    session: Session = Depends(get_session),
):
    """All analysed clauses.

    Sorted by impact by default rather than document order: the point of the
    tool is that document order is exactly what hides the important clauses.
    """
    if session.get(Document, doc_id) is None:
        raise HTTPException(404, "No such document")

    rows = [(c, a) for c, a in clause_rows(session, doc_id) if a is not None]
    if clause_type:
        rows = [r for r in rows if r[1].clause_type == clause_type]

    if sort == "impact":
        rows.sort(key=lambda r: r[1].impact_score, reverse=True)
    else:
        rows.sort(key=lambda r: r[0].order_idx)

    return [_summary(c, a) for c, a in rows]


@router.get("/clauses/{clause_id}", response_model=ClauseDetail)
def get_clause(clause_id: str, session: Session = Depends(get_session)):
    """One clause, including the verbatim source text that justifies it.

    `source_text` is what makes the explanation checkable: the UI shows it
    beside the plain-language version so a reader can confirm for themselves
    that nothing was invented.
    """
    clause = session.get(Clause, clause_id)
    if clause is None:
        raise HTTPException(404, "No such clause")
    analysis = session.exec(
        select(ClauseAnalysis).where(ClauseAnalysis.clause_id == clause_id)
    ).first()
    if analysis is None:
        raise HTTPException(409, "This clause has not been analysed")

    base = _summary(clause, analysis)
    return ClauseDetail(
        **base.model_dump(),
        source_text=clause.text,
        char_start=clause.char_start,
        char_end=clause.char_end,
        page_start=clause.page_start,
        page_end=clause.page_end,
        likelihood=analysis.likelihood,
        severity=analysis.severity,
    )
