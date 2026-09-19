"""Pipeline orchestration: runs all four stages and persists the results.

This is the only place the four stages are wired together. Each stage stays a
pure function of its input - `ingest(path)`, `segment(result)`,
`analyze(segments)`, `score(segments, analyses)` - and none of them knows the
database exists. That is what lets the eval harness run the identical pipeline
with no web server and no database at all.

STATUS REPORTING
----------------
Analysis takes minutes on a local model, so progress is written to the
`Document` row as it goes and the UI polls for it. Each stage updates
`status`, `stage_detail` and `progress` before it begins, so a user watching
the screen sees which stage is running rather than an unbroken spinner.

Progress is weighted by how long each stage actually takes, not by stage count.
Ingest and segment finish in milliseconds; analysis is essentially all of the
wall time. Giving each of the four stages 25% would show a bar that leaps to
50% instantly and then appears frozen for two minutes.
"""

import json
import logging
import traceback

from sqlmodel import Session, delete, select

from app.db import engine
from app.models import Clause, ClauseAnalysis, Document
from app.pipeline.analyze import analyze
from app.pipeline.ingest import ingest
from app.pipeline.score import score
from app.pipeline.segment import segment
from app.taxonomy import DocStatus

log = logging.getLogger(__name__)

# Analysis dominates, so it owns the bulk of the progress bar.
_PROGRESS_SEGMENTED = 0.05
_PROGRESS_ANALYSIS_START = 0.10
_PROGRESS_ANALYSIS_END = 0.95


def _update(doc_id: str, **fields) -> None:
    """Write status fields in their own short-lived session.

    Deliberately not reusing a long-lived session held across the whole run:
    the polling endpoint reads this row from a different connection, and it can
    only observe changes that have actually been committed.
    """
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            return
        for key, value in fields.items():
            setattr(doc, key, value)
        session.add(doc)
        session.commit()


async def process_document(doc_id: str, pdf_path: str) -> None:
    """Run the full pipeline for one document, recording progress as it goes.

    Never raises. This runs as a background task with nobody waiting on it, so
    an escaping exception would vanish into the event loop and leave the
    document stuck reporting "analyzing" forever. Failures are recorded on the
    row instead, where the polling UI can surface them.
    """
    try:
        _update(doc_id, status=DocStatus.INGESTING,
                stage_detail="Reading the PDF", progress=0.01)
        ingested = ingest(pdf_path)

        _update(doc_id, status=DocStatus.SEGMENTING,
                stage_detail="Finding clause boundaries",
                progress=_PROGRESS_SEGMENTED,
                page_count=ingested.page_count,
                raw_text=ingested.raw_text)
        segments = segment(ingested)

        _persist_clauses(doc_id, segments)
        _update(doc_id, status=DocStatus.ANALYZING,
                stage_detail=f"Reading {len(segments)} clauses",
                progress=_PROGRESS_ANALYSIS_START,
                clause_count=len(segments))

        def on_progress(done: int, total: int) -> None:
            span = _PROGRESS_ANALYSIS_END - _PROGRESS_ANALYSIS_START
            _update(
                doc_id,
                stage_detail=f"Read {done} of {total} clauses",
                progress=_PROGRESS_ANALYSIS_START + span * (done / max(total, 1)),
            )

        analyses = await analyze(segments, progress=on_progress)

        _update(doc_id, status=DocStatus.SCORING,
                stage_detail="Ranking by claim-denial impact",
                progress=_PROGRESS_ANALYSIS_END)
        scored = score(segments, analyses)

        _persist_analyses(doc_id, segments, analyses, scored)

        analysed = len(analyses)
        detail = f"Analysed {analysed} of {len(segments)} clauses"
        if analysed < len(segments):
            # Stated plainly rather than hidden. An incomplete analysis means
            # the user is looking at a policy that may be riskier than shown.
            detail += " - some could not be read"
        _update(doc_id, status=DocStatus.READY, stage_detail=detail, progress=1.0)

    except Exception as exc:
        log.exception("pipeline failed for %s", doc_id)
        _update(
            doc_id,
            status=DocStatus.FAILED,
            stage_detail="Processing failed",
            error=f"{type(exc).__name__}: {exc}",
            progress=0.0,
        )
        log.debug(traceback.format_exc())


def _persist_clauses(doc_id: str, segments) -> None:
    """Store clause structure. Written before analysis begins.

    Splitting the write means a run that dies mid-analysis still leaves the
    parsed document behind, and re-analysis never has to re-parse.
    """
    with Session(engine) as session:
        # Idempotent: re-processing a document replaces its rows rather than
        # accumulating a second copy alongside the first.
        session.exec(delete(ClauseAnalysis).where(ClauseAnalysis.doc_id == doc_id))
        session.exec(delete(Clause).where(Clause.doc_id == doc_id))
        for seg in segments:
            session.add(
                Clause(
                    id=f"{doc_id}:{seg.order_idx}",
                    doc_id=doc_id,
                    order_idx=seg.order_idx,
                    section_path=seg.section_path,
                    number=seg.number,
                    heading=seg.heading,
                    text=seg.text,
                    page_start=seg.page_start,
                    page_end=seg.page_end,
                    char_start=seg.char_start,
                    char_end=seg.char_end,
                    bboxes_json=json.dumps(seg.bboxes),
                )
            )
        session.commit()


def _persist_analyses(doc_id: str, segments, analyses, scored) -> None:
    from app.config import settings
    from app.llm.prompts import PROMPT_VERSION

    with Session(engine) as session:
        for seg in segments:
            key = str(seg.order_idx)
            analysis = analyses.get(key)
            sc = scored.get(key)
            if analysis is None or sc is None:
                continue
            session.add(
                ClauseAnalysis(
                    clause_id=f"{doc_id}:{seg.order_idx}",
                    doc_id=doc_id,
                    clause_type=analysis.clause_type,
                    plain_language=analysis.plain_language,
                    what_it_means=analysis.what_it_means,
                    triggers_json=json.dumps(analysis.triggers),
                    limits_json=json.dumps(analysis.monetary_limits),
                    time_windows_json=json.dumps(analysis.time_windows),
                    waiting_periods_json=json.dumps(analysis.waiting_periods_days),
                    copay_percent=analysis.copay_percent,
                    copay_min_age_at_inception=analysis.copay_min_age_at_inception,
                    cap_percent_of_sum_insured=analysis.cap_percent_of_sum_insured,
                    icu_cap_percent_of_sum_insured=(
                        analysis.icu_cap_percent_of_sum_insured
                    ),
                    cap_max_inr_per_day=analysis.cap_max_inr_per_day,
                    icu_cap_max_inr_per_day=analysis.icu_cap_max_inr_per_day,
                    cover_window_days=analysis.cover_window_days,
                    cover_window_anchor=analysis.cover_window_anchor,
                    exceptions_json=json.dumps(analysis.exceptions),
                    likelihood=analysis.likelihood,
                    severity=analysis.severity,
                    buriedness=sc.buriedness,
                    impact_score=sc.impact_score,
                    reading_grade=sc.reading_grade,
                    position_signal=sc.position_signal,
                    reading_signal=sc.reading_signal,
                    crossref_signal=sc.crossref_signal,
                    jargon_signal=sc.jargon_signal,
                    # Recorded per row so a document analysed under an older
                    # prompt is identifiable without guesswork.
                    model=settings.model,
                    prompt_version=PROMPT_VERSION,
                )
            )
        session.commit()


def clause_rows(session: Session, doc_id: str):
    """Clauses joined with their analyses, ordered by impact.

    An outer join, not an inner one: a clause the model failed on still exists
    in the document and must remain visible. Dropping it would quietly present
    a more complete-looking policy than was actually analysed.
    """
    statement = (
        select(Clause, ClauseAnalysis)
        .join(ClauseAnalysis, ClauseAnalysis.clause_id == Clause.id, isouter=True)
        .where(Clause.doc_id == doc_id)
    )
    return session.exec(statement).all()
