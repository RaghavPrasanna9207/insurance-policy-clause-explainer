"""Scenario endpoint: ask a what-if question about one policy."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models import Document, ScenarioRun
from app.pipeline.run import clause_rows
from app.pipeline.scenario import ShortlistClause, run_scenario
from app.schemas import CitationOut, ScenarioRequest, ScenarioResponse
from app.taxonomy import DocStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["scenarios"])


@router.post("/documents/{doc_id}/scenarios", response_model=ScenarioResponse)
async def create_scenario(
    doc_id: str,
    request: ScenarioRequest,
    session: Session = Depends(get_session),
):
    """Answer one what-if question against this policy's clauses.

    Synchronous, unlike document upload. This is two model calls over a prompt
    the size of the policy - tens of seconds, not minutes - so the client can
    wait. Adding a second polling flow for it would be machinery without a
    reader.
    """
    document = session.get(Document, doc_id)
    if document is None:
        raise HTTPException(404, "No such document")
    if document.status != DocStatus.READY:
        # Answering from a half-analysed policy would mean reasoning over a
        # subset of the clauses while appearing to consider all of them.
        raise HTTPException(409, "This policy has not finished being analysed")

    rows = [(c, a) for c, a in clause_rows(session, doc_id) if a is not None]
    if not rows:
        raise HTTPException(409, "This policy has no analysed clauses")

    # The citation id the model sees is the policy's OWN clause number, so the
    # id it cites and the number printed inside the clause text are the same
    # string. `ref` carries the database id so the answer can be mapped back to
    # a real row for the page number and heading.
    #
    # Numbers are made unique defensively: a malformed document could repeat
    # one, and a duplicate id would make a citation ambiguous.
    seen: dict[str, int] = {}
    shortlist_clauses: list[ShortlistClause] = []
    meta: dict[str, tuple] = {}
    for clause, analysis in rows:
        base = clause.number or f"c{clause.order_idx}"
        seen[base] = seen.get(base, 0) + 1
        citation_id = base if seen[base] == 1 else f"{base}#{seen[base]}"
        shortlist_clauses.append(
            ShortlistClause(
                clause_id=citation_id,
                ref=clause.id,
                number=clause.number,
                clause_type=analysis.clause_type,
                text=clause.text,
                impact_score=analysis.impact_score,
                waiting_period_days=analysis.waiting_period_days,
                exceptions=json.loads(analysis.exceptions_json or "[]"),
            )
        )
        meta[citation_id] = (clause.id, clause.number, clause.heading, clause.page_start + 1)

    result = await run_scenario(request.scenario, shortlist_clauses)

    citations = []
    for citation in result.citations:
        db_id, number, heading, page = meta.get(citation.clause_id, (None, "", "", 0))
        citations.append(
            CitationOut(
                clause_id=citation.clause_id,
                clause_db_id=db_id,
                number=number,
                heading=heading,
                page=page,
                effect=citation.effect,
                quote=citation.quote,
                verified=citation.verified,
                unverified_reason=citation.unverified_reason,
            )
        )

    run_id = uuid.uuid4().hex[:12]
    session.add(
        ScenarioRun(
            id=run_id,
            doc_id=doc_id,
            scenario_text=request.scenario,
            facts_json=json.dumps(result.facts),
            verdict=result.verdict,
            rationale=result.reasoning,
            citations_json=json.dumps([c.model_dump() for c in citations]),
            missing_facts_json=json.dumps(result.missing_facts),
            clauses_considered=result.clauses_considered,
            verified=result.verified,
        )
    )
    session.commit()

    return ScenarioResponse(
        id=run_id,
        scenario=request.scenario,
        verdict=result.verdict,
        reasoning=result.reasoning,
        citations=citations,
        missing_facts=result.missing_facts,
        facts={k: v for k, v in result.facts.items() if v not in (None, "", "unknown")},
        verified=result.verified,
        clauses_considered=result.clauses_considered,
    )
