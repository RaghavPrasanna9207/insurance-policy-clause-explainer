"""Scenario endpoint: ask a what-if question about one policy."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models import Document, ScenarioRun
from app.pipeline.run import load_clauses
from app.pipeline.scenario import run_scenario, stated
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

    # The same list the scenario eval measures, from the same function.
    clauses = load_clauses(session, doc_id)
    if not clauses:
        raise HTTPException(409, "This policy has no analysed clauses")

    result = await run_scenario(request.scenario, clauses)

    by_id = {clause.clause_id: clause for clause in clauses}
    citations = []
    for citation in result.citations:
        cited = by_id.get(citation.clause_id)
        citations.append(
            CitationOut(
                clause_id=citation.clause_id,
                clause_db_id=cited.db_id if cited else None,
                number=cited.number if cited else "",
                heading=cited.heading if cited else "",
                page=cited.page if cited else 0,
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
        facts={k: v for k, v in result.facts.items() if stated(v)},
        verified=result.verified,
        clauses_considered=result.clauses_considered,
    )
