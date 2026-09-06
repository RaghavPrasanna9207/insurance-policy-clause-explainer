"""Stage 5: the scenario simulator.

    scenario ──> [5a extract facts] ──> [5b shortlist] ──> [5c reason] ──> [5d verify]
                 LLM, schema             pure filter        LLM, id-enum     substring

WHY THERE IS NO RETRIEVAL HERE
------------------------------
The obvious architecture for "answer a question about a document" is retrieval:
embed the clauses, embed the question, fetch the top k. This project does not,
and the reason is arithmetic rather than taste.

A health policy has roughly 40 clauses averaging ~150 tokens, so the ENTIRE
document is about 6,000 tokens. qwen2.5 has a 32,000-token context. Every clause
that could possibly matter fits in a single prompt with room to spare.

Given that, retrieval could only make the answer worse. Top-k means choosing a
k, and any k below "all of them" can drop the one clause that decides the case -
which in this domain means confidently telling someone they are covered because
the exclusion did not make the cut. Retrieval solves a problem this document
does not have, and introduces a failure mode it did not previously have.

So `shortlist()` sorts by impact and takes everything that fits the budget. On a
normal policy that is all of it. The impact ordering only starts to matter for a
document large enough to overflow the context, and then it keeps the clauses
most likely to cost the reader money.

TWO CALLS, NOT ONE
------------------
Fact extraction is separated from reasoning deliberately. A single prompt doing
both would let the model quietly invent a missing fact on its way to an answer,
and nothing downstream could tell that had happened. Splitting them makes
"the person never said how long they have held the policy" an explicit,
inspectable null that the reasoning prompt is then told about by name.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.grounding import QuoteCheck, verify_citations
from app.llm import client
from app.llm.prompts import (
    FACTS_SYSTEM,
    PROMPT_VERSION,
    REASON_SYSTEM,
    render_reasoning_request,
    render_scenario,
)
from app.taxonomy import Verdict

log = logging.getLogger(__name__)

# How a cited clause bears on the outcome. Enum-constrained like everything
# else categorical, so the UI can style these without string-matching prose.
CITATION_EFFECTS = ["denies", "delays", "reduces", "requires", "permits"]

# Cap on how many clauses the model may cite. Without an upper bound a model
# that is unsure tends to cite everything, which reads as thorough and is
# actually an abdication - the point of the answer is which clauses DECIDE it.
MAX_CITATIONS = 6

# Rough characters-per-token for budgeting. Deliberately conservative: an
# underestimate would silently truncate the clause list and drop the exclusion
# that decides the case.
CHARS_PER_TOKEN = 3.5


@dataclass
class ShortlistClause:
    """A clause as offered to the reasoning step.

    `clause_id` is the POLICY'S OWN clause number ("4.7") wherever the document
    provides one, not an internal identifier.

    That is a correctness fix, not cosmetics. An earlier version used opaque ids
    (`c14`) while the clause text itself began "2.4 Day Care Procedures", which
    left the model translating between two numbering systems mid-sentence. It
    got it wrong: it cited `c14` and quoted clause 2.4's text. The id enum
    accepted it because `c14` was a real id - the enum guarantees the address
    exists, never that it is the right one - and only the verbatim quote check
    caught the mismatch.

    Using the document's own numbers removes the translation step entirely: the
    id the model cites and the number printed inside the clause are the same
    string.
    """

    clause_id: str  # the policy's clause number where it has one
    ref: str  # the database id, for mapping the answer back
    number: str
    clause_type: str
    text: str
    impact_score: float


@dataclass
class Citation:
    clause_id: str
    quote: str
    effect: str
    verified: bool = True
    unverified_reason: str = ""


@dataclass
class ScenarioResult:
    verdict: str
    reasoning: str
    citations: list[Citation] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    # False when any quotation failed the verbatim check. The UI must present
    # such an answer as unverified rather than as established fact.
    verified: bool = True
    clauses_considered: int = 0


# --- 5a: facts ------------------------------------------------------------

FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "procedure": {"type": ["string", "null"]},
        "condition": {"type": ["string", "null"]},
        "body_system": {"type": ["string", "null"]},
        # Nullable, not a -1 sentinel. Verified that Ollama's constrained
        # decoding emits a real null for these: the extractor must be able to
        # say "not stated" rather than invent 0, because zero months since
        # inception changes which waiting periods apply.
        "months_since_policy_start": {"type": ["integer", "null"]},
        "age": {"type": ["integer", "null"]},
        "hospitalised": {"type": ["boolean", "null"]},
        "hours_since_admission": {"type": ["integer", "null"]},
        "estimated_cost_inr": {"type": ["integer", "null"]},
        "pre_existing_condition": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
        },
        "notes": {"type": "string"},
    },
    "required": [
        "procedure", "condition", "body_system", "months_since_policy_start",
        "age", "hospitalised", "hours_since_admission", "estimated_cost_inr",
        "pre_existing_condition", "notes",
    ],
}


async def extract_facts(scenario: str) -> dict[str, Any]:
    return await client.complete_json(
        [
            {"role": "system", "content": FACTS_SYSTEM},
            {"role": "user", "content": render_scenario(scenario)},
        ],
        FACTS_SCHEMA,
        prompt_version=PROMPT_VERSION,
    )


# --- 5b: shortlist (no LLM) ----------------------------------------------


def shortlist(
    clauses: list[ShortlistClause], token_budget: int | None = None
) -> list[ShortlistClause]:
    """Choose which clauses the reasoning step sees.

    Sorted by impact, then truncated to fit the context budget. On a normal
    policy nothing is dropped at all - the whole document fits. The ordering
    exists so that if a very large document ever does overflow, what survives
    is the clauses most likely to cost the reader money, rather than whichever
    ones happened to come first.

    Returned in document order, because a person reading the answer expects
    clause 3.2 to be discussed before 6.1.
    """
    budget = token_budget or settings.scenario_token_budget
    ranked = sorted(clauses, key=lambda c: c.impact_score, reverse=True)

    kept: list[ShortlistClause] = []
    used = 0
    for clause in ranked:
        cost = int(len(clause.text) / CHARS_PER_TOKEN) + 20  # +20 for the header
        if used + cost > budget and kept:
            break
        kept.append(clause)
        used += cost

    if len(kept) < len(clauses):
        log.info("shortlist dropped %d clause(s) for budget", len(clauses) - len(kept))

    kept.sort(key=lambda c: _sort_key(c.number))
    return kept


def _sort_key(number: str) -> tuple:
    """Sort clause numbers numerically: 3.2 before 3.10, and both before 4.1."""
    try:
        return (0, tuple(int(part) for part in number.split(".") if part))
    except ValueError:
        return (1, (0,))


# --- 5c: reason -----------------------------------------------------------


def _reasoning_schema(clause_ids: list[str]) -> dict[str, Any]:
    """Build the reasoning schema, with citations locked to these clauses.

    THE CENTRAL GUARANTEE OF THIS PROJECT LIVES ON THE `clause_id` LINE.

    Its enum is exactly the ids placed in this prompt. Ollama enforces
    JSON-Schema enums during sampling, so at the moment the model is emitting a
    citation, every token that would spell a different id has probability zero.
    A fabricated citation is not caught after the fact and retried - it is
    unrepresentable in the output grammar.

    Most RAG systems detect bad citations post-hoc. This makes them impossible
    to generate. The trade is that the schema must be rebuilt per request, which
    is why this is a function rather than a constant.
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": Verdict.values()},
            "reasoning": {"type": "string"},
            "deciding_clauses": {
                "type": "array",
                # No minItems: insufficient_information legitimately cites
                # nothing, because the honest answer is that no clause decides
                # it. The pairing of verdict and citation count is checked in
                # code below, where a conditional rule can be expressed.
                "maxItems": MAX_CITATIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "clause_id": {"type": "string", "enum": clause_ids},
                        "quote": {"type": "string"},
                        "effect": {"type": "string", "enum": CITATION_EFFECTS},
                    },
                    "required": ["clause_id", "quote", "effect"],
                },
            },
            "missing_information": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "reasoning", "deciding_clauses", "missing_information"],
    }


async def reason(
    scenario: str,
    facts: dict[str, Any],
    clauses: list[ShortlistClause],
    *,
    nudge_citations: bool = False,
) -> dict[str, Any]:
    ids = [c.clause_id for c in clauses]
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user", "content": render_reasoning_request(scenario, facts, clauses)},
    ]
    if nudge_citations:
        # Appended as a second user turn rather than edited into the first, so
        # the retry is a different cache key and cannot be served the very
        # response that failed.
        messages.append({"role": "user", "content": CITATION_NUDGE})
    return await client.complete_json(
        messages, _reasoning_schema(ids), prompt_version=PROMPT_VERSION
    )


CITATION_NUDGE = """Your previous answer named a clause in its reasoning but left deciding_clauses
empty. Any clause you rely on MUST appear in deciding_clauses, with its
clause_id and an exact quote copied from that clause's text.

If no clause in the list actually decides this, the verdict is
insufficient_information."""


# --- the whole stage ------------------------------------------------------


async def run_scenario(
    scenario: str, clauses: list[ShortlistClause]
) -> ScenarioResult:
    """Answer one scenario against one policy."""
    facts = await extract_facts(scenario)
    missing = sorted(
        key for key, value in facts.items()
        if key != "notes" and value in (None, "", "unknown")
    )

    considered = shortlist(clauses)
    if not considered:
        return ScenarioResult(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            reasoning="This policy has no analysed clauses to check against.",
            facts=facts,
            missing_facts=missing,
        )

    payload = await reason(scenario, facts, considered)

    # 5d: verification. Every quotation must actually occur in the clause it
    # was attributed to.
    source_by_id = {c.clause_id: c.text for c in considered}
    raw_citations = payload.get("deciding_clauses", [])
    checks: list[QuoteCheck] = verify_citations(raw_citations, source_by_id)

    citations = [
        Citation(
            clause_id=check.clause_id,
            quote=raw.get("quote", ""),
            effect=raw.get("effect", "permits"),
            verified=check.verified,
            unverified_reason=check.reason,
        )
        for raw, check in zip(raw_citations, checks)
    ]

    verdict = payload.get("verdict", Verdict.INSUFFICIENT_INFORMATION)

    # A definite verdict with nothing to back it is not a definite verdict.
    # The schema cannot express "citations are required unless the verdict is
    # insufficient_information" - that is a conditional rule, and constrained
    # decoding takes a fixed grammar - so it is enforced here instead.
    #
    # Retry BEFORE downgrading. The observed failure was not a model that had
    # no reason: asked "will you pay for a nose job", it answered not_covered
    # and named the cosmetic-surgery exclusion in its prose, but left
    # deciding_clauses empty. Downgrading immediately threw away a correct
    # answer over a formatting slip. One pointed retry recovers it.
    if verdict != Verdict.INSUFFICIENT_INFORMATION and not citations:
        log.info("verdict %s arrived with no citations; retrying once", verdict)
        payload = await reason(scenario, facts, considered, nudge_citations=True)
        raw_citations = payload.get("deciding_clauses", [])
        checks = verify_citations(raw_citations, source_by_id)
        citations = [
            Citation(
                clause_id=check.clause_id,
                quote=raw.get("quote", ""),
                effect=raw.get("effect", "permits"),
                verified=check.verified,
                unverified_reason=check.reason,
            )
            for raw, check in zip(raw_citations, checks)
        ]
        verdict = payload.get("verdict", Verdict.INSUFFICIENT_INFORMATION)

        if verdict != Verdict.INSUFFICIENT_INFORMATION and not citations:
            # Still nothing. Downgrade is the safe direction: it turns an
            # unsupported claim into an admission of ignorance, rather than
            # leaving a confident answer standing on nothing.
            log.warning("verdict %s still had no citations; downgrading", verdict)
            verdict = Verdict.INSUFFICIENT_INFORMATION

    return ScenarioResult(
        verdict=verdict,
        reasoning=payload.get("reasoning", "").strip(),
        citations=citations,
        facts=facts,
        missing_facts=missing,
        verified=all(c.verified for c in citations),
        clauses_considered=len(considered),
    )
