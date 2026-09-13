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
    REASON_SYSTEM,
    render_reasoning_request,
    render_scenario,
)
from app.pipeline import reduction, waiting, window
from app.taxonomy import Verdict

log = logging.getLogger(__name__)

# How a cited clause bears on the outcome. Enum-constrained like everything
# else categorical, so the UI can style these without string-matching prose.
CITATION_EFFECTS = ["denies", "delays", "reduces", "requires", "permits"]

# Cap on how many clauses the model may cite. Without an upper bound a model
# that is unsure tends to cite everything, which reads as thorough and is
# actually an abdication - the point of the answer is which clauses DECIDE it.
#
# Lowered from 6 to 4 for two reasons at once: four is already more deciding
# clauses than a real question has, and it shrinks the worst-case response
# enough to stay clear of the generation cap.
MAX_CITATIONS = 4

# Rough characters-per-token for budgeting. Deliberately conservative: an
# underestimate would silently truncate the clause list and drop the exclusion
# that decides the case.
CHARS_PER_TOKEN = 3.5

# Days per unit, for normalising whatever the person happened to say.
# Approximate on purpose: no real policy turns on a day either side of a
# 36-month bar, and precision here would imply a certainty the source
# sentence ("about five years ago") never had.
_DAYS_PER_UNIT = {"days": 1, "weeks": 7, "months": 30, "years": 365}


def policy_age_days(facts: dict[str, Any]) -> int | None:
    """How long the policy has been held, in days, or None if unstated.

    None is a real answer and must not be replaced with a default. Every
    waiting period then comes back UNKNOWN, which is what lets the verdict
    be insufficient_information rather than a guess.
    """
    value = facts.get("policy_age_value")
    unit = facts.get("policy_age_unit")
    if not value or value <= 0 or unit not in _DAYS_PER_UNIT:
        return None
    return value * _DAYS_PER_UNIT[unit]


def expense_offset_days(facts: dict[str, Any]) -> int | None:
    """How many days before admission or after discharge an expense fell.

    The same conversion as `policy_age_days`, applied to a different
    measurement: this one counts from a hospital stay, not from the day the
    policy began. None when unstated, never a default.
    """
    value = facts.get("expense_timing_value")
    unit = facts.get("expense_timing_unit")
    if not value or value <= 0 or unit not in _DAYS_PER_UNIT:
        return None
    return value * _DAYS_PER_UNIT[unit]

# Facts that can actually decide an outcome under an Indian health policy.
#
# Defined once and imported by the prompt renderer, because these were briefly
# two separate lists: the prompt was narrowed to these five, but the list shown
# to the USER still included every null field, so the interface offered "body
# system" and "estimated cost inr" as information it needed to answer. Two
# lists that must agree should not be two lists - the same rule that applies to
# the context window and its token budget.
DECISIVE_FACTS = (
    "policy_age_value",
    "age",
    "pre_existing_condition",
    "hospitalised",
    "hours_since_admission",
)


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
    # Structured facts extracted at analysis time, so stage 5 can reason over
    # them arithmetically instead of asking the model to re-read the prose.
    waiting_period_days: int | None = None
    exceptions: list[str] = field(default_factory=list)
    # The operands a reduction is computed from, carried through from analysis.
    # See app/pipeline/reduction.py for what is done with them.
    copay_percent: int | None = None
    copay_min_age_at_inception: int | None = None
    cap_percent_of_sum_insured: int | None = None
    icu_cap_percent_of_sum_insured: int | None = None
    # A cover window around one hospital stay, in days, and which side of the
    # stay it counts from. See app/pipeline/window.py.
    cover_window_days: int | None = None
    cover_window_anchor: str | None = None


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
        # HOW LONG THE POLICY HAS BEEN HELD, as a value and a unit.
        #
        # Not "months_since_policy_start", which is what this was, and
        # which produced 2 for "two weeks after my policy started" - the
        # number read correctly and the unit dropped. A fortnight-old
        # policy then counted as two months old and cleared a 30-day
        # waiting period it should have failed.
        #
        # This is the same fix already applied to clause durations, on the
        # other side of the same comparison. Converting units is arithmetic
        # and belongs to Python; reporting what the sentence said is
        # reading, and belongs to the model.
        "policy_age_value": {"type": ["integer", "null"]},
        "policy_age_unit": {
            "type": ["string", "null"],
            "enum": ["days", "weeks", "months", "years", None],
        },
        "age": {"type": ["integer", "null"]},
        # AGE WHEN THE POLICY STARTED, which is a different number from `age`
        # and is the one a senior-citizen co-payment actually keys on. Someone
        # 68 today may have bought the policy at 50; treating the two as
        # interchangeable fires a 20% co-payment at a person who does not owe
        # one. Where only one of the two is stated, Python derives the other
        # from the policy's age - see reduction.age_at_inception.
        "age_at_policy_start": {"type": ["integer", "null"]},
        "hospitalised": {"type": ["boolean", "null"]},
        "hours_since_admission": {"type": ["integer", "null"]},
        "estimated_cost_inr": {"type": ["integer", "null"]},
        # THE SUM INSURED, as a value and a unit. Indian policy schedules are
        # written "5 lakh", never 500000, and asking the model for the rupee
        # figure is asking it to multiply - the same mistake that read
        # "two weeks" as two months on the other side of a waiting-period
        # comparison. It reports what the sentence said; Python multiplies.
        "sum_insured_value": {"type": ["integer", "null"]},
        "sum_insured_unit": {
            "type": ["string", "null"],
            "enum": ["rupees", "thousand", "lakh", "crore", None],
        },
        # The per-day accommodation charge, in plain rupees, and whether it was
        # intensive care - because the room cap and the ICU cap are different
        # percentages of the same sum insured.
        "room_rent_per_day_inr": {"type": ["integer", "null"]},
        "room_is_icu": {"type": ["boolean", "null"]},
        # WHEN AN EXPENSE FELL RELATIVE TO A HOSPITAL STAY: a value, a unit,
        # and which side of the stay. "A scan 120 days after I was discharged"
        # is (120, days, after_discharge). The model reports the sentence;
        # window.py compares it against the policy's window. Kept apart from
        # policy_age because the two count from different starting points,
        # and the anchor is an enum so "after discharge" cannot be paraphrased
        # into something the comparison does not recognise.
        "expense_timing_value": {"type": ["integer", "null"]},
        "expense_timing_unit": {
            "type": ["string", "null"],
            "enum": ["days", "weeks", "months", "years", None],
        },
        "expense_timing_anchor": {
            "type": ["string", "null"],
            "enum": ["before_admission", "after_discharge", None],
        },
        "pre_existing_condition": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
        },
        "notes": {"type": "string"},
    },
    "required": [
        "procedure", "condition", "body_system",
        "policy_age_value", "policy_age_unit",
        "age", "age_at_policy_start", "hospitalised", "hours_since_admission",
        "estimated_cost_inr", "sum_insured_value", "sum_insured_unit",
        "room_rent_per_day_inr", "room_is_icu",
        "expense_timing_value", "expense_timing_unit", "expense_timing_anchor",
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
    use_cache: bool = True,
) -> dict[str, Any]:
    ids = [c.clause_id for c in clauses]
    held = policy_age_days(facts)
    # Every waiting period compared against the stated policy age, in Python,
    # before the model sees anything. The result is handed over as settled fact.
    # Prefer days when stated, else convert months. Neither is guessed: if
    # the person said nothing, both stay None and every waiting period comes
    # back UNKNOWN rather than being compared against an invented figure.
    checks = waiting.evaluate(clauses, held)
    # And the second family of comparisons, added after the first was fixed:
    # a waiting period decides whether the claim is PAID, a reduction decides
    # whether it is paid IN FULL. Only the first question was being asked, so
    # a senior citizen was told "covered" while losing a fifth of the claim.
    cuts = reduction.evaluate(clauses, facts, held)
    # The third family: whether an expense before admission or after discharge
    # fell inside the policy's window. Silent unless the person raised it.
    windows = window.evaluate(
        clauses, facts.get("expense_timing_anchor"), expense_offset_days(facts)
    )
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {
            "role": "user",
            "content": render_reasoning_request(
                scenario, facts, clauses,
                waiting.render(checks), reduction.render(cuts),
                window.render(windows),
            ),
        },
    ]
    if nudge_citations:
        # Appended as a second user turn rather than edited into the first, so
        # the retry is a different cache key and cannot be served the very
        # response that failed.
        messages.append({"role": "user", "content": CITATION_NUDGE})
    return await client.complete_json(
        messages, _reasoning_schema(ids), use_cache=use_cache,
    )


CITATION_NUDGE = """Your previous answer named a clause in its reasoning but left deciding_clauses
empty. Any clause you rely on MUST appear in deciding_clauses, with its
clause_id and an exact quote copied from that clause's text.

If no clause in the list actually decides this, the verdict is
insufficient_information."""


# --- the whole stage ------------------------------------------------------


async def run_scenario(
    scenario: str, clauses: list[ShortlistClause], *, resample: bool = False
) -> ScenarioResult:
    """Answer one scenario against one policy.

    `resample` bypasses the cache for the REASONING step only, so the same
    prompt is answered afresh while the facts stay as extracted. It exists for
    the eval: a cached answer re-read is a copy, not a second opinion, and the
    only way to learn whether a verdict is stable is to ask again. Nothing
    resampled is written back, so the stored answer remains the baseline.
    """
    facts = await extract_facts(scenario)
    missing = [k for k in DECISIVE_FACTS if facts.get(k) in (None, "", "unknown")]

    considered = shortlist(clauses)
    if not considered:
        return ScenarioResult(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            reasoning="This policy has no analysed clauses to check against.",
            facts=facts,
            missing_facts=missing,
        )

    payload = await reason(scenario, facts, considered, use_cache=not resample)

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
        payload = await reason(
            scenario, facts, considered, nudge_citations=True,
            use_cache=not resample,
        )
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
