"""Stage 5: the scenario simulator.

    scenario ──> [5a extract facts] ──> [5b shortlist] ──> [5c reason] ──> [5d verify]
                 LLM, schema             pure filter        LLM, id-enum     substring

WHY THERE IS NO RETRIEVAL HERE
------------------------------
The obvious architecture for "answer a question about a document" is retrieval:
embed the clauses, embed the question, fetch the top k. This project does not,
and the reason is arithmetic rather than taste.

A policy's clauses are shown to the model whole, and the premise that allows it
is size. The synthetic golden policy is about 3,100 tokens. Real wordings, first
measured in M15, are 69,000-129,000 characters - roughly 16,000-30,000 tokens at
the 4.25 characters per token measured on policy text - and at the 8,192-token
window of the time the scenario step saw a quarter of the smallest. The window is now 28,672
(`settings.num_ctx`, measured in M16 as the largest that runs entirely on this
machine's GPU), which holds IRDAI's standard Arogya Sanjeevani wording whole.
`clause_token_budget()` derives the room for clause text from it.

Given that, retrieval could only make the answer worse. Top-k means choosing a
k, and any k below "all of them" can drop the one clause that decides the case -
which in this domain means confidently telling someone they are covered because
the exclusion did not make the cut.

So `shortlist()` takes everything that fits. For a document too large even for
the wider window, it gives up definitions and administrative clauses first,
then the lowest impact, and says so in a warning.

TWO CALLS, NOT ONE
------------------
Fact extraction is separated from reasoning deliberately. A single prompt doing
both would let the model quietly invent a missing fact on its way to an answer,
and nothing downstream could tell that had happened. Splitting them makes
"the person never said how long they have held the policy" an explicit,
inspectable null that the reasoning prompt is then told about by name.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from app.config import settings
from app.grounding import QuoteCheck, verified_prefix, verify_citations
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

# Length caps on the two free-text fields, enforced by the grammar like the
# enums: a string that has reached its cap can only close. M16: a Star answer
# ran on inside "reasoning" past 1,600, 3,200 and 6,400 tokens, and the eval
# died with it. Retrying with a higher ceiling cannot fix a loop; making it
# unrepresentable does. Sized from the 229 answers stored at the time: the
# longest reasoning was 974 characters and the longest quote 832 (a whole
# clause, which the prompt already discourages).
MAX_REASONING_CHARS = 1_500
MAX_QUOTE_CHARS = 1_200

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
    value = facts.get("time_since_policy_start_value")
    unit = facts.get("time_since_policy_start_unit")
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
    "time_since_policy_start_value",
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
    # Every waiting period the clause sets, in days - usually one. See
    # ClauseAnalysis.waiting_periods_days.
    waiting_periods_days: list[int] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    # The operands a reduction is computed from, carried through from analysis.
    # See app/pipeline/reduction.py for what is done with them.
    copay_percent: int | None = None
    copay_min_age_at_inception: int | None = None
    cap_percent_of_sum_insured: int | None = None
    icu_cap_percent_of_sum_insured: int | None = None
    cap_max_inr_per_day: int | None = None
    icu_cap_max_inr_per_day: int | None = None
    # A cover window around one hospital stay, in days, and which side of the
    # stay it counts from. See app/pipeline/window.py.
    cover_window_days: int | None = None
    cover_window_anchor: str | None = None
    # The section this clause was segmented under. Used to tell whether an
    # annexure another clause refers to is part of the document at all.
    section_path: str = ""


def citation_ids(numbered: list[tuple[str, int]]) -> list[str]:
    """The id the model cites for each clause: its own number, made unique.

    Takes (number, order_idx) per clause, in document order. A repeated number
    gets a suffix - "10", "10#2" - because the id enum and the quote check both
    look a clause up by id, and two clauses under one id would make a citation
    ambiguous and verify a quote against the wrong text.

    Real wordings repeat numbers routinely (the Star Health standard policy
    restarts its numbering in each section). This lived inline in the scenario
    endpoint while the eval built ids its own way without the suffix; on the
    synthetic policy, which repeats no number, the two could not disagree, so
    nothing showed they had drifted. One function, so they cannot.
    """
    seen: dict[str, int] = {}
    ids = []
    for number, order_idx in numbered:
        base = number or f"c{order_idx}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return ids


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
        #
        # THE NAME IS PART OF THE INSTRUCTION. This was `policy_age_value`,
        # and the model read "policy age" as "age at the policy": for "I
        # bought this policy at 67 and am claiming three years later" it wrote
        # 67 here with no unit, and the three years were lost. Renamed to
        # `policy_held_for_value`, the ages stopped landing here - and "ten
        # days after my policy started" stopped landing here too, because
        # "held for" only matched "I have held it for". With the same prompt
        # examples, this name read 31 of 33 test sentences in both orderings
        # tried; the old name read 29 and 30, most of its misses a subtraction
        # between two ages that nobody asked it to do. A schema key is the
        # last thing the model reads before it writes the value.
        "time_since_policy_start_value": {"type": ["integer", "null"]},
        "time_since_policy_start_unit": {
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
        "time_since_policy_start_value", "time_since_policy_start_unit",
        "age", "age_at_policy_start", "hospitalised", "hours_since_admission",
        "estimated_cost_inr", "sum_insured_value", "sum_insured_unit",
        "room_rent_per_day_inr", "room_is_icu",
        "expense_timing_value", "expense_timing_unit", "expense_timing_anchor",
        "pre_existing_condition", "notes",
    ],
}


# Words that tie an age to the START of a policy. An age at policy start is
# only something a person can have said if they talked about the policy
# beginning at all.
_POLICY_START = re.compile(
    r"\b(bought|buy|purchased?|took|taken|take|got|started?|began|begin|"
    r"signed|joined|inception|enrolled|insured me)\b",
    re.IGNORECASE,
)


def correct_misfiled_age(facts: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Move an age filed as "at policy start" to "now" when nothing says the policy started.

    Measured: "My mother, who is 75" and "My father, 82" came back with the age
    in `age_at_policy_start` and `age` empty - which fires a senior-citizen
    co-payment at someone whose age at inception was never mentioned. A prompt
    example made it worse; renaming the `age` field made every third-person
    sentence fail. So the extraction is checked against the words it came from,
    the way exceptions are (app/grounding.py): if the question never mentions
    the policy beginning, the only age it can contain is the age now.

    Only moves; never invents. Both ages given, or start words present -> left
    exactly as extracted.
    """
    start_age = facts.get("age_at_policy_start")
    if start_age and not facts.get("age") and not _POLICY_START.search(scenario):
        return facts | {"age": start_age, "age_at_policy_start": None}
    return facts


async def extract_facts(scenario: str, *, use_cache: bool = True) -> dict[str, Any]:
    facts = await client.complete_json(
        [
            {"role": "system", "content": FACTS_SYSTEM},
            {"role": "user", "content": render_scenario(scenario)},
        ],
        FACTS_SCHEMA,
        use_cache=use_cache,
    )
    return correct_misfiled_age(facts, scenario)


# --- 5b: shortlist (no LLM) ----------------------------------------------


# Prompt tokens that are neither the system prompt nor clause text: the person's
# words, the facts, and the computed waiting-period, window and reduction lines.
# Measured in M15: the synthetic policy's pneumonia prompt was 5,270 tokens, of
# which about 2,700 were clause text and 1,900 system prompt, leaving ~650 with
# the clause headers included. 1,000 leaves room for a question that produces
# more computed lines - and an underestimate is refused as truncation, not
# silently answered.
QUESTION_TOKENS = 1_000

# The clause types given up first when a policy is too large for the context.
# M15 ranked by impact alone, and on three real wordings that dropped every
# coverage clause: impact measures what a clause can cost you, and the clause
# granting cover costs nothing. No question is answerable without the clause
# that says what is paid for; definitions and administration rarely decide one.
DROPPED_FIRST = frozenset({"definition", "procedural"})


def clause_token_budget() -> int:
    """How many tokens of clause text the reasoning prompt can hold.

    Derived from everything else that must fit, not reserved beside it. The
    reservation this replaces was a fixed 2,500 tokens, set in M5; by M15 the
    system prompt alone had grown to about 1,900 and the answer ceiling is
    1,600, so the reservation could no longer hold what it was reserving for.
    Measuring the system prompt here means growing it shrinks the budget.
    """
    system_prompt = int(len(REASON_SYSTEM) / CHARS_PER_TOKEN)
    return settings.num_ctx - settings.num_predict - system_prompt - QUESTION_TOKENS


def shortlist(
    clauses: list[ShortlistClause], token_budget: int | None = None
) -> list[ShortlistClause]:
    """Choose which clauses the reasoning step sees.

    Everything, whenever it fits - which is the design: there is no retrieval,
    because any selection can drop the clause that decides the case. With the
    context window measured in M16, IRDAI's standard Arogya Sanjeevani wording
    fits whole.

    When a policy does not fit, clauses are given up in a stated order -
    definitions and procedural clauses first, then lowest impact - and a clause
    too large for the remaining room is skipped rather than ending the
    selection, so one long annexure cannot push out every short exclusion
    ranked after it. That is logged as a warning: the answer may be missing
    the clause that decides it.

    `clauses` must arrive in document order, and the selection keeps it. Real
    wordings restart their numbering in every section, so sorting by clause
    number - as this once did - interleaves coverage 1, exclusion 1 and
    condition 1.
    """
    budget = token_budget or clause_token_budget()
    ranked = sorted(
        range(len(clauses)),
        key=lambda i: (clauses[i].clause_type in DROPPED_FIRST, -clauses[i].impact_score),
    )

    keep: set[int] = set()
    used = 0
    for i in ranked:
        cost = int(len(clauses[i].text) / CHARS_PER_TOKEN) + 20  # +20 for the header
        if used + cost > budget and keep:
            continue
        keep.add(i)
        used += cost

    if len(keep) < len(clauses):
        dropped = Counter(c.clause_type for i, c in enumerate(clauses) if i not in keep)
        log.warning(
            "policy too large for the context: %d of %d clauses left out (%s)",
            len(clauses) - len(keep), len(clauses),
            ", ".join(f"{n} {t}" for t, n in dropped.most_common()),
        )
    return [c for i, c in enumerate(clauses) if i in keep]


# Standard English function words: long enough to pass the length filter,
# meaningless as evidence of what a clause is about. Not tuned to any policy.
_FUNCTION_WORDS = frozenset({
    "after", "before", "because", "another", "itself", "taken", "taking", "their",
    "there", "these", "those", "which", "while", "would", "could", "should", "about",
    "again", "other", "under", "until", "being", "having", "every", "where", "whose",
    "since", "though",
})
# A word shared with the question counts only if at most this many clauses use it.
_RARE_IN_POLICY = 2
_MIN_SHARED = 2


def _stems(text: str) -> dict[str, str]:
    """Six-letter stems of the words in `text`, each mapped to a word it came from.

    Six letters, so "Ayurvedic" meets "Ayurveda" and "accreditation" meets
    "accredited", without a stemming library for one comparison.
    """
    out: dict[str, str] = {}
    for word in re.findall(r"[a-z]{5,}", text.lower()):
        if word not in _FUNCTION_WORDS:
            out.setdefault(word[:6], word)
    return out


def named_in_question(
    facts: dict[str, Any], clauses: list[ShortlistClause]
) -> dict[str, list[str]]:
    """Clauses using an unusual word from the procedure or condition the person named.

    One word is enough here, where the whole-question rule below needs two,
    because these words are not drawn from anywhere in the question: they are
    what the question is about. On the Star policy the clause that decides a
    hernia question is the only one naming "hernia", and the maternity
    exclusion the only one naming "caesarean"; neither shares a second unusual
    word with its question, so the two-word rule never named them.

    The word must still be rare (used by at most two clauses), and definitions
    and procedural clauses are never named - the same types the shortlist gives
    up first, because they rarely decide a claim. Measured on the 69 synthetic
    questions and 10 Star questions, from their stored facts: it names a clause
    for 18 of them, and for 13 of those the named clauses include one the
    answer must cite. Of the 21 clauses named, 13 must be cited, 4 are about the
    same treatment without deciding the case (a cataract waiting period on a
    cataract cost question), and 4 are noise ("admitted", "removal",
    "existing", "anaesthesia") - which is why this, too, only names clauses.
    """
    wanted = _stems(" ".join(str(facts.get(k) or "") for k in ("procedure", "condition")))
    per_clause = {c.clause_id: _stems(c.text) for c in clauses}
    used_by = Counter(s for stems in per_clause.values() for s in stems)
    rare = {s for s in wanted if 0 < used_by[s] <= _RARE_IN_POLICY}

    named: dict[str, list[str]] = {}
    for clause in clauses:
        common = rare & per_clause[clause.clause_id].keys()
        if common and clause.clause_type not in DROPPED_FIRST:
            named[clause.clause_id] = sorted(wanted[s] for s in common)
    return named


def shared_words(
    scenario: str, clauses: list[ShortlistClause], facts: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Clauses that share several unusual words with the question, or one with
    the procedure or condition it names (named_in_question).

    Measured case: Ayurvedic treatment at an unaccredited private clinic. The
    verdict was right and the citation was the definition of a hospital, in
    every run on record - and again when the model was asked for citations in a
    separate turn, so it was not a slip. The AYUSH clause shares five unusual
    words with that question.

    NOT RETRIEVAL. This project deliberately has none: every clause is still in
    the prompt, and nothing is ranked or dropped. It is a lookup reported as a
    lookup - these words appear in both - and deciding what a shared word means
    stays with the model.

    Precision over recall, on purpose. A word counts only if at most two
    clauses use it, and a clause is named only with at least two such words:
    measured on the 40 eval questions, a single shared word hinted at 38 of
    them, including clauses they must not cite ("years" appears in the
    co-payment clause). Two words hinted at 8, each the clause that decides the
    case. Those thresholds were chosen on that same set, so 8 of 8 is partly
    fitted - which is why the hint only names clauses and never sets anything.
    """
    per_clause = {c.clause_id: _stems(c.text) for c in clauses}
    used_by = Counter(s for stems in per_clause.values() for s in stems)
    mine = _stems(scenario)
    rare_mine = {s for s in mine if 0 < used_by[s] <= _RARE_IN_POLICY}

    shared: dict[str, list[str]] = {}
    for clause_id, stems in per_clause.items():
        common = rare_mine & stems.keys()
        if len(common) >= _MIN_SHARED:
            shared[clause_id] = sorted(mine[s] for s in common)
    for clause_id, words in named_in_question(facts or {}, clauses).items():
        shared[clause_id] = sorted(set(shared.get(clause_id, [])) | set(words))
    # In policy order, as the clauses are shown.
    order = {c.clause_id: i for i, c in enumerate(clauses)}
    return dict(sorted(shared.items(), key=lambda item: order[item[0]]))


_ANNEXURE = re.compile(r"\bannexure\s+([ivxl]+|\d+)\b", re.IGNORECASE)


def absent_annexures(clauses: list[ShortlistClause]) -> dict[str, list[str]]:
    """Clauses that refer to an annexure the document does not contain.

    An annexure counts as present only if some clause was segmented under a
    heading naming it (segment.py recognises "ANNEXURE <n>" as a section head).
    Whether a section exists is a lookup; what it would have said is not, so
    this reports only the absence.
    """
    present = {m.group(1).upper() for c in clauses for m in _ANNEXURE.finditer(c.section_path)}
    absent: dict[str, list[str]] = {}
    for clause in clauses:
        labels = sorted({m.group(1).upper() for m in _ANNEXURE.finditer(clause.text)} - present)
        if labels:
            absent[clause.clause_id] = [f"Annexure {label}" for label in labels]
    return absent


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
            "reasoning": {"type": "string", "maxLength": MAX_REASONING_CHARS},
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
                        "quote": {"type": "string", "maxLength": MAX_QUOTE_CHARS},
                        "effect": {"type": "string", "enum": CITATION_EFFECTS},
                    },
                    "required": ["clause_id", "quote", "effect"],
                },
            },
            "missing_information": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "reasoning", "deciding_clauses", "missing_information"],
    }


@dataclass
class Computed:
    """Everything worked out in Python before the model sees the question.

    Computed once and shared: the reasoning prompt is rendered from it, and the
    model's answer is checked against it afterwards (find_contradictions).
    """

    waiting: list[waiting.WaitingCheck]
    reductions: list[reduction.ReductionCheck]
    windows: list[window.WindowCheck]
    # Clause id -> annexures it refers to that the document does not contain.
    absent: dict[str, list[str]] = field(default_factory=dict)


def compute(facts: dict[str, Any], clauses: list[ShortlistClause]) -> Computed:
    held = policy_age_days(facts)
    return Computed(
        absent=absent_annexures(clauses),
        # Every waiting period compared against the stated policy age, in
        # Python, before the model sees anything. Nothing is guessed: if the
        # person said nothing, every waiting period comes back UNKNOWN rather
        # than being compared against an invented figure.
        waiting=waiting.evaluate(clauses, held, named_in_question(facts, clauses)),
        # And the second family of comparisons, added after the first was
        # fixed: a waiting period decides whether the claim is PAID, a
        # reduction decides whether it is paid IN FULL. Only the first question
        # was being asked, so a senior citizen was told "covered" while losing
        # a fifth of the claim.
        reductions=reduction.evaluate(clauses, facts, held),
        # The third family: whether an expense before admission or after
        # discharge fell inside the policy's window. Silent unless raised.
        windows=window.evaluate(
            clauses, facts.get("expense_timing_anchor"), expense_offset_days(facts)
        ),
    )


async def reason(
    scenario: str,
    facts: dict[str, Any],
    clauses: list[ShortlistClause],
    computed: Computed,
    *,
    nudge: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    ids = [c.clause_id for c in clauses]
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {
            "role": "user",
            "content": render_reasoning_request(
                scenario, facts, clauses,
                waiting.render(computed.waiting),
                reduction.render(computed.reductions),
                window.render(computed.windows),
            ),
        },
    ]
    if nudge:
        # Appended as a second user turn rather than edited into the first, so
        # the retry is a different cache key and cannot be served the very
        # response that failed.
        messages.append({"role": "user", "content": nudge})
    return await client.complete_json(
        messages, _reasoning_schema(ids), use_cache=use_cache,
    )


CITATION_NUDGE = """Your previous answer named a clause in its reasoning but left deciding_clauses
empty. Any clause you rely on MUST appear in deciding_clauses, with its
clause_id and an exact quote copied from that clause's text.

If no clause in the list actually decides this, the verdict is
insufficient_information."""


CONSISTENCY_NUDGE = """Your previous answer contradicts results that were calculated, not estimated:

{problems}

Answer the question again, and do not rely on anything these results rule out."""

# Effects that say a clause is working against this claim - and how to say so
# to the model. A clause the arithmetic has cleared cannot be doing any of them.
_ADVERSE_EFFECTS = {"denies": "refused", "delays": "delayed", "reduces": "reduced"}


def find_contradictions(
    citations: list[Citation], verdict: str, computed: Computed
) -> tuple[list[str], list[Citation]]:
    """Where the model's answer disagrees with what Python already worked out.

    Returns (problems, irrelevant): a sentence for each contradiction, to put in
    front of the model, and the citations that are provably wrong - a clause
    cited as working against the claim that the arithmetic has cleared.

    WHY THIS EXISTS. The prompt already says "never contradict these results",
    and the model still did, three samples of three:
      - told "Already satisfied: 3.2", it refused a claim five years into a
        36-month pre-existing disease wait, citing 3.2;
      - told 5.1 was "ruled out by the numbers" (8,000 within a 10,000 cap), it
        said the room would be paid at 80%, citing 5.1.
    An instruction can be ignored. A check in code cannot, and it only has to
    compare the citations against results that already exist.

    It never changes a verdict. It finds the disagreement; the model answers
    again with the disagreement in front of it.
    """
    served = {
        c.clause_id: c for c in computed.waiting
        if c.status is waiting.WaitingStatus.SERVED
    }
    ruled_out = {
        c.clause_id: c for c in computed.reductions
        if c.status is reduction.ReductionStatus.DOES_NOT_APPLY
    }
    problems: list[str] = []
    irrelevant: list[Citation] = []
    for citation in citations:
        if citation.effect not in _ADVERSE_EFFECTS:
            continue
        cleared = served.get(citation.clause_id) or ruled_out.get(citation.clause_id)
        if cleared is not None:
            problems.append(
                f"- You cited clause {citation.clause_id} as a reason this claim is "
                f"{_ADVERSE_EFFECTS[citation.effect]}, but it was calculated: "
                f"{cleared.describe()}."
            )
            irrelevant.append(citation)

    # A refusal and a reduction cannot both decide one claim: a co-payment or a
    # cap takes a share of a claim that IS paid. The model wrote exactly this
    # contradiction for a nose job bought at 70 - "not covered ... however the
    # 20% co-payment applies" - and answered conditional. No arithmetic is
    # involved, so nothing is dropped; the model is asked to decide.
    # Covered, while citing a reduction the arithmetic says DOES apply. The
    # citation and the calculation agree the claim is cut; only the verdict
    # disagrees. Measured on an ICU stay at 9,000 a day against a 6,000 cap,
    # three fresh samples of three. Nothing is dropped: the citation is right.
    if verdict == Verdict.COVERED:
        applies = {
            c.clause_id: c for c in computed.reductions
            if c.status is reduction.ReductionStatus.APPLIES
        }
        for citation in citations:
            check = applies.get(citation.clause_id)
            if citation.effect == "reduces" and check is not None:
                problems.append(
                    f"- You cited clause {citation.clause_id} as reducing this claim, "
                    f"and it was calculated that it does: {check.describe()}. A claim "
                    f"paid less than in full is conditional, not covered."
                )

    denying = [c.clause_id for c in citations if c.effect == "denies"]
    reducing = [c.clause_id for c in citations if c.effect == "reduces"]
    if denying and reducing and verdict != Verdict.NOT_COVERED:
        problems.append(
            f"- You cited {', '.join(denying)} as refusing this claim and "
            f"{', '.join(reducing)} as reducing it. Both cannot decide it: a "
            f"co-payment, cap or sub-limit reduces a claim that IS paid, and a "
            f"refused claim has nothing to reduce. Decide first whether it is refused."
        )

    # Paying a claim on the strength of a clause that points to a list this
    # document does not contain. Measured on `day-care-not-listed`: day care is
    # paid for treatment "listed in Annexure II", there is no Annexure II, and
    # the model answered that six hours on a drip is covered. A note printed
    # under the clause in every prompt was tried first, and reverted: it did
    # not fix this case and moved four unrelated ones. Checking only answers
    # that actually lean on such a clause reaches only those answers. Nothing
    # is dropped - the absence proves the answer unchecked, not wrong.
    #
    # Not asked when a sub-limit elsewhere names the described treatment. "Treatment
    # of cataract shall be limited to 25,000" presupposes cataract treatment is
    # paid, so the missing day-care list is not what the answer rests on. Asked
    # anyway, the retry turned a correct cataract answer into a cosmetic-surgery
    # refusal, two samples of two.
    names_treatment = any(c.named for c in computed.reductions)
    if verdict in (Verdict.COVERED, Verdict.CONDITIONAL) and not names_treatment:
        for citation in citations:
            names = computed.absent.get(citation.clause_id)
            # Any effect but "denies": a held-out case leaned on the day-care
            # clause labelled "delays" and slipped past a check for "permits".
            if citation.effect != "denies" and names:
                listed = " and ".join(names)
                problems.append(
                    f"- You relied on clause {citation.clause_id} to pay this claim. "
                    f"Clause {citation.clause_id} refers to {listed}, which is not "
                    f"included in this document, so anything that depends on what "
                    f"{listed} lists cannot be checked here."
                )
    return problems, irrelevant


def unraised_reductions(citations: list[Citation], computed: Computed) -> list[Citation]:
    """Citations of a reduction as REDUCING this claim when nothing described bears on it.

    No room, no sum insured, no age: the arithmetic comes back NOT_RAISED, and a
    room cap or co-payment cited as cutting this claim is a reduction asserted
    with no described fact behind it. Measured on a claim about late paperwork
    (cited beside the documents clause that decides it) and on a held-out war
    injury "subject to the room rent limit and the senior co-payment".

    These are FILTERED, not retried. A retry that told the model "nothing the
    person described bears on this" was measured, and it pushed right answers
    to insufficient_information - a held-out late-notice case, and two main
    cases to two samples of three. Filtering changes no verdict.
    """
    not_raised = {
        c.clause_id for c in computed.reductions
        if c.status is reduction.ReductionStatus.NOT_RAISED
    }
    return [c for c in citations if c.effect == "reduces" and c.clause_id in not_raised]


def _names_a_refusal(citations: list[Citation], clauses: list[ShortlistClause]) -> bool:
    """True if some cited exclusion, waiting period or condition is said to refuse or delay."""
    refusing_types = {"exclusion", "waiting_period", "condition"}
    types = {c.clause_id: c.clause_type for c in clauses}
    return any(
        c.effect in ("denies", "delays") and types.get(c.clause_id) in refusing_types
        for c in citations
    )


def _rests_only_on_a_missing_list(
    citations: list[Citation], verdict: str, computed: Computed,
    clauses: list[ShortlistClause],
) -> bool:
    """True if a paying answer has nothing behind it but a list the document lacks.

    Checked only after the model has been told about the missing annexure and
    asked again. If every clause it still relies on to pay is one that points to
    that list - or a definition, which explains a word and pays nothing - then
    nothing checkable supports the answer. That is the same position as an
    answer with no citations at all, and it goes the same safe way: downgraded
    to insufficient_information. A held-out short procedure and the main set's
    six hours on a drip both answered conditional "on the procedure being listed
    in Annexure II" - true, and exactly what insufficient_information means.

    Not when a sub-limit names the described treatment: that clause presupposes
    the treatment is paid, so the missing list is not what the answer rests on.
    """
    if verdict not in (Verdict.COVERED, Verdict.CONDITIONAL):
        return False
    if any(c.named for c in computed.reductions):
        return False
    definitions = {c.clause_id for c in clauses if c.clause_type == "definition"}
    noise = unraised_reductions(citations, computed)
    basis = [c for c in citations if c.effect != "denies" and c not in noise]
    return any(c.clause_id in computed.absent for c in basis) and all(
        c.clause_id in computed.absent or c.clause_id in definitions for c in basis
    )


def _read_answer(
    payload: dict[str, Any], source_by_id: dict[str, str]
) -> tuple[list[Citation], str]:
    """The model's citations, each verified against its clause, and its verdict."""
    raw_citations = payload.get("deciding_clauses", [])
    checks: list[QuoteCheck] = verify_citations(raw_citations, source_by_id)
    citations = []
    for raw, check in zip(raw_citations, checks):
        quote, verified, reason_text = raw.get("quote", ""), check.verified, check.reason
        if not verified:
            # The clause's own words with something appended: keep the words.
            # See grounding.verified_prefix for why a retry could not do this.
            kept = verified_prefix(quote, source_by_id.get(check.clause_id, ""))
            if kept is not None:
                quote, verified, reason_text = kept, True, ""
        citations.append(Citation(
            clause_id=check.clause_id,
            quote=quote,
            effect=raw.get("effect", "permits"),
            verified=verified,
            unverified_reason=reason_text,
        ))
    return citations, payload.get("verdict", Verdict.INSUFFICIENT_INFORMATION)


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

    computed = compute(facts, considered)
    payload = await reason(
        scenario, facts, considered, computed, use_cache=not resample
    )

    # 5d: verification. Every quotation must actually occur in the clause it
    # was attributed to.
    source_by_id = {c.clause_id: c.text for c in considered}
    citations, verdict = _read_answer(payload, source_by_id)

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
            scenario, facts, considered, computed, nudge=CITATION_NUDGE,
            use_cache=not resample,
        )
        citations, verdict = _read_answer(payload, source_by_id)

        if verdict != Verdict.INSUFFICIENT_INFORMATION and not citations:
            # Still nothing. Downgrade is the safe direction: it turns an
            # unsupported claim into an admission of ignorance, rather than
            # leaving a confident answer standing on nothing.
            log.warning("verdict %s still had no citations; downgrading", verdict)
            verdict = Verdict.INSUFFICIENT_INFORMATION

    # 5e: consistency. The answer is checked against what Python already
    # worked out, and a contradiction gets one retry with the calculation in
    # front of the model - the same shape as the citation retry above.
    problems, _ = find_contradictions(citations, verdict, computed)
    # Remembered before the retry, because the retry can drop the clause that
    # pointed to the missing list - and still not have answered the question.
    leaned_on_missing_list = verdict in (Verdict.COVERED, Verdict.CONDITIONAL) and any(
        c.clause_id in computed.absent and c.effect != "denies" for c in citations
    )
    if problems:
        log.info("answer contradicts computed results; retrying once: %s", problems)
        payload = await reason(
            scenario, facts, considered, computed,
            nudge=CONSISTENCY_NUDGE.format(problems="\n".join(problems)),
            use_cache=not resample,
        )
        citations, verdict = _read_answer(payload, source_by_id)

        # Still citing a clause the arithmetic cleared. Under a COVERED verdict
        # that is a wrong label, not a wrong clause: the answer says nothing is
        # cut or refused, the arithmetic says the clause cuts nothing, and both
        # agree it permits. Measured on `room-rent-within-cap`, four fresh
        # samples of four: the retry answered covered - "8,000 does not exceed
        # the 10,000 limit ... paid in full" - and still labelled clause 5.1
        # "reduces". Dropping that citation left a right answer with nothing
        # behind it, and the downgrade below turned it into
        # insufficient_information. So it is relabelled instead.
        #
        # Under any other verdict the citation is doing work the arithmetic
        # rules out, so it goes. Either way the verdict is left alone - code
        # can show a citation is wrong, but not what the right answer is.
        _, irrelevant = find_contradictions(citations, verdict, computed)
        if irrelevant and verdict == Verdict.COVERED:
            citations = [
                replace(c, effect="permits") if c in irrelevant else c for c in citations
            ]
        elif irrelevant:
            log.warning("dropping citation(s) the arithmetic rules out: %s",
                        [c.clause_id for c in irrelevant])
            citations = [c for c in citations if c not in irrelevant]
            if verdict != Verdict.INSUFFICIENT_INFORMATION and not citations:
                verdict = Verdict.INSUFFICIENT_INFORMATION

        if _rests_only_on_a_missing_list(citations, verdict, computed, considered):
            log.warning("verdict %s rests only on a list this document lacks; downgrading", verdict)
            verdict = Verdict.INSUFFICIENT_INFORMATION
        elif (
            leaned_on_missing_list
            and verdict == Verdict.NOT_COVERED
            and not _names_a_refusal(citations, considered)
        ):
            # Measured on `day-care-not-listed`, four fresh samples of four: told
            # the day-care list is missing, the retry answered not_covered
            # "because in-patient cover needs 24 hours", citing that cover clause
            # as permitting. That abandons the question rather than answering
            # it: no exclusion, waiting period or condition is named as refusing.
            # Scoped to answers that leaned on a missing list, because elsewhere
            # right refusals carry sloppy labels too - a refusal on a cover
            # window cites the window clause as "permits".
            log.warning("refusal after the missing-list retry names no refusing clause; downgrading")
            verdict = Verdict.INSUFFICIENT_INFORMATION

    # Only when something else still stands: a verdict is never left with nothing
    # behind it by this filter, so it can never trigger a downgrade.
    noise = unraised_reductions(citations, computed)
    if noise and len(noise) < len(citations):
        citations = [c for c in citations if c not in noise]

    return ScenarioResult(
        verdict=verdict,
        reasoning=payload.get("reasoning", "").strip(),
        citations=citations,
        facts=facts,
        missing_facts=missing,
        verified=all(c.verified for c in citations),
        clauses_considered=len(considered),
    )
