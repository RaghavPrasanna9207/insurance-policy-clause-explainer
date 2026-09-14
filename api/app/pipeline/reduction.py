"""Deciding which clauses REDUCE a payout rather than refuse it. No LLM.

WHY THIS MODULE EXISTS
----------------------
The scenario simulator could answer "will this claim be refused?" and could not
reliably answer "will this claim be paid in full?". Those are different
questions, and only the first one was being asked.

The measured failure that named the problem:

    "I bought this policy at 67 and I am claiming three years later"
        expected: conditional  (a 20% co-payment applies to every claim)
        got:      covered

Nothing in the pipeline was wrong about the facts. The co-payment clause was in
the prompt, the waiting periods had been computed and found satisfied, and the
model concluded - correctly, as far as it went - that nothing BLOCKED the claim.
It then stopped. A real policyholder reading that answer files a claim
expecting the full amount and loses a fifth of it.

Widening the eval set from 16 cases to 40 showed this was not one case. Of the
thirteen cases that turn on a reduction, the failures ran in both directions:

    room-rent-within-cap   a room UNDER the cap read as a breach   (false positive)
    icu-rate-breach        a rate OVER the cap read as fine        (false negative)
    oral-chemo-limit       a named sub-limit missed entirely
    senior-copay           the co-payment missed entirely

WHAT IS ARITHMETIC HERE, AND WHAT IS NOT
----------------------------------------
This is the project's governing principle applied to a second family of
comparisons. `waiting.py` already took integer comparison of DURATIONS away
from the model. The same mistake was still being made with MONEY and with AGE,
and nobody had noticed because the 16-case set contained no case that needed
either comparison.

    reading "one percent of the Sum Insured per day" out of prose  -> the model
    deciding whether 8,000 > 1% of 10,00,000                       -> this module
    reading "completed sixty years at first inception"             -> the model
    deciding whether 58 >= 60                                      -> this module
    deciding whether a drip counts as "cataract treatment"         -> the model

The last line matters as much as the others. Not every reduction reduces to a
comparison: whether a procedure falls under the cataract sub-limit or the
modern-treatment limit is a question about language, and this module must not
pretend otherwise. For those it returns JUDGEMENT - "this cap exists, it is
your call whether it bites" - which is a different thing from UNKNOWN.

FIVE STATUSES, BECAUSE FEWER WOULD LIE
--------------------------------------
    APPLIES         computed, and it definitely bites
    DOES_NOT_APPLY  computed, and it definitely does not
    UNKNOWN         an operand was raised but left incomplete
    NOT_RAISED      nothing the person said bears on this cap at all
    JUDGEMENT       no comparison exists; this one turns on language

Collapsing UNKNOWN into DOES_NOT_APPLY would be the expensive mistake. Someone
who did not mention their room category has not thereby stayed within the cap,
and telling them the claim is fine is exactly the confident guess this project
exists to prevent.

Collapsing JUDGEMENT into UNKNOWN would be a quieter one: it would report a
missing fact where the fact is present and only the reading is open, which
sends the reasoning step looking for information it already has.

NOT_RAISED versus UNKNOWN is the distinction this module was FIRST BUILT
WITHOUT, and its absence cost 0.125 of verdict accuracy in a measured run. Both
mean "no comparison was made", so treating them as one status looked harmless.
They differ in what that failure MEANS: someone who gave a sum insured but no
room rate has left a real question open, while someone describing a broken hip
has not left the room-rent cap open - they were never talking about it. Printed
identically, the second kind manufactures doubt on nearly every question,
because most descriptions mention neither a room nor a sum insured. See
render() for what that did to the numbers.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReductionStatus(StrEnum):
    APPLIES = "applies"
    DOES_NOT_APPLY = "does_not_apply"
    UNKNOWN = "unknown"
    JUDGEMENT = "judgement"
    # The person said nothing that could bear on this cap at all - no room, no
    # sum insured, no age. Distinct from UNKNOWN, and the distinction is the
    # difference between a useful block and a useless one. See render().
    NOT_RAISED = "not_raised"


class ReductionKind(StrEnum):
    COPAY = "copay"
    ROOM_CAP = "room_cap"
    OTHER_CAP = "other_cap"


# Indian numbering, because a policy schedule says "5 lakh" and never 500000.
# Converting it is arithmetic and belongs here, for the same reason converting
# "two weeks" to days belongs in Python: the model's job is to report that the
# person said five and lakh, not to multiply.
_MULTIPLIER = {"rupees": 1, "thousand": 1_000, "lakh": 100_000, "crore": 10_000_000}


def sum_insured_rupees(facts: dict[str, Any]) -> int | None:
    """The sum insured in rupees, or None if it was never stated."""
    value = facts.get("sum_insured_value")
    unit = facts.get("sum_insured_unit")
    if not value or value <= 0 or unit not in _MULTIPLIER:
        return None
    return value * _MULTIPLIER[unit]


def age_at_inception(facts: dict[str, Any], policy_age_days: int | None) -> int | None:
    """How old the person was when the policy STARTED, or None.

    This is the operand the co-payment clause actually keys on, and it is not
    the same number as the person's age today. "I am 68 now" says nothing about
    whether they were over 60 when they bought the policy - they could have
    been 50. Treating current age as age at inception would fire a 20%
    co-payment at someone who does not owe it.

    Two routes to it, both arithmetic:
      - the person said it outright ("I bought this policy at 67")
      - or they gave their age now and how long they have held the policy,
        and the subtraction is ours to do

    Neither available -> None, and the check comes back UNKNOWN rather than
    guessing at a number that decides a fifth of someone's claim.
    """
    stated = facts.get("age_at_policy_start")
    if stated and stated > 0:
        return stated

    age_now = facts.get("age")
    if age_now and age_now > 0 and policy_age_days is not None:
        derived = age_now - policy_age_days // 365
        # A negative or absurd result means one of the two inputs was misread.
        # Returning None is better than returning nonsense that then decides
        # a co-payment.
        if 0 < derived <= age_now:
            return derived
    return None


@dataclass
class ReductionCheck:
    clause_id: str
    kind: ReductionKind
    status: ReductionStatus
    detail: str
    # For a JUDGEMENT clause: words of the described treatment that its own
    # text contains. A lookup, not a decision - see _named_in.
    named: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"clause {self.clause_id}: {self.detail}"


# Words too general to say WHICH treatment a clause is about. "Gallbladder
# operation" shares "operation" with "operation theatre charges", and that is
# not the proportionate-deduction clause naming gallbladder surgery.
_GENERIC_WORDS = frozenset({
    "surgery", "surgeries", "treatment", "treatments", "procedure", "procedures",
    "operation", "therapy", "hospital", "hospitalisation", "hospitalization",
    "admission", "admitted", "care", "medical", "expenses", "room", "rent", "policy",
    "claim", "charges", "condition", "disease", "illness", "injury", "health",
    "unknown", "insured", "person",
})


def _named_in(clause_text: str, facts: dict[str, Any]) -> list[str]:
    """Words of the person's described treatment that this clause's text contains.

    Measured case: oral chemotherapy, and the modern-treatment clause caps "oral
    chemotherapy" by name - yet the model, told only to "read" the sub-limit
    clauses, answered that none applied, three samples of three. Whether a
    clause mentions a word the person used is a lookup, so code does it.
    Whether that clause caps this claim stays with the model.

    Uses the extracted procedure and condition rather than the whole question,
    so "sum insured" and "held the policy" cannot match anything.
    """
    described = " ".join(str(facts.get(k) or "") for k in ("procedure", "condition"))
    mine = {w for w in re.findall(r"[a-z]{4,}", described.lower()) if w not in _GENERIC_WORDS}
    return sorted(mine & set(re.findall(r"[a-z]{4,}", clause_text.lower())))


def _human_rupees(amount: int) -> str:
    """Render an amount the way the person would say it.

    The comparison happens in rupees, but "500000" is not how a policy schedule
    reads, and a prompt full of bare integers invites the model to start doing
    the conversion itself - which is the job this module took away from it.
    """
    if amount >= 10_000_000 and amount % 10_000_000 == 0:
        return f"{amount // 10_000_000} crore"
    if amount >= 100_000 and amount % 100_000 == 0:
        return f"{amount // 100_000} lakh"
    return f"{amount:,} rupees"


def _copay_check(clause, facts, policy_age_days) -> ReductionCheck | None:
    percent = getattr(clause, "copay_percent", None)
    if not percent:
        return None

    threshold = getattr(clause, "copay_min_age_at_inception", None)
    inception_age = age_at_inception(facts, policy_age_days)

    if threshold is None:
        # The clause imposes a co-payment but states no age condition, so it
        # applies to everyone and there is nothing to compare.
        return ReductionCheck(
            clause.clause_id, ReductionKind.COPAY, ReductionStatus.APPLIES,
            f"a {percent}% co-payment applies to the admissible claim amount",
        )

    if inception_age is None:
        # Nothing was said about age in any form. That is not an unresolved
        # question about the co-payment, it is the co-payment not being at
        # issue - and reporting it as doubt is what turned questions about
        # broken hips into "insufficient_information".
        if not facts.get("age") and not facts.get("age_at_policy_start"):
            return ReductionCheck(
                clause.clause_id, ReductionKind.COPAY,
                ReductionStatus.NOT_RAISED, "no age was mentioned",
            )
        # An age WAS given but could not be pinned to inception - "I am 68 and
        # I don't remember when I took the policy out". Here the doubt is real
        # and load-bearing: they may or may not owe a fifth of the claim.
        return ReductionCheck(
            clause.clause_id, ReductionKind.COPAY, ReductionStatus.UNKNOWN,
            f"a {percent}% co-payment applies if the person had reached "
            f"{threshold} at inception, but their age when the policy STARTED "
            f"was not stated, so this cannot be determined",
        )

    if inception_age >= threshold:
        # "IF this claim is payable at all" is load-bearing. This used to say
        # "so this claim is paid at 80%", which asserts the one thing the
        # arithmetic never established - that the claim is paid. Asked about a
        # nose job bought at 70, the model took the line at its word and
        # answered conditional over the cosmetic exclusion, three times of
        # three.
        return ReductionCheck(
            clause.clause_id, ReductionKind.COPAY, ReductionStatus.APPLIES,
            f"aged {inception_age} at inception against a threshold of "
            f"{threshold} -> the {percent}% co-payment DOES apply to this "
            f"person. IF this claim is payable at all, it is paid at "
            f"{100 - percent}% of the admissible amount",
        )

    return ReductionCheck(
        clause.clause_id, ReductionKind.COPAY, ReductionStatus.DOES_NOT_APPLY,
        f"aged {inception_age} at inception against a threshold of {threshold} "
        f"-> the {percent}% co-payment does NOT apply",
    )


def _room_cap_check(clause, facts) -> ReductionCheck | None:
    in_icu = bool(facts.get("room_is_icu"))
    percent = getattr(
        clause, "icu_cap_percent_of_sum_insured" if in_icu else
        "cap_percent_of_sum_insured", None
    )
    if not percent:
        return None

    sum_insured = sum_insured_rupees(facts)
    charged = facts.get("room_rent_per_day_inr")
    label = "ICU charges" if in_icu else "room rent"

    if not sum_insured or not charged or charged <= 0:
        # NEITHER figure given: the person never mentioned a room or a sum
        # insured, so this cap is simply not what their question is about.
        # Saying "cannot be determined" here is technically true and actively
        # harmful - it manufactures doubt about a clause nobody raised.
        if not sum_insured and not charged:
            return ReductionCheck(
                clause.clause_id, ReductionKind.ROOM_CAP,
                ReductionStatus.NOT_RAISED,
                "no room charge or sum insured was mentioned",
            )
        # ONE of the two given: they did raise it, and left it incomplete.
        # That gap is worth naming, because the answer may turn on it.
        missing = "the sum insured" if not sum_insured else "the per-day charge"
        return ReductionCheck(
            clause.clause_id, ReductionKind.ROOM_CAP, ReductionStatus.UNKNOWN,
            f"{label} are capped at {percent}% of the sum insured per day, but "
            f"{missing} was not stated, so whether the cap is exceeded cannot "
            f"be determined",
        )

    limit = sum_insured * percent // 100
    if charged > limit:
        return ReductionCheck(
            clause.clause_id, ReductionKind.ROOM_CAP, ReductionStatus.APPLIES,
            f"{percent}% of {_human_rupees(sum_insured)} is "
            f"{_human_rupees(limit)} per day; {label} of "
            f"{_human_rupees(charged)} per day EXCEED that, so the excess is "
            f"not paid",
        )

    # "does NOT exceed that limit" on purpose: it is the condition the
    # proportionate-deduction clause is written in ("where the room rent
    # exceeds the limit specified in Clause 5.1"), so the model can match the
    # result to that clause without code having to interpret it.
    return ReductionCheck(
        clause.clause_id, ReductionKind.ROOM_CAP, ReductionStatus.DOES_NOT_APPLY,
        f"{percent}% of {_human_rupees(sum_insured)} is "
        f"{_human_rupees(limit)} per day; {label} of {_human_rupees(charged)} "
        f"per day does NOT exceed that limit, so this cap costs nothing here",
    )


def evaluate(clauses, facts: dict[str, Any], policy_age_days: int | None
             ) -> list[ReductionCheck]:
    """Work out, for every clause that reduces a payout, whether it bites.

    `clauses` is any iterable of objects exposing `clause_id` and `clause_type`;
    clauses that reduce nothing are skipped here rather than filtered by the
    caller, the same arrangement `waiting.evaluate` uses.
    """
    checks: list[ReductionCheck] = []
    for clause in clauses:
        check = _copay_check(clause, facts, policy_age_days)
        if check is None:
            check = _room_cap_check(clause, facts)

        if check is not None:
            checks.append(check)
            continue

        # A sub-limit with no comparable numbers attached: a cataract cap, a
        # modern-treatment restriction. Whether it applies is a question about
        # what the treatment WAS, which is language and stays with the model.
        # It is still listed, because the failure being fixed here is not the
        # model getting these wrong - it is nobody asking.
        if getattr(clause, "clause_type", "") == "sub_limit":
            checks.append(
                ReductionCheck(
                    clause.clause_id, ReductionKind.OTHER_CAP,
                    ReductionStatus.JUDGEMENT,
                    "caps or reduces what is paid for certain treatments - "
                    "decide from the clause text whether it covers this one",
                    named=_named_in(getattr(clause, "text", ""), facts),
                )
            )
    return checks


def render(checks: list[ReductionCheck]) -> str:
    """Format the computed reductions for the reasoning prompt.

    THE ORDERING AND THE EMPHASIS ARE PART OF THE MESSAGE, not decoration.

    An earlier module in this pipeline learned that the expensive way. Served
    waiting periods were each given their own emphatic line saying the clause
    "does NOT block the claim"; with four of them listed, the model read four
    statements that nothing blocked the claim and started answering "covered"
    to questions about co-payments. Every statement was true and the aggregate
    misled, and verdict accuracy fell while the arithmetic behind it was
    perfect.

    So the reductions that BITE lead and get a line each. The ones ruled out
    collapse to a single line - they are non-events, and giving each its own
    line would rebuild exactly the failure above in a new place.

    IT WAS REBUILT IN A NEW PLACE ANYWAY, and the measurement caught it. The
    first version of this function collapsed the ruled-out reductions exactly
    as described above, and then handed every UNKNOWN and every JUDGEMENT its
    own emphatic line - without noticing that those two are far more common
    than DOES_NOT_APPLY, because most people describing a broken hip mention
    neither their sum insured nor their age. A question decided entirely by the
    alcohol exclusion arrived carrying five lines of doubt:

        - CANNOT TELL: clause 5.1: room rent are capped at 1% ... not stated
        - CANNOT TELL: clause 5.3: a 20% co-payment applies if ... not stated
        - YOUR CALL: clause 5.2 ... 5.4 ... 5.5

    Verdict accuracy fell from 0.725 to 0.600, and every single new failure
    landed on `conditional` or `insufficient_information` - the two verdicts
    that block reads as. The lesson from the waiting-period module had been
    quoted in this file's docstring and applied to the wrong status.

    Hence NOT_RAISED, which prints nothing at all. Silence about a room is not
    ambiguity about a room, and a clause nobody's question touches should not
    appear in the answer's reasoning at all.
    """
    if not checks:
        return ""

    applies = [c for c in checks if c.status is ReductionStatus.APPLIES]
    unknown = [c for c in checks if c.status is ReductionStatus.UNKNOWN]
    judgement = [c for c in checks if c.status is ReductionStatus.JUDGEMENT]
    ruled_out = [c for c in checks if c.status is ReductionStatus.DOES_NOT_APPLY]

    # Nothing in this policy reduces anything on these facts, and nothing is
    # even arguably at issue. Say nothing rather than print a heading whose
    # only content is that it has no content.
    if not applies and not unknown and not ruled_out and not judgement:
        return ""

    # NOTHING WAS COMPUTED, SO NOTHING IS SAID. Reaching here means every
    # arithmetic check returned NOT_RAISED - the question named no amount, no
    # room and no age - and the only survivors are JUDGEMENT clauses, whose
    # full text is already in this same prompt a few hundred tokens below.
    #
    # An earlier version emitted one line here reminding the model that those
    # clauses cap what they name. It carried no computed finding and repeated
    # nothing the model could not already read, and it cost three cases:
    # initial-waiting-period, dental-no-accident and non-disclosure were each
    # a confident, correct refusal without it and `insufficient_information`
    # with it. A line that adds no information still adds emphasis, and
    # emphasis is not free.
    # A sub-limit whose own text names the described treatment is a finding
    # too - a lookup rather than arithmetic, but not a bare reminder.
    named = [c for c in judgement if c.named]
    unnamed = [c for c in judgement if not c.named]
    computed = applies or unknown or ruled_out or named
    if not computed:
        return ""

    lines = [
        "WHAT REDUCES THE PAYOUT (arithmetic, not opinion).",
        "",
        "SCOPE: these clauses cut the AMOUNT paid. None of them refuses a claim,",
        "and none of them applies to a claim that is refused for some other",
        "reason - there is nothing to take a percentage of. If an exclusion pays",
        "zero, the verdict is not_covered and every line below is irrelevant.",
        "",
        "Where a reduction below DOES apply and the claim is otherwise payable,",
        "the verdict is conditional, never covered. 'Covered' told to someone who",
        "will be paid 80% is a wrong answer that costs them the other 20%.",
        "",
    ]

    for check in applies:
        lines.append(f"- APPLIES: {check.describe()}")
    for check in unknown:
        lines.append(f"- CANNOT TELL: {check.describe()}")

    # A room within its cap gets its own line with the numbers. Every other
    # ruled-out reduction stays collapsed to its id, for the reason above - but
    # a room cap is only ever computed when the person named their room rate,
    # so the room is what they are asking about, and a bare "5.1" left the
    # model to redo the comparison: "paid at 80% of 8,000", three times of three.
    within_room = [c for c in ruled_out if c.kind is ReductionKind.ROOM_CAP]
    collapsed = [c for c in ruled_out if c.kind is not ReductionKind.ROOM_CAP]
    for check in within_room:
        lines.append(f"- WITHIN THE LIMIT: {check.describe()}")
    if collapsed:
        ids = ", ".join(c.clause_id for c in collapsed)
        lines.append(
            f"- Ruled out by the numbers, so IRRELEVANT here and not worth "
            f"citing: {ids}"
        )

    # One line, not one line each. These carry no computed finding - they are
    # a reminder to read a clause that is already in the prompt - and three
    # separate "YOUR CALL" lines gave a reminder the visual weight of a result.
    for check in named:
        words = ", ".join(f'"{w}"' for w in check.named)
        lines.append(
            f"- NAMES THIS TREATMENT: clause {check.clause_id} mentions {words}, "
            f"which is in what the person described. Read it: whether it caps "
            f"this claim is your call from the clause text."
        )
    if unnamed:
        ids = ", ".join(c.clause_id for c in unnamed)
        lines.append(
            f"- Also capped by {ids}, but only for the treatments those clauses "
            f"name. Read them; if this treatment is not one of them, they "
            f"decide nothing here."
        )

    # Not said beside a named sub-limit: "nothing reduces this claim" right
    # under "this clause names your treatment" is two lines disagreeing.
    if not applies and not unknown and not named:
        lines.append(
            "- Nothing here reduces this claim. Some other clause decides it."
        )

    return "\n".join(lines)
