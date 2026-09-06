"""Deciding whether a waiting period has been served. Contains no LLM.

WHY THIS MODULE EXISTS
----------------------
The scenario simulator scored 0.688 on verdict accuracy, and three of its five
failures were the same mistake:

    "held the policy 5 years, 36-month waiting period"   -> said NOT covered
    "chest infection 2 weeks in, 30-day waiting period"  -> said covered
    "cataract 4 years in, 24-month waiting period"       -> said NOT covered

None of those is a language problem. Each is a comparison between two numbers,
and the model was getting them wrong roughly half the time - which is what
asking a 7B model to do arithmetic looks like.

This project's governing principle says deterministic wherever possible, a model
only where language understanding is genuinely required. Stages 2 and 4 already
follow it. Stage 5 had quietly broken it, so the split is restored here:

    reading "thirty six months" out of legal prose   -> the model's job
    deciding whether 60 >= 36                        -> this module's job

The model still decides everything that needs judgment. It is simply no longer
asked to compare integers, and is told the answers instead.

WHY "UNKNOWN" IS A FIRST-CLASS RESULT
-------------------------------------
If the person never said how long they have held the policy, the comparison
cannot be made. Returning UNKNOWN rather than guessing is what lets the
reasoning step answer `insufficient_information` honestly - and a fourth failure
was the model asserting a verdict when the timing had never been stated.
"""

from dataclasses import dataclass
from enum import StrEnum


class WaitingStatus(StrEnum):
    SERVED = "served"
    NOT_SERVED = "not_served"
    # The person did not say how long they have held the policy, so no
    # comparison is possible. Not a failure - the honest answer.
    UNKNOWN = "unknown"


@dataclass
class WaitingCheck:
    clause_id: str
    required_days: int
    held_days: int | None
    status: WaitingStatus

    def describe(self) -> str:
        """A sentence for the reasoning prompt, stating the arithmetic done."""
        if self.status is WaitingStatus.UNKNOWN:
            return (
                f"clause {self.clause_id}: requires {_human(self.required_days)}, "
                f"but how long the policy has been held was NOT STATED, so whether "
                f"this bar has lifted cannot be determined"
            )
        if self.status is WaitingStatus.SERVED:
            # Narrow wording on purpose. An earlier version said this clause
            # "does NOT block the claim", which reads as a verdict on the whole
            # question rather than on this one clause. With four served periods
            # listed, the model saw four statements that nothing blocks the
            # claim and answered "covered" to questions about co-payments and
            # room-rent caps. Correct facts, overreaching phrasing.
            return (
                f"clause {self.clause_id}: requires {_human(self.required_days)}, "
                f"policy held {_human(self.held_days)} -> this waiting period no "
                f"longer applies (it says nothing about any other clause)"
            )
        return (
            f"clause {self.clause_id}: requires {_human(self.required_days)}, "
            f"policy held {_human(self.held_days)} -> this waiting period still "
            f"applies and blocks treatment covered by THIS clause"
        )


def _human(days: int | None) -> str:
    """Render a day count the way the policy phrases it.

    The comparison happens in days, but "1080 days" is not how anyone reads a
    36-month waiting period - and a prompt that says it invites the model to
    start converting units again, which is the exact job this module took away
    from it.
    """
    if days is None:
        return "an unstated period"
    if days >= 365 and days % 365 == 0:
        n = days // 365
        return f"{n} year" + ("s" if n > 1 else "")
    if days >= 30 and days % 30 == 0:
        n = days // 30
        return f"{n} month" + ("s" if n > 1 else "")
    return f"{days} day" + ("s" if days != 1 else "")


def evaluate(clauses, days_held: int | None) -> list[WaitingCheck]:
    """Compare every waiting period against how long the policy has been held.

    `clauses` is any iterable of objects exposing `clause_id` and
    `waiting_period_months`; only those with a duration are considered, so
    exclusions and sub-limits are ignored here rather than filtered by the
    caller.
    """
    checks: list[WaitingCheck] = []
    for clause in clauses:
        required = getattr(clause, "waiting_period_days", None)
        if not required or required <= 0:
            continue

        if days_held is None:
            status = WaitingStatus.UNKNOWN
        elif days_held >= required:
            status = WaitingStatus.SERVED
        else:
            status = WaitingStatus.NOT_SERVED

        checks.append(
            WaitingCheck(
                clause_id=clause.clause_id,
                required_days=required,
                held_days=days_held,
                status=status,
            )
        )
    return checks


def render(checks: list[WaitingCheck]) -> str:
    """Format the computed results for the reasoning prompt.

    Presented as settled fact rather than as something to work out - but scoped
    tightly to what it actually settles.

    That scoping was learned the hard way. A first version stated each served
    period as "this clause does NOT block the claim", which is true of the
    clause and reads as a verdict on the question. With four served periods
    listed, the model saw four statements that nothing blocked the claim and
    began answering "covered" to questions about co-payments and room-rent caps
    - dropping verdict accuracy from 0.688 to 0.625 while the arithmetic it was
    added to fix worked perfectly.

    Correct facts can still mislead if their scope is implied rather than
    stated. The SCOPE paragraph exists to say what this block does NOT decide.
    """
    if not checks:
        return ""

    blocking = [c for c in checks if c.status is WaitingStatus.NOT_SERVED]
    unknown = [c for c in checks if c.status is WaitingStatus.UNKNOWN]
    served = [c for c in checks if c.status is WaitingStatus.SERVED]

    lines = [
        "WAITING PERIODS, ALREADY CALCULATED FOR YOU (arithmetic, not opinion).",
        "",
        "SCOPE: waiting periods ONLY. This says nothing about exclusions, payout",
        "caps, co-payments or notice conditions - those are decided by other",
        "clauses that you must still read. A waiting period that no longer",
        "applies is NOT a reason to answer 'covered'.",
        "",
    ]

    # Served periods collapse to one line. They decide nothing, and giving each
    # its own emphatic line made the model cite waiting periods on questions
    # about co-payments and notice deadlines while missing the clause that
    # actually answered them.
    if served:
        ids = ", ".join(c.clause_id for c in served)
        lines.append(
            f"- Already satisfied, so IRRELEVANT here and not worth citing: {ids}"
        )

    for check in blocking:
        lines.append(f"- {check.describe()}")
    for check in unknown:
        lines.append(f"- {check.describe()}")

    if not blocking and not unknown:
        lines.append("- No waiting period blocks this claim. Some other clause decides it.")

    return "\n".join(lines)
