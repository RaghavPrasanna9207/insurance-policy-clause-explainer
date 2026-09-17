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

import re
from dataclasses import dataclass, field
from enum import StrEnum

# A pre-existing diseases waiting period is recognised by its opening words, the
# clause's own heading. Not by any mention: the IRDAI specified-disease clause
# mentions pre-existing diseases in its body, only to say which of two periods
# wins, and is not about them.
_PRE_EXISTING = re.compile(r"pre[\s-]?existing", re.IGNORECASE)
_OPENING_CHARS = 60


class WaitingStatus(StrEnum):
    SERVED = "served"
    NOT_SERVED = "not_served"
    # The person did not say how long they have held the policy, so no
    # comparison is possible. Not a failure - the honest answer.
    UNKNOWN = "unknown"
    # The clause sets more than one period and only the shorter ones have
    # passed. Which one covers this treatment is a question about the clause's
    # lists - language - so the arithmetic stops here and says so.
    PARTLY_SERVED = "partly_served"


@dataclass
class WaitingCheck:
    clause_id: str
    # Every period the clause sets, shortest first. Usually one; IRDAI's
    # specified-disease clause sets 24 months for one list and 36 for another.
    required_days: list[int]
    held_days: int | None
    status: WaitingStatus
    # The clause's own carve-outs, already checked against its text at
    # analysis time (app/grounding.py, verify_exception).
    exceptions: list[str] = field(default_factory=list)
    # Words from the procedure or condition the person named that this clause
    # also uses - a lookup (scenario.named_in_question), not a judgement.
    named: list[str] = field(default_factory=list)
    # A pre-existing diseases waiting period, which bars only an illness the
    # person already had.
    pre_existing: bool = False

    def describe(self) -> str:
        """A sentence for the reasoning prompt, stating the arithmetic done."""
        who = f"clause {self.clause_id}"
        if self.named:
            words = ", ".join(f'"{w}"' for w in self.named)
            who += f" (it names {words}, from the question)"
        requires = _either(self.required_days)
        if len(self.required_days) > 1:
            requires += " (different periods for different treatments)"

        if self.status is WaitingStatus.PARTLY_SERVED:
            # M16, Star's specified-disease clause: 24 months for hernia, 36 for
            # joint replacement. Stored as one number, 36, a hernia 30 months in
            # would have been told a lifted bar still stood.
            served = [d for d in self.required_days if d <= self.held_days]
            waiting = [d for d in self.required_days if d > self.held_days]
            return (
                f"{who}: requires {requires}. Policy held {_human(self.held_days)}, "
                f"so the {_either(served)} period is served and the {_either(waiting)} "
                f"period is NOT. Which period this clause sets for this treatment "
                f"decides whether it still blocks the claim: read the clause"
            )

        if self.pre_existing and self.status is not WaitingStatus.SERVED:
            # Regression: on the Star policy, "this waiting period still
            # applies and blocks treatment" for the pre-existing diseases bar
            # was the reason given for a hernia, a dengue fever and a stay in
            # Dubai, none of them described as an illness from before the
            # policy. A qualifier appended to that sentence changed nothing:
            # dengue and Dubai kept citing it, the model's own reasoning
            # quoting the opening words. So the line now OPENS with whom the
            # bar concerns - the served-period lesson again, that the first
            # words of a line are the ones acted on.
            held = (
                "how long the policy has been held was NOT STATED"
                if self.status is WaitingStatus.UNKNOWN
                else f"policy held {_human(self.held_days)}, so not yet served"
            )
            return (
                f"{who}: concerns ONLY an illness the person already had when the "
                f"policy began. Unless the description says this one had begun by "
                f"then, it is not the reason for this claim. (Requires "
                f"{requires}; {held}.)"
            )
        if self.status is WaitingStatus.UNKNOWN:
            return (
                f"{who}: requires {requires}, "
                f"but how long the policy has been held was NOT STATED, so whether "
                f"this bar has lifted cannot be determined"
            )
        if self.status is WaitingStatus.SERVED and self.named:
            # The narrow served wording below, on a line of its own. A first
            # version said this period "no longer stands in the way of this
            # treatment", and `star-hernia-after-wait` went from conditional to
            # covered, three samples of three: read as permission, not as one
            # bar removed.
            return (
                f"{who}: requires {requires}, policy held "
                f"{_human(self.held_days)} -> this waiting period is served and no "
                f"longer applies (it says nothing about any other clause)"
            )
        if self.status is WaitingStatus.SERVED:
            # Narrow wording on purpose. An earlier version said this clause
            # "does NOT block the claim", which reads as a verdict on the whole
            # question rather than on this one clause. With four served periods
            # listed, the model saw four statements that nothing blocks the
            # claim and answered "covered" to questions about co-payments and
            # room-rent caps. Correct facts, overreaching phrasing.
            return (
                f"clause {self.clause_id}: requires {requires}, "
                f"policy held {_human(self.held_days)} -> this waiting period no "
                f"longer applies (it says nothing about any other clause)"
            )
        blocked = (
            f"{who}: requires {requires}, "
            f"policy held {_human(self.held_days)} -> this waiting period still "
            f"applies and blocks treatment covered by THIS clause"
        )
        if not self.exceptions:
            return blocked
        # Measured: without this, "hit by a car ten days in" was refused under
        # a 30-day bar that says "except claims arising out of an Accident" -
        # the model followed "blocks" over the clause's own carve-out, three
        # times out of three. Not served and excepted are both true; this line
        # had only been saying the first. Whether the situation IS an accident
        # is language, so it is named here and decided by the model.
        carve_outs = "; ".join(f'"{e}"' for e in self.exceptions)
        return (
            f"{blocked}, EXCEPT where the situation falls within this clause's "
            f"own exception: {carve_outs}. Decide from the description whether it does"
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


def _either(days: list[int]) -> str:
    """"36 months", or "24 months or 36 months" for a clause that sets two."""
    return " or ".join(_human(d) for d in days)


def evaluate(
    clauses, days_held: int | None, named: dict[str, list[str]] | None = None
) -> list[WaitingCheck]:
    """Compare every waiting period against how long the policy has been held.

    `clauses` is any iterable of objects exposing `clause_id` and
    `waiting_periods_days`; only those with a duration are considered, so
    exclusions and sub-limits are ignored here rather than filtered by the
    caller. `named` maps a clause id to the words it shares with what the
    person described.
    """
    named = named or {}
    checks: list[WaitingCheck] = []
    for clause in clauses:
        required = sorted({d for d in getattr(clause, "waiting_periods_days", None) or [] if d > 0})
        if not required:
            continue

        # Served only when the LONGEST period has passed, and not served only
        # when even the shortest has not: in between, the answer depends on
        # which period covers this treatment.
        if days_held is None:
            status = WaitingStatus.UNKNOWN
        elif days_held >= required[-1]:
            status = WaitingStatus.SERVED
        elif days_held < required[0]:
            status = WaitingStatus.NOT_SERVED
        else:
            status = WaitingStatus.PARTLY_SERVED

        checks.append(
            WaitingCheck(
                clause_id=clause.clause_id,
                required_days=required,
                held_days=days_held,
                status=status,
                exceptions=list(getattr(clause, "exceptions", None) or []),
                named=list(named.get(clause.clause_id, [])),
                pre_existing=bool(_PRE_EXISTING.search(
                    getattr(clause, "text", "")[:_OPENING_CHARS])),
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

    # The model tends to cite the first bar it reads. A period naming what the
    # person described goes first; a pre-existing diseases period, which may
    # not concern them at all, goes last. Otherwise the policy's own order.
    def first(check: WaitingCheck) -> tuple[bool, bool]:
        return (not check.named, check.pre_existing)

    # A partly served period may still block, so it is listed with the ones
    # that do - and "no waiting period blocks this claim" is not said beside it.
    blocking = sorted(
        (c for c in checks
         if c.status in (WaitingStatus.NOT_SERVED, WaitingStatus.PARTLY_SERVED)),
        key=first,
    )
    unknown = sorted((c for c in checks if c.status is WaitingStatus.UNKNOWN), key=first)
    # A served period that names the person's treatment is the reason a claim
    # is not barred, so it keeps a line of its own.
    served_named = [c for c in checks if c.status is WaitingStatus.SERVED and c.named]
    served = [c for c in checks if c.status is WaitingStatus.SERVED and not c.named]

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

    for check in [*served_named, *blocking, *unknown]:
        lines.append(f"- {check.describe()}")

    if not blocking and not unknown:
        lines.append("- No waiting period blocks this claim. Some other clause decides it.")

    return "\n".join(lines)
