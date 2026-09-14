"""Deciding whether an expense fell inside a cover window. Contains no LLM.

WHY THIS MODULE EXISTS
----------------------
Indian health policies pay for treatment around a hospital stay only within a
stated window:

    "Medical Expenses incurred during the sixty days immediately preceding
     the date of admission"                                    (clause 2.2)
    "Medical Expenses incurred during the ninety days immediately following
     the date of discharge"                                    (clause 2.3)

The eval case `post-hospitalisation-too-late` - a follow-up scan 120 days after
discharge - failed in every run on record. The model answered `covered` or
`conditional`. The correct answer is `not_covered`, because 120 is more than 90.

That is not a language problem. It is the same mistake `waiting.py` was written
to fix (is 5 years more than 36 months) and `reduction.py` after it (is 8,000
more than 1% of 10 lakh): a comparison between two numbers, handed to a 7B
model inside a paragraph of legal prose. This is the third family of that
mistake, and it gets the same split:

    reading "the ninety days immediately following discharge"  -> the model
    reading "a scan 120 days after I was discharged"           -> the model
    deciding whether 120 <= 90                                 -> this module

WHY THE ANCHOR IS PART OF THE COMPARISON
----------------------------------------
A window has a side. Fifty days BEFORE admission says nothing about the window
AFTER discharge, and comparing them would be comparing numbers that measure
different things - the operand problem this project has hit before, when a
co-payment turned out to key on age at inception rather than age now. So each
window and each expense carries an anchor, and a comparison is only made when
the anchors match.

WHY MOST CHECKS PRINT NOTHING
-----------------------------
Most questions never mention expenses before or after a stay. Those checks come
back NOT_RAISED and render to nothing at all. The reduction module learned this
at a measured cost: telling the model about caps nobody had asked about turned
confident correct answers into `insufficient_information`. A window nobody's
question touches is not doubt about that window - it is not what the question
is about.
"""

from dataclasses import dataclass
from enum import StrEnum


class Anchor(StrEnum):
    BEFORE_ADMISSION = "before_admission"
    AFTER_DISCHARGE = "after_discharge"


_SIDE = {
    Anchor.BEFORE_ADMISSION: "before admission",
    Anchor.AFTER_DISCHARGE: "after discharge",
}


class WindowStatus(StrEnum):
    WITHIN = "within"
    OUTSIDE = "outside"
    # The person placed the expense on this side of the stay but never said
    # how far from it. A real gap that can decide the answer.
    UNKNOWN = "unknown"
    # The person said nothing about expenses on this side of a stay. Not a gap:
    # this window is simply not what the question is about. Prints nothing.
    NOT_RAISED = "not_raised"


@dataclass
class WindowCheck:
    clause_id: str
    anchor: str
    window_days: int
    offset_days: int | None
    status: WindowStatus

    def describe(self) -> str:
        """A sentence for the reasoning prompt, stating the arithmetic done."""
        side = _SIDE[Anchor(self.anchor)]
        rule = (
            f"clause {self.clause_id} pays only for expenses within "
            f"{self.window_days} days {side}"
        )
        if self.status is WindowStatus.OUTSIDE:
            return (
                f"OUTSIDE THE WINDOW: {rule}. These were {self.offset_days} days "
                f"{side}, so clause {self.clause_id} does NOT pay for them."
            )
        if self.status is WindowStatus.WITHIN:
            # "Its other conditions still apply" is load-bearing. Clause 2.3
            # also requires the same condition and an accepted in-patient
            # claim; meeting the timing is not meeting the clause.
            return (
                f"WITHIN THE WINDOW: {rule}. These were {self.offset_days} days "
                f"{side}, so the timing is satisfied. The clause's other "
                f"conditions still apply."
            )
        return (
            f"CANNOT TELL: {rule}, but how many days {side} these expenses "
            f"fell was NOT STATED."
        )


def evaluate(
    clauses, anchor: str | None, offset_days: int | None
) -> list[WindowCheck]:
    """Compare every cover window against when the person's expense fell.

    `clauses` is any iterable of objects exposing `clause_id`,
    `cover_window_days` and `cover_window_anchor`; clauses without a window are
    skipped here rather than filtered by the caller.

    `anchor` and `offset_days` describe the expense: which side of the stay,
    and how many days from it. Either may be None.
    """
    checks: list[WindowCheck] = []
    for clause in clauses:
        window = getattr(clause, "cover_window_days", None)
        clause_anchor = getattr(clause, "cover_window_anchor", None)
        if not window or window <= 0 or clause_anchor not in _SIDE:
            continue

        if anchor != clause_anchor:
            status = WindowStatus.NOT_RAISED
        elif offset_days is None:
            status = WindowStatus.UNKNOWN
        # Inclusive. "During the ninety days immediately following discharge"
        # includes day ninety; excluding it would refuse a claim on the last
        # day the policy promises to pay.
        elif offset_days <= window:
            status = WindowStatus.WITHIN
        else:
            status = WindowStatus.OUTSIDE

        checks.append(
            WindowCheck(
                clause_id=clause.clause_id,
                anchor=clause_anchor,
                window_days=window,
                offset_days=offset_days,
                status=status,
            )
        )
    return checks


def render(checks: list[WindowCheck]) -> str:
    """Format the raised checks for the reasoning prompt; nothing if none."""
    raised = [c for c in checks if c.status is not WindowStatus.NOT_RAISED]
    if not raised:
        return ""
    lines = ["WHEN THE EXPENSES FELL (arithmetic, not opinion)."]
    lines += [f"- {check.describe()}" for check in raised]
    return "\n".join(lines)
