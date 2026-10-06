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
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

# A pre-existing diseases waiting period is recognised by its opening words, the
# clause's own heading. Not by any mention: the IRDAI specified-disease clause
# mentions pre-existing diseases in its body, only to say which of two periods
# wins, and is not about them.
# "PED" too, case-sensitive: HDFC heads its option "PED waiting period
# modification", and the lower-case letters turn up inside words ("speed").
_PRE_EXISTING = re.compile(r"(?i:pre[\s-]?existing)|\bPED\b")
_OPENING_CHARS = 60
# A period that bars only the treatments its clause lists (IRDAI's "Specified
# disease/procedure waiting period", maternity), recognised the same way. Only
# these get the narrower line; anything not recognised keeps the firm one,
# because calling an all-illness period list-only would pay claims it bars.
_LISTED = re.compile(r"specified|maternity", re.IGNORECASE)
# A period outside the policy's waiting-period and exclusions section bars only
# what its own clause covers: HDFC's 36 months for planned treatment abroad,
# under an optional cover in Section B, and a 30-day wait in a chronic-care
# add-on, both read as "blocks" and both refusing an accident on day 10.
# Measured on five policies, every general period sits in a section named for
# waiting periods or exclusions. An unknown section changes nothing.
# ponytail: a general period in an oddly named section would get the narrow
# line; name that section here if a real policy has one.
_GENERAL_SECTION = re.compile(r"wait|exclu", re.IGNORECASE)
_HEADED_WAIT = re.compile(r"waiting\s+period", re.IGNORECASE)
_DURATION = re.compile(r"\b(\d+)\s*(day|month|year)s?\b", re.IGNORECASE)
_DAYS_PER = {"day": 1, "month": 30, "year": 365}
# The list is quoted into the line only up to this length. M23 set 600, which
# left Star's ~1,970-character clause unquoted, and its dengue answer applied
# the list to dengue. The ~560 tokens are room the prompt has: a policy over
# PICK_ABOVE_TOKENS of clause text is narrowed long before the 28,672-token
# window, and an overflow is refused by the client, never silently truncated.
# ponytail: a longer clause keeps the unquoted line.
_QUOTE_LIST_CHARS = 2_500
# Where a clause says "List of ...", only the list is quoted. HDFC's quote, with
# the five conditions before its list, ran to 2,400 characters and drowned the
# 30-day line beside it: typhoid on day 20 was paid 3 times in 3.
_LIST_STARTS = re.compile(r"list\s+of\b", re.IGNORECASE)
# A run of definitions ("X means ...") is not a waiting period, even when one
# term it defines is. Star's definitions page, split for length, left a piece
# defining "Specific Waiting Period" typed waiting_period with 36 months, and
# every claim within three years was told it "still applies and blocks"
# (Failure 86). Measured on three policies: that piece says "means" 8 times,
# every real waiting period 0 times.
_DEFINES = re.compile(r"\bmeans\b", re.IGNORECASE)
_DEFINITIONS_BLOCK = 2
# A waiting period says so. HDFC's plan-comparison table has rows like
# "1.6 Post-Hospitalization 180 days 180 days ...", typed waiting_period: a cover
# window read as a 180-day bar on every claim. Measured on five policies, every
# real waiting period says "wait" in its heading or text, and those rows don't.
# ponytail: one that never says "wait" loses its computed line (its text still
# reaches the model); add its wording here if a real policy has one.
_WAITS = re.compile(r"wait", re.IGNORECASE)


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
    # Whether the person's condition predates the policy - the scenario fact,
    # after code has settled it where it can ("yes", "no" or "unknown").
    condition_predates_policy: str = "unknown"
    # What the period is for, from the clause heading ("Maternity Waiting
    # Period"). M22: a line saying only "blocks treatment covered by THIS
    # clause" was read as blocking everything - a 36-month maternity wait was
    # given as the reason to refuse unrelated claims on a new policy.
    subject: str = ""
    # Bars only the treatments its clause lists.
    listed: bool = False
    # The clause's own words, quoted into the line of a short list-type period.
    listing: str = ""

    def concerns_nothing_here(self) -> bool:
        """A pre-existing diseases bar, for a condition settled as beginning after the policy."""
        return self.pre_existing and self.condition_predates_policy == "no"

    def describe(self) -> str:
        """A sentence for the reasoning prompt, stating the arithmetic done."""
        who = f"clause {self.clause_id}"
        if self.subject:
            who += f" ({self.subject})"
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

        if self.concerns_nothing_here():
            # M22: settled in code - the person said so plainly, or two stated
            # durations decided it. Hedging here ("unless the description
            # says...") was measured to lose: told nothing firmer, the model
            # refused a kidney stone "never had before taking the policy" under
            # this bar.
            return (
                f"{who}: concerns ONLY an illness the person already had when the "
                f"policy began. This one began after the policy did, so this "
                f"waiting period does not concern this claim"
            )
        # No firm line for "yes" on a SERVED wait: measured in M22, such a
        # claim, told the condition was pre-existing, was refused under the
        # cosmetic exclusion 5 times in 5. An unserved one is different: HDFC,
        # "diabetes for six years", policy held one year, got the hedge below
        # and was paid 3 times in 3, the model reading an optional PED
        # modification as having cut the wait to 12 months.
        if (self.pre_existing and self.condition_predates_policy == "yes"
                and self.status is WaitingStatus.NOT_SERVED):
            return (
                f"{who}: concerns ONLY an illness the person already had when the "
                f"policy began. This one had begun by then, and the policy has been "
                f"held {_human(self.held_days)} of the {requires} required, so this "
                f"waiting period still applies and blocks this claim"
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
        if self.listed and not self.named and not self.exceptions:
            # M22: told "still applies and blocks treatment covered by THIS
            # clause", the model refused a thyroid problem, malaria and
            # appendicitis under the specified-disease period, copying the
            # line into its reasoning; none is on the clause's list. The
            # pre-existing lesson again: open with whom the bar concerns.
            # Not when the question's words appear in the clause - then it
            # probably is listed, and the firm line stays.
            # M23: told only that the list exists, the model still refused
            # appendicitis under 3.3 - quoting the list in its answer. With
            # the list in this line, 3 of 3 answers were right. Naming the
            # person's treatment beside it as well was measured worse (it
            # broke gallstones), so the comparison is left to the model.
            lists = f': "{self.listing}"' if self.listing else ""
            return (
                f"{who}: bars ONLY the treatments this clause lists{lists}. If this "
                f"treatment is not one of them, this period is not the reason "
                f"for this claim. (Requires {requires}; policy held "
                f"{_human(self.held_days)}, so not yet served.)"
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
    clauses, days_held: int | None, named: dict[str, list[str]] | None = None,
    condition_predates_policy: str = "unknown",
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
        heading = unicodedata.normalize("NFKC", getattr(clause, "heading", "") or "").strip()
        text = " ".join(unicodedata.normalize("NFKC", getattr(clause, "text", "") or "").split())
        if not required and _HEADED_WAIT.search(heading):
            # The analysis typed HDFC's "Specified Disease/Procedure waiting
            # period" an exclusion and gave it no period, so it got no line,
            # and a hernia 14 months in was told no waiting period applied.
            # Reading "24 months" off a clause headed as a waiting period is
            # extraction, not judgement; measured on five policies, it fires on
            # that clause alone.
            required = sorted({int(n) * _DAYS_PER[u.lower()] for n, u in _DURATION.findall(text)})
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

        # NFKC (above): PDFs set "fi" as one ligature character, so Star's
        # "Speciﬁed disease" never matched "specified" and got the firm line.
        if len(_DEFINES.findall(text)) >= _DEFINITIONS_BLOCK or not _WAITS.search(f"{heading} {text}"):
            continue
        section = getattr(clause, "section_path", "") or ""
        listed = (bool(_LISTED.search(f"{heading} {text[:_OPENING_CHARS]}"))
                  or bool(section) and not _GENERAL_SECTION.search(section))
        body = text.removeprefix(heading).strip()
        listing = body[m.start():] if (m := _LIST_STARTS.search(body)) else body
        checks.append(
            WaitingCheck(
                clause_id=clause.clause_id,
                required_days=required,
                held_days=days_held,
                status=status,
                exceptions=list(getattr(clause, "exceptions", None) or []),
                named=list(named.get(clause.clause_id, [])),
                pre_existing=bool(_PRE_EXISTING.search(text[:_OPENING_CHARS])),
                condition_predates_policy=condition_predates_policy,
                listed=listed,
                listing=listing if listed and len(listing) <= _QUOTE_LIST_CHARS else "",
                # The heading repeats the clause number; the line already has it.
                subject=re.sub(r"^[\d.]+\s*", "", heading),
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

    # A pre-existing diseases bar for a condition settled as new is not a bar
    # at all, served or not: it gets its own line saying so, and is kept out of
    # the blocking and unknown lists that would otherwise count it.
    not_concerned = [c for c in checks if c.concerns_nothing_here()]
    checks = [c for c in checks if not c.concerns_nothing_here()]

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

    for check in [*not_concerned, *served_named, *blocking, *unknown]:
        lines.append(f"- {check.describe()}")

    if not blocking and not unknown:
        lines.append("- No waiting period blocks this claim. Some other clause decides it.")

    return "\n".join(lines)
