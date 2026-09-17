"""M5 fix: waiting periods are decided arithmetically, not by the model.

Three of five scenario failures were the same mistake - comparing a policy age
against a waiting period and getting it wrong. That is arithmetic, and this
project's rule is that arithmetic does not go to a language model. These tests
pin the comparison down exactly, which is only possible because it is code.
"""

from dataclasses import dataclass, field

from app.pipeline.waiting import WaitingStatus, evaluate, render


@dataclass
class FakeClause:
    clause_id: str
    waiting_periods_days: list[int] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    text: str = ""


def test_an_elapsed_waiting_period_is_served():
    """The exact case the model got wrong: 5 years against a 36-month bar."""
    checks = evaluate([FakeClause("3.2", [1080])], days_held=1800)
    assert checks[0].status is WaitingStatus.SERVED


def test_an_unelapsed_waiting_period_is_not_served():
    checks = evaluate([FakeClause("3.2", [1080])], days_held=240)
    assert checks[0].status is WaitingStatus.NOT_SERVED


def test_the_boundary_is_inclusive():
    """Exactly 36 months of continuous coverage satisfies "until the expiry of
    thirty six months". A model asked this would be guessing; here it is a
    decision made once, in one place, and documented."""
    assert evaluate([FakeClause("3.2", [1080])], 1080)[0].status is WaitingStatus.SERVED
    assert evaluate([FakeClause("3.2", [1080])], 1079)[0].status is WaitingStatus.NOT_SERVED


def test_unstated_policy_age_yields_unknown_not_a_guess():
    """If the person never said how long they have held the policy, no
    comparison is possible. UNKNOWN is what lets the reasoning step answer
    insufficient_information honestly instead of asserting a verdict."""
    checks = evaluate([FakeClause("3.2", [1080])], days_held=None)
    assert checks[0].status is WaitingStatus.UNKNOWN
    assert "NOT STATED" in checks[0].describe()


def test_clauses_without_a_waiting_period_are_ignored():
    """Exclusions and sub-limits carry no duration and must not appear as
    waiting periods that were somehow satisfied."""
    clauses = [FakeClause("4.1"), FakeClause("5.1", []), FakeClause("3.2", [720])]
    assert [c.clause_id for c in evaluate(clauses, 900)] == ["3.2"]


def test_zero_or_negative_durations_are_ignored():
    """Guards against a model emitting 0 for "no waiting period", which would
    otherwise render as a bar that is always trivially served."""
    assert evaluate([FakeClause("3.2", [0])], 900) == []
    assert evaluate([FakeClause("3.2", [-6])], 900) == []


def test_render_states_the_conclusion_not_the_puzzle():
    """The block hands over an answer, not an exercise."""
    block = render(evaluate([FakeClause("3.2", [1080])], 1800))
    assert "IRRELEVANT" in block
    assert "ALREADY CALCULATED" in block


def test_render_states_what_it_does_NOT_decide():
    """Regression test for a fix that made things worse.

    A first version told the model each served waiting period "does NOT block
    the claim". True of that clause, and read as a verdict on the whole
    question: with four served periods listed, the model concluded "covered"
    for questions about co-payments and room-rent caps. Verdict accuracy fell
    from 0.688 to 0.625 while the arithmetic worked perfectly.

    Correct facts can mislead when their scope is implied rather than stated.
    The block must therefore say out loud which clause types it does not settle.
    """
    block = render(evaluate([FakeClause("3.2", [1080])], 1800))

    assert "SCOPE" in block
    for other in ("exclusion", "cap", "co-payment", "notice condition"):
        assert other in block, f"scope note must disclaim {other}"
    # Case-insensitive: the assertion is about the claim being made, not about
    # how the sentence happens to be capitalised.
    assert "not a reason to answer 'covered'" in block.lower()
    # And the per-clause wording must stay narrow.
    assert "does NOT block the claim" not in block


def test_render_is_empty_when_there_is_nothing_to_say():
    """No waiting periods means no block, rather than an empty heading that
    would spend prompt tokens saying nothing."""
    assert render([]) == ""


def test_days_and_months_are_not_confused():
    """Regression test for a unit bug that passed for the wrong reason.

    Clause 3.1 reads "the first thirty days". An earlier schema asked only for
    a number of months, so it came back as 30 and the comparison read it as
    thirty MONTHS. The eval case still passed - a two-week-old policy is short
    of both 30 days and 30 months - which is the worst kind of green test.

    The model now reports the value and its unit separately and Python
    converts, so a 30-day bar and a 30-month bar can never collapse together.
    """
    thirty_days = FakeClause("3.1", [30])
    thirty_months = FakeClause("3.2", [900])

    # Held 60 days: past a 30-day bar, nowhere near a 30-month one.
    checks = {c.clause_id: c.status for c in evaluate([thirty_days, thirty_months], 60)}
    assert checks["3.1"] is WaitingStatus.SERVED
    assert checks["3.2"] is WaitingStatus.NOT_SERVED


def test_durations_render_in_the_units_a_policy_uses():
    """1080 days is not how anyone reads a 36-month waiting period, and showing
    it that way invites the model to start converting units again."""
    from app.pipeline.waiting import _human

    assert _human(1080) == "36 months"
    assert _human(30) == "1 month"
    assert _human(365) == "1 year"
    assert _human(14) == "14 days"


def test_served_periods_collapse_to_one_line():
    """Prominence follows decisiveness.

    A served waiting period decides nothing - it removes one clause from
    consideration. Giving each its own emphatic line made the model cite
    waiting periods on questions about co-payments and notice deadlines while
    missing the clause that actually answered them. Four non-events were
    outweighing the one clause that mattered.
    """
    served = [FakeClause("3.1", [30]), FakeClause("3.2", [1080]), FakeClause("3.3", [720])]
    block = render(evaluate(served, 1800))

    # One combined line, not three.
    assert block.count("IRRELEVANT") == 1
    assert "3.1, 3.2, 3.3" in block


def test_blocking_periods_still_get_their_own_line():
    """The half that does decide something keeps full treatment."""
    clauses = [FakeClause("3.1", [30]), FakeClause("3.2", [1080])]
    block = render(evaluate(clauses, 60))  # past 30 days, short of 36 months

    assert "IRRELEVANT here" in block          # 3.1 satisfied
    assert "still applies" in block            # 3.2 blocks
    assert "3.2" in block


INITIAL = FakeClause("3.1", [30], ["claims arising out of an Accident"])


def test_a_blocking_period_names_its_own_exception():
    """Regression test for `accident-in-initial-period`.

    Hit by a car ten days into a policy whose 30-day initial waiting period
    says "except claims arising out of an Accident". The block said only that
    3.1 "still applies and blocks treatment", and the model followed that line
    over the clause's own carve-out, three samples out of three. The arithmetic
    was right and incomplete: the bar is not served, AND the bar has an
    exception. Whether an accident happened is language, so the block names the
    exception and leaves that call to the model.
    """
    block = render(evaluate([INITIAL], days_held=10))
    assert "still applies" in block
    assert "claims arising out of an Accident" in block


def test_a_served_or_unknown_period_does_not_repeat_its_exception():
    """A served bar decides nothing, and an unknown one is about a missing
    fact. Printing the carve-out on either adds emphasis to a clause that is
    not deciding the case - the lesson of the served-periods line above."""
    assert "Accident" not in render(evaluate([INITIAL], days_held=400))
    assert "Accident" not in render(evaluate([INITIAL], days_held=None))


# --- whom a waiting period concerns (M16, the Star policy) ------------------

# Shaped like the IRDAI standard wording: the pre-existing diseases exclusion,
# and the specified-disease list, which mentions pre-existing diseases in its
# body only to say which of two periods wins.
PED = FakeClause("1#2", [1080], text=(
    "1. Pre-Existing Diseases - Code Excl 01 A. Expenses related to the treatment of a "
    "pre-existing disease and its direct complications shall be excluded."))
LISTED = FakeClause("2#2", [720], text=(
    "2. Specified disease / procedure waiting period - Code Excl 02 A. Expenses related to "
    "the treatment of the following listed conditions shall be excluded. C. If any of the "
    "specified disease/procedure falls under the waiting period specified for pre-existing "
    "diseases, then the longer of the two waiting periods shall apply. 12. Hernia of all types"))


def test_a_pre_existing_disease_period_says_whom_it_concerns():
    """Regression test for the Star policy's hernia, dengue and Dubai questions.

    Each came back with the right verdict and the pre-existing diseases
    exclusion as its reason, although nobody had mentioned an illness from
    before the policy. The block had told the model that this period "still
    applies and blocks treatment covered by THIS clause", listed first, and
    the model took the first bar it was given. The arithmetic was right; the
    line left out that this bar only concerns an illness the person already
    had. Whether this one is such an illness is language, so the line says
    whom the bar concerns and leaves that call to the model.
    """
    for held in (420, None):
        block = render(evaluate([PED], days_held=held))
        assert "concerns ONLY an illness the person already had" in block
        assert "still applies and blocks" not in block


def test_a_clause_that_only_mentions_pre_existing_diseases_is_not_one():
    """Judged by the clause's opening, not by any mention in its body."""
    assert "already had when the policy began" not in render(evaluate([LISTED], 420))


def test_a_served_pre_existing_period_says_nothing_extra():
    assert "already had" not in render(evaluate([PED], days_held=1200))


def test_a_period_naming_the_questions_condition_comes_first_and_says_so():
    """The model cites the first bar it reads. The one that names what the
    person described goes first; the pre-existing one, which may not concern
    them at all, goes last."""
    block = render(evaluate([PED, LISTED], 420, named={"2#2": ["hernia"]}))
    assert block.index("clause 2#2") < block.index("clause 1#2")
    assert '"hernia"' in block


def test_a_served_period_naming_the_condition_is_not_called_irrelevant():
    """Regression test for `star-hernia-after-wait`. Three years in, the
    hernia waiting period is over, and that is the reason the claim is
    payable at all - yet the block filed it under "IRRELEVANT here and not
    worth citing" with every other served period."""
    block = render(evaluate([PED, LISTED], 1200, named={"2#2": ["hernia"]}))
    irrelevant = next(line for line in block.splitlines() if "IRRELEVANT" in line)
    assert "1#2" in irrelevant and "2#2" not in irrelevant
    assert "clause 2#2" in block and '"hernia"' in block and "served and no longer applies" in block


def test_naming_is_optional():
    """Callers that know nothing about the question still get the old block."""
    assert evaluate([LISTED], 420) == evaluate([LISTED], 420, named={})


# --- a clause that sets two periods (M16, the Star policy) ------------------

# Star's specified-disease clause: 24 months for one list (hernia), 36 for
# another (joint replacement). It was stored as 36 alone.
TWO_LISTS = FakeClause("2#2", [720, 1080])


def test_a_clause_with_two_periods_is_partly_served_between_them():
    """Regression test for the 24/36-month clause stored as 36.

    Thirty months in, a hernia claim is payable and a joint replacement is
    not. With one number the block said the bar "still applies" to both.
    Which list a treatment is on is language, so the block gives both results
    and leaves that reading to the model.
    """
    check = evaluate([TWO_LISTS], days_held=900)[0]
    assert check.status is WaitingStatus.PARTLY_SERVED

    line = check.describe()
    assert "the 24 months period is served" in line
    assert "the 36 months period is NOT" in line
    assert "still applies" not in line


def test_a_clause_with_two_periods_is_decided_when_both_agree():
    """Short of both, or past both, the answer does not depend on the list."""
    assert evaluate([TWO_LISTS], days_held=420)[0].status is WaitingStatus.NOT_SERVED
    assert evaluate([TWO_LISTS], days_held=1080)[0].status is WaitingStatus.SERVED
    assert "24 months or 36 months" in evaluate([TWO_LISTS], days_held=420)[0].describe()


def test_a_partly_served_period_is_listed_with_the_ones_that_block():
    """It may still block, so the block must not also say that nothing does."""
    block = render(evaluate([TWO_LISTS], days_held=900))
    assert "clause 2#2" in block
    assert "No waiting period blocks this claim" not in block
    assert "IRRELEVANT" not in block
