"""M5 fix: waiting periods are decided arithmetically, not by the model.

Three of five scenario failures were the same mistake - comparing a policy age
against a waiting period and getting it wrong. That is arithmetic, and this
project's rule is that arithmetic does not go to a language model. These tests
pin the comparison down exactly, which is only possible because it is code.
"""

from dataclasses import dataclass

from app.pipeline.waiting import WaitingStatus, evaluate, render


@dataclass
class FakeClause:
    clause_id: str
    waiting_period_days: int | None = None


def test_an_elapsed_waiting_period_is_served():
    """The exact case the model got wrong: 5 years against a 36-month bar."""
    checks = evaluate([FakeClause("3.2", 1080)], days_held=1800)
    assert checks[0].status is WaitingStatus.SERVED


def test_an_unelapsed_waiting_period_is_not_served():
    checks = evaluate([FakeClause("3.2", 1080)], days_held=240)
    assert checks[0].status is WaitingStatus.NOT_SERVED


def test_the_boundary_is_inclusive():
    """Exactly 36 months of continuous coverage satisfies "until the expiry of
    thirty six months". A model asked this would be guessing; here it is a
    decision made once, in one place, and documented."""
    assert evaluate([FakeClause("3.2", 1080)], 1080)[0].status is WaitingStatus.SERVED
    assert evaluate([FakeClause("3.2", 1080)], 1079)[0].status is WaitingStatus.NOT_SERVED


def test_unstated_policy_age_yields_unknown_not_a_guess():
    """If the person never said how long they have held the policy, no
    comparison is possible. UNKNOWN is what lets the reasoning step answer
    insufficient_information honestly instead of asserting a verdict."""
    checks = evaluate([FakeClause("3.2", 1080)], days_held=None)
    assert checks[0].status is WaitingStatus.UNKNOWN
    assert "NOT STATED" in checks[0].describe()


def test_clauses_without_a_waiting_period_are_ignored():
    """Exclusions and sub-limits carry no duration and must not appear as
    waiting periods that were somehow satisfied."""
    clauses = [FakeClause("4.1"), FakeClause("5.1", None), FakeClause("3.2", 720)]
    assert [c.clause_id for c in evaluate(clauses, 900)] == ["3.2"]


def test_zero_or_negative_durations_are_ignored():
    """Guards against a model emitting 0 for "no waiting period", which would
    otherwise render as a bar that is always trivially served."""
    assert evaluate([FakeClause("3.2", 0)], 900) == []
    assert evaluate([FakeClause("3.2", -6)], 900) == []


def test_render_states_the_conclusion_not_the_puzzle():
    """The block hands over an answer, not an exercise."""
    block = render(evaluate([FakeClause("3.2", 1080)], 1800))
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
    block = render(evaluate([FakeClause("3.2", 1080)], 1800))

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
    thirty_days = FakeClause("3.1", 30)
    thirty_months = FakeClause("3.2", 900)

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
    served = [FakeClause("3.1", 30), FakeClause("3.2", 1080), FakeClause("3.3", 720)]
    block = render(evaluate(served, 1800))

    # One combined line, not three.
    assert block.count("IRRELEVANT") == 1
    assert "3.1, 3.2, 3.3" in block


def test_blocking_periods_still_get_their_own_line():
    """The half that does decide something keeps full treatment."""
    clauses = [FakeClause("3.1", 30), FakeClause("3.2", 1080)]
    block = render(evaluate(clauses, 60))  # past 30 days, short of 36 months

    assert "IRRELEVANT here" in block          # 3.1 satisfied
    assert "still applies" in block            # 3.2 blocks
    assert "3.2" in block
