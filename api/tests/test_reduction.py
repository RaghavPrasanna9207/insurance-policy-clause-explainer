"""M8: reductions are decided arithmetically, not by the model.

The scenario simulator could say whether a claim would be REFUSED and could not
reliably say whether it would be paid IN FULL. Widening the eval set from 16
cases to 40 showed the failure ran both ways: a room rent under the cap read as
a breach, an ICU rate over the cap read as fine, a 20% co-payment missed
entirely.

Every one of those is a comparison between two numbers, which is the same
mistake `waiting.py` was written to fix - in a different family of comparisons
(money and age rather than duration) that nobody had noticed was the same
family. These tests pin the comparisons down exactly, which is only possible
because they are code.
"""

from dataclasses import dataclass

from app.pipeline.reduction import (
    ReductionStatus,
    age_at_inception,
    evaluate,
    render,
    sum_insured_rupees,
)


@dataclass
class FakeClause:
    clause_id: str
    clause_type: str = "sub_limit"
    copay_percent: int | None = None
    copay_min_age_at_inception: int | None = None
    cap_percent_of_sum_insured: int | None = None
    icu_cap_percent_of_sum_insured: int | None = None
    cap_max_inr_per_day: int | None = None
    icu_cap_max_inr_per_day: int | None = None


COPAY = FakeClause("5.3", copay_percent=20, copay_min_age_at_inception=60)
ROOM = FakeClause(
    "5.1", cap_percent_of_sum_insured=1, icu_cap_percent_of_sum_insured=2
)


# --- the co-payment comparison -------------------------------------------


def test_over_the_age_threshold_at_inception_applies():
    """The exact case the model got wrong: bought at 67, claiming later."""
    checks = evaluate([COPAY], {"age_at_policy_start": 67}, policy_age_days=1095)
    assert checks[0].status is ReductionStatus.APPLIES
    assert "80%" in checks[0].detail


def test_an_applicable_copay_does_not_claim_the_claim_is_paid():
    """Regression test for `senior-but-excluded`.

    Bought at 70, then a nose job for appearance: refused outright by the
    cosmetic exclusion. The line said the co-payment applies "so this claim is
    paid at 80%" - but the arithmetic only established the AGE condition.
    Whether this claim is paid at all was never computed, and here it is not.
    The model answered conditional, three samples of three. A computed line
    must not state more than was computed."""
    detail = evaluate([COPAY], {"age_at_policy_start": 70}, policy_age_days=1095)[0].detail
    assert "so this claim is paid" not in detail
    assert "IF this claim is payable at all" in detail


def test_a_copay_on_every_claim_does_not_decide_whether_this_one_is_paid():
    """Regression test for the Star policy (M16): 5% of each and every claim.

    Its line was the only APPLIES on the page, identical for every question,
    and a caesarean, a stay abroad and an undated gallbladder removal were all
    answered "conditional because a 5% co-payment applies". The line now says
    what the age-based one says, and why it cannot be the reason."""
    universal = FakeClause("9", copay_percent=5)
    check = evaluate([universal], {}, policy_age_days=None)[0]

    assert check.status is ReductionStatus.APPLIES
    assert "EVERY claim this policy pays" in check.detail
    assert "IF this claim is payable at all, it is paid at 95%" in check.detail


def test_under_the_age_threshold_does_not_apply():
    """The false-positive direction, which matters just as much. A system
    pushed to hunt for co-payments will start finding them for people who do
    not owe one, and 58 is not 60."""
    checks = evaluate([COPAY], {"age_at_policy_start": 58}, policy_age_days=1460)
    assert checks[0].status is ReductionStatus.DOES_NOT_APPLY


def test_the_age_boundary_is_inclusive():
    """"Completed sixty years" is satisfied at exactly 60. Decided once, here,
    rather than guessed at differently on every request."""
    at_60 = evaluate([COPAY], {"age_at_policy_start": 60}, 1095)[0]
    at_59 = evaluate([COPAY], {"age_at_policy_start": 59}, 1095)[0]
    assert at_60.status is ReductionStatus.APPLIES
    assert at_59.status is ReductionStatus.DOES_NOT_APPLY


def test_current_age_is_not_age_at_inception():
    """The operand bug this field exists to prevent. Someone 68 today who never
    said when they bought the policy could have bought it at 50, and firing a
    20% co-payment at them is a wrong answer that costs real money."""
    checks = evaluate([COPAY], {"age": 68}, policy_age_days=None)
    assert checks[0].status is ReductionStatus.UNKNOWN


def test_age_at_inception_is_derived_when_only_current_age_is_given():
    """72 now, held four years -> 68 at inception. Subtraction is arithmetic
    and belongs in Python, not in a prompt."""
    assert age_at_inception({"age": 72}, policy_age_days=1460) == 68


def test_a_stated_inception_age_beats_a_derived_one():
    """"I bought this policy at 70 and I am 72 now" states the operand
    outright. Deriving it instead would introduce rounding for no reason."""
    facts = {"age": 72, "age_at_policy_start": 70}
    assert age_at_inception(facts, policy_age_days=730) == 70


def test_absurd_derivations_are_refused_rather_than_returned():
    """A misread policy age can produce a negative or impossible age. Returning
    None sends the check to UNKNOWN, where a missing fact belongs; returning
    the nonsense would let it decide a fifth of someone's claim."""
    assert age_at_inception({"age": 30}, policy_age_days=40 * 365) is None


# --- the room and ICU cap comparisons -------------------------------------


def test_a_room_over_the_cap_applies():
    facts = {
        "sum_insured_value": 5, "sum_insured_unit": "lakh",
        "room_rent_per_day_inr": 9000,
    }
    checks = evaluate([ROOM], facts, policy_age_days=1460)
    assert checks[0].status is ReductionStatus.APPLIES


def test_a_room_within_the_cap_does_not_apply():
    """1% of 10 lakh is 10,000 and the room cost 8,000. The model read this as
    a breach; the arithmetic cannot."""
    facts = {
        "sum_insured_value": 10, "sum_insured_unit": "lakh",
        "room_rent_per_day_inr": 8000,
    }
    checks = evaluate([ROOM], facts, policy_age_days=1095)
    assert checks[0].status is ReductionStatus.DOES_NOT_APPLY


def test_a_room_within_the_cap_is_stated_with_its_numbers():
    """Regression test for `room-rent-within-cap`.

    Collapsed to "Ruled out ...: 5.1", the finding reached the model without its
    reason, and the model did its own proportion - "paid at 80% of 8,000" -
    reading the proportionate-deduction clause, which applies only "where the
    room rent EXCEEDS the limit specified in Clause 5.1". The room is what this
    person asked about, so the result gets its own line, in the same words the
    dependent clause uses."""
    facts = {
        "sum_insured_value": 10, "sum_insured_unit": "lakh",
        "room_rent_per_day_inr": 8000,
    }
    block = render(evaluate([ROOM, COPAY], facts | {"age_at_policy_start": 40}, 1095))
    assert "does NOT exceed" in block and "10,000" in block and "8,000" in block
    # A ruled-out co-payment is still a non-event, collapsed to its id.
    assert "not worth citing: 5.3" in block


def test_icu_is_compared_against_the_icu_rate():
    """2% of 5 lakh is 10,000, so 12,000 a day in intensive care exceeds it -
    while the same charge against the 1% ROOM rate would be a different
    comparison entirely. Using one rate for both is how this was got wrong."""
    facts = {
        "sum_insured_value": 5, "sum_insured_unit": "lakh",
        "room_rent_per_day_inr": 12000, "room_is_icu": True,
    }
    assert evaluate([ROOM], facts, 2190)[0].status is ReductionStatus.APPLIES

    # The same money, in an ordinary room, is compared against 1% = 5,000 and
    # is still a breach - but by a different margin and a different clause rate.
    facts["room_is_icu"] = False
    assert evaluate([ROOM], facts, 2190)[0].status is ReductionStatus.APPLIES


def test_a_rupee_maximum_lowers_the_cap_below_the_percentage():
    """Regression test for M15's Failure 52: a cap the policy does not have.

    A real wording caps room rent at "2% of the Sum Insured subject to maximum
    of Rs.5000/- per day". Modelling only the percentage, the arithmetic told a
    3-lakh policyholder their cap was 6,000 - so a 5,500 room was "within the
    cap", decided by the part of the system that exists so the model is never
    trusted with arithmetic. The limit is whichever of the two is lower.
    """
    capped = FakeClause("1", cap_percent_of_sum_insured=2, cap_max_inr_per_day=5_000,
                        icu_cap_percent_of_sum_insured=5, icu_cap_max_inr_per_day=10_000)
    three_lakh = {"sum_insured_value": 3, "sum_insured_unit": "lakh"}

    check = evaluate([capped], three_lakh | {"room_rent_per_day_inr": 5_500}, 1460)[0]
    assert check.status is ReductionStatus.APPLIES
    assert "5,000" in check.detail and "6,000" in check.detail

    # ICU against its own pair: 5% of 3 lakh is 15,000, the maximum 10,000.
    icu = three_lakh | {"room_rent_per_day_inr": 12_000, "room_is_icu": True}
    assert evaluate([capped], icu, 1460)[0].status is ReductionStatus.APPLIES

    # Where the percentage is the lower limit, the percentage decides:
    # 2% of 2 lakh is 4,000, under the 5,000 maximum.
    two_lakh = {"sum_insured_value": 2, "sum_insured_unit": "lakh"}
    assert evaluate([capped], two_lakh | {"room_rent_per_day_inr": 4_500}, 1460)[0].status \
        is ReductionStatus.APPLIES
    assert evaluate([capped], two_lakh | {"room_rent_per_day_inr": 3_900}, 1460)[0].status \
        is ReductionStatus.DOES_NOT_APPLY


def test_an_unstated_sum_insured_yields_unknown_not_a_pass():
    """Someone who did not mention their sum insured has not thereby stayed
    within the cap. UNKNOWN, never DOES_NOT_APPLY."""
    checks = evaluate([ROOM], {"room_rent_per_day_inr": 9000}, 1460)
    assert checks[0].status is ReductionStatus.UNKNOWN


def test_lakh_and_crore_are_converted_here_not_by_the_model():
    """Indian schedules say "5 lakh", never 500000. Converting is arithmetic,
    the same lesson as reading "two weeks" as 2 + weeks."""
    assert sum_insured_rupees({"sum_insured_value": 5, "sum_insured_unit": "lakh"}) == 500_000
    assert sum_insured_rupees({"sum_insured_value": 1, "sum_insured_unit": "crore"}) == 10_000_000
    assert sum_insured_rupees({"sum_insured_value": None, "sum_insured_unit": "lakh"}) is None


# --- the clauses with no arithmetic to do ---------------------------------


def test_a_sub_limit_with_no_numbers_is_a_judgement_not_an_unknown():
    """Whether a treatment falls under a cataract sub-limit is a question about
    language, not a comparison. Reporting it as UNKNOWN would send the
    reasoning step hunting for a fact that is already present; the open
    question is the reading, not the data."""
    cataract = FakeClause("5.4", clause_type="sub_limit")
    checks = evaluate([cataract], {}, policy_age_days=1460)
    assert checks[0].status is ReductionStatus.JUDGEMENT


def test_clauses_that_reduce_nothing_are_skipped():
    exclusion = FakeClause("4.1", clause_type="exclusion")
    waiting = FakeClause("3.2", clause_type="waiting_period")
    assert evaluate([exclusion, waiting], {}, 1460) == []


# --- how the result is presented -----------------------------------------


def test_ruled_out_reductions_collapse_to_one_line():
    """Prominence is proportional to decisiveness, and this is not a style
    preference - it is a fix for a measured regression. When served waiting
    periods were each given their own emphatic line, the model read four true
    statements that nothing blocked the claim and started answering "covered"
    to questions about co-payments. Verdict accuracy fell from 0.688 to 0.625
    while the arithmetic behind those lines was perfect.

    So reductions that bite lead and get a line each; ones ruled out share a
    single line naming them as irrelevant.

    One exception, measured: a room within its cap gets its own line with the
    numbers (see test_a_room_within_the_cap_is_stated_with_its_numbers),
    because a room cap is only computed when the person named their room rate
    - it is the subject of their question, not a non-event."""
    other_copay = FakeClause("5.6", copay_percent=10, copay_min_age_at_inception=65)
    facts = {
        "age_at_policy_start": 50,
        "sum_insured_value": 10, "sum_insured_unit": "lakh",
        "room_rent_per_day_inr": 8000,
    }
    block = render(evaluate([COPAY, other_copay, ROOM], facts, 1460))
    assert block.count("APPLIES") == 0
    assert "not worth citing: 5.3, 5.6" in block
    assert "WITHIN THE LIMIT: clause 5.1" in block


def test_the_block_says_a_reduction_never_refuses_a_claim():
    """The scoping paragraph, and the reason it is not optional: a co-payment
    must not turn a refused cosmetic-surgery claim into `conditional`. There is
    nothing to take 20% of."""
    block = render(evaluate([COPAY], {"age_at_policy_start": 70}, 1095))
    assert "None of them refuses a claim" in block
    assert "conditional, never covered" in block


def test_no_reductions_renders_nothing():
    """An empty block rather than a heading with nothing under it - the prompt
    should not carry a section that says only that it is empty."""
    assert render([]) == ""


# --- silence versus doubt -------------------------------------------------
#
# The distinction below was missing from the first version of this module, and
# its absence dropped verdict accuracy from 0.725 to 0.600 in a measured run.
# Every new failure landed on `conditional` or `insufficient_information` -
# because a question about a broken hip was arriving with five lines of doubt
# about caps nobody had asked about.


def test_a_cap_nobody_raised_is_silent_not_uncertain():
    """No room, no sum insured, no age: these caps are not unresolved, they are
    not what the question is about. Reporting "cannot be determined" here is
    true and actively harmful - it invents doubt on nearly every question,
    since most people describing an injury mention neither."""
    checks = evaluate([COPAY, ROOM], {}, policy_age_days=1825)
    assert {c.status for c in checks} == {ReductionStatus.NOT_RAISED}


def test_a_half_stated_operand_is_still_a_real_unknown():
    """The other side of the same line. Someone who gave a sum insured and no
    room rate HAS raised the cap and left it open, and that gap can decide the
    answer. NOT_RAISED must not swallow this."""
    facts = {"sum_insured_value": 5, "sum_insured_unit": "lakh"}
    assert evaluate([ROOM], facts, 1460)[0].status is ReductionStatus.UNKNOWN


def test_not_raised_prints_nothing_at_all():
    """The whole point of the status. A clause nobody's question touches should
    not appear in the reasoning the answer is built from."""
    block = render(evaluate([COPAY, ROOM], {}, policy_age_days=1825))
    assert "5.1" not in block and "5.3" not in block


def test_framing_is_sized_to_what_is_at_stake():
    """Nine lines explaining how reductions interact with a verdict earn their
    space when a co-payment has actually been computed."""
    computed = render(evaluate([COPAY], {"age_at_policy_start": 67}, 1095))
    assert "WHAT REDUCES THE PAYOUT" in computed


def test_nothing_computed_says_nothing_at_all():
    """When every arithmetic check returns NOT_RAISED and only JUDGEMENT
    clauses survive, the block is EMPTY - not short, empty.

    An earlier version emitted one line here naming the capped clauses and
    telling the model to read them. It carried no computed finding and pointed
    at clause text already present in the same prompt. Measured on the 40-case
    scenario set it cost three cases: initial-waiting-period, dental-no-accident
    and non-disclosure were each a confident, correct refusal without that line
    and `insufficient_information` with it."""
    assert render(evaluate([FakeClause("5.4")], {}, 1825)) == ""
    assert render(evaluate([FakeClause("5.2"), FakeClause("5.4")], {}, 1825)) == ""


def test_judgement_clauses_share_one_line_when_something_was_computed():
    """Alongside a real finding the judgement clauses still appear, and still
    share a single line. Three "YOUR CALL" lines gave a reminder the visual
    weight of a result, which it is not - the clause text is already in the
    prompt."""
    clauses = [COPAY, FakeClause("5.2"), FakeClause("5.4"), FakeClause("5.5")]
    block = render(evaluate(clauses, {"age_at_policy_start": 67}, 1095))
    assert "5.2, 5.4, 5.5" in block
    assert block.count("Also capped by") == 1


@dataclass
class TextClause(FakeClause):
    text: str = ""


CATARACT = TextClause(
    "5.4", text="5.4 Cataract Sub-limit Expenses in respect of treatment of cataract shall be "
                "limited to twenty five thousand rupees per eye."
)
MODERN = TextClause(
    "5.5", text="5.5 Modern Treatment Limit Expenses incurred on advanced treatment methods "
                "including robotic surgery, stem cell therapy, oral chemotherapy and deep brain "
                "stimulation shall be restricted to fifty percent of the Sum Insured."
)
PROPORTIONATE = TextClause(
    "5.2", text="5.2 Proportionate Deduction Where the Insured Person is admitted to a room "
                "category whose rent exceeds the limit, surgeon fees and operation theatre "
                "charges shall be reduced."
)


def test_a_sub_limit_naming_the_described_treatment_says_so():
    """Regression test for `oral-chemo-limit`.

    Oral chemotherapy, and clause 5.5 caps "oral chemotherapy" by name. The
    block said only "Also capped by 5.2, 5.4, 5.5, but only for the treatments
    those clauses name. Read them" - and the model answered that no sub-limit
    applied, three samples of three. Whether a clause's text contains a word
    the person used is a lookup, so code does it; whether the cap bites is
    still the model's call."""
    facts = {"procedure": "oral chemotherapy", "age_at_policy_start": 41}
    block = render(evaluate([COPAY, PROPORTIONATE, CATARACT, MODERN], facts, 3285))
    assert "NAMES THIS TREATMENT: clause 5.5" in block
    assert "chemotherapy" in block
    assert "Also capped by 5.2, 5.4," in block  # the others stay on the shared line
    assert "Nothing here reduces this claim" not in block


def test_generic_words_do_not_count_as_naming_a_treatment():
    """"Gallbladder operation" shares the word "operation" with "operation
    theatre charges". That is not the proportionate-deduction clause naming
    gallbladder surgery."""
    facts = {"procedure": "gallbladder operation", "condition": "unknown"}
    checks = evaluate([PROPORTIONATE, CATARACT, MODERN], facts, 1460)
    assert all(not c.named for c in checks)


def test_a_named_sub_limit_is_worth_saying_even_when_nothing_else_was_computed():
    """The silence rule above is for blocks with no finding. A clause whose
    text names the described treatment is a finding."""
    block = render(evaluate([CATARACT], {"procedure": "cataract operation"}, 1460))
    assert "NAMES THIS TREATMENT: clause 5.4" in block
