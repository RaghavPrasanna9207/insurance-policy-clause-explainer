"""M16: a clause reading is checked against the clause's own words.

The model reads each clause once; these checks run on its answer before
anything is stored or computed from it, the way exceptions already are
(app/grounding.py, verify_exception). Each case below was measured wrong on a
real policy, or on the synthetic one, and each pointed the scenario step at the
wrong clause.
"""

from app.pipeline.analyze import ClauseAnalysis, correct_daily_cap, correct_stay_period


def _reading(**fields) -> ClauseAnalysis:
    base = dict(clause_key="7", clause_type="coverage", plain_language="", what_it_means="")
    return ClauseAnalysis(**(base | fields))


# Shaped like Star Health's pre-hospitalisation clause.
PRE_HOSPITAL = (
    "4. Pre Hospitalization: The Company shall indemnify pre-hospitalization medical "
    "expenses incurred, related to an admissible hospitalization requiring inpatient care, "
    "for a fixed period of 30 days prior to the date of admissible hospitalization covered "
    "under the Policy."
)
POST_HOSPITAL = (
    "5. Post Hospitalization: The Company shall indemnify post hospitalization medical "
    "expenses incurred for a fixed period of 60 days from the date of discharge."
)
# The IRDAI standard initial waiting period.
INITIAL_WAITING = (
    "3. 30-day waiting period - Code Excl 03 A. Expenses related to the treatment of any "
    "illness within 30 days from the first Policy commencement date shall be excluded "
    "except claims arising due to an accident."
)


def test_a_period_before_admission_is_a_cover_window_not_a_waiting_period():
    """Regression test for Star's clause 4.

    Stored as a 30-day waiting period of type waiting_period, it was listed to
    a dengue question 20 days into the policy as a bar that "still applies",
    above the real 30-day exclusion.
    """
    reading = _reading(clause_type="waiting_period",
                       waiting_period_value=30, waiting_period_unit="days")
    correct_stay_period(reading, PRE_HOSPITAL)

    assert reading.waiting_periods_days == []
    assert (reading.cover_window_days, reading.cover_window_anchor) == (30, "before_admission")
    assert reading.clause_type == "coverage"


def test_a_period_after_discharge_becomes_an_after_discharge_window():
    reading = _reading(waiting_period_value=60, waiting_period_unit="days")
    correct_stay_period(reading, POST_HOSPITAL)

    assert reading.waiting_period_days is None
    assert (reading.cover_window_days, reading.cover_window_anchor) == (60, "after_discharge")


def test_a_real_waiting_period_is_left_alone():
    """Only positive evidence moves a period. The policy's start is named here."""
    reading = _reading(clause_type="exclusion",
                       waiting_period_value=30, waiting_period_unit="days")
    correct_stay_period(reading, INITIAL_WAITING)

    assert reading.waiting_period_days == 30
    assert reading.cover_window_days is None
    assert reading.clause_type == "exclusion"


def test_a_waiting_period_that_also_mentions_admission_is_left_alone():
    """A clause naming both the policy's start and a stay is not guessed at:
    dropping a real bar tells someone a refused claim is payable."""
    text = ("Expenses incurred during the first 30 days from the commencement of the policy, "
            "including those prior to the date of admission, are excluded.")
    reading = _reading(waiting_period_value=30, waiting_period_unit="days")
    correct_stay_period(reading, text)
    assert reading.waiting_period_days == 30


def test_a_window_already_read_is_not_overwritten():
    reading = _reading(waiting_period_value=30, waiting_period_unit="days",
                       cover_window_value=30, cover_window_unit="days",
                       cover_window_anchor="before_admission")
    correct_stay_period(reading, PRE_HOSPITAL + " " + POST_HOSPITAL)

    assert reading.waiting_period_days is None
    assert (reading.cover_window_days, reading.cover_window_anchor) == (30, "before_admission")


def test_a_clause_naming_both_sides_gets_no_guessed_window():
    reading = _reading(waiting_period_value=30, waiting_period_unit="days")
    correct_stay_period(reading, PRE_HOSPITAL + " " + POST_HOSPITAL)

    assert reading.waiting_period_days is None
    assert reading.cover_window_days is None


# Shaped like Star's cataract clause and the synthetic modern-treatment limit.
CATARACT = ("3. Cataract Treatment: The Company shall indemnify medical expenses incurred for "
            "treatment of Cataract, subject to a limit of 25% of Sum Insured or Rs.40,000/-, "
            "whichever is lower, per each eye in one Policy Year.")
MODERN = ("5.5 Modern Treatment Limit Expenses incurred on advanced treatment methods shall be "
          "restricted to fifty percent of the Sum Insured per Policy Year.")
ROOM = ("i. Room Rent, Boarding, Nursing Expenses up to 2% of the Sum Insured subject to "
        "maximum of Rs.5000/-, per day.")


def test_a_cap_that_is_not_per_day_is_not_a_room_cap():
    """Regression test for Star's clause 3: "room rent are capped at 25% of
    the sum insured per day", told to a cataract question."""
    for text, percent in ((CATARACT, 25), (MODERN, 50)):
        reading = _reading(clause_type="sub_limit", cap_percent_of_sum_insured=percent)
        correct_daily_cap(reading, text)

        assert reading.cap_percent_of_sum_insured is None
        # Still a cap: stage 5 lists a sub-limit without numbers for the model to read.
        assert reading.clause_type == "sub_limit"


def test_a_per_day_room_cap_is_kept():
    reading = _reading(clause_type="sub_limit", cap_percent_of_sum_insured=2,
                       icu_cap_percent_of_sum_insured=5, cap_max_inr_per_day=5000,
                       icu_cap_max_inr_per_day=10000)
    correct_daily_cap(reading, ROOM)

    assert (reading.cap_percent_of_sum_insured, reading.cap_max_inr_per_day) == (2, 5000)
    assert (reading.icu_cap_percent_of_sum_insured, reading.icu_cap_max_inr_per_day) == (5, 10000)


def test_a_clause_that_sets_two_waiting_periods_carries_both():
    """Regression test for Star's specified-disease clause, stored as 36
    months although its hernia list waits 24."""
    reading = _reading(clause_type="waiting_period",
                       waiting_period_value=36, waiting_period_unit="months",
                       time_windows=["24 months", "36 months"])
    assert reading.waiting_periods_days == [720, 1080]


def test_only_bare_durations_add_a_waiting_period():
    """A time window that says more than a duration may be a different kind of
    period altogether, and a clause with no waiting period has none."""
    reading = _reading(waiting_period_value=36, waiting_period_unit="months",
                       time_windows=["within 30 days of discharge", "24 hours", "1 Year"])
    assert reading.waiting_periods_days == [365, 1080]

    assert _reading(time_windows=["24 months"]).waiting_periods_days == []
