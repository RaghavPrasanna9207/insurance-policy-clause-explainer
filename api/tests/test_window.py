"""Cover windows are decided arithmetically, not by the model.

`post-hospitalisation-too-late` - a scan 120 days after discharge against a
90-day window - failed in every recorded eval run. These tests pin the
comparison down exactly, which is only possible because it is code.
"""

from dataclasses import dataclass

from app.pipeline.window import WindowStatus, evaluate, render


@dataclass
class FakeClause:
    clause_id: str
    cover_window_days: int | None = None
    cover_window_anchor: str | None = None


PRE = FakeClause("2.2", 60, "before_admission")
POST = FakeClause("2.3", 90, "after_discharge")
UNRELATED = FakeClause("4.1")


def status_of(clause_id, checks):
    return next(c.status for c in checks if c.clause_id == clause_id)


def test_an_expense_past_the_window_is_outside():
    """The eval case itself: 120 days after discharge, 90-day window."""
    checks = evaluate([POST], "after_discharge", 120)
    assert status_of("2.3", checks) is WindowStatus.OUTSIDE


def test_an_expense_inside_the_window_is_within():
    """The other eval case: tests 50 days before admission, 60-day window."""
    checks = evaluate([PRE], "before_admission", 50)
    assert status_of("2.2", checks) is WindowStatus.WITHIN


def test_the_last_day_of_the_window_is_inside_it():
    """"During the ninety days following discharge" includes day ninety."""
    assert status_of("2.3", evaluate([POST], "after_discharge", 90)) is WindowStatus.WITHIN
    assert status_of("2.3", evaluate([POST], "after_discharge", 91)) is WindowStatus.OUTSIDE


def test_windows_on_the_other_side_of_the_stay_are_not_compared():
    """Fifty days BEFORE admission says nothing about the window AFTER
    discharge. Comparing them would compare numbers measuring different things."""
    checks = evaluate([PRE, POST], "before_admission", 50)
    assert status_of("2.2", checks) is WindowStatus.WITHIN
    assert status_of("2.3", checks) is WindowStatus.NOT_RAISED


def test_a_question_about_something_else_raises_no_window():
    """Most questions never mention a stay's before or after."""
    checks = evaluate([PRE, POST], None, None)
    assert {c.status for c in checks} == {WindowStatus.NOT_RAISED}


def test_a_side_without_a_number_is_a_real_unknown():
    """"A scan after I was discharged" raises the window and leaves it open.
    NOT_RAISED must not swallow that - the gap can decide the answer."""
    assert status_of("2.3", evaluate([POST], "after_discharge", None)) is WindowStatus.UNKNOWN


def test_clauses_without_a_window_are_skipped():
    assert evaluate([UNRELATED], "after_discharge", 120) == []


def test_nothing_raised_renders_nothing_at_all():
    """The reduction module's measured lesson: a note about something the
    question never touched still shifts the answer. Silence, not a heading."""
    assert render(evaluate([PRE, POST], None, None)) == ""


def test_the_block_names_the_clause_and_the_numbers():
    block = render(evaluate([PRE, POST], "after_discharge", 120))
    assert "clause 2.3" in block and "90 days" in block and "120 days" in block
    assert "does NOT pay" in block
    # The pre-admission window was not raised, so it must not appear.
    assert "2.2" not in block


def test_meeting_the_timing_is_not_reported_as_meeting_the_clause():
    """Clause 2.3 also needs the same condition and an accepted in-patient
    claim. WITHIN must say the other conditions still apply."""
    block = render(evaluate([POST], "after_discharge", 30))
    assert "other conditions still apply" in block
