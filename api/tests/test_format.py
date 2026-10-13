"""The same words, set differently, must read the same.

Where a line breaks, whether an apostrophe is curly, whether "fi" is one
character or two: none of that is in the policy or the question, it is in the
PDF and the keyboard. Each test here is a rule that once gave a different
answer for one of those, found by running every rule over three policies'
clauses and the same text reflowed (M44).

No Ollama needed.
"""

from app.grounding import plain, verify_quote
from app.pipeline import analyze, scenario, score, waiting, window


def test_plain_text_keeps_the_words_and_drops_the_typesetting():
    assert plain("I don’t  have\nit") == "I don't have it"
    assert plain("speciﬁed in the Schedule") == "specified in the Schedule"
    assert plain("Pre–Existing") == "Pre-Existing"


def test_a_quotation_across_a_line_end_hyphen_is_verified():
    clause = (
        "Excl01 Expenses related to the treatment of a pre-\nexisting disease and "
        "its direct complications shall be excluded until the expiry of 36 months."
    )
    ok, _ = verify_quote("treatment of a pre-existing disease and its direct complications", clause)
    assert ok
    # Different words still fail: the join is applied to both sides alike.
    ok, _ = verify_quote("treatment of a long-existing disease and its direct complications", clause)
    assert not ok


def test_a_curly_apostrophe_does_not_hide_a_negation():
    typed = "I don’t have another health policy with anyone."
    assert scenario.mentions_another_policy(typed)            # the fault, on the raw text
    assert not scenario.mentions_another_policy(plain(typed))  # what run_scenario now passes on


def test_a_phrase_wrapped_across_a_line_is_still_the_phrase():
    assert scenario._ANOTHER_POLICY.search("where the insured holds multiple\npolicies")
    assert len(score._RE_CROSSREF.findall(plain("subject\nto the limits speciﬁed in Annexure I"))) == 3


def test_pre_existing_is_recognised_when_broken_at_a_line_end():
    # waiting.evaluate collapses whitespace first, so the break reads "pre- existing".
    for text in ("Pre-Existing Diseases", "Pre- Existing Diseases", "pre existing", "preexisting"):
        assert waiting._PRE_EXISTING.search(text), text


def test_a_duration_is_read_however_its_number_is_set():
    for text in ("24 months", "24-month waiting period", "twenty four (24) months"):
        assert waiting._DURATION.findall(text) == [("24", "month")], text
    # A heading's number is not a duration.
    assert waiting._DURATION.findall("Section 3 - Day Care Treatment") == []
    assert analyze._BARE_DURATION.fullmatch("24-month")
    assert window._STATED_WINDOW.search("sixty (60) days prior to admission")
    assert window._STATED_WINDOW.search("a 60-day period following the date of discharge")


def test_a_ligature_does_not_change_the_reading_grade():
    assert score.flesch_kincaid_grade("The speciﬁed beneﬁts apply.") == score.flesch_kincaid_grade(
        "The specified benefits apply."
    )


def test_a_daily_cap_is_recognised_with_a_hyphen():
    assert analyze._PER_DAY.search("Rs. 5,000 per-day limit")
