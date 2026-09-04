"""M2 gate, part 1: the scoring formula.

None of these need Ollama. Stage 4 is pure arithmetic, which is the entire
reason it was kept out of the model's hands - a formula can be pinned down by
unit tests, a model's arithmetic cannot.
"""

import pytest

from app.pipeline.analyze import ClauseAnalysis
from app.pipeline.score import (
    BURIEDNESS_BOOST,
    TYPE_WEIGHT,
    flesch_kincaid_grade,
    score,
)
from app.pipeline.segment import Segment
from app.taxonomy import ClauseType


def _segment(idx: int, text: str, *, heading: str = "", number: str = "") -> Segment:
    return Segment(
        order_idx=idx,
        section_path="SECTION 4 - EXCLUSIONS",
        number=number,
        heading=heading,
        text=text,
        page_start=0,
        page_end=0,
        char_start=0,
        char_end=len(text),
    )


def _analysis(idx: int, clause_type: str, likelihood: int, severity: int) -> ClauseAnalysis:
    return ClauseAnalysis(
        clause_key=str(idx),
        clause_type=clause_type,
        plain_language="...",
        what_it_means="...",
        likelihood=likelihood,
        severity=severity,
    )


# --- Flesch-Kincaid ---------------------------------------------------------


def test_legal_prose_scores_harder_than_plain_english():
    """The signal must actually distinguish the thing it claims to measure."""
    plain = "You must tell us within one day. If you do not, we may refuse to pay."
    legal = (
        "Written notice of any claim must be given to the Company or its authorised "
        "Third Party Administrator within twenty four hours of admission in the case "
        "of Emergency hospitalisation, failing which the Company may at its sole "
        "discretion repudiate the claim in its entirety."
    )
    assert flesch_kincaid_grade(legal) > flesch_kincaid_grade(plain) + 4


def test_grade_is_never_negative():
    """Very short simple sentences drive the formula below zero mathematically.

    A negative reading grade is meaningless and would subtract from buriedness,
    so it is floored at 0.
    """
    assert flesch_kincaid_grade("You pay. We pay.") >= 0.0
    assert flesch_kincaid_grade("") == 0.0


# --- The impact formula -----------------------------------------------------


def test_clause_type_dominates_ranking_at_equal_ratings():
    """An exclusion and a definition rated identically must not rank equally.

    Type weight encodes what the model is not asked to judge: the structural
    role a clause plays in deciding a claim.
    """
    text = "The Company shall not be liable for such expenses under this policy."
    segments = [_segment(0, text), _segment(1, text)]
    analyses = {
        "0": _analysis(0, ClauseType.EXCLUSION, 4, 4),
        "1": _analysis(1, ClauseType.DEFINITION, 4, 4),
    }

    scored = score(segments, analyses)
    assert scored["0"].impact_score > scored["1"].impact_score


def test_score_stays_within_zero_to_one_hundred():
    """The formula is normalised rather than clamped.

    Clamping would collapse several genuinely different clauses onto exactly
    100, destroying the ordering at the top of the list - which is the only part
    of the list a user actually reads.
    """
    text = (
        "Notwithstanding anything contained herein, and subject to the provisions "
        "specified in Annexure III as defined in Clause 2, the Company shall not "
        "be liable in accordance with the terms referred to in Section 7."
    )
    worst = score([_segment(99, text)], {"99": _analysis(99, ClauseType.EXCLUSION, 5, 5)})
    best = score([_segment(0, "You are covered.")], {"0": _analysis(0, ClauseType.DEFINITION, 1, 1)})

    assert 0.0 <= best["0"].impact_score <= worst["99"].impact_score <= 100.0
    # Lowest possible ratings on the lowest-weighted type is the floor.
    assert best["0"].impact_score == pytest.approx(0.0)


def test_buriedness_multiplies_rather_than_adds():
    """A buried trivial clause must not outrank a plain consequential one.

    Burial should reorder clauses of similar consequence, never manufacture
    importance. Multiplication preserves that; addition would not.
    """
    buried_trivia = "1.1 Hospital means an institution as defined in Annexure I, subject to Clause 2."
    plain_danger = "We will not pay for cosmetic surgery."

    scored = score(
        [_segment(0, plain_danger), _segment(50, buried_trivia)],
        {
            "0": _analysis(0, ClauseType.EXCLUSION, 4, 5),
            "50": _analysis(50, ClauseType.DEFINITION, 2, 2),
        },
    )
    assert scored["0"].impact_score > scored["50"].impact_score


def test_later_clauses_score_as_more_buried():
    """Position is one of the four buriedness signals, all else being equal."""
    text = "The Company shall not be liable for these expenses."
    segments = [_segment(i, text) for i in range(10)]
    analyses = {str(i): _analysis(i, ClauseType.EXCLUSION, 3, 3) for i in range(10)}

    scored = score(segments, analyses)
    assert scored["9"].buriedness > scored["0"].buriedness
    assert scored["9"].impact_score > scored["0"].impact_score


def test_cross_references_raise_buriedness():
    """Every hop to another clause is a chance for a reader to give up."""
    direct = "We will not pay for dental treatment."
    referred = (
        "We will not pay for dental treatment, subject to the exceptions "
        "specified in Annexure III and as defined in Clause 2 of Section 1."
    )
    scored = score(
        [_segment(0, direct), _segment(1, referred)],
        {
            "0": _analysis(0, ClauseType.EXCLUSION, 3, 3),
            "1": _analysis(1, ClauseType.EXCLUSION, 3, 3),
        },
    )
    assert scored["1"].crossref_signal > scored["0"].crossref_signal


def test_defined_terms_are_detected_from_the_document_itself():
    """Jargon is measured per document, not from a fixed word list.

    Which terms are load-bearing depends on what THIS policy chose to define,
    so the signal is derived from its own definition clauses.
    """
    definition = _segment(
        0,
        "1.1 Hospital\nHospital means any institution with ten in-patient beds.",
        heading="1.1 Hospital",
        number="1.1",
    )
    user = _segment(1, "We pay for treatment taken in a Hospital only.")
    plain = _segment(2, "We pay for treatment taken anywhere in India.")

    scored = score(
        [definition, user, plain],
        {
            "0": _analysis(0, ClauseType.DEFINITION, 2, 2),
            "1": _analysis(1, ClauseType.COVERAGE, 3, 3),
            "2": _analysis(2, ClauseType.COVERAGE, 3, 3),
        },
    )
    assert scored["1"].jargon_signal > scored["2"].jargon_signal


def test_unanalysed_clauses_are_omitted_not_zero_scored():
    """A clause with no analysis must not silently appear as harmless.

    Scoring it 0 would place it at the bottom of the risk list, indistinguishable
    from a genuinely trivial clause. Omitting it keeps the gap detectable.
    """
    segments = [_segment(0, "Covered."), _segment(1, "Not covered.")]
    scored = score(segments, {"0": _analysis(0, ClauseType.COVERAGE, 3, 3)})

    assert "0" in scored
    assert "1" not in scored


def test_every_taxonomy_member_has_a_weight():
    """A new clause type must not silently fall back to a default weight."""
    for clause_type in ClauseType:
        assert clause_type in TYPE_WEIGHT, f"{clause_type} has no TYPE_WEIGHT"


def test_scoring_is_deterministic():
    """Run twice, get the same numbers. The whole point of keeping stage 4
    out of the model's hands."""
    text = "Room rent is limited to 1% of the Sum Insured per day, subject to Clause 5."
    segments = [_segment(3, text)]
    analyses = {"3": _analysis(3, ClauseType.SUB_LIMIT, 5, 3)}

    first = score(segments, analyses)["3"]
    second = score(segments, analyses)["3"]
    assert first == second


def test_buriedness_boost_is_bounded():
    """Buriedness can lift a score by at most BURIEDNESS_BOOST, never invert
    the type hierarchy on its own."""
    assert 0 < BURIEDNESS_BOOST <= 0.5
