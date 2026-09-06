"""M5 gate, part 1: the verbatim quote check.

No Ollama needed. This is the layer that catches a real clause id attached to
invented wording, and it has to be exactly as strict as it claims: too loose
and it waves fabrication through while reporting success, too tight and it
cries wolf on correct quotations until people stop believing it.
"""

import pytest

from app.grounding import MIN_QUOTE_CHARS, normalize, verify_citations, verify_quote

CLAUSE = (
    "3.2 Pre-existing Disease Waiting Period\n"
    "Expenses related to the treatment of a Pre-existing Disease and its direct "
    "complications shall be excluded until the\n"
    "expiry of thirty six months of continuous coverage after the date of inception "
    "of the first policy with the Company."
)


# --- what must pass --------------------------------------------------------


def test_exact_quote_verifies():
    quote = "Expenses related to the treatment of a Pre-existing Disease and its direct"
    assert verify_quote(quote, CLAUSE)[0]


def test_quote_spanning_a_line_break_verifies():
    """PDF extraction reflows lines, so a clause contains newlines the original
    did not. A quotation that reads straight through one must still verify."""
    quote = "shall be excluded until the expiry of thirty six months of continuous coverage"
    assert verify_quote(quote, CLAUSE)[0]


def test_typographic_punctuation_is_tolerated():
    """Models silently convert curly quotes and en-dashes to ASCII. That is a
    rendering difference, not a difference in what was said."""
    source = "The Company’s liability — as defined — shall not exceed the Sum Insured."
    quote = "The Company's liability - as defined - shall not exceed the Sum Insured."
    assert verify_quote(quote, source)[0]


def test_trailing_reference_tag_is_tolerated():
    """Regression test for a misleading failure.

    A quotation that was 233 of 245 characters byte-perfect - the entire clause,
    correctly copied - failed verification because the model appended its own
    reference marker: "... with the Company. [3.2 (3.2)]".

    Reporting that as "quote does not appear in the cited clause" implies
    fabrication where the substance was exact. A check that cries wolf is one
    people learn to ignore, so only a bracketed group at the very END is
    stripped, and everything before it must still match exactly.
    """
    quote = (
        "Expenses related to the treatment of a Pre-existing Disease and its direct "
        "complications shall be excluded until the expiry of thirty six months of "
        "continuous coverage after the date of inception of the first policy with "
        "the Company. [3.2 (3.2)]"
    )
    verified, reason = verify_quote(quote, CLAUSE)
    assert verified, reason


# --- what must fail --------------------------------------------------------


def test_fabricated_numbers_are_caught():
    """The failure this whole module exists for: a real clause, invented terms.

    "twelve months" instead of "thirty six months" is the kind of error that
    would tell someone they are covered when they are not.
    """
    quote = (
        "Expenses related to the treatment of a Pre-existing Disease shall be "
        "excluded until the expiry of twelve months of continuous coverage"
    )
    verified, reason = verify_quote(quote, CLAUSE)
    assert not verified
    assert "does not appear" in reason


def test_fabrication_cannot_hide_behind_a_bracket():
    """The trailing-tag tolerance must not become a hole.

    Stripping "(see 3.2)" leaves the invented sentence, which must still fail.
    """
    quote = "The Company will pay all diabetes claims immediately (see 3.2)"
    assert not verify_quote(quote, CLAUSE)[0]


def test_reordered_words_are_caught():
    """Normalisation collapses whitespace and punctuation, never word order."""
    quote = "continuous coverage of thirty six months until excluded be shall complications"
    assert not verify_quote(quote, CLAUSE)[0]


def test_a_quote_too_short_to_prove_anything_is_rejected():
    """"The Company" appears in every clause of every policy.

    Accepting it would report success while verifying nothing, and a false green
    light is worse than no check at all.
    """
    verified, reason = verify_quote("The Company", CLAUSE)
    assert not verified
    assert str(MIN_QUOTE_CHARS) in reason


def test_empty_quote_is_rejected():
    assert not verify_quote("", CLAUSE)[0]
    assert not verify_quote("   ", CLAUSE)[0]


# --- normalisation ---------------------------------------------------------


def test_normalize_collapses_whitespace_but_keeps_words():
    assert normalize("  The   Company\nshall\tpay  ") == "the company shall pay"


@pytest.mark.parametrize("fancy,plain", [("’", "'"), ("“", '"'), ("—", "-")])
def test_normalize_unifies_punctuation(fancy: str, plain: str):
    assert normalize(f"a{fancy}b") == normalize(f"a{plain}b")


# --- citation batch --------------------------------------------------------


def test_verify_citations_reports_per_citation():
    """One result per citation, not one verdict for the whole answer.

    The UI marks the specific claim that failed rather than discarding an
    answer that was otherwise sound.
    """
    sources = {"3.2": CLAUSE, "4.1": "The Company shall not be liable for cosmetic surgery."}
    checks = verify_citations(
        [
            {"clause_id": "3.2", "quote": "shall be excluded until the expiry of thirty six months"},
            {"clause_id": "4.1", "quote": "The Company shall pay for all cosmetic surgery costs"},
        ],
        sources,
    )
    assert [c.verified for c in checks] == [True, False]


def test_citing_a_clause_outside_the_document_fails_loudly():
    """Should be impossible while the id enum is built from these same clauses.

    Checked anyway, because it is the assumption the entire grounding story
    rests on, and an assumption worth relying on is worth asserting.
    """
    checks = verify_citations(
        [{"clause_id": "99.9", "quote": "something quite long and entirely made up here"}],
        {"3.2": CLAUSE},
    )
    assert not checks[0].verified
    assert "not in this document" in checks[0].reason
