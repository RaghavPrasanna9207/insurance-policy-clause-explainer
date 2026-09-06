"""M5 gate, part 2: the scenario pipeline's deterministic parts.

The shortlist and the schema construction contain no model call, so they can be
pinned down exactly. What the model *decides* is measured by
`evals/run_scenario_eval.py`, not asserted here - tests check that it works,
evals check how well.
"""

import pytest

from app.pipeline.scenario import (
    MAX_CITATIONS,
    ShortlistClause,
    _reasoning_schema,
    _sort_key,
    shortlist,
)
from app.taxonomy import Verdict


def _clause(number: str, impact: float, text: str = "x" * 200) -> ShortlistClause:
    return ShortlistClause(
        clause_id=number,
        ref=f"doc:{number}",
        number=number,
        clause_type="exclusion",
        text=text,
        impact_score=impact,
    )


# --- shortlist -------------------------------------------------------------


def test_a_normal_policy_is_never_truncated():
    """The central architectural claim, asserted.

    The design says every clause that could matter fits in one prompt, which is
    why there is no retrieval step. A 40-clause policy must therefore come back
    whole - if it does not, the reasoning step is silently deciding a case with
    clauses missing.
    """
    clauses = [_clause(f"{i//10 + 1}.{i % 10}", float(i)) for i in range(40)]
    assert len(shortlist(clauses)) == 40


def test_budget_keeps_the_highest_impact_clauses():
    """Truncation should only ever happen on a document too large to fit, and
    what survives must be the clauses most likely to cost the reader money."""
    clauses = [_clause("1.1", 10.0), _clause("1.2", 90.0), _clause("1.3", 50.0)]
    # A budget that fits roughly one clause of ~200 chars.
    kept = shortlist(clauses, token_budget=80)
    assert [c.number for c in kept] == ["1.2"]


def test_shortlist_returns_document_order():
    """Selection is by impact; presentation is by document order, because a
    reader expects clause 3.2 discussed before 6.1."""
    clauses = [_clause("6.1", 90.0), _clause("3.2", 50.0), _clause("4.10", 70.0)]
    assert [c.number for c in shortlist(clauses)] == ["3.2", "4.10", "6.1"]


def test_clause_numbers_sort_numerically_not_alphabetically():
    """"4.10" must come after "4.2". String ordering would put it before."""
    assert _sort_key("4.2") < _sort_key("4.10")
    assert _sort_key("3.9") < _sort_key("4.1")


def test_unnumbered_clauses_sort_last_without_crashing():
    assert _sort_key("c7") > _sort_key("9.9")


def test_empty_input_is_handled():
    assert shortlist([]) == []


# --- the reasoning schema: where the guarantee lives -----------------------


def test_citation_ids_are_locked_to_the_supplied_clauses():
    """THE central grounding mechanism.

    `clause_id` is an enum of exactly the ids placed in this prompt. Ollama
    enforces JSON-Schema enums during sampling, so at the moment a citation is
    generated every token spelling any other id has probability zero. A
    fabricated citation is not detected and retried - it is unrepresentable.
    """
    ids = ["3.2", "4.1", "6.1"]
    schema = _reasoning_schema(ids)
    citation = schema["properties"]["deciding_clauses"]["items"]["properties"]

    assert citation["clause_id"]["enum"] == ids
    assert "9.9" not in citation["clause_id"]["enum"]


def test_verdict_is_enum_constrained():
    """Including insufficient_information.

    Constrained decoding cannot abstain: the grammar admits only these values,
    so "I don't know" has to be one of them or the model is forced to guess
    covered/not_covered on a question the document never answers.
    """
    schema = _reasoning_schema(["3.2"])
    assert schema["properties"]["verdict"]["enum"] == Verdict.values()
    assert Verdict.INSUFFICIENT_INFORMATION in schema["properties"]["verdict"]["enum"]


def test_citations_are_capped_but_not_floored():
    """No minItems, deliberately.

    insufficient_information legitimately cites nothing, because the honest
    answer is that no clause decides it. The pairing of a definite verdict with
    at least one citation is a CONDITIONAL rule, which a fixed grammar cannot
    express, so it is enforced in `run_scenario` instead.
    """
    array = _reasoning_schema(["3.2"])["properties"]["deciding_clauses"]
    assert array["maxItems"] == MAX_CITATIONS
    assert "minItems" not in array


def test_every_schema_string_field_that_is_categorical_has_an_enum():
    """The project-wide rule: an unconstrained categorical field invites the
    model to invent a value that then flows downstream as an unknown key."""
    items = _reasoning_schema(["3.2"])["properties"]["deciding_clauses"]["items"]
    assert "enum" in items["properties"]["effect"]
    assert "enum" in items["properties"]["clause_id"]
