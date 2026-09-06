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


def test_only_decisive_facts_are_reported_as_missing():
    """The list shown to the user and the list shown to the model are one list.

    They were briefly two. The prompt had been narrowed to the five facts that
    can decide an Indian health claim, but the user-facing list still contained
    every null field - so the interface told people it needed to know their
    "body system" and "estimated cost inr" before it could answer.
    """
    from app.pipeline.scenario import DECISIVE_FACTS

    assert "body_system" not in DECISIVE_FACTS
    assert "estimated_cost_inr" not in DECISIVE_FACTS
    assert "notes" not in DECISIVE_FACTS
    assert "policy_age_value" in DECISIVE_FACTS

    # The prompt renderer must use the same source, not a copy of it.
    from app.llm.prompts import render_reasoning_request

    facts = {k: None for k in DECISIVE_FACTS} | {"body_system": None, "notes": ""}
    rendered = render_reasoning_request("x", facts, [])
    assert "body_system" not in rendered
    assert "policy_age_value" in rendered


def test_policy_age_keeps_its_unit():
    """Regression test for a units bug on the SCENARIO side of the comparison.

    "two weeks after my policy started" was extracted as
    months_since_policy_start = 2 - the number read correctly and the unit
    dropped. A fortnight-old policy then counted as two months old and cleared
    a 30-day initial waiting period it should have failed.

    The identical bug had already been fixed on the clause side ("thirty days"
    read as 30 months). Fixing one operand of a comparison and leaving the
    other is not fixing the comparison.
    """
    from app.pipeline.scenario import policy_age_days

    assert policy_age_days({"policy_age_value": 2, "policy_age_unit": "weeks"}) == 14
    assert policy_age_days({"policy_age_value": 2, "policy_age_unit": "months"}) == 60
    assert policy_age_days({"policy_age_value": 5, "policy_age_unit": "years"}) == 1825
    assert policy_age_days({"policy_age_value": 30, "policy_age_unit": "days"}) == 30

    # A fortnight must NOT clear a 30-day bar.
    assert policy_age_days({"policy_age_value": 2, "policy_age_unit": "weeks"}) < 30


def test_unstated_policy_age_stays_none():
    """None is a real answer and must never be defaulted.

    Every waiting period then evaluates to UNKNOWN, which is what allows an
    honest insufficient_information rather than a verdict built on a guess.
    """
    from app.pipeline.scenario import policy_age_days

    assert policy_age_days({}) is None
    assert policy_age_days({"policy_age_value": None, "policy_age_unit": None}) is None
    assert policy_age_days({"policy_age_value": 5, "policy_age_unit": "fortnights"}) is None
    assert policy_age_days({"policy_age_value": 0, "policy_age_unit": "months"}) is None
