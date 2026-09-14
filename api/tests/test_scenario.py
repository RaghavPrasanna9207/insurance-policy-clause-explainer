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
    assert "time_since_policy_start_value" in DECISIVE_FACTS

    # The prompt renderer must use the same source, not a copy of it.
    from app.llm.prompts import render_reasoning_request

    facts = {k: None for k in DECISIVE_FACTS} | {"body_system": None, "notes": ""}
    rendered = render_reasoning_request("x", facts, [])
    assert "body_system" not in rendered
    assert "time_since_policy_start_value" in rendered


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

    assert policy_age_days({"time_since_policy_start_value": 2, "time_since_policy_start_unit": "weeks"}) == 14
    assert policy_age_days({"time_since_policy_start_value": 2, "time_since_policy_start_unit": "months"}) == 60
    assert policy_age_days({"time_since_policy_start_value": 5, "time_since_policy_start_unit": "years"}) == 1825
    assert policy_age_days({"time_since_policy_start_value": 30, "time_since_policy_start_unit": "days"}) == 30

    # A fortnight must NOT clear a 30-day bar.
    assert policy_age_days({"time_since_policy_start_value": 2, "time_since_policy_start_unit": "weeks"}) < 30


def test_unstated_policy_age_stays_none():
    """None is a real answer and must never be defaulted.

    Every waiting period then evaluates to UNKNOWN, which is what allows an
    honest insufficient_information rather than a verdict built on a guess.
    """
    from app.pipeline.scenario import policy_age_days

    assert policy_age_days({}) is None
    assert policy_age_days({"time_since_policy_start_value": None, "time_since_policy_start_unit": None}) is None
    assert policy_age_days({"time_since_policy_start_value": 5, "time_since_policy_start_unit": "fortnights"}) is None
    assert policy_age_days({"time_since_policy_start_value": 0, "time_since_policy_start_unit": "months"}) is None


def test_notes_that_answer_the_question_are_not_shown_to_the_reasoner():
    """Regression test for `not-in-document`.

    Asked "Does this policy cover my car being stolen from the hospital car
    park?", the fact extractor wrote in its free-text notes: "This policy does
    not cover car theft from the hospital car park." The extractor never sees
    the policy, so that sentence is invented - and it reached the reasoning
    step under FACTS UNDERSTOOD, where the verdict followed it.

    Removing notes altogether was measured and cost three cases, because notes
    usually restate the situation in words that help find the right clause. So
    only a note claiming coverage the person never mentioned is dropped.
    """
    from app.llm.prompts import render_reasoning_request

    scenario = "Does this policy cover my car being stolen from the hospital car park?"
    facts = {"notes": "This policy does not cover car theft from the hospital car park."}
    assert "does not cover car theft" not in render_reasoning_request(scenario, facts, [])


def test_notes_repeating_the_persons_own_words_are_kept():
    from app.llm.prompts import render_reasoning_request

    scenario = "The insurer told me the scan is not covered. I have held the policy two years."
    facts = {"notes": "Insurer said the scan is not covered."}
    assert "Insurer said the scan is not covered." in render_reasoning_request(scenario, facts, [])

    # And a note with no coverage language at all is untouched.
    facts = {"notes": "Stayed in a room costing 9,000 rupees a night for surgery."}
    assert "9,000 rupees a night" in render_reasoning_request("x", facts, [])


# --- 5e: the answer checked against the arithmetic --------------------------

PED_TEXT = (
    "3.2 Pre-existing Disease Waiting Period Expenses related to the treatment of a "
    "Pre-existing Disease shall be excluded until the expiry of thirty six months."
)
INPATIENT_TEXT = "2.1 In-patient Hospitalisation The Company shall indemnify Medical Expenses."
PED_QUOTE = "shall be excluded until the expiry of thirty six months"
INPATIENT_QUOTE = "The Company shall indemnify Medical Expenses"


def _clauses():
    from app.pipeline.scenario import ShortlistClause

    return [
        ShortlistClause("2.1", "t:1", "2.1", "coverage", INPATIENT_TEXT, 1.0),
        ShortlistClause("3.2", "t:2", "3.2", "waiting_period", PED_TEXT, 1.0,
                        waiting_period_days=1080),
    ]


def _citation(clause_id, effect):
    from app.pipeline.scenario import Citation

    return Citation(clause_id=clause_id, quote="q", effect=effect)


def _computed(years_held):
    from app.pipeline.scenario import compute

    return compute(
        {"time_since_policy_start_value": years_held, "time_since_policy_start_unit": "years"},
        _clauses(),
    )


def test_citing_a_served_waiting_period_as_a_refusal_is_a_contradiction():
    """`ped-waiting-served`: five years in, told 3.2 was satisfied, refused under 3.2."""
    from app.pipeline.scenario import find_contradictions

    cited = _citation("3.2", "denies")
    problems, irrelevant = find_contradictions([cited], "not_covered", _computed(5))
    assert len(problems) == 1 and "3.2" in problems[0] and "no longer applies" in problems[0]
    assert irrelevant == [cited]


def test_citing_a_ruled_out_reduction_is_a_contradiction():
    """`room-rent-within-cap`: 8,000 a day against a 10,000 cap, cited as reducing."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    room = ShortlistClause("5.1", "t:3", "5.1", "sub_limit", "Room rent 1% of SI.", 1.0,
                           cap_percent_of_sum_insured=1)
    facts = {"sum_insured_value": 10, "sum_insured_unit": "lakh", "room_rent_per_day_inr": 8000}
    problems, irrelevant = find_contradictions(
        [_citation("5.1", "reduces")], "conditional", compute(facts, [room])
    )
    assert "WITHIN" in problems[0]
    assert [c.clause_id for c in irrelevant] == ["5.1"]


def test_a_clause_that_is_still_live_is_not_a_contradiction():
    """Six months into a 36-month wait, citing 3.2 as a refusal is the right answer."""
    from app.pipeline.scenario import find_contradictions

    assert find_contradictions([_citation("3.2", "denies")], "not_covered", _computed(0.5)) == ([], [])


def test_a_cleared_clause_cited_as_permitting_is_not_a_contradiction():
    from app.pipeline.scenario import find_contradictions

    assert find_contradictions([_citation("3.2", "permits")], "covered", _computed(5)) == ([], [])


def test_a_refusal_and_a_reduction_together_is_a_contradiction():
    """`senior-but-excluded`: a nose job bought at 70 - "not covered ... however
    the 20% co-payment applies" - answered conditional."""
    from app.pipeline.scenario import find_contradictions

    cited = [_citation("4.1", "denies"), _citation("5.3", "reduces")]
    problems, irrelevant = find_contradictions(cited, "conditional", _computed(5))
    assert len(problems) == 1 and "4.1" in problems[0] and "5.3" in problems[0]
    # No arithmetic proves which is wrong, so nothing is dropped.
    assert irrelevant == []


def test_a_discretionary_refusal_alone_is_not_a_contradiction():
    """`late-notice` is rightly conditional while citing 6.1 as denying: the
    Company MAY repudiate. Without a reduction beside it there is no clash."""
    from app.pipeline.scenario import find_contradictions

    assert find_contradictions([_citation("6.1", "denies")], "conditional", _computed(5)) == ([], [])
    cited = [_citation("4.1", "denies"), _citation("5.3", "reduces")]
    assert find_contradictions(cited, "not_covered", _computed(5)) == ([], [])


def _fake_model(monkeypatch, answers):
    """Replace the model: facts first, then the given reasoning answers in order."""
    from app.pipeline import scenario

    sent = []
    queue = list(answers)

    async def fake_complete_json(messages, schema, **kwargs):
        if schema is scenario.FACTS_SCHEMA:
            return {"time_since_policy_start_value": 5, "time_since_policy_start_unit": "years"}
        sent.append(messages)
        return queue.pop(0)

    monkeypatch.setattr(scenario.client, "complete_json", fake_complete_json)
    return sent


def _answer(verdict, clause_id, quote, effect):
    return {
        "verdict": verdict, "reasoning": "r", "missing_information": [],
        "deciding_clauses": [{"clause_id": clause_id, "quote": quote, "effect": effect}],
    }


def test_a_contradiction_gets_one_retry_with_the_calculation_in_front_of_it(monkeypatch):
    import asyncio

    from app.pipeline.scenario import run_scenario

    sent = _fake_model(monkeypatch, [
        _answer("not_covered", "3.2", PED_QUOTE, "denies"),
        _answer("covered", "2.1", INPATIENT_QUOTE, "permits"),
    ])
    result = asyncio.run(run_scenario("held five years", _clauses()))

    assert result.verdict == "covered"
    assert len(sent) == 2
    assert "no longer applies" in sent[1][-1]["content"]  # the retry names the calculation


def test_a_citation_still_contradicting_after_the_retry_is_dropped(monkeypatch):
    """The verdict is never overridden. The citation code can PROVE wrong goes,
    and a definite verdict left with nothing behind it is downgraded."""
    import asyncio

    from app.pipeline.scenario import run_scenario

    stubborn = _answer("not_covered", "3.2", PED_QUOTE, "denies")
    sent = _fake_model(monkeypatch, [stubborn, stubborn])
    result = asyncio.run(run_scenario("held five years", _clauses()))

    assert len(sent) == 2  # one retry, not a loop
    assert result.citations == []
    assert result.verdict == "insufficient_information"


def test_a_consistent_answer_makes_no_second_call(monkeypatch):
    import asyncio

    from app.pipeline.scenario import run_scenario

    sent = _fake_model(monkeypatch, [_answer("covered", "2.1", INPATIENT_QUOTE, "permits")])
    assert asyncio.run(run_scenario("held five years", _clauses())).verdict == "covered"
    assert len(sent) == 1


def test_the_policy_duration_field_is_not_named_like_an_age():
    """The model reads a schema key as an instruction.

    Named `policy_age_value`, it received the person's AGE: "I bought this
    policy at 67 and am claiming three years later" came back with 67 in it and
    no unit, and the three years were lost. Every duration field is kept free
    of the word "age" so that cannot be reintroduced by a tidy-minded rename.
    """
    from app.pipeline.scenario import FACTS_SCHEMA

    durations = [k for k in FACTS_SCHEMA["properties"] if k.endswith(("_value", "_unit"))]
    assert "time_since_policy_start_value" in durations
    assert not [k for k in durations if "age" in k.split("_")]
