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
    citation_ids,
    clause_token_budget,
    shortlist,
)
from app.taxonomy import Verdict


def test_repeated_clause_numbers_get_distinct_citation_ids():
    """Real wordings restart numbering per section; ids must still be unique.

    The id enum and the quote check both look a clause up by id. Two clauses
    under "10" would let a quote from one be checked against the other.
    """
    numbered = [("10", 0), ("11", 1), ("10", 2), ("", 3), ("10", 4)]
    ids = citation_ids(numbered)
    assert ids == ["10", "11", "10#2", "c3", "10#3"]
    assert len(set(ids)) == len(ids)


def _clause(number: str, impact: float, text: str = "x" * 200,
            clause_type: str = "exclusion") -> ShortlistClause:
    return ShortlistClause(
        clause_id=number,
        ref=f"doc:{number}",
        number=number,
        clause_type=clause_type,
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


def test_shortlist_keeps_document_order_when_numbering_restarts():
    """Selection is by impact; presentation is in the order the clauses arrived.

    It used to sort by clause number, which is document order only while
    numbers never repeat. Real wordings restart them per section - coverage 1-9,
    exclusions 1-23, conditions 1-28 in one of them - so number order put
    "coverage 1, exclusion 1, condition 1" side by side.
    """
    in_document_order = [_clause("1", 20.0, clause_type="coverage"), _clause("2", 30.0),
                         _clause("1", 90.0), _clause("c7", 10.0), _clause("1", 50.0)]
    kept = shortlist(in_document_order, token_budget=10_000)
    assert [c.impact_score for c in kept] == [20.0, 30.0, 90.0, 10.0, 50.0]


def test_an_oversized_policy_gives_up_definitions_before_coverage():
    """Regression test for M15's measurement: impact order dropped all coverage.

    Impact measures what a clause can cost the reader, so the clause granting
    cover ranks lowest - and on three real wordings, every coverage clause was
    the first to go. Definitions and procedural clauses now give way first.
    """
    clauses = [
        _clause("1", 5.0, clause_type="coverage"),
        _clause("2", 60.0, clause_type="definition"),
        _clause("3", 70.0, clause_type="procedural"),
        _clause("4", 80.0, clause_type="exclusion"),
    ]
    # Room for two clauses of ~77 tokens each.
    kept = shortlist(clauses, token_budget=160)
    assert [c.clause_type for c in kept] == ["coverage", "exclusion"]


def test_one_clause_too_large_does_not_end_the_selection():
    """A long annexure that does not fit must not push out every short clause after it."""
    clauses = [_clause("1", 90.0), _clause("A", 80.0, text="x" * 3_000), _clause("2", 70.0)]
    kept = shortlist(clauses, token_budget=160)
    assert [c.number for c in kept] == ["1", "2"]


def test_the_clause_budget_is_what_the_window_has_left():
    """Derived, not reserved: a longer system prompt must shrink it."""
    from app.config import settings
    from app.llm.prompts import REASON_SYSTEM
    from app.pipeline.scenario import CHARS_PER_TOKEN, QUESTION_TOKENS

    assert clause_token_budget() == (
        settings.num_ctx - settings.num_predict
        - int(len(REASON_SYSTEM) / CHARS_PER_TOKEN) - QUESTION_TOKENS
    )
    assert clause_token_budget() > 0


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
    assert "does NOT exceed" in problems[0]
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


DAY_CARE_TEXT = (
    "2.4 Day Care Procedures The Company shall indemnify Medical Expenses incurred for "
    "Day Care Treatment listed in Annexure II to this policy."
)


def test_an_annexure_referred_to_but_not_included_is_found():
    from app.pipeline.scenario import ShortlistClause, absent_annexures

    day_care = ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
    assert absent_annexures([day_care]) == {"2.4": ["Annexure II"]}

    listed = ShortlistClause("(a)", "t:2", "(a)", "coverage", "(a) Cataract surgery", 1.0,
                             section_path="ANNEXURE II - LIST OF DAY CARE PROCEDURES")
    assert absent_annexures([day_care, listed]) == {}


def test_paying_on_a_clause_whose_list_is_missing_is_questioned():
    """`day-care-not-listed`: six hours on a drip, answered covered on the
    strength of a clause that pays only for treatments on a list this document
    does not contain."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    day_care = ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
    computed = compute({}, [day_care])
    problems, irrelevant = find_contradictions([_citation("2.4", "permits")], "covered", computed)
    assert "Annexure II" in problems[0] and "not included" in problems[0]
    assert irrelevant == []  # unchecked is not the same as wrong


def test_a_refusal_citing_that_clause_is_not_questioned():
    """Non-medical items are refused by a clause that also refers to a missing
    annexure. Refusing on the items it names outright needs no list."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    day_care = ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
    computed = compute({}, [day_care])
    assert find_contradictions([_citation("2.4", "permits")], "not_covered", computed) == ([], [])


ROOM_TEXT = "5.1 Room Rent Limit Room rent shall be limited to one percent of the Sum Insured per day."


def test_covered_while_citing_a_reduction_that_applies_is_a_contradiction():
    """ICU at 9,000 a day against a 6,000 cap, three fresh samples of three:
    answered covered, citing the cap as reducing the claim. The arithmetic
    agrees it reduces - which is exactly why the answer cannot be covered."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    room = ShortlistClause("5.1", "t:1", "5.1", "sub_limit", ROOM_TEXT, 1.0,
                           cap_percent_of_sum_insured=1)
    facts = {"sum_insured_value": 5, "sum_insured_unit": "lakh", "room_rent_per_day_inr": 9000}
    cited = [_citation("5.1", "reduces")]
    problems, irrelevant = find_contradictions(cited, "covered", compute(facts, [room]))
    assert "EXCEED" in problems[0] and "conditional" in problems[0]
    assert irrelevant == []  # the citation is right; the verdict disagrees with it
    assert find_contradictions(cited, "conditional", compute(facts, [room])) == ([], [])


def test_a_reduction_nobody_raised_cited_as_reducing_is_found():
    """`documents-late` cited the room cap and the senior co-payment as reducing
    a claim about late paperwork; a held-out war injury did the same. The person
    mentioned no room and no age, and the arithmetic says exactly that."""
    from app.pipeline.scenario import ShortlistClause, compute, unraised_reductions

    room = ShortlistClause("5.1", "t:1", "5.1", "sub_limit", ROOM_TEXT, 1.0,
                           cap_percent_of_sum_insured=1)
    cited = _citation("5.1", "reduces")
    assert unraised_reductions([cited], compute({}, [room])) == [cited]

    facts = {"sum_insured_value": 5, "sum_insured_unit": "lakh", "room_rent_per_day_inr": 9000}
    assert unraised_reductions([cited], compute(facts, [room])) == []


def test_an_unraised_reduction_is_filtered_not_retried(monkeypatch):
    """Retrying on it pushed right answers to insufficient_information, so it is
    removed quietly - and only when another citation still stands."""
    import asyncio

    from app.pipeline.scenario import ShortlistClause, run_scenario

    clauses = _clauses() + [ShortlistClause("5.1", "t:9", "5.1", "sub_limit", ROOM_TEXT, 1.0,
                                            cap_percent_of_sum_insured=1)]
    answer = {
        "verdict": "covered", "reasoning": "r", "missing_information": [],
        "deciding_clauses": [
            {"clause_id": "2.1", "quote": INPATIENT_QUOTE, "effect": "permits"},
            {"clause_id": "5.1", "quote": "Room rent shall be limited to one percent", "effect": "reduces"},
        ],
    }
    sent = _fake_model(monkeypatch, [answer])
    result = asyncio.run(run_scenario("held five years", clauses))
    assert len(sent) == 1
    assert [c.clause_id for c in result.citations] == ["2.1"]

    alone = answer | {"verdict": "conditional", "deciding_clauses": [answer["deciding_clauses"][1]]}
    _fake_model(monkeypatch, [alone])
    result = asyncio.run(run_scenario("held five years", clauses))
    assert [c.clause_id for c in result.citations] == ["5.1"] and result.verdict == "conditional"


def test_leaning_on_a_missing_list_is_questioned_whatever_the_effect_label():
    """A held-out short procedure cited the day-care clause as "delays", not
    "permits", and slipped past a check that looked only for "permits"."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    day_care = ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
    problems, _ = find_contradictions([_citation("2.4", "delays")], "conditional",
                                      compute({}, [day_care]))
    assert "Annexure II" in problems[0]


def test_a_claim_resting_only_on_a_missing_list_is_downgraded_after_the_retry(monkeypatch):
    """Asked again, the answer still pays on nothing but a clause whose list is
    not in the document (and a definition). That is an answer with nothing
    checkable behind it - the same case as an answer with no citations - so it
    goes the same safe way."""
    import asyncio

    from app.pipeline.scenario import ShortlistClause, run_scenario

    clauses = [
        ShortlistClause("1.4", "t:0", "1.4", "definition", "1.4 Day Care Treatment means treatment under twenty four hours.", 1.0),
        ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0),
    ]
    leaning = {
        "verdict": "conditional", "reasoning": "r", "missing_information": [],
        "deciding_clauses": [
            {"clause_id": "1.4", "quote": "Day Care Treatment means treatment under twenty four hours", "effect": "permits"},
            {"clause_id": "2.4", "quote": "Day Care Treatment listed in Annexure II", "effect": "delays"},
        ],
    }
    sent = _fake_model(monkeypatch, [leaning, leaning])
    result = asyncio.run(run_scenario("three hours, home the same day", clauses))
    assert len(sent) == 2
    assert result.verdict == "insufficient_information"


def test_a_refusal_that_abandons_the_missing_list_names_no_refusing_clause(monkeypatch):
    """`day-care-not-listed`, four fresh samples of four: told the day-care list
    is missing, the retry answered not_covered "because in-patient cover needs
    24 hours", citing that cover clause as permitting. A refusal with no clause
    refusing is as unsupported as a payment resting on a missing list."""
    import asyncio

    from app.pipeline.scenario import run_scenario

    clauses = _clauses() + [_day_care_clause()]
    first = {
        "verdict": "conditional", "reasoning": "r", "missing_information": [],
        "deciding_clauses": [{"clause_id": "2.4", "quote": "Day Care Treatment listed in Annexure II", "effect": "permits"}],
    }
    abandoned = _answer("not_covered", "2.1", INPATIENT_QUOTE, "permits")
    _fake_model(monkeypatch, [first, abandoned])
    assert asyncio.run(run_scenario("six hours on a drip", clauses)).verdict == "insufficient_information"


def test_a_refusal_naming_a_refusing_clause_stands(monkeypatch):
    import asyncio

    from app.pipeline.scenario import ShortlistClause, run_scenario

    exclusion = ShortlistClause("4.9", "t:8", "4.9", "exclusion",
                                "4.9 Drips are not covered under this policy at all.", 1.0)
    clauses = _clauses() + [_day_care_clause(), exclusion]
    first = {
        "verdict": "conditional", "reasoning": "r", "missing_information": [],
        "deciding_clauses": [{"clause_id": "2.4", "quote": "Day Care Treatment listed in Annexure II", "effect": "permits"}],
    }
    refused = _answer("not_covered", "4.9", "Drips are not covered under this policy", "denies")
    _fake_model(monkeypatch, [first, refused])
    assert asyncio.run(run_scenario("six hours on a drip", clauses)).verdict == "not_covered"


def _day_care_clause():
    from app.pipeline.scenario import ShortlistClause

    return ShortlistClause("2.4", "t:7", "2.4", "coverage", DAY_CARE_TEXT, 1.0)


def test_a_claim_with_another_basis_is_not_downgraded(monkeypatch):
    import asyncio

    from app.pipeline.scenario import ShortlistClause, run_scenario

    clauses = _clauses() + [ShortlistClause("2.4", "t:9", "2.4", "coverage", DAY_CARE_TEXT, 1.0)]
    both = {
        "verdict": "covered", "reasoning": "r", "missing_information": [],
        "deciding_clauses": [
            {"clause_id": "2.1", "quote": INPATIENT_QUOTE, "effect": "permits"},
            {"clause_id": "2.4", "quote": "Day Care Treatment listed in Annexure II", "effect": "permits"},
        ],
    }
    _fake_model(monkeypatch, [both, both])
    assert asyncio.run(run_scenario("held five years", clauses)).verdict == "covered"


def test_a_treatment_the_policy_names_elsewhere_is_not_questioned():
    """`cataract-served`: a day-care cataract operation. The cataract sub-limit
    names the treatment and so presupposes it is paid; the missing day-care
    list is not what that answer rests on. Questioned anyway, the retry turned
    a correct answer into a cosmetic-surgery refusal."""
    from app.pipeline.scenario import ShortlistClause, compute, find_contradictions

    day_care = ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
    cataract = ShortlistClause("5.4", "t:2", "5.4", "sub_limit",
                               "5.4 Expenses in respect of treatment of cataract shall be limited.", 1.0)
    computed = compute({"procedure": "cataract operation"}, [day_care, cataract])
    cited = [_citation("2.4", "permits"), _citation("5.4", "reduces")]
    assert find_contradictions(cited, "conditional", computed) == ([], [])


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


def test_a_covered_answer_keeps_a_cleared_clause_relabelled_as_permitting(monkeypatch):
    """`room-rent-within-cap`, four fresh samples of four: the retry answered
    covered with the right reason and still labelled the room cap "reduces".
    Dropping it left a right answer with nothing behind it, and the downgrade
    turned it into insufficient_information."""
    import asyncio

    from app.pipeline.scenario import run_scenario

    _fake_model(monkeypatch, [
        _answer("not_covered", "3.2", PED_QUOTE, "denies"),
        _answer("covered", "3.2", PED_QUOTE, "denies"),
    ])
    result = asyncio.run(run_scenario("held five years", _clauses()))
    assert result.verdict == "covered"
    assert [(c.clause_id, c.effect) for c in result.citations] == [("3.2", "permits")]


def test_a_consistent_answer_makes_no_second_call(monkeypatch):
    import asyncio

    from app.pipeline.scenario import run_scenario

    sent = _fake_model(monkeypatch, [_answer("covered", "2.1", INPATIENT_QUOTE, "permits")])
    assert asyncio.run(run_scenario("held five years", _clauses())).verdict == "covered"
    assert len(sent) == 1


def test_an_age_at_policy_start_needs_the_policy_to_have_started():
    """"My father, 82, was admitted with pneumonia" came back as 82 AT POLICY
    START - a senior-citizen co-payment for a man whose inception age nobody
    mentioned. Two prompt fixes made it worse. The sentence says nothing about
    buying or starting a policy, so the only age in it is his age now."""
    from app.pipeline.scenario import correct_misfiled_age

    misfiled = {"age": None, "age_at_policy_start": 82}
    assert correct_misfiled_age(misfiled, "My father, 82, was admitted with pneumonia.") == {
        "age": 82, "age_at_policy_start": None,
    }


def test_a_stated_inception_age_is_left_alone():
    from app.pipeline.scenario import correct_misfiled_age

    for said in (
        "I bought this policy at 67 and I am claiming three years later.",
        "My mother took this policy out at 61 and was hospitalised two years later.",
        "I signed up for this insurance at 60. Two years later I had a heart attack.",
    ):
        facts = {"age": None, "age_at_policy_start": 61}
        assert correct_misfiled_age(facts, said) == facts
    # And both ages given: nothing to decide between.
    both = {"age": 72, "age_at_policy_start": 70}
    assert correct_misfiled_age(both, "I am 72 and was 70 back then.") == both


AYUSH_TEXT = (
    "2.6 AYUSH Treatment The Company shall indemnify in-patient expenses for Ayurveda, Unani, "
    "Siddha and Homeopathy treatment taken in a government hospital or an institute accredited "
    "by the Quality Council of India."
)
HOSPITAL_TEXT = "1.1 Hospital means any institution registered with the local authorities."


def _policy(*texts):
    from app.pipeline.scenario import ShortlistClause

    return [ShortlistClause(t.split()[0], f"t:{i}", t.split()[0], "coverage", t, 1.0)
            for i, t in enumerate(texts)]


def test_a_clause_sharing_several_unusual_words_with_the_question_is_named():
    """Regression test for `ayush-private-clinic`: Ayurvedic treatment at a clinic
    with no Quality Council accreditation. The answer was right and cited the
    definition of a hospital, every run on record, instead of the AYUSH clause
    that decides it - even when asked for citations in a turn of its own."""
    from app.pipeline.scenario import shared_words

    question = ("I had in-patient Ayurvedic treatment at a private clinic that is not "
                "government-run and has no Quality Council accreditation.")
    shared = shared_words(question, _policy(AYUSH_TEXT, HOSPITAL_TEXT))
    assert list(shared) == ["2.6"]
    assert "ayurvedic" in shared["2.6"] and "accreditation" in shared["2.6"]


def test_one_shared_word_or_a_common_word_is_not_enough():
    """One rare word in common is a coincidence as often as a signal, and a
    word most clauses use says nothing about which clause this is."""
    from app.pipeline.scenario import shared_words

    common = [f"{n} Every clause here mentions the hospital and the treatment." for n in
              ("1.2", "1.3", "1.4")]
    assert shared_words("I had treatment in hospital.", _policy(*common, AYUSH_TEXT)) == {}
    assert shared_words("I went to a government office.", _policy(AYUSH_TEXT, HOSPITAL_TEXT)) == {}


def test_shared_words_are_listed_above_the_clauses_and_drop_nothing():
    from app.llm.prompts import render_reasoning_request

    question = "Ayurvedic treatment, no Quality Council accreditation."
    rendered = render_reasoning_request(question, {}, _policy(AYUSH_TEXT, HOSPITAL_TEXT))
    head, clauses = rendered.split("POLICY CLAUSES AVAILABLE TO YOU:")
    assert "clause 2.6" in head
    assert "1.1 Hospital means" in clauses  # every clause is still shown


def test_words_spliced_onto_a_true_quotation_are_trimmed_with_no_second_call(monkeypatch):
    """`post-hospitalisation-too-late` quoted clause 2.3 with "by the Company"
    spliced on from clause 2.2, in every run, and a retry repeated it byte for
    byte. The true part is kept; nothing is asked again; the verdict is untouched."""
    import asyncio

    from app.pipeline.scenario import run_scenario

    spliced = {"clause_id": "2.1", "quote": INPATIENT_QUOTE + " by the Company", "effect": "permits"}
    sent = _fake_model(monkeypatch, [
        _answer("covered", "2.1", INPATIENT_QUOTE, "permits") | {"deciding_clauses": [spliced]},
    ])
    result = asyncio.run(run_scenario("held five years", _clauses()))

    assert len(sent) == 1
    assert result.verdict == "covered"
    assert [(c.quote, c.verified) for c in result.citations] == [(INPATIENT_QUOTE, True)]


def test_an_invented_quotation_stays_flagged(monkeypatch):
    import asyncio

    from app.pipeline.scenario import run_scenario

    bad = {"clause_id": "2.1", "quote": "The Company shall pay for absolutely everything", "effect": "permits"}
    _fake_model(monkeypatch, [_answer("covered", "2.1", INPATIENT_QUOTE, "permits") | {"deciding_clauses": [bad]}])
    result = asyncio.run(run_scenario("held five years", _clauses()))
    assert not result.citations[0].verified
    assert not result.verified


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
