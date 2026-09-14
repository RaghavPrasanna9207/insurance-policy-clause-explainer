"""The eval's comparison and stability logic.

These functions decide whether a prompt change is reported as a fix, a break,
or noise, so a bug in them would quietly mislead every later decision - the
exact failure the history file was built to prevent. They are pure functions
over plain dicts, so they can be pinned down without running a model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from run_scenario_eval import (  # noqa: E402
    compare_with_previous,
    majority,
    stability_section,
    unstable_cases,
    watchlist,
)


def entry(cases: dict[str, tuple[str, bool]], subset: bool = False, acc: float = 0.5):
    return {
        "at": "2026-09-13 10:00 UTC",
        "prompt_version": "vtest",
        "subset": subset,
        "verdict_accuracy": acc,
        "cases": {cid: {"got": got, "fresh": fresh} for cid, (got, fresh) in cases.items()},
    }


def row(case_id: str, expected: str, got: str, fresh: bool):
    return {
        "id": case_id, "expected": expected, "got": got,
        "verdict_ok": got == expected, "fresh": fresh,
    }


# --- what could have changed ---------------------------------------------


def test_a_replayed_case_is_never_reported_as_fixed_or_broken():
    """The whole point. A replayed answer is stored bytes, so even if it
    somehow differed it must not be counted as the change's effect."""
    history = [entry({"a": ("covered", True)})]
    res = {"rows": [row("a", "not_covered", "covered", fresh=False)]}
    cmp = compare_with_previous(res, history)
    assert cmp["replayed"] == ["a"]
    assert not cmp["fixed"] and not cmp["broken"]


def test_regenerated_cases_are_split_into_fixed_broken_and_unchanged():
    history = [entry({
        "was-wrong": ("covered", True),
        "was-right": ("not_covered", True),
        "stays-right": ("covered", True),
        "stays-wrong": ("covered", True),
    })]
    res = {"rows": [
        row("was-wrong", "not_covered", "not_covered", fresh=True),
        row("was-right", "not_covered", "conditional", fresh=True),
        row("stays-right", "covered", "covered", fresh=True),
        row("stays-wrong", "not_covered", "conditional", fresh=True),
    ]}
    cmp = compare_with_previous(res, history)
    assert cmp["fixed"] == ["was-wrong"]
    assert cmp["broken"] == ["was-right"]
    assert cmp["still_right"] == ["stays-right"]
    assert cmp["still_wrong"] == ["stays-wrong"]


def test_a_replay_that_disagrees_with_the_record_is_flagged_not_hidden():
    """Happens when a change is reverted (the old prompt's stored answers
    replay) or an unrecorded run refilled the cache. Found by reverting a
    change: nine replayed cases differed from the run just before, and an
    earlier wording of this warning blamed an unrecorded run that did not
    exist. Either way the named baseline is not where the answer came from."""
    history = [entry({"a": ("covered", True)})]
    res = {"rows": [row("a", "covered", "conditional", fresh=False)]}
    assert compare_with_previous(res, history)["drifted"] == ["a"]


def test_the_baseline_for_a_case_is_the_latest_run_that_contained_it():
    """A subset run between two full runs must not hide a case's history."""
    history = [
        entry({"a": ("covered", True), "b": ("covered", True)}),
        entry({"b": ("not_covered", True)}, subset=True),
    ]
    res = {"rows": [row("a", "not_covered", "not_covered", fresh=True)]}
    assert compare_with_previous(res, history)["fixed"] == ["a"]


# --- stability counts fresh samples only ---------------------------------


def test_replayed_copies_never_make_a_case_look_stable_or_unstable():
    """The bug in the first version: a replayed run copied the previous one,
    and a copy always agrees with what it copied."""
    history = [
        entry({"a": ("covered", True)}),
        entry({"a": ("not_covered", False)}),  # replayed - ignored
    ]
    assert unstable_cases(history) == {}
    assert "No case has been generated fresh more than once" in stability_section(history)


def test_two_fresh_samples_that_disagree_make_a_case_unstable():
    history = [
        entry({"a": ("covered", True)}),
        entry({"a": ("conditional", True)}),
    ]
    assert unstable_cases(history) == {"a": ["conditional", "covered"]}


# --- watchlist -----------------------------------------------------------


def test_watchlist_holds_failing_and_wobbly_cases_and_skips_reliable_ones():
    cases = [
        {"id": "reliable", "expected_verdict": "covered"},
        {"id": "failing", "expected_verdict": "not_covered"},
        {"id": "wobbly", "expected_verdict": "covered"},
    ]
    history = [
        entry({"reliable": ("covered", True), "failing": ("covered", True),
               "wobbly": ("conditional", True)}),
        entry({"reliable": ("covered", True), "failing": ("covered", True),
               "wobbly": ("covered", True)}),
    ]
    assert watchlist(cases, history) == ["failing", "wobbly"]


# --- asking each case more than once --------------------------------------


def samples(*runs: dict[str, str], expected: str = "covered"):
    """Each positional arg is one run: {case_id: verdict it got}."""
    return [{"rows": [row(cid, expected, got, True) for cid, got in run.items()]} for run in runs]


def test_a_unanimous_right_answer_is_right():
    out = majority(samples({"a": "covered"}, {"a": "covered"}, {"a": "covered"}))
    assert out == [{"id": "a", "expected": "covered", "verdicts": ["covered"] * 3,
                    "majority": "covered", "agreed": 3, "ok": True}]


def test_two_of_three_decides_it_in_either_direction():
    right = majority(samples({"a": "covered"}, {"a": "conditional"}, {"a": "covered"}))
    wrong = majority(samples({"a": "conditional"}, {"a": "covered"}, {"a": "conditional"}))
    assert right[0]["ok"] and right[0]["agreed"] == 2
    assert not wrong[0]["ok"] and wrong[0]["majority"] == "conditional"


def test_a_three_way_split_has_no_majority_and_counts_as_wrong():
    """Picking one of three disagreeing answers would be choosing a result, not
    measuring one. Even if one of the three happened to be right."""
    out = majority(samples({"a": "covered"}, {"a": "conditional"}, {"a": "not_covered"}))
    assert out[0]["majority"] is None
    assert not out[0]["ok"]
