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
    citation_problems,
    compare_with_previous,
    majority,
    render_comparison,
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


def test_citation_problems_name_the_case_and_how_often():
    """A rate says one guarded case in five cited a forbidden clause. Which
    case, and in how many samples, is what anyone fixing it needs - and a
    summary that printed only the rate left that unrecorded."""
    def cited(cid, cites, required=(), forbidden=()):
        return {"id": cid, "cited": sorted(cites), "required": sorted(required),
                "wrongly_cited": sorted(set(cites) & set(forbidden))}

    runs = [
        {"rows": [cited("a", {"5.3"}, forbidden={"5.3"}), cited("b", {"2.1"}, required={"6.6"})]},
        {"rows": [cited("a", set(), forbidden={"5.3"}), cited("b", {"2.1"}, required={"6.6"})]},
    ]
    assert citation_problems(runs) == {
        "a": ["cited forbidden 5.3 in 1/2"],
        "b": ["missing 6.6 in 2/2"],
    }


def test_an_unanswered_case_is_wrong_and_cites_nothing():
    """M16: one answer that never finished used to end the whole eval. It is now
    a row like any other - wrong, missing its citation, counted by majority."""
    from run_scenario_eval import NO_ANSWER, _unanswered

    case = {"id": "a", "scenario": "", "expected_verdict": "not_covered",
            "must_cite": ["2#2"], "why": ""}
    unanswered = _unanswered(case, "LlmError: truncated")

    assert unanswered["got"] == NO_ANSWER and not unanswered["verdict_ok"]
    assert not unanswered["citation_ok"]
    assert citation_problems([{"rows": [unanswered]}]) == {"a": ["missing 2#2 in 1/1"]}
    right = {"rows": [row("a", "not_covered", "not_covered", True)]}
    assert majority([{"rows": [unanswered]}, right, right])[0]["ok"]



def test_a_comparison_across_ollama_versions_says_so():
    """M17: an Ollama update alone moved 10 of 69 verdicts, so a comparison that
    spans one must not read as the effect of whatever else changed."""
    history = [dict(entry({"a": ("covered", True)}), ollama_version="0.34.1")]

    def compared(version):
        res = {"rows": [row("a", "covered", "covered", fresh=True)], "ollama_version": version}
        return render_comparison(compare_with_previous(res, history))

    across = compared("0.34.2")
    assert "Ollama 0.34.1" in across and "Ollama 0.34.2" in across
    assert "Ollama 0.34.2" not in compared("0.34.1")

# --- the order each sample asks its questions in ------------------------------


def _fake_eval(monkeypatch, tmp_path, n_cases: int):
    """The eval's run() with the model replaced, recording the order it asks in."""
    import asyncio
    import json
    from types import SimpleNamespace

    import run_scenario_eval as eval_module

    ids = [f"q{i}" for i in range(n_cases)]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [
        {"id": i, "scenario": i, "expected_verdict": "covered", "must_cite": [], "why": ""}
        for i in ids
    ]}), encoding="utf-8")
    asked: list[str] = []

    async def no_clauses(pdf):
        return []

    async def no_facts(scenario):
        return None

    async def no_server_version():
        return "test"

    async def answer(scenario, clauses, resample=False):
        asked.append(scenario)
        return SimpleNamespace(verdict="covered", citations=[], verified=True, reasoning="")

    monkeypatch.setattr(eval_module, "build_clauses", no_clauses)
    monkeypatch.setattr(eval_module, "extract_facts", no_facts)
    monkeypatch.setattr(eval_module, "run_scenario", answer)
    monkeypatch.setattr(eval_module.client, "runtime_version", no_server_version)

    def repeats(n):
        return asyncio.run(eval_module.run_repeats(n, True, None, cases_path=path))

    return ids, asked, repeats


def test_repeated_samples_ask_in_different_orders(monkeypatch, tmp_path):
    """M16, Failure 68: asked in the same order, each sample found the same
    server history and the three samples agreed on every verdict - copies, not
    draws. The first sample keeps the file's order; each later one its own."""
    ids, asked, repeats = _fake_eval(monkeypatch, tmp_path, 12)

    results = repeats(3)

    first, second, third = asked[0:12], asked[12:24], asked[24:36]
    assert first == ids
    assert sorted(second) == sorted(ids) and sorted(third) == sorted(ids)
    assert len({tuple(first), tuple(second), tuple(third)}) == 3
    # Reported in the file's order whatever order it was asked in, so every
    # sample's table reads the same way.
    assert all([r["id"] for r in res["rows"]] == ids for res in results)


def test_a_sample_number_always_asks_in_the_same_order(monkeypatch, tmp_path):
    """Seeded by the sample number, so a run can be repeated exactly."""
    ids, asked, repeats = _fake_eval(monkeypatch, tmp_path, 12)

    repeats(2)
    once = asked[12:24]
    asked.clear()
    repeats(2)

    assert asked[12:24] == once
