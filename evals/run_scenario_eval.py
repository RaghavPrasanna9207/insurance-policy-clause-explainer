"""Scenario simulator eval.

Separate from run_eval.py because it measures different things. The
classification eval asks "did we label this clause correctly". This asks three
questions that matter more, and that a labelling score cannot see:

  VERDICT ACCURACY     did it reach the answer a competent human reader would?
  CITATION RECALL      did it cite the clause that actually decides the case?
  FABRICATION RATE     how often did the model quote something not in the policy?
  DETECTION INTEGRITY  was every fabricated quote caught and flagged?

THE LAST TWO ARE NOT THE SAME MEASUREMENT, and an earlier version of this file
conflated them under one name with a hard 100% target. That was a category
error worth spelling out.

  Fabrication rate measures THE MODEL. A 7B model will sometimes attach an
  invented quote to a real clause id - measured here at 1 case in 16, where it
  cited a waiting-period clause for a co-payment question and quoted words that
  were not in it. Driving this to zero is a quality goal, not a guarantee.

  Detection integrity measures THE SYSTEM. Every fabricated quote must be
  caught and surfaced as unverified, never presented as fact. THIS is the
  property with a hard 100% target, and it is the one the grounding design
  actually promises.

Reporting a caught fabrication as a system failure would have been exactly
backwards: it is the check doing its job. What would be a real failure is an
unverifiable quote reaching a reader unflagged.

Usage:
    python evals/run_scenario_eval.py
    python evals/run_scenario_eval.py --no-cache
"""

import argparse
import asyncio
import json
from collections import Counter
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))

from app.config import settings  # noqa: E402
from app.grounding import verify_quote  # noqa: E402
from app.llm.prompts import PROMPT_VERSION  # noqa: E402
from app.llm import cache, client  # noqa: E402
from app.pipeline.analyze import analyze  # noqa: E402
from app.pipeline.ingest import ingest  # noqa: E402
from app.pipeline.scenario import (  # noqa: E402
    ShortlistClause,
    citation_ids,
    extract_facts,
    run_scenario,
)
from app.pipeline.score import score  # noqa: E402
from app.pipeline.segment import segment  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
GOLDEN_PDF = GOLDEN_DIR / "synthetic-health-policy.pdf"
CASES_PATH = GOLDEN_DIR / "scenarios.json"
# Held-out batches, by number. Each is run only to measure, never to tune; see
# the _why at the top of each file for when it was written and what it has seen.
HELDOUT_PATHS = {
    "1": GOLDEN_DIR / "scenarios-heldout.json",
    "2": GOLDEN_DIR / "scenarios-heldout-2.json",
}
REPORT_PATH = REPO_ROOT / "evals" / "scenario-report.md"


async def build_clauses(pdf: Path = GOLDEN_PDF) -> list[ShortlistClause]:
    """Run the map pipeline once; every scenario reuses the result."""
    result = ingest(pdf)
    segments = segment(result)
    analyses = await analyze(segments)
    scored = score(segments, analyses)

    analysed = [
        seg for seg in segments
        if str(seg.order_idx) in analyses and str(seg.order_idx) in scored
    ]
    # The same ids the scenario endpoint gives the model, from the same function.
    ids = citation_ids([(seg.number, seg.order_idx) for seg in analysed])

    clauses = []
    for seg, clause_id in zip(analysed, ids):
        analysis = analyses[str(seg.order_idx)]
        sc = scored[str(seg.order_idx)]
        clauses.append(
            ShortlistClause(
                clause_id=clause_id,
                ref=f"{pdf.stem}:{seg.order_idx}",
                number=seg.number,
                clause_type=analysis.clause_type,
                text=seg.text,
                impact_score=sc.impact_score,
                waiting_period_days=analysis.waiting_period_days,
                exceptions=analysis.exceptions,
                copay_percent=analysis.copay_percent,
                copay_min_age_at_inception=analysis.copay_min_age_at_inception,
                cap_percent_of_sum_insured=analysis.cap_percent_of_sum_insured,
                icu_cap_percent_of_sum_insured=analysis.icu_cap_percent_of_sum_insured,
                cover_window_days=analysis.cover_window_days,
                cover_window_anchor=analysis.cover_window_anchor,
                section_path=seg.section_path,
            )
        )
    return clauses


async def run(
    use_cache: bool, only: list[str] | None = None, resample: bool = False,
    cases_path: Path = CASES_PATH, pdf: Path = GOLDEN_PDF,
) -> dict:
    if not use_cache:
        cache.clear()

    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if only is not None:
        known = {c["id"] for c in cases}
        unknown = [i for i in only if i not in known]
        if unknown:
            raise SystemExit(f"unknown case id(s): {', '.join(unknown)}")
        cases = [c for c in cases if c["id"] in only]
    print(f"model: {settings.model}")
    print(f"preparing policy {pdf.name}...")
    clauses = await build_clauses(pdf)
    print(f"  {len(clauses)} analysed clauses\n")

    source_by_id = {c.clause_id: c.text for c in clauses}

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case['id']}", flush=True)
        # Facts are extracted FIRST, outside the count, so the count below sees
        # only the reasoning step - the one that produces the verdict.
        #
        # Counting every call over-reports. A change to the fact extractor's
        # prompt regenerates every case's facts, and nearly every case still
        # extracts the same facts, sends a byte-identical reasoning prompt, and
        # gets its stored verdict back. That case could not have moved, but a
        # count that included the fact call would have called it regenerated.
        # run_scenario's own call to extract_facts is then a cache hit.
        await extract_facts(case["scenario"])
        calls_before = client.model_calls
        result = await run_scenario(case["scenario"], clauses, resample=resample)
        # True if the reasoning reached the model. False means the verdict is
        # the stored bytes of an earlier run and cannot have moved. See
        # compare_with_previous for why that distinction carries the eval.
        fresh = client.model_calls > calls_before

        # Re-verify every quotation INDEPENDENTLY rather than trusting the flag
        # the pipeline set. A self-reported guarantee is not a measurement: if
        # the verifier were silently broken, its own flag would happily say
        # everything is fine. This recomputes the check and compares.
        truly_bad = 0
        flagged_bad = 0
        for citation in result.citations:
            ok, _ = verify_quote(citation.quote, source_by_id.get(citation.clause_id, ""))
            if not ok:
                truly_bad += 1
                if not citation.verified:
                    flagged_bad += 1

        cited = {c.clause_id for c in result.citations}
        required = set(case["must_cite"])
        # Clauses the answer must NOT lean on. Recall alone cannot see this
        # failure: a system that cites every clause it can reach scores a
        # perfect recall while being useless, and one pushed to hunt for
        # co-payments starts finding them for people who were 58 at inception.
        forbidden = set(case.get("must_not_cite", []))
        rows.append({
            "id": case["id"],
            "expected": case["expected_verdict"],
            "got": result.verdict,
            "verdict_ok": result.verdict == case["expected_verdict"],
            "required": sorted(required),
            "forbidden": sorted(forbidden),
            "wrongly_cited": sorted(forbidden & cited),
            "cited": sorted(cited),
            "citation_ok": required.issubset(cited),
            "grounded": result.verified,
            "unverified": [
                {"clause": c.clause_id, "reason": c.unverified_reason}
                for c in result.citations if not c.verified
            ],
            "reasoning": result.reasoning,
            "why": case["why"],
            "truly_bad": truly_bad,
            "flagged_bad": flagged_bad,
            "fresh": fresh,
        })

    elapsed = time.perf_counter() - started
    total = len(rows)
    # Citation recall is only meaningful for cases that require a citation;
    # "insufficient_information with nothing to cite" would otherwise inflate it.
    citable = [r for r in rows if r["required"]]
    # Likewise, only the cases that name a forbidden clause can fail this way.
    guarded = [r for r in rows if r["forbidden"]]

    return {
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "seconds": round(elapsed, 1),
        "rows": rows,
        # Another case file is never the report's 40 cases, so it is recorded
        # like a subset and never overwrites the main report.
        "subset": only is not None or cases_path != CASES_PATH or pdf != GOLDEN_PDF,
        "verdict_accuracy": sum(r["verdict_ok"] for r in rows) / max(total, 1),
        "citation_recall": (
            sum(r["citation_ok"] for r in citable) / len(citable) if citable else 1.0
        ),
        # The false-positive direction of citation quality. Reported alongside
        # recall rather than folded into it, because the two fail for opposite
        # reasons and a single blended number would let one hide the other.
        "false_citation_rate": (
            sum(bool(r["wrongly_cited"]) for r in guarded) / len(guarded)
            if guarded
            else 0.0
        ),
        "guarded_count": len(guarded),
        # Renamed from "groundedness", which merged two different questions.
        # This one is about the MODEL: how often did it quote something that is
        # not in the policy.
        "fabrication_rate": sum(not r["grounded"] for r in rows) / max(total, 1),
        # This one is about the SYSTEM: of the answers containing a fabricated
        # quote, how many were correctly flagged as unverified. Anything below
        # 1.000 means an invented quotation reached the surface looking like
        # evidence, which is the failure the whole design exists to prevent.
        "detection_integrity": (
            sum(r["flagged_bad"] for r in rows) / sum(r["truly_bad"] for r in rows)
            if sum(r["truly_bad"] for r in rows)
            else 1.0
        ),
        "fabricated_quotes": sum(r["truly_bad"] for r in rows),
        "citable_count": len(citable),
        "total": total,
    }


HISTORY_PATH = REPO_ROOT / "evals" / "run-history.json"
HISTORY_KEEP = 10


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A convenience record, not a result. A corrupted one is not worth
        # failing an eval over; start again.
        return []


def record_run(res: dict) -> list[dict]:
    """Append this run's per-case outcomes to the run history, and return it.

    WHY A HISTORY FILE EXISTS AT ALL. This eval was compared against itself for
    eight milestones on the assumption that running it twice gives the same
    answer. It does not: three runs of identical code scored 0.725, 0.700 and
    0.750. Every ordinary run already knows exactly which cases it got right,
    so keeping that costs nothing, and it is what the comparison and stability
    sections below are computed from.

    `fresh` is recorded per case because a replayed answer is not a new
    measurement. See stability_section for what went wrong without it.
    """
    history = load_history()
    history.append({
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "prompt_version": PROMPT_VERSION,
        "subset": res["subset"],
        "verdict_accuracy": res["verdict_accuracy"],
        "cases": {
            r["id"]: {"got": r["got"], "fresh": r["fresh"]} for r in res["rows"]
        },
    })
    history = history[-HISTORY_KEEP:]
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def unstable_cases(history: list[dict]) -> dict[str, list[str]]:
    """Cases that produced more than one verdict across FRESH samples.

    Replayed answers are skipped: they are copies of an earlier sample, and
    counting a copy as agreement is how five identical-looking runs can be one
    measurement repeated five times.
    """
    seen: dict[str, set[str]] = {}
    for entry in history:
        for case_id, outcome in entry["cases"].items():
            if outcome["fresh"]:
                seen.setdefault(case_id, set()).add(outcome["got"])
    return {cid: sorted(v) for cid, v in seen.items() if len(v) > 1}


def watchlist(cases: list[dict], history: list[dict]) -> list[str]:
    """The cases worth re-running for a quick check: failing, or wobbly.

    Derived from the history rather than written down, because a hand-kept list
    of "the cases that fail" is out of date the moment something is fixed. The
    cases that pass reliably are the ones a quick check can safely skip; the
    full set is still run at milestones.
    """
    expected = {c["id"]: c["expected_verdict"] for c in cases}
    latest: dict[str, str] = {}
    for entry in history:
        for case_id, outcome in entry["cases"].items():
            latest[case_id] = outcome["got"]
    failing = {cid for cid, got in latest.items() if got != expected.get(cid)}
    wobbly = set(unstable_cases(history))
    return [c["id"] for c in cases if c["id"] in failing | wobbly]


def compare_with_previous(res: dict, history: list[dict]) -> dict:
    """Split this run's cases by whether they COULD have changed.

    THE PROBLEM THIS SOLVES. A prompt change used to be judged by comparing two
    aggregate scores, and the aggregate moves by two or three cases between
    runs of identical code. A two-case improvement and no improvement at all
    looked the same.

    But most of the noise is avoidable, because the cache is content-addressed.
    A case whose every model request was answered from the cache sent exactly
    the same text as before and got back exactly the stored bytes, so its
    verdict cannot have moved - not "is unlikely to have", cannot. Only the
    cases whose requests actually reached the model are new samples, and those
    are precisely the cases the change touched.

    So instead of one blended score, a change is reported as: how many cases it
    reached, and of those, which were fixed and which were broken. The
    untouched cases contribute no noise at all.

    `history` must be the history BEFORE this run was recorded.
    """
    wobbly = unstable_cases(history)
    out = {
        "baseline": history[-1] if history else None,
        "replayed": [], "drifted": [], "no_baseline": [],
        "fixed": [], "broken": [], "still_right": [], "still_wrong": [],
        "wobbly": wobbly,
    }
    for row in res["rows"]:
        previous = next(
            (e["cases"][row["id"]] for e in reversed(history) if row["id"] in e["cases"]),
            None,
        )
        if previous is None:
            out["no_baseline"].append(row["id"])
            continue
        if not row["fresh"]:
            out["replayed"].append(row["id"])
            # Replayed, yet different from the previous run: the stored answer
            # came from some OTHER run than the one being compared against.
            # Two ordinary causes - a change was reverted, so the old prompt's
            # answers replay; or the cache was filled by a run that was never
            # recorded (interrupted, or --no-report). Either way the named
            # baseline is not where this answer came from, so it is surfaced
            # rather than silently counted as a fix or a break.
            if previous["got"] != row["got"]:
                out["drifted"].append(row["id"])
            continue
        was_right = previous["got"] == row["expected"]
        if row["verdict_ok"] and not was_right:
            out["fixed"].append(row["id"])
        elif was_right and not row["verdict_ok"]:
            out["broken"].append(row["id"])
        elif row["verdict_ok"]:
            out["still_right"].append(row["id"])
        else:
            out["still_wrong"].append(row["id"])
    return out


def render_comparison(cmp: dict) -> str:
    base = cmp["baseline"]
    if base is None:
        return (
            "No earlier run on record, so there is nothing to compare against.\n"
            "The next run will be compared with this one."
        )

    def names(ids: list[str]) -> str:
        return ", ".join(
            f"{i} (has wobbled before)" if i in cmp["wobbly"] else i for i in ids
        )

    regenerated = (
        len(cmp["fixed"]) + len(cmp["broken"])
        + len(cmp["still_right"]) + len(cmp["still_wrong"])
    )
    lines = [
        f"Compared with the previous run ({base['at']}, {base['prompt_version']}):",
        "",
        f"  {len(cmp['replayed']):3} replayed from cache   same text sent, stored answer returned - cannot have moved",
        f"  {regenerated:3} regenerated         the only cases this run could have changed",
    ]
    if regenerated:
        lines += [
            f"        fixed        {len(cmp['fixed']):3}  {names(cmp['fixed'])}",
            f"        broken       {len(cmp['broken']):3}  {names(cmp['broken'])}",
            f"        still right  {len(cmp['still_right']):3}",
            f"        still wrong  {len(cmp['still_wrong']):3}",
        ]
    if cmp["drifted"]:
        lines += [
            "",
            f"  NOTE: {len(cmp['drifted'])} replayed case(s) differ from the previous run:",
            f"  {', '.join(cmp['drifted'])}",
            "  Their stored answers come from an earlier run than the one named",
            "  above - usually because a change was reverted and the old prompt's",
            "  answers replayed, or because an unrecorded run filled the cache.",
            "  They are not counted as fixed or broken.",
        ]
    if cmp["no_baseline"]:
        lines.append(f"\n  {len(cmp['no_baseline'])} case(s) never run before: {', '.join(cmp['no_baseline'])}")
    if regenerated:
        lines += [
            "",
            "  A regenerated case is ONE sample from a model that does not answer",
            "  identically twice. Before believing a fix or break, ask again:",
            "      --only <ids> --resample",
            "  A plain --only re-run replays the stored answer, which is a copy,",
            "  not a second opinion.",
        ]
    return "\n".join(lines)


def stability_section(history: list[dict]) -> str:
    """Report which cases answer consistently, and which are coin flips.

    COUNTING ONLY FRESH SAMPLES, which the first version of this function did
    not do. It counted every recorded run, and a run served entirely from the
    cache copies the previous run's answers exactly - so two recorded runs
    where the second was a replay reported "every case stable across 2 runs"
    from what was really one measurement. Copies always agree with the thing
    they copy. A case is only called stable here once it has been generated
    fresh at least twice.
    """
    full_runs = [e for e in history if not e["subset"]]
    fresh_counts: dict[str, int] = {}
    for entry in history:
        for case_id, outcome in entry["cases"].items():
            if outcome["fresh"]:
                fresh_counts[case_id] = fresh_counts.get(case_id, 0) + 1
    measured = sorted(cid for cid, n in fresh_counts.items() if n >= 2)
    wobbly = unstable_cases(history)

    lines = ["## Stability", ""]
    if not measured:
        lines.append(
            "No case has been generated fresh more than once yet, so nothing can "
            "be said about stability. Replayed runs do not count - they return "
            "stored answers. Run with `--no-cache`, or change something, and "
            "this section fills in."
        )
        return "\n".join(lines) + "\n"

    if len(full_runs) >= 2:
        accs = [e["verdict_accuracy"] for e in full_runs]
        lines += [
            f"Full runs on record: {len(full_runs)}. "
            f"Verdict accuracy ranged **{min(accs):.3f} - {max(accs):.3f}**.",
            "",
        ]
    lines += [
        f"**{len(measured)}** cases have at least two fresh samples. "
        f"Of those, **{len(wobbly)}** gave different verdicts on different runs.",
        "",
    ]
    if wobbly:
        lines += [
            "A case listed here is not a result: whether it counts as correct "
            "depends on which run you look at. A change worth fewer cases than "
            "this list cannot be measured by comparing aggregate scores.",
            "",
            "| Case | Verdicts seen | Fresh samples |",
            "|---|---|---:|",
        ]
        for case_id, seen in sorted(wobbly.items()):
            lines.append(
                f"| `{case_id}` | {', '.join(f'`{v}`' for v in seen)} | "
                f"{fresh_counts[case_id]} |"
            )
    return "\n".join(lines) + "\n"


def report(res: dict) -> str:
    lines = [
        "# Scenario simulator evaluation",
        "",
        f"- **Model**: `{res['model']}`",
        f"- **Prompt version**: `{res['prompt_version']}`",
        f"- **Decoding**: num_ctx {res['num_ctx']}, num_predict {res['num_predict']}",
        f"- **Cases**: {res['total']}",
        f"- **Wall time**: {res['seconds']}s",
        "",
        "## Results",
        "",
        f"| Metric | Score | Target |",
        f"|---|---:|---|",
        f"| Verdict accuracy | **{res['verdict_accuracy']:.3f}** | quality measure |",
        f"| Citation recall | **{res['citation_recall']:.3f}** | quality measure "
        f"({res['citable_count']} cases require a citation) |",
        f"| False citation rate | **{res['false_citation_rate']:.3f}** | quality measure "
        f"({res['guarded_count']} cases name a clause that must NOT be relied on) |",
        f"| Fabrication rate | **{res['fabrication_rate']:.3f}** | quality measure "
        f"({res['fabricated_quotes']} invented quote(s)) |",
        f"| **Detection integrity** | **{res['detection_integrity']:.3f}** "
        f"| **must be 1.000** |",
        "",
        "**These last two measure different things, and only one is a guarantee.**",
        "",
        "*Fabrication rate* is about the model: how often it attached an invented",
        "quotation to a real clause id. A 7B model will sometimes do this, and",
        "driving it to zero is a quality goal.",
        "",
        "*Detection integrity* is about the system: of the quotations that genuinely",
        "could not be found in the policy, how many were caught and shown to the",
        "reader as unverified. This is the property the grounding design actually",
        "promises, and it is the only hard target here. It is computed by",
        "re-verifying every quotation independently and comparing against the flag",
        "the pipeline set - a self-reported guarantee is not a measurement.",
        "",
        "A caught fabrication is the check working, not the system failing.",
        "",
        "## Case by case",
        "",
        "| Case | Expected | Got | Cited | Grounded |",
        "|---|---|---|---|---|",
    ]
    for row in res["rows"]:
        verdict_mark = "" if row["verdict_ok"] else " ⚠"
        cite = ", ".join(row["cited"]) or "—"
        if row["required"] and not row["citation_ok"]:
            cite += f" (missing {', '.join(row['required'])})"
        if row["wrongly_cited"]:
            cite += f" (**must not cite {', '.join(row['wrongly_cited'])}**)"
        lines.append(
            f"| `{row['id']}` | {row['expected']} | {row['got']}{verdict_mark} "
            f"| {cite} | {'yes' if row['grounded'] else '**NO**'} |"
        )

    misses = [r for r in res["rows"] if not r["verdict_ok"]]
    if misses:
        lines += ["", "## Verdict misses", ""]
        for row in misses:
            lines += [
                f"**`{row['id']}`** — expected `{row['expected']}`, got `{row['got']}`",
                "",
                f"- Why the expected answer is right: {row['why']}",
                f"- What the model said: {row['reasoning']}",
                "",
            ]

    ungrounded = [r for r in res["rows"] if not r["grounded"]]
    if ungrounded:
        lines += ["", "## Ungrounded citations (correctness failures)", ""]
        for row in ungrounded:
            for item in row["unverified"]:
                lines.append(f"- `{row['id']}` clause {item['clause']}: {item['reason']}")

    lines += ["", "---", "", "Regenerate with `python evals/run_scenario_eval.py`.", ""]
    return "\n".join(lines)


def majority(samples: list[dict]) -> list[dict]:
    """Per case across repeated runs: every verdict seen, and the majority one.

    WHY. The model does not answer identically twice (M9, M10), so one full run
    is one sample and its score moves by two or three cases on its own. Asking
    each case several times and taking the majority measures what the system
    USUALLY answers, which is the thing a reader of the headline number assumes
    it means.

    A strict majority is required. With three samples and four possible
    verdicts a three-way split can happen, and it counts as wrong: picking one
    of three disagreeing answers would be choosing a result, not measuring one.
    """
    verdicts: dict[str, list[str]] = {}
    expected: dict[str, str] = {}
    for res in samples:
        for r in res["rows"]:
            verdicts.setdefault(r["id"], []).append(r["got"])
            expected[r["id"]] = r["expected"]

    out = []
    for case_id, got in verdicts.items():
        top, count = Counter(got).most_common(1)[0]
        winner = top if count * 2 > len(got) else None
        out.append({
            "id": case_id, "expected": expected[case_id], "verdicts": got,
            "majority": winner, "agreed": count, "ok": winner == expected[case_id],
        })
    return out


def citation_problems(samples: list[dict]) -> dict[str, list[str]]:
    """Per case: each missing or forbidden citation, and in how many samples.

    The per-sample rates say HOW MANY cases cited wrongly; this says WHICH, so
    a wrong citation seen in two samples of three is on record by name.
    """
    counts: dict[str, Counter] = {}
    for res in samples:
        for r in res["rows"]:
            missing = sorted(set(r["required"]) - set(r["cited"]))
            for clause_id in missing:
                counts.setdefault(r["id"], Counter())[f"missing {clause_id}"] += 1
            for clause_id in r["wrongly_cited"]:
                counts.setdefault(r["id"], Counter())[f"cited forbidden {clause_id}"] += 1
    n = len(samples)
    return {
        case_id: [f"{problem} in {times}/{n}" for problem, times in sorted(c.items())]
        for case_id, c in counts.items()
    }


def render_majority(samples: list[dict]) -> str:
    rows = majority(samples)
    n = len(samples)
    lines = [f"Majority of {n} samples per case:", ""]
    for r in rows:
        flag = "  " if r["ok"] else "<-"
        shown = r["majority"] or "no majority"
        lines.append(
            f"  {flag} {r['id']:30} {r['expected']:26} got {shown}  ({r['agreed']}/{n} agreed)"
        )
    right = sum(r["ok"] for r in rows)
    unanimous = sum(r["agreed"] == n for r in rows)
    lines += [
        "",
        f"  majority verdict accuracy : {right}/{len(rows)} = {right / max(len(rows), 1):.3f}",
        f"  unanimous cases           : {unanimous}/{len(rows)}",
        "",
        "  per sample:  verdict  recall  false-cit  failed-quotes  detection (must be 1.000)",
    ]
    for i, res in enumerate(samples, 1):
        lines.append(
            f"    sample {i}   {res['verdict_accuracy']:.3f}    {res['citation_recall']:.3f}"
            f"   {res['false_citation_rate']:.3f}      {res['fabricated_quotes']}"
            f"              {res['detection_integrity']:.3f}"
        )
    problems = citation_problems(samples)
    if problems:
        lines += ["", "  citation problems:"]
        lines += [f"    {cid}: {'; '.join(p)}" for cid, p in problems.items()]
    return "\n".join(lines)


async def run_repeats(
    n: int, use_cache: bool, only: list[str] | None, cases_path: Path = CASES_PATH,
    pdf: Path = GOLDEN_PDF,
) -> list[dict]:
    """The first sample may replay the cache; every later one is asked afresh."""
    results = [await run(use_cache=use_cache, only=only, cases_path=cases_path, pdf=pdf)]
    for _ in range(n - 1):
        results.append(
            await run(use_cache=use_cache, only=only, resample=True,
                      cases_path=cases_path, pdf=pdf)
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--only",
        help="comma-separated case ids to run, e.g. --only non-disclosure,late-notice",
    )
    parser.add_argument(
        "--resample", action="store_true",
        help="answer the reasoning step afresh instead of replaying the cache "
             "(combine with --only or --watchlist to test whether a verdict is stable)",
    )
    parser.add_argument(
        "--watchlist", action="store_true",
        help="run only the cases that failed or wobbled in the recorded history",
    )
    parser.add_argument(
        "--heldout", choices=sorted(HELDOUT_PATHS), default=None,
        help="run a held-out batch (1 or 2) instead of the main 40; never tune on these",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="ask each case N times (first may replay the cache, the rest afresh) "
             "and score the majority verdict; combine with --only to split a long run",
    )
    parser.add_argument(
        "--policy", type=Path, default=GOLDEN_PDF,
        help="a different policy PDF, e.g. one from evals/real (requires --cases)",
    )
    parser.add_argument(
        "--cases", type=Path, default=None,
        help="a scenario file written for --policy",
    )
    args = parser.parse_args()
    # The golden cases cite the golden policy's clause numbers, so asking them
    # of any other document would score answers against the wrong answer key.
    if args.policy != GOLDEN_PDF and args.cases is None:
        parser.error("--policy needs --cases written for that policy")

    # Read before this run is recorded: the comparison is against what came
    # BEFORE, and the watchlist is chosen from it.
    history = load_history()
    cases_path = args.cases or (HELDOUT_PATHS[args.heldout] if args.heldout else CASES_PATH)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]

    only = None
    if args.only:
        only = [i.strip() for i in args.only.split(",") if i.strip()]
    elif args.watchlist:
        only = watchlist(cases, history)
        if not only:
            print("The watchlist is empty: no recorded failures or wobbles.")
            return
        print(f"watchlist: {len(only)} of {len(cases)} cases\n")

    if args.repeats > 1:
        results = asyncio.run(
            run_repeats(args.repeats, not args.no_cache, only, cases_path, args.policy)
        )
        print()
        print(render_majority(results))
        if not args.no_report:
            # Every sample is recorded: the fresh ones are exactly what the
            # stability section counts. The single-run report is not written,
            # because it describes one sample and this was several.
            for res in results:
                record_run(res)
        return

    res = asyncio.run(
        run(use_cache=not args.no_cache, only=only, resample=args.resample,
            cases_path=cases_path, pdf=args.policy)
    )

    print()
    scope = f"  ({res['total']}-case subset)" if res["subset"] else ""
    print(f"  verdict accuracy : {res['verdict_accuracy']:.3f}{scope}")
    print(f"  citation recall  : {res['citation_recall']:.3f} "
          f"({res['citable_count']} cases)")
    print(f"  false citations  : {res['false_citation_rate']:.3f} "
          f"({res['guarded_count']} guarded cases)")
    print(f"  fabrication rate : {res['fabrication_rate']:.3f} "
          f"({res['fabricated_quotes']} invented quote(s))")
    print(f"  detection integ. : {res['detection_integrity']:.3f}  (must be 1.000)")
    print()
    for row in res["rows"]:
        flag = "  " if row["verdict_ok"] else "<-"
        ground = "" if row["grounded"] else "  UNGROUNDED"
        source = "" if row["fresh"] else "  (replayed)"
        print(f"  {flag} {row['id']:26} {row['expected']:26} got {row['got']}{ground}{source}")

    comparison = render_comparison(compare_with_previous(res, history))
    print()
    print(comparison)

    if not args.no_report:
        history = record_run(res)
        # A subset run is recorded - its fresh samples are real measurements -
        # but it does not overwrite the full report, which would otherwise
        # silently start describing 13 cases under a heading about 40.
        if not res["subset"]:
            REPORT_PATH.write_text(
                report(res)
                + "\n---\n\n## Compared with the previous run\n\n```\n"
                + comparison
                + "\n```\n\n---\n\n"
                + stability_section(history),
                encoding="utf-8",
            )
            print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
