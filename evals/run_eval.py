"""Evaluation harness. Measures the pipeline against the golden set.

WHY THIS IS THE MOST IMPORTANT SCRIPT IN THE REPO
-------------------------------------------------
Anyone can wire a model to a PDF parser and produce output that looks right.
The difference between that and an engineered system is being able to answer
"how well does it work, and how do you know" with a number rather than a
demonstration.

This script exists so that every subsequent decision - a prompt edit, a bigger
model, a different batch size - is settled by measurement instead of impression.
Impressions are formed by looking at a handful of outputs, which is exactly the
sample where a plausible-sounding wrong answer is most convincing.

WHAT IS MEASURED
----------------
Per-type precision, recall and F1 for clause classification, plus macro-F1 (the
unweighted mean across types).

Macro-F1 is the headline number rather than plain accuracy, and the choice
matters. The golden policy has 8 exclusions but only 4 waiting periods. Accuracy
lets a model do well on the common types and ignore the rare ones. Macro-F1
weights every type equally, so failing on waiting periods costs as much as
failing on exclusions - which reflects reality, since a missed waiting period
misleads a policyholder just as badly.

The confusion pairs printed at the end matter as much as the score: WHICH types
get mixed up tells you what to fix, and a `sub_limit` misread as `coverage` is a
far more dangerous error than `procedural` misread as `condition`.

Usage:
    python evals/run_eval.py                 # classification eval
    python evals/run_eval.py --no-cache      # ignore cached model responses
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))

from app.config import settings  # noqa: E402
from app.llm import cache  # noqa: E402
from app.pipeline.analyze import analyze  # noqa: E402
from app.pipeline.ingest import ingest  # noqa: E402
from app.pipeline.score import score  # noqa: E402
from app.pipeline.segment import segment  # noqa: E402
from app.llm.prompts import PROMPT_VERSION  # noqa: E402
from app.taxonomy import ClauseType  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
GOLDEN_PDF = GOLDEN_DIR / "synthetic-health-policy.pdf"
GOLDEN_LABELS = GOLDEN_DIR / "synthetic-health-policy.labels.json"
REPORT_PATH = REPO_ROOT / "evals" / "classification-report.md"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 - with the degenerate cases handled explicitly.

    A type the model never predicted has precision 0/0. Reporting that as 0 is
    correct here: predicting a type zero times is a total failure to recognise
    it, not an undefined situation.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_ranking(segments, scored) -> dict:
    """Measure the ranking, not just the labelling.

    Classification F1 and ranking quality are genuinely different things, and
    only the second one is the product. A system can label all 39 clauses
    perfectly and still bury the room-rent cap at number 14, beneath a clause
    that merely lets the insurer request a medical examination at its own
    expense. F1 would report that as flawless.

    Expectations live in `golden/ranking-expectations.json`, hand-authored from
    how Indian health insurance claims actually go wrong, and deliberately
    written before inspecting any ranking - otherwise the metric rationalises
    the output instead of testing it.
    """
    path = GOLDEN_DIR / "ranking-expectations.json"
    if not path.exists():
        return {}

    spec = json.loads(path.read_text(encoding="utf-8"))
    top_n = spec["top_n"]

    by_idx = {seg.order_idx: seg for seg in segments}
    ranked = sorted(scored.values(), key=lambda s: s.impact_score, reverse=True)
    positions = {
        by_idx[s.order_idx].number: rank
        for rank, s in enumerate(ranked, 1)
        if by_idx.get(s.order_idx) and by_idx[s.order_idx].number
    }

    hits, misses = [], []
    for item in spec["must_rank_top"]:
        rank = positions.get(item["number"])
        entry = {**item, "rank": rank}
        (hits if rank and rank <= top_n else misses).append(entry)

    correct_exclusions, violations = [], []
    for item in spec["must_not_rank_top"]:
        rank = positions.get(item["number"])
        entry = {**item, "rank": rank}
        (violations if rank and rank <= top_n else correct_exclusions).append(entry)

    total = len(spec["must_rank_top"]) + len(spec["must_not_rank_top"])
    passed = len(hits) + len(correct_exclusions)
    return {
        "top_n": top_n,
        "score": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "hits": hits,
        "misses": misses,
        "violations": violations,
    }


async def run(use_cache: bool, batch_size: int | None = None) -> dict:
    if not GOLDEN_PDF.exists():
        print("golden PDF missing; generating...")
        import subprocess

        subprocess.run(
            [sys.executable, str(GOLDEN_DIR / "build_synthetic_policy.py")],
            check=True,
        )

    labels = json.loads(GOLDEN_LABELS.read_text(encoding="utf-8"))["clauses"]
    expected_by_number = {label["number"]: label["expected_type"] for label in labels}

    if not use_cache:
        cache.clear()

    print(f"model: {settings.model}")
    print(f"ingesting {GOLDEN_PDF.name} ...")
    result = ingest(GOLDEN_PDF)
    segments = segment(result)
    print(f"  {result.page_count} pages, {len(segments)} segments")

    effective_batch = batch_size or settings.analyze_batch_size
    print(f"analysing (batch={effective_batch}, "
          f"concurrency={settings.analyze_concurrency}) ...")
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} clauses", end="", flush=True)

    analyses = await analyze(segments, batch_size=batch_size, progress=progress)
    elapsed = time.perf_counter() - started
    print(f"\r  {len(analyses)}/{len(segments)} clauses analysed in {elapsed:.1f}s")

    scored = score(segments, analyses)

    # --- classification metrics, over labelled clauses only ---
    by_number = {seg.number: seg for seg in segments if seg.number}
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    unanalysed: list[str] = []

    for number, expected in expected_by_number.items():
        seg = by_number.get(number)
        analysis = analyses.get(str(seg.order_idx)) if seg else None
        if analysis is None:
            unanalysed.append(number)
            fn[expected] += 1
            continue

        predicted = analysis.clause_type
        if predicted == expected:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1
            confusions[(expected, predicted)] += 1

    per_type = {}
    for clause_type in ClauseType.values():
        support = sum(1 for e in expected_by_number.values() if e == clause_type)
        precision, recall, f1 = prf(tp[clause_type], fp[clause_type], fn[clause_type])
        per_type[clause_type] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    present = [t for t in ClauseType.values()
               if per_type[t]["support"] or fp[t]]
    macro_f1 = sum(per_type[t]["f1"] for t in present) / max(len(present), 1)
    accuracy = sum(tp.values()) / max(len(expected_by_number), 1)

    ranking = evaluate_ranking(segments, scored)

    return {
        "model": settings.model,
        # Recorded, not referenced. This file previously said "see
        # PROMPT_VERSION in prompts.py", which is a pointer rather than a
        # record: the report was generated under v4 while the code had moved to
        # v6, and nothing in it could reveal that. A report that cannot state
        # the conditions of its own run is not evidence of anything.
        "prompt_version": PROMPT_VERSION,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "batch_size": settings.analyze_batch_size,
        "clauses_expected": len(expected_by_number),
        "clauses_analysed": len(expected_by_number) - len(unanalysed),
        "unanalysed": unanalysed,
        "seconds": round(elapsed, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_type": per_type,
        "confusions": confusions,
        "ranking": ranking,
        "segments": segments,
        "analyses": analyses,
        "scored": scored,
    }


def report(res: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- **Model**: `{res['model']}`",
        f"- **Prompt version**: `{res['prompt_version']}`",
        f"- **Decoding**: num_ctx {res['num_ctx']}, num_predict "
        f"{res['num_predict']}, batch size {res['batch_size']}",
        f"- **Clauses**: {res['clauses_analysed']}/{res['clauses_expected']} analysed",
        f"- **Wall time**: {res['seconds']}s",
        "",
        "## Classification",
        "",
        f"**Macro-F1: {res['macro_f1']:.3f}** &nbsp;&nbsp; Accuracy: {res['accuracy']:.3f}",
        "",
        "Macro-F1 is the headline: it weights every clause type equally, so the",
        "4 waiting periods count as much as the 8 exclusions. Accuracy would let",
        "a model coast on the common types.",
        "",
        "| Clause type | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for clause_type, m in res["per_type"].items():
        if not m["support"]:
            continue
        lines.append(
            f"| `{clause_type}` | {m['support']} | {m['precision']:.2f} "
            f"| {m['recall']:.2f} | {m['f1']:.2f} |"
        )

    if res["confusions"]:
        lines += ["", "## Confusions", "",
                  "| Expected | Predicted | n |", "|---|---|---:|"]
        for (expected, predicted), n in res["confusions"].most_common():
            lines.append(f"| `{expected}` | `{predicted}` | {n} |")

    if res["unanalysed"]:
        lines += ["", f"**Unanalysed clauses:** {', '.join(res['unanalysed'])}"]

    rank = res.get("ranking") or {}
    if rank:
        lines += [
            "",
            "## Ranking quality",
            "",
            f"**{rank['passed']}/{rank['total']} expectations met** "
            f"(top-{rank['top_n']})",
            "",
            "Classification F1 measures labelling. This measures the ranking, which",
            "is the actual product: a system can label every clause perfectly and",
            "still bury the room-rent cap beneath a harmless administrative power.",
            "Expectations are hand-authored in `golden/ranking-expectations.json`",
            "from how claims actually go wrong, written before any ranking was seen.",
            "",
        ]
        if rank["misses"]:
            lines += [
                f"### Should be in the top {rank['top_n']}, but is not",
                "",
                "| Clause | Actual rank | Why it matters |",
                "|---|---:|---|",
            ]
            for m in rank["misses"]:
                lines.append(f"| `{m['number']}` | {m['rank'] or '-'} | {m['reason']} |")
            lines.append("")
        if rank["violations"]:
            lines += [
                f"### In the top {rank['top_n']}, but should not be",
                "",
                "| Clause | Actual rank | Why it does not belong |",
                "|---|---:|---|",
            ]
            for v in rank["violations"]:
                lines.append(f"| `{v['number']}` | {v['rank']} | {v['reason']} |")
            lines.append("")

    # --- the actual product: the ranked risk list ---
    ranked = sorted(
        res["scored"].values(), key=lambda s: s.impact_score, reverse=True
    )[:10]
    seg_by_idx = {s.order_idx: s for s in res["segments"]}

    lines += [
        "",
        "## Top 10 by impact score",
        "",
        "This is the app's actual output: the clauses most likely to cost a",
        "policyholder money, ranked. `buriedness` is computed, not model-judged.",
        "",
        "| # | Clause | Type | Impact | Buried | Grade |",
        "|---|---|---|---:|---:|---:|",
    ]
    for rank, sc in enumerate(ranked, 1):
        seg = seg_by_idx[sc.order_idx]
        analysis = res["analyses"][str(sc.order_idx)]
        label = seg.heading or (seg.number or f"#{seg.order_idx}")
        lines.append(
            f"| {rank} | {label[:44]} | `{analysis.clause_type}` "
            f"| {sc.impact_score:.1f} | {sc.buriedness:.2f} | {sc.reading_grade:.0f} |"
        )

    lines += [
        "",
        "---",
        "",
        "Regenerate with `python evals/run_eval.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached model responses")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override clauses per LLM call (default from config)")
    parser.add_argument("--no-report", action="store_true",
                        help="print metrics only; do not overwrite report.md")
    args = parser.parse_args()

    res = asyncio.run(run(use_cache=not args.no_cache, batch_size=args.batch_size))

    print()
    print(f"  macro-F1 : {res['macro_f1']:.3f}")
    print(f"  accuracy : {res['accuracy']:.3f}")
    print()
    for clause_type, m in res["per_type"].items():
        if m["support"]:
            print(f"  {clause_type:16} n={m['support']:<3} "
                  f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
    rank = res.get("ranking") or {}
    if rank:
        print()
        print(f"  ranking  : {rank['passed']}/{rank['total']} expectations met "
              f"(top-{rank['top_n']})")
        for miss in rank["misses"]:
            print(f"    MISSING from top-{rank['top_n']}: "
                  f"{miss['number']} (actual rank {miss['rank']})")
        for bad in rank["violations"]:
            print(f"    SHOULD NOT be top-{rank['top_n']}: "
                  f"{bad['number']} (actual rank {bad['rank']})")

    if res["confusions"]:
        print("\n  confusions (expected -> predicted):")
        for (expected, predicted), n in res["confusions"].most_common():
            print(f"    {expected:16} -> {predicted:16} x{n}")

    if args.no_report:
        print("\n(--no-report: report.md left unchanged)")
        return

    REPORT_PATH.write_text(report(res), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
