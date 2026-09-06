"""Scenario simulator eval.

Separate from run_eval.py because it measures different things. The
classification eval asks "did we label this clause correctly". This asks three
questions that matter more, and that a labelling score cannot see:

  VERDICT ACCURACY     did it reach the answer a competent human reader would?
  CITATION RECALL      did it cite the clause that actually decides the case?
  GROUNDEDNESS         is every quotation genuinely present in the policy?

Groundedness is the one with a hard target of 100%. The other two are quality
measures where a local 7B model is allowed to be imperfect; groundedness is a
correctness property. An answer whose quotations cannot be found in the
document is not a worse answer, it is a fabricated one, and the whole grounding
design exists so that number stays at 1.000.

Usage:
    python evals/run_scenario_eval.py
    python evals/run_scenario_eval.py --no-cache
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))

from app.config import settings  # noqa: E402
from app.llm import cache  # noqa: E402
from app.pipeline.analyze import analyze  # noqa: E402
from app.pipeline.ingest import ingest  # noqa: E402
from app.pipeline.scenario import ShortlistClause, run_scenario  # noqa: E402
from app.pipeline.score import score  # noqa: E402
from app.pipeline.segment import segment  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
GOLDEN_PDF = GOLDEN_DIR / "synthetic-health-policy.pdf"
CASES_PATH = GOLDEN_DIR / "scenarios.json"
REPORT_PATH = REPO_ROOT / "evals" / "scenario-report.md"


async def build_clauses() -> list[ShortlistClause]:
    """Run the map pipeline once; every scenario reuses the result."""
    result = ingest(GOLDEN_PDF)
    segments = segment(result)
    analyses = await analyze(segments)
    scored = score(segments, analyses)

    clauses = []
    for seg in segments:
        analysis = analyses.get(str(seg.order_idx))
        sc = scored.get(str(seg.order_idx))
        if analysis is None or sc is None:
            continue
        clauses.append(
            ShortlistClause(
                clause_id=seg.number or f"c{seg.order_idx}",
                ref=f"golden:{seg.order_idx}",
                number=seg.number,
                clause_type=analysis.clause_type,
                text=seg.text,
                impact_score=sc.impact_score,
            )
        )
    return clauses


async def run(use_cache: bool) -> dict:
    if not use_cache:
        cache.clear()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    print(f"model: {settings.model}")
    print("preparing policy...")
    clauses = await build_clauses()
    print(f"  {len(clauses)} analysed clauses\n")

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case['id']}", flush=True)
        result = await run_scenario(case["scenario"], clauses)

        cited = {c.clause_id for c in result.citations}
        required = set(case["must_cite"])
        rows.append({
            "id": case["id"],
            "expected": case["expected_verdict"],
            "got": result.verdict,
            "verdict_ok": result.verdict == case["expected_verdict"],
            "required": sorted(required),
            "cited": sorted(cited),
            "citation_ok": required.issubset(cited),
            "grounded": result.verified,
            "unverified": [
                {"clause": c.clause_id, "reason": c.unverified_reason}
                for c in result.citations if not c.verified
            ],
            "reasoning": result.reasoning,
            "why": case["why"],
        })

    elapsed = time.perf_counter() - started
    total = len(rows)
    # Citation recall is only meaningful for cases that require a citation;
    # "insufficient_information with nothing to cite" would otherwise inflate it.
    citable = [r for r in rows if r["required"]]

    return {
        "model": settings.model,
        "seconds": round(elapsed, 1),
        "rows": rows,
        "verdict_accuracy": sum(r["verdict_ok"] for r in rows) / max(total, 1),
        "citation_recall": (
            sum(r["citation_ok"] for r in citable) / len(citable) if citable else 1.0
        ),
        "groundedness": sum(r["grounded"] for r in rows) / max(total, 1),
        "citable_count": len(citable),
        "total": total,
    }


def report(res: dict) -> str:
    lines = [
        "# Scenario simulator evaluation",
        "",
        f"- **Model**: `{res['model']}`",
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
        f"| Groundedness | **{res['groundedness']:.3f}** | **must be 1.000** |",
        "",
        "Groundedness is the only hard target. Verdict accuracy and citation",
        "recall are quality measures where a local 7B model is allowed to be",
        "imperfect. Groundedness is a correctness property: an answer whose",
        "quotations cannot be found in the policy is not a weaker answer, it is a",
        "fabricated one.",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    res = asyncio.run(run(use_cache=not args.no_cache))

    print()
    print(f"  verdict accuracy : {res['verdict_accuracy']:.3f}")
    print(f"  citation recall  : {res['citation_recall']:.3f} "
          f"({res['citable_count']} cases)")
    print(f"  groundedness     : {res['groundedness']:.3f}  (must be 1.000)")
    print()
    for row in res["rows"]:
        flag = "  " if row["verdict_ok"] else "<-"
        ground = "" if row["grounded"] else "  UNGROUNDED"
        print(f"  {flag} {row['id']:26} {row['expected']:26} got {row['got']}{ground}")

    if not args.no_report:
        REPORT_PATH.write_text(report(res), encoding="utf-8")
        print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
