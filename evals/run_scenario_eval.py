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
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))

from app.config import settings  # noqa: E402
from app.grounding import verify_quote  # noqa: E402
from app.llm.prompts import PROMPT_VERSION  # noqa: E402
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

    source_by_id = {c.clause_id: c.text for c in clauses}

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case['id']}", flush=True)
        result = await run_scenario(case["scenario"], clauses)

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
            "truly_bad": truly_bad,
            "flagged_bad": flagged_bad,
        })

    elapsed = time.perf_counter() - started
    total = len(rows)
    # Citation recall is only meaningful for cases that require a citation;
    # "insufficient_information with nothing to cite" would otherwise inflate it.
    citable = [r for r in rows if r["required"]]

    return {
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "seconds": round(elapsed, 1),
        "rows": rows,
        "verdict_accuracy": sum(r["verdict_ok"] for r in rows) / max(total, 1),
        "citation_recall": (
            sum(r["citation_ok"] for r in citable) / len(citable) if citable else 1.0
        ),
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
    print(f"  fabrication rate : {res['fabrication_rate']:.3f} "
          f"({res['fabricated_quotes']} invented quote(s))")
    print(f"  detection integ. : {res['detection_integrity']:.3f}  (must be 1.000)")
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
