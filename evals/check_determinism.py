"""Is this system reproducible? Measured, not assumed.

WHY THIS EXISTS. For eight milestones this project took `temperature = 0.0` as
proof that a given prompt produced a given answer. It is not. Two full runs of
the 40-case scenario eval, over identical code and byte-identical prompts,
disagreed on three cases - two correct answers became wrong and one wrong
answer became correct. A prompt change worth two cases cannot be measured
through noise worth three, so every number this project has ever compared was
resting on an assumption nobody had checked.

WHAT IT CHECKS. Two layers, because they fail for different reasons and a
single pass/fail would let one hide the other:

  MODEL     the same messages, sent N times with the cache bypassed, must come
            back byte-identical. This is llama.cpp's decoding, with no pipeline
            involvement at all. It fails when a near-tie between two tokens is
            broken by a random seed.

  PIPELINE  the analysis stage run N times must produce identical structured
            output. This is the same model plus THIS PROJECT'S concurrency:
            `analyze_concurrency` requests are in flight at once, Ollama batches
            whatever arrives together, and batch composition changes the order
            of the floating-point reductions inside the matmuls. Float addition
            is not associative, so a different order is a different logit in the
            low bits - and near-ties flip again.

The model layer can pass while the pipeline layer fails. That is the outcome
that tells you concurrency, not sampling, is the culprit.

Usage:
    python evals/check_determinism.py            # 3 repeats
    python evals/check_determinism.py --repeats 5
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from app.config import settings  # noqa: E402
from app.llm import client  # noqa: E402
from app.llm.prompts import REASON_SYSTEM, render_reasoning_request  # noqa: E402
from app.pipeline.analyze import analyze  # noqa: E402
from app.pipeline.ingest import ingest  # noqa: E402
from app.pipeline.scenario import _reasoning_schema, shortlist  # noqa: E402
from app.pipeline.segment import segment  # noqa: E402

GOLDEN_PDF = REPO_ROOT / "evals" / "golden" / "synthetic-health-policy.pdf"

# One scenario, chosen because it is decided by a plain exclusion with no
# arithmetic anywhere near it. If even this is not reproducible, nothing is.
SCENARIO = (
    "I broke my leg skiing on holiday and needed surgery. "
    "I've had the policy for three years."
)


def _canonical(payload: dict) -> str:
    """Sort keys so an ordering difference is not reported as a content one."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


async def check_model(clauses, repeats: int) -> bool:
    """Same messages, N times, cache bypassed."""
    considered = shortlist(clauses)
    ids = [c.clause_id for c in considered]
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {
            "role": "user",
            # No waiting or reduction block: this layer is testing the decoder,
            # so the prompt is held as simple as the real path allows.
            "content": render_reasoning_request(SCENARIO, {}, considered, "", ""),
        },
    ]
    schema = _reasoning_schema(ids)

    outs = []
    for i in range(repeats):
        payload = await client.complete_json(
            messages, schema, use_cache=False
        )
        outs.append(_canonical(payload))
        print(f"  model  run {i + 1}/{repeats}: {len(outs[-1])} chars", flush=True)

    return _report("MODEL", outs)


async def check_pipeline(segments, repeats: int, concurrency: int) -> bool:
    """The analysis stage, N times, at a given concurrency.

    `concurrency` is a parameter rather than a setting because it is the
    VARIABLE UNDER TEST. Comparing a run at 2 against a run at 1 is the whole
    experiment: if 1 is reproducible and 2 is not, the nondeterminism is Ollama
    batching our concurrent requests together, not the decoder.
    """
    # The first 12 clauses rather than all 39: enough to put several requests
    # in flight at once, which is the condition being tested, without paying
    # for a full analysis pass three times over.
    subset = segments[:12]
    outs = []
    for i in range(repeats):
        analyses = await analyze(subset, use_cache=False, concurrency=concurrency)
        outs.append(
            _canonical({k: asdict(v) for k, v in analyses.items()})
        )
        print(f"  pipe   run {i + 1}/{repeats}: {len(outs[-1])} chars", flush=True)

    return _report(f"PIPELINE (concurrency={concurrency})", outs)


async def check_sequence(clauses, cases, count: int) -> bool:
    """THE LAYER THAT MATTERS MOST, AND THE ONE THIS FILE ORIGINALLY LACKED.

    The two layers above send the SAME input repeatedly. An eval never does
    that - it sends forty DIFFERENT prompts in a fixed order, once each. Those
    are different properties, and only the second one is what a measurement
    rests on:

        repeat stability    asking the same question twice gives one answer
        sequence stability  asking forty questions gives the same forty
                            answers as it did last time

    A system can have the first and lack the second. llama.cpp reuses the
    cached KV of a matching prompt prefix, so what a request costs - and
    therefore how its prefill is batched, and therefore its logits in the low
    bits - depends on what was asked immediately before it. Repeating one
    prompt holds that predecessor fixed and hides the whole effect.

    This runs the first `count` real eval scenarios in order, twice, with the
    cache bypassed, and compares pass 1 against pass 2 case by case. That is
    exactly the operation `run_scenario_eval.py` performs, and exactly the
    property its numbers depend on.
    """
    considered = shortlist(clauses)
    schema = _reasoning_schema([c.clause_id for c in considered])
    chosen = cases[:count]

    async def one_pass(label: str) -> list[str]:
        out = []
        for i, case in enumerate(chosen, 1):
            messages = [
                {"role": "system", "content": REASON_SYSTEM},
                {
                    "role": "user",
                    "content": render_reasoning_request(
                        case["scenario"], {}, considered, "", ""
                    ),
                },
            ]
            payload = await client.complete_json(
                messages, schema, use_cache=False
            )
            out.append(_canonical(payload))
            print(f"  seq {label} {i}/{len(chosen)}: {case['id']}", flush=True)
        return out

    first = await one_pass("A")
    second = await one_pass("B")

    # COMPARED AT THE LEVEL THE METRICS READ, NOT BYTE FOR BYTE.
    #
    # An earlier version of this comparison used the raw JSON and reported 7 of
    # 8 cases differing - which is true and almost useless, because most of
    # that difference lives in the free-text `reasoning` field that no metric
    # consumes. Two answers that reach the same verdict citing the same clauses
    # with the same quotes are the same answer as far as every number in
    # `evals/REPORT.md` is concerned, however differently they are worded.
    #
    # Measuring byte equality when the metric reads a projection of the output
    # overstates the noise and hides how much of it would actually move a
    # score.
    def scored_part(raw: str) -> str:
        payload = json.loads(raw)
        cited = sorted(
            (c.get("clause_id"), c.get("quote"), c.get("effect"))
            for c in payload.get("deciding_clauses", [])
        )
        return _canonical({"verdict": payload.get("verdict"), "cited": cited})

    differing = [
        chosen[i]["id"]
        for i in range(len(chosen))
        if scored_part(first[i]) != scored_part(second[i])
    ]
    worded = sum(1 for i in range(len(chosen)) if first[i] != second[i])
    print(
        f"  (raw JSON differed on {worded}/{len(chosen)}; "
        f"the count below is what the metrics would see)"
    )
    if not differing:
        print(f"  SEQUENCE ({len(chosen)} cases, 2 passes): identical  OK\n")
        return True

    print(
        f"  SEQUENCE ({len(chosen)} cases, 2 passes): "
        f"{len(differing)} case(s) differ  FAIL"
    )
    for case_id in differing:
        print(f"    {case_id}")
    print()
    return False


def _report(label: str, outs: list[str]) -> bool:
    distinct = sorted(set(outs))
    if len(distinct) == 1:
        print(f"  {label}: identical across {len(outs)} runs  OK\n")
        return True

    print(f"  {label}: {len(distinct)} DIFFERENT outputs across {len(outs)} runs  FAIL")
    a, b = distinct[0], distinct[1]
    # Print the first divergence rather than two whole documents - the point is
    # WHERE they part company, and a diff of two 2KB JSON blobs buries it.
    for pos, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            print(f"    first divergence at char {pos}:")
            print(f"      run A: ...{a[max(0, pos - 60):pos + 60]}")
            print(f"      run B: ...{b[max(0, pos - 60):pos + 60]}")
            break
    else:
        print(f"    one output is a prefix of the other ({len(a)} vs {len(b)} chars)")
    print()
    return False


async def main(repeats: int, concurrency: int, sequence: int) -> int:
    print(f"model: {settings.model}")
    print(f"seed: {settings.seed}  temperature: {settings.temperature}")
    print(f"analyze_concurrency: {concurrency}\n")

    result = ingest(GOLDEN_PDF)
    segments = segment(result)

    # Cached: this is setup, not the thing under test.
    analyses = await analyze(segments)
    from run_scenario_eval import build_clauses  # noqa: E402
    clauses = await build_clauses()
    print(f"prepared {len(clauses)} clauses, {len(analyses)} analyses\n")

    cases = json.loads(
        (REPO_ROOT / "evals" / "golden" / "scenarios.json").read_text(encoding="utf-8")
    )["cases"]

    model_ok = await check_model(clauses, repeats)
    pipe_ok = await check_pipeline(segments, repeats, concurrency)
    seq_ok = await check_sequence(clauses, cases, sequence)

    if model_ok and pipe_ok and seq_ok:
        print("REPRODUCIBLE: identical output across runs at both layers.")
        return 0
    print("NOT REPRODUCIBLE. Eval deltas smaller than this noise mean nothing.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--concurrency", type=int, default=settings.analyze_concurrency,
        help="analysis concurrency to test (default: the configured value)",
    )
    parser.add_argument(
        "--sequence", type=int, default=8,
        help="how many real eval scenarios to run twice, in order",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(main(args.repeats, args.concurrency, args.sequence))
    )
