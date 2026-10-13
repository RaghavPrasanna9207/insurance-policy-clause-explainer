"""Which eval questions would a code change actually reach? Answered without a model.

WHY THIS EXISTS. A 7B model does not answer the same prompt identically twice,
so re-running every question after a change mixes the change's effect with
noise. The method this project settled on (M18 onward) is to find the cases
whose REQUESTS changed, and re-measure only those: an unchanged request replays
its stored answer and cannot have moved. Finding them was a throwaway script
every milestone. This is that script, kept.

WHAT IT DOES. For every question in every case file, on two checkouts of this
repo - a baseline and the one this file is in - it works out the requests the
pipeline would send, and lists the cases where they differ:

    facts    the fact-extraction request
    pick     the clause-picking request (long policies only)
    reason   the first reasoning request - the one that produces the verdict

The reasoning call is intercepted and never reaches the model, so no answer is
generated. Facts and picks are real calls, because the reasoning request is
built from their answers; they replay from the cache unless their own request
changed, in which case the model is asked and the count is reported.

Run it from the repo root, so both checkouts share one `data/llm_cache.db`:

    git worktree add ../baseline main
    python evals/compare_requests.py ../baseline
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# (case file, policy PDF), relative to a checkout. Real policies are skipped
# when their PDF has not been fetched (evals/real/fetch_policies.py).
SETS = [
    ("evals/golden/scenarios.json", "evals/golden/synthetic-health-policy.pdf"),
    *[(f"evals/golden/scenarios-heldout{n}.json", "evals/golden/synthetic-health-policy.pdf")
      for n in ("", "-2", "-3", "-4")],
    ("evals/real/scenarios-star-arogya-sanjeevani.json", "samples/real/star-arogya-sanjeevani.pdf"),
    ("evals/real/scenarios-star-heldout-abroad.json", "samples/real/star-arogya-sanjeevani.pdf"),
    ("evals/real/scenarios-star-heldout-maternity.json", "samples/real/star-arogya-sanjeevani.pdf"),
    ("evals/real/scenarios-hdfc-optima-secure.json", "samples/real/hdfc-optima-secure.pdf"),
]


def _digest(messages, schema) -> str:
    # Messages and schema only: what the code under comparison decides. The
    # model, options and runtime version are the same for both checkouts.
    payload = json.dumps([messages, schema], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def collect(repo: Path) -> dict:
    """{case file: {case id: {stage: digest}}}, using the code in `repo`."""
    sys.path[:0] = [str(repo / "api"), str(repo / "evals")]
    from app.llm import client
    from app.pipeline import scenario
    from run_scenario_eval import build_clauses

    real, seen = client.complete_json, {}

    async def spy(messages, schema, **kwargs):
        if schema is scenario.FACTS_SCHEMA:
            stage = "facts"
        elif "clause_ids" in schema.get("properties", {}):
            stage = "pick"
        else:
            # First request only: a retry depends on the answer, and there is
            # none. This verdict needs no citations, so no retry is asked for.
            seen.setdefault("reason", _digest(messages, schema))
            return {"verdict": "insufficient_information", "reasoning": "", "deciding_clauses": []}
        seen.setdefault(stage, _digest(messages, schema))
        return await real(messages, schema, **kwargs)

    out, clauses_by_pdf = {}, {}
    for cases_file, pdf in SETS:
        if not (repo / cases_file).exists() or not Path(pdf).exists():
            continue
        if pdf not in clauses_by_pdf:
            clauses_by_pdf[pdf] = await build_clauses(Path(pdf))  # analysis: real, cached
        client.complete_json = spy
        try:
            out[cases_file] = {}
            for case in json.loads((repo / cases_file).read_text(encoding="utf-8"))["cases"]:
                seen = {}
                await scenario.run_scenario(case["scenario"], clauses_by_pdf[pdf])
                out[cases_file][case["id"]] = seen
        finally:
            client.complete_json = real
    out["_model_calls"] = client.model_calls
    return out


def report(base: dict, new: dict) -> int:
    changed_total = total = 0
    for cases_file in new:
        if cases_file.startswith("_"):
            continue
        ours, theirs = new[cases_file], base.get(cases_file, {})
        changed = {
            case_id: [s for s in ("facts", "pick", "reason") if stages.get(s) != theirs[case_id].get(s)]
            for case_id, stages in ours.items() if case_id in theirs
        }
        changed = {k: v for k, v in changed.items() if v}
        added = [c for c in ours if c not in theirs]
        total += len(ours)
        changed_total += len(changed)
        print(f"{cases_file}: {len(changed)} of {len(ours)} cases changed"
              + (f", {len(added)} new" if added else ""))
        for case_id, stages in changed.items():
            print(f"    {case_id}: {', '.join(stages)}")
    print(f"\n{changed_total} of {total} cases send a different request.")
    print(f"model calls made to work this out: baseline {base['_model_calls']}, "
          f"this checkout {new['_model_calls']}")
    return changed_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("baseline", type=Path, help="a checkout of the code to compare against")
    parser.add_argument("--collect", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.collect:
        # Child process: two versions of `app` cannot be imported into one.
        args.collect.write_text(json.dumps(asyncio.run(collect(args.baseline.resolve()))))
        raise SystemExit(0)

    with tempfile.TemporaryDirectory() as tmp:
        results = []
        for repo in (args.baseline, Path(__file__).resolve().parents[1]):
            target = Path(tmp) / f"{len(results)}.json"
            subprocess.run([sys.executable, __file__, str(repo), "--collect", str(target)],
                           check=True, stdout=subprocess.DEVNULL)
            results.append(json.loads(target.read_text()))
    report(*results)
