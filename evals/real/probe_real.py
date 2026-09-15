"""Run the pipeline on real policy wordings and report what it does to them.

WHY A PROBE, AND NOT AN EVAL
----------------------------
Every other number in this project was measured on one synthetic policy: 39
clauses, 5 pages, about 3,400 tokens. Real wordings are 24-53 pages. There are
no labels for them yet, so this cannot score anything. What it can do is
measure the assumptions the design rests on, before any of them is relied on:

  - does the segmenter find clause boundaries, or one giant clause per page?
  - is a clause's number still a unique identifier, when citations are
    enum-locked to clause numbers?
  - does every clause fit in the scenario simulator's context budget - the
    premise of having no retrieval at all?

Nothing here is tuned. It reports.

Policy text is printed to the terminal for reading, and never written to a
file: the wordings belong to their insurers.

Usage:
    python evals/real/fetch_policies.py           # first
    python evals/real/probe_real.py --structure-only   # seconds, no model
    python evals/real/probe_real.py                    # adds analysis; long when uncached
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings  # noqa: E402
from app.pipeline.ingest import ingest  # noqa: E402
from app.pipeline.scenario import CHARS_PER_TOKEN, shortlist  # noqa: E402
from app.pipeline.segment import segment  # noqa: E402
from fetch_policies import SOURCES, pdf_path  # noqa: E402
from run_scenario_eval import GOLDEN_PDF, build_clauses  # noqa: E402


def tokens(chars: int) -> int:
    # The same estimate the shortlist budgets with, so "fits" here means what
    # it means there.
    return int(chars / CHARS_PER_TOKEN)


def structure(pdf: Path) -> None:
    result = ingest(pdf)
    segments = segment(result)
    lengths = [len(s.text) for s in segments]
    numbers = Counter(s.number for s in segments if s.number)
    shared = {n: c for n, c in numbers.items() if c > 1}
    split = sum(1 for s in segments if s.heading.endswith("(cont.)"))
    offsets_ok = all(result.raw_text[s.char_start:s.char_end] == s.text for s in segments)

    print(f"  pages {result.page_count}, {len(result.raw_text):,} chars "
          f"(~{tokens(len(result.raw_text)):,} tokens), body font {result.body_size():.1f}pt")
    print(f"  segments {len(segments)}; length median {statistics.median(lengths):.0f}, "
          f"p90 {sorted(lengths)[int(len(lengths) * 0.9)]}, max {max(lengths)} chars")
    print(f"  force-split at the character cap: {split}")
    print(f"  unnumbered: {sum(1 for s in segments if not s.number)}")
    print(f"  distinct sections: {len({s.section_path for s in segments})}")
    print(f"  numbers used by more than one segment: {len(shared)} "
          f"(covering {sum(shared.values())} segments)")
    if shared:
        worst = sorted(shared.items(), key=lambda kv: -kv[1])[:8]
        print("    most shared:", ", ".join(f"{n} x{c}" for n, c in worst))
    print(f"  offset invariant holds: {offsets_ok}")


async def analysis(pdf: Path) -> None:
    started = time.perf_counter()
    clauses = await build_clauses(pdf)
    elapsed = time.perf_counter() - started

    types = Counter(c.clause_type for c in clauses)
    ids = Counter(c.clause_id for c in clauses)
    kept = shortlist(clauses)
    kept_types = Counter(c.clause_type for c in kept)
    total_tokens = sum(tokens(len(c.text)) + 20 for c in clauses)

    print(f"  analysed {len(clauses)} clauses in {elapsed:.0f}s")
    print("  types:", ", ".join(f"{t} {n}" for t, n in types.most_common()))
    print(f"  citation ids shared by more than one clause: "
          f"{sum(1 for n in ids.values() if n > 1)} ids, "
          f"{sum(n for n in ids.values() if n > 1)} clauses")
    print(f"  scenario fit: all clauses ~{total_tokens:,} tokens against a budget of "
          f"{settings.scenario_token_budget:,}")
    print(f"  shortlist keeps {len(kept)} of {len(clauses)}; dropped by type:")
    for t, n in types.most_common():
        print(f"    {t:15} kept {kept_types[t]:3} of {n:3}")
    print("  top 15 by impact:")
    for c in sorted(clauses, key=lambda c: -c.impact_score)[:15]:
        snippet = " ".join(c.text.split())[:70]
        print(f"    {c.impact_score:5.1f}  {c.clause_id:10} {c.clause_type:15} {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--only", help="one policy id from sources.json")
    args = parser.parse_args()

    policies = json.loads(SOURCES.read_text(encoding="utf-8"))["policies"]
    if args.only:
        policies = [p for p in policies if p["id"] == args.only]
    targets = [("synthetic golden policy (baseline)", GOLDEN_PDF)]
    targets += [(f"{p['insurer']} - {p['product']}", pdf_path(p["id"])) for p in policies]

    missing = [str(path) for _, path in targets if not path.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing) + " - run fetch_policies.py first")

    for name, pdf in targets:
        print(f"\n=== {name}")
        structure(pdf)
        if not args.structure_only:
            asyncio.run(analysis(pdf))


if __name__ == "__main__":
    main()
