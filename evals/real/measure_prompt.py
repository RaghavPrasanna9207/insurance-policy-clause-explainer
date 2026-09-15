"""Measure the scenario reasoning prompt in real tokens, not estimated ones.

The context budget is planned in characters: `CHARS_PER_TOKEN = 3.5` turns clause
text into a token estimate, and `clause_token_budget()` subtracts the system
prompt, an allowance for the question and the answer's ceiling from the window.
Ollama reports how many prompt tokens it actually evaluated, so this asks it.

Each policy's prompt is sent TWICE, identically. Ollama can reuse the start of
a prompt it has just processed, and if its reported count then covers only
the part it did not reuse, a single reading would understate the prompt. Two
readings of the same prompt show whether that happens.

The reasoning call is made with the cache off, so nothing here is stored and no
eval answer can change. Run it while nothing else is using the model: a second
request in flight changes the batch and therefore the output (M9).

Usage:
    python evals/real/measure_prompt.py
"""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings  # noqa: E402
from app.llm import client  # noqa: E402
from app.llm.prompts import REASON_SYSTEM  # noqa: E402
from app.pipeline.scenario import (  # noqa: E402
    CHARS_PER_TOKEN,
    clause_token_budget,
    compute,
    extract_facts,
    reason,
    shortlist,
)
from fetch_policies import SOURCES, pdf_path  # noqa: E402
from run_scenario_eval import GOLDEN_PDF, build_clauses  # noqa: E402

# Generic enough to be a fair question of any health policy.
SCENARIO = "I was admitted to hospital for three days with pneumonia. I have held the policy for four years."


async def measure(name: str, pdf: Path) -> None:
    clauses = await build_clauses(pdf)
    considered = shortlist(clauses)
    facts = await extract_facts(SCENARIO)
    computed = compute(facts, considered)

    clause_chars = sum(len(c.text) for c in considered)
    readings = []
    for _ in range(2):
        await reason(SCENARIO, facts, considered, computed, use_cache=False)
        readings.append(client.last_prompt_tokens)

    print(f"\n=== {name}")
    print(f"  clauses in prompt  {len(considered)} of {len(clauses)}")
    print(f"  clause text        {clause_chars:,} chars, estimated {int(clause_chars / CHARS_PER_TOKEN):,} tokens"
          f" (budget {clause_token_budget():,})")
    print(f"  system prompt      {len(REASON_SYSTEM):,} chars")
    print(f"  Ollama reported    {readings[0]} then {readings[1]} prompt tokens (same prompt twice)")
    if readings[0]:
        print(f"  room for answer    {settings.num_ctx - readings[0]:,} of num_predict {settings.num_predict:,}")


async def main() -> None:
    policies = json.loads(SOURCES.read_text(encoding="utf-8"))["policies"]
    targets = [("synthetic golden policy", GOLDEN_PDF)]
    targets += [(p["id"], pdf_path(p["id"])) for p in policies if pdf_path(p["id"]).exists()]
    print(f"num_ctx {settings.num_ctx}, num_predict {settings.num_predict}, "
          f"clause budget {clause_token_budget():,}")
    for name, pdf in targets:
        await measure(name, pdf)


if __name__ == "__main__":
    asyncio.run(main())
