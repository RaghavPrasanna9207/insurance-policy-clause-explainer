"""Does the fact extractor read what the person actually said?

WHY THIS EXISTS
---------------
The scenario eval scores verdicts. A wrong FACT reaches it only as a wrong
verdict three steps later, with a plausible-sounding reason attached, and that
is how this went unnoticed through several milestones:

    "I bought this policy at 67 and I am claiming three years later"
        policy_age_value: 67, unit: null      <- the age, in the duration field
    "I was hit by a car ten days after my policy started"
        policy_age_value: null                <- stated, and dropped

Nine of the forty eval questions had the policy's age wrong or missing, and the
reasoning step - told truthfully that it was "NOT STATED" - repeated that back
as its reason. Every downstream check did its job on a false input.

So this checks the extraction step alone, on three kinds of sentence:

    eval      the eval's own questions, whose facts must be right
    heldout   sentences in no eval case and no prompt example, written to test
              whether a prompt change generalises or just memorises
    tuned     sentences that WERE held out until a prompt change was made
              while looking at them; kept, but no longer evidence of generalising

Nothing is cached or written back: every sentence is a fresh model call. It
takes about three minutes.

Usage:
    python evals/check_fact_extraction.py
"""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "api"))

from app.llm import client  # noqa: E402
from app.llm.prompts import FACTS_SYSTEM, PROMPT_VERSION, render_scenario  # noqa: E402
from app.pipeline.scenario import FACTS_SCHEMA  # noqa: E402

CASES = {
    c["id"]: c["scenario"]
    for c in json.loads(
        (REPO_ROOT / "evals" / "golden" / "scenarios.json").read_text(encoding="utf-8")
    )["cases"]
}

HELD = "time_since_policy_start"


def expect(value=None, unit=None, **ages):
    """The fields a sentence must produce. Ages are checked only when given."""
    return {f"{HELD}_value": value, f"{HELD}_unit": unit, **ages}


# (group, label, sentence, expected fields)
SENTENCES = [
    ("eval", "ped-waiting-served", CASES["ped-waiting-served"], expect(5, "years")),
    ("eval", "accident-in-initial-period", CASES["accident-in-initial-period"], expect(10, "days")),
    ("eval", "non-disclosure", CASES["non-disclosure"], expect(2, "years")),
    ("eval", "copay-and-room-breach", CASES["copay-and-room-breach"], expect(5, "years", age_at_policy_start=65)),
    ("eval", "senior-copay", CASES["senior-copay"], expect(3, "years", age_at_policy_start=67)),
    ("eval", "copay-just-under-sixty", CASES["copay-just-under-sixty"], expect(4, "years", age_at_policy_start=58)),
    ("eval", "senior-but-excluded", CASES["senior-but-excluded"], expect(3, "years", age_at_policy_start=70)),
    # 72 now and 70 at the start: the duration is derivable, not stated.
    ("eval", "copay-applies-emergency", CASES["copay-applies-emergency"], expect(age=72, age_at_policy_start=70)),
    ("eval", "dental-no-accident", CASES["dental-no-accident"], expect()),  # "for years" is no number
    ("eval", "ped-waiting-not-served", CASES["ped-waiting-not-served"], expect(8, "months")),
    ("eval", "initial-waiting-period", CASES["initial-waiting-period"], expect(2, "weeks")),
    ("eval", "maternity-too-early", CASES["maternity-too-early"], expect(14, "months")),
    ("eval", "cosmetic-exclusion", CASES["cosmetic-exclusion"], expect()),
    ("eval", "no-timing-given", CASES["no-timing-given"], expect()),
    ("eval", "copay-unknown-inception-age", CASES["copay-unknown-inception-age"], expect(age=68, age_at_policy_start=None)),
    ("eval", "cataract-served", CASES["cataract-served"], expect(4, "years", age=40)),
    ("eval", "not-in-document", CASES["not-in-document"], expect()),
    ("eval", "post-hospitalisation-too-late", CASES["post-hospitalisation-too-late"], expect(5, "years")),
    ("heldout", "eighteen-months-ago", "I took out this policy eighteen months ago and now need a hernia operation.", expect(18, "months")),
    ("heldout", "cover-started-45-days", "My cover started 45 days ago and I have been diagnosed with typhoid.", expect(45, "days")),
    ("heldout", "63-held-eleven", "I'm 63 and I've had the policy for eleven years. I need a hip replacement.", expect(11, "years", age=63, age_at_policy_start=None)),
    ("heldout", "62-bypass-six-later", "I was 62 when I bought the policy, and I needed a bypass six years later.", expect(6, "years", age_at_policy_start=62)),
    ("heldout", "three-weeks-into", "Three weeks into my new policy I was admitted with dengue.", expect(3, "weeks")),
    ("heldout", "70-bought-at-55", "I am 70. I bought this policy at 55 and now I need knee surgery.", expect(age=70, age_at_policy_start=55)),
    ("heldout", "shoulder-no-timing", "I needed surgery on my shoulder.", expect()),
    ("heldout", "four-months-since-began", "It has been four months since my policy began and I need gallstone surgery.", expect(4, "months")),
    ("heldout", "66-got-at-64", "I'm 66, I got this policy when I was 64, and I've been admitted with a fever.", expect(age=66, age_at_policy_start=64)),
    ("heldout", "signed-up-at-60", "I signed up for this insurance at 60. Two years later I had a heart attack.", expect(2, "years", age_at_policy_start=60)),
    ("heldout", "appendicitis-20-days", "Admitted with appendicitis 20 days after my cover began.", expect(20, "days")),
    ("heldout", "bought-in-2019", "We bought the family floater in 2019 and my son was hospitalised.", expect()),
    # Looked at while fixing third-person ages, so no longer held out.
    ("tuned", "mother-75", "My mother, who is 75, was hospitalised with a fracture.", expect(age=75, age_at_policy_start=None)),
    ("heldout", "seven-years-at-59", "Seven years after I first took out this policy, I needed a knee replacement at age 59.", expect(7, "years", age=59, age_at_policy_start=None)),
    ("heldout", "five-months-into-cover", "Five months into cover, I was admitted for asthma.", expect(5, "months")),
    # Third-person ages, written before the prompt change they test. The last
    # one DOES state an age at the start, so a fix cannot pass by never filling
    # that field for someone else.
    ("heldout", "father-82", "My father, 82, was admitted with pneumonia.", expect(age=82, age_at_policy_start=None)),
    ("heldout", "son-9-tonsils", "Our son is 9 and needs his tonsils removed. We have held the policy three years.", expect(3, "years", age=9, age_at_policy_start=None)),
    ("heldout", "wife-58", "My wife is 58; she was diagnosed with breast cancer last month.", expect(age=58, age_at_policy_start=None)),
    ("heldout", "grandmother-90", "My grandmother, aged 90, fractured her wrist.", expect(age=90, age_at_policy_start=None)),
    ("heldout", "mother-took-it-at-61", "My mother took this policy out at 61 and was hospitalised two years later.", expect(2, "years", age_at_policy_start=61)),
]


async def main() -> None:
    print(f"prompt version: {PROMPT_VERSION}\n")
    score = {group: [0, 0] for group, *_ in SENTENCES}
    for group, label, sentence, expected in SENTENCES:
        facts = await client.complete_json(
            [
                {"role": "system", "content": FACTS_SYSTEM},
                {"role": "user", "content": render_scenario(sentence)},
            ],
            FACTS_SCHEMA,
            # A cached answer re-read is a copy of an old sample, not a check.
            use_cache=False,
        )
        wrong = {k: facts.get(k) for k, v in expected.items() if facts.get(k) != v}
        score[group][0] += not wrong
        score[group][1] += 1
        shown = ", ".join(f"{k}={v}" for k, v in wrong.items())
        print(f"  {'ok ' if not wrong else 'BAD'} {group:<8} {label:<30} {shown}", flush=True)

    print()
    for group, (right, total) in score.items():
        print(f"  {group:<8} {right}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
