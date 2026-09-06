"""Stage 3: classify and explain each clause. The only LLM stage in the map pipeline.

WHAT THIS STAGE DECIDES
-----------------------
For every clause: what role it plays (the taxonomy in app/taxonomy.py), what it
means in plain English, and two judgments - how LIKELY it is to affect a typical
policyholder, and how SEVERE the consequence is when it does.

Those two numbers are the ONLY model-supplied input to the impact score. Stage 4
combines them with signals it computes itself. The split is deliberate: judging
"how bad is a 20% co-payment for a senior citizen" needs language understanding,
while "how far into the document is this clause" needs arithmetic. Asking a 7B
model to do the second is how you get confidently wrong numbers.

BATCHING - AND WHY THE DEFAULT IS 1
-----------------------------------
This module was originally written to send clauses in batches of 5, reasoning
that batching amortises the long system prompt across several clauses and so
must be faster. That reasoning sounded obvious and was wrong. Measured on the
39-clause golden policy:

    batch size 5    macro-F1 0.973    121.9s
    batch size 1    macro-F1 1.000    128.5s

Batching bought ~5% wall time and cost a clause. The amortisation argument
failed because Ollama already caches the repeated system-prompt prefix between
calls, so re-sending it is nearly free - the saving being optimised for did not
exist.

The accuracy loss is the more interesting half. The clause that batching got
wrong (5.2 "Proportionate Deduction", a sub_limit read as a condition)
classifies CORRECTLY when sent on its own. Nothing about the clause was hard;
it was the company it kept. Sharing a generation with neighbouring clauses lets
the model's reading of one bleed into the next, and clauses in a policy are
adjacent precisely because they are related - which makes the interference worse,
not better.

So the default is 1: one clause per call, `analyze_concurrency` of them in
flight. Batching remains supported via `batch_size` because it is the right
trade for a much larger document, where the wall-time picture changes.

The general lesson, and the reason `evals/` exists: a performance argument that
has not been measured is a guess, however obvious it sounds.

WHY THE `id` FIELD IS ENUM-CONSTRAINED
--------------------------------------
Each batch's schema restricts `id` to exactly the clause ids in that batch.
Ollama enforces JSON-Schema enums during sampling, so the model physically
cannot attribute an analysis to a clause that is not in front of it. Without
this, a model that loses track mid-batch will happily emit a plausible-looking
id, and an explanation would be silently attached to the wrong clause.

This is the same mechanism the scenario simulator uses to make fabricated
citations impossible, applied here to keep analyses correctly attributed.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.llm import client
from app.llm.prompts import CLASSIFY_SYSTEM, PROMPT_VERSION, render_clause_batch
from app.pipeline.segment import Segment
from app.taxonomy import ClauseType

log = logging.getLogger(__name__)


@dataclass
class ClauseAnalysis:
    """The model's reading of one clause."""

    clause_key: str  # matches Segment.order_idx, as a string
    clause_type: str
    plain_language: str
    what_it_means: str
    triggers: list[str] = field(default_factory=list)
    monetary_limits: list[str] = field(default_factory=list)
    time_windows: list[str] = field(default_factory=list)
    # A waiting period exactly as the clause states it: "thirty days" is
    # (30, "days"), "thirty six months" is (36, "months"). Normalised to days
    # by `waiting_period_days`, never by the model.
    waiting_period_value: int | None = None
    waiting_period_unit: str | None = None
    # Conditions under which the clause does NOT apply.
    exceptions: list[str] = field(default_factory=list)
    likelihood: int = 3
    severity: int = 3

    @property
    def waiting_period_days(self) -> int | None:
        """The waiting period in days, or None if this clause imposes none.

        Conversion lives here rather than in the prompt because it is
        arithmetic. 30-day months and 365-day years are close enough: the
        comparison decides whether a bar has lifted, and no real policy turns
        on a day either side.
        """
        if not self.waiting_period_value or self.waiting_period_value <= 0:
            return None
        per = {"days": 1, "months": 30, "years": 365}.get(self.waiting_period_unit or "")
        return self.waiting_period_value * per if per else None


def _batch_schema(ids: list[str]) -> dict[str, Any]:
    """Build the JSON Schema for one batch.

    Every categorical field carries an `enum`. That is not stylistic: an
    unconstrained string field invites the model to invent a value, and an
    invented clause type flows straight into the scoring formula as an unknown
    key. Constrained decoding makes the invalid value unrepresentable instead.
    """
    return {
        "type": "object",
        "properties": {
            "clauses": {
                "type": "array",
                # Cardinality is part of the grammar too, and it has to be.
                # Without these, the model handed a title page returned
                # {"clauses": []} - completely valid under a schema that only
                # constrains the SHAPE of each element, and a silent data loss:
                # the clause simply vanished from the pipeline with no error.
                # Pinning min == max == len(ids) makes "return nothing" and
                # "return four of the five" unrepresentable.
                #
                # Verified enforced: given a prompt insisting there were no
                # clauses at all, the model still emitted exactly 3 entries.
                #
                # Forcing an entry for genuinely non-clause text (a title page)
                # is the right trade. It lands as `procedural` with low ratings
                # and scores near zero, so it is harmless - whereas a silently
                # dropped clause is invisible for the rest of the pipeline's
                # life, which is the exact failure this project exists to stop.
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        # Restricted to this batch: the model cannot attribute
                        # an analysis to a clause it was not shown.
                        "id": {"type": "string", "enum": ids},
                        "clause_type": {"type": "string", "enum": ClauseType.values()},
                        "plain_language": {"type": "string"},
                        "what_it_means": {"type": "string"},
                        "triggers": {"type": "array", "items": {"type": "string"}},
                        "monetary_limits": {"type": "array", "items": {"type": "string"}},
                        "time_windows": {"type": "array", "items": {"type": "string"}},
                        # STRUCTURED, not prose. A waiting period's length as a
                        # number of months, so stage 5 can COMPARE it instead of
                        # asking the model whether 5 years exceeds 36 months.
                        #
                        # This split is the project's governing principle applied
                        # where it had been forgotten: reading "thirty six months"
                        # out of legal prose is language understanding and belongs
                        # to the model; deciding 60 >= 36 is arithmetic and does
                        # not. Three of five scenario failures were the model
                        # getting that arithmetic wrong.
                        # VALUE AND UNIT, not a bare number of months.
                        #
                        # A first version asked only for "waiting_period_months",
                        # and clause 3.1 - "the first thirty days" - came back as
                        # 30. The comparison then read that as thirty MONTHS. It
                        # happened to give the right answer on the test case, for
                        # entirely the wrong reason, which is the worst kind of
                        # passing test.
                        #
                        # Policies mix days, months and years freely. Asking the
                        # model to normalise them is asking it to do arithmetic
                        # again; asking it to report what the clause SAYS is
                        # reading, which is its job. Python converts.
                        "waiting_period_value": {"type": ["integer", "null"]},
                        "waiting_period_unit": {
                            "type": ["string", "null"],
                            "enum": ["days", "months", "years", None],
                        },
                        # Carve-outs: "unless necessitated by an Accident".
                        # Extracted as their own field because an exclusion with
                        # an exception was read as an unconditional bar, and a
                        # clause's escape hatch is exactly what a policyholder
                        # needs to see.
                        "exceptions": {"type": "array", "items": {"type": "string"}},
                        # Integer enums, not bare integers: a 1-5 scale that can
                        # return 7 is not a 1-5 scale, and stage 4 normalises
                        # these assuming the stated range.
                        "likelihood": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "severity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    },
                    "required": [
                        "id", "clause_type", "plain_language", "what_it_means",
                        "triggers", "monetary_limits", "time_windows",
                        "waiting_period_value", "waiting_period_unit", "exceptions",
                        "likelihood", "severity",
                    ],
                },
            }
        },
        "required": ["clauses"],
    }


def _parse(payload: dict[str, Any]) -> dict[str, ClauseAnalysis]:
    out: dict[str, ClauseAnalysis] = {}
    for item in payload.get("clauses", []):
        out[item["id"]] = ClauseAnalysis(
            clause_key=item["id"],
            clause_type=item["clause_type"],
            plain_language=item["plain_language"].strip(),
            what_it_means=item["what_it_means"].strip(),
            triggers=[t.strip() for t in item.get("triggers", []) if t.strip()],
            monetary_limits=[t.strip() for t in item.get("monetary_limits", []) if t.strip()],
            time_windows=[t.strip() for t in item.get("time_windows", []) if t.strip()],
            waiting_period_value=item.get("waiting_period_value"),
            waiting_period_unit=item.get("waiting_period_unit"),
            exceptions=[e.strip() for e in item.get("exceptions", []) if e.strip()],
            likelihood=item["likelihood"],
            severity=item["severity"],
        )
    return out


async def _analyze_batch(batch: list[Segment]) -> dict[str, ClauseAnalysis]:
    ids = [str(seg.order_idx) for seg in batch]
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": render_clause_batch(batch)},
    ]
    payload = await client.complete_json(
        messages, _batch_schema(ids), prompt_version=PROMPT_VERSION
    )
    return _parse(payload)


async def analyze(
    segments: list[Segment],
    *,
    batch_size: int | None = None,
    concurrency: int | None = None,
    progress=None,
) -> dict[str, ClauseAnalysis]:
    """Analyse every clause. Returns a mapping of str(order_idx) -> ClauseAnalysis."""
    batch_size = batch_size or settings.analyze_batch_size
    concurrency = concurrency or settings.analyze_concurrency

    batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, ClauseAnalysis] = {}
    done = 0

    async def run(batch: list[Segment]) -> dict[str, ClauseAnalysis]:
        nonlocal done
        # Ollama can serve requests concurrently, but they contend for the same
        # GPU. The semaphore keeps a few in flight without thrashing VRAM.
        async with semaphore:
            try:
                out = await _analyze_batch(batch)
            except Exception as exc:
                # One failed batch must not abandon the other 34 clauses. The
                # gaps are filled by _retry_missing below.
                log.warning("batch starting at %s failed: %s", batch[0].order_idx, exc)
                out = {}
        done += len(batch)
        if progress:
            progress(done, len(segments))
        return out

    for out in await asyncio.gather(*(run(b) for b in batches)):
        results.update(out)

    await _retry_missing(segments, results)
    return results


async def _retry_missing(
    segments: list[Segment], results: dict[str, ClauseAnalysis]
) -> None:
    """Second pass, one clause at a time, for anything the batches missed.

    Clauses go missing when a batch call fails outright, or when the model
    simply returns four entries for five inputs - which constrained decoding
    does not prevent, because an array of four objects is perfectly valid under
    the schema. The schema controls the SHAPE of each element, never how many
    elements the model chooses to emit.

    Retrying alone removes the interference from whichever neighbouring clause
    confused it, and usually succeeds.
    """
    missing = [s for s in segments if str(s.order_idx) not in results]
    if not missing:
        return

    log.info("re-analysing %d clause(s) individually", len(missing))
    for seg in missing:
        try:
            results.update(await _analyze_batch([seg]))
        except Exception as exc:
            log.error("clause %s could not be analysed: %s", seg.order_idx, exc)

    still_missing = [s.order_idx for s in segments if str(s.order_idx) not in results]
    if still_missing:
        # Loud, because a silently dropped clause is exactly the failure mode
        # this whole project exists to prevent: the user sees a policy that
        # looks safer than it is.
        log.error("UNANALYSED CLAUSES: %s", still_missing)
