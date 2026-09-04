"""Prompt templates and their version.

PROMPT_VERSION is part of every cache key. Bump it whenever a prompt's wording
changes, and the next run recomputes instead of silently serving results
produced by the old instructions. Forgetting to bump it is the classic way to
spend an afternoon debugging a change that never actually took effect.

--------------------------------------------------------------------------
Why CLASSIFY_SYSTEM is so specific about what each label means
--------------------------------------------------------------------------
Measured with a bare prompt ("You classify Indian health insurance policy
clauses") and only the enum constraining output:

    "Room rent limited to 1% of Sum Insured per day"   -> coverage   WRONG
    "Notice must be given within 24 hours ..."         -> coverage   WRONG
    "No claim payable ... first 36 months"             -> exclusion  WRONG

The enum did its job perfectly - every answer was a legal member of our
taxonomy. It just was not the *right* member.

The lesson, which generalises well beyond this project: **constrained decoding
guarantees the shape of an answer, never the judgment behind it.** The model saw
seven plausible English words and picked by vibe, because nothing told it that
`sub_limit` in this system means "caps a payout" and `condition` means "a duty
whose breach voids the claim".

Both wrong answers also failed in the *dangerous* direction - a clause that caps
your payout and a clause that can void your claim were both filed under
"coverage", the reassuring bucket. That is precisely the harm this project
exists to prevent, reproduced by our own classifier.

So the definitions below carry the real weight, and the tie-breakers exist
because these categories genuinely overlap: a room-rent cap *is* describing
coverage, it just happens to be describing its ceiling.
"""

from app.taxonomy import ClauseType

PROMPT_VERSION = "v2-typed-definitions"

CLASSIFY_SYSTEM = """\
You are an expert on Indian (IRDAI-regulated) health insurance policy wordings.
Your job is to classify each clause by the role it plays in deciding a claim,
and to restate it in plain English for someone with no insurance knowledge.

CLAUSE TYPES - use these exact meanings, not the everyday sense of the words:

- coverage: states something the policy DOES pay for. Use only when the clause
  grants a benefit and places no cap and no time bar on it.

- exclusion: states something the policy will NEVER pay for, permanently. If the
  bar lifts after some period of time, it is a waiting_period, NOT an exclusion.

- waiting_period: coverage that begins only after a stated period has elapsed
  (e.g. pre-existing diseases covered after 36 months, maternity after 24
  months). The giveaway is a duration after which the policyholder IS covered.

- sub_limit: caps or reduces what is paid, below the sum insured. Room-rent
  caps, ICU caps, disease-wise ceilings, co-payment and deductible percentages.
  A clause that both grants a benefit AND caps it is a sub_limit.

- condition: an obligation on the POLICYHOLDER whose breach can void an
  otherwise valid claim - notice deadlines, document submission, prior
  authorisation, disclosure duties. The tell is a duty plus a consequence.

- definition: defines a term used elsewhere in the policy (e.g. what counts as a
  "Hospital", "Accident", "Pre-existing Disease").

- procedural: administrative machinery with no direct effect on whether a
  specific claim is paid - renewal, portability, cancellation, grievance
  redressal, free-look period.

TIE-BREAKERS (apply in this order - the more specific label always wins):
1. Does it cap or reduce an amount?              -> sub_limit
2. Does coverage start after a stated period?    -> waiting_period
3. Does it impose a duty that can void a claim?  -> condition
4. Is it a permanent bar on payment?             -> exclusion
5. Does it define a term?                        -> definition
6. Is it pure administration?                    -> procedural
7. Otherwise                                     -> coverage

PLAIN LANGUAGE RULES:
- Address the reader as "you". Aim at a 13-year-old's reading level.
- Keep every number, percentage, and time period exactly as written.
- State the consequence plainly. If a claim can be refused, say so.
- Never soften, never add reassurance the clause does not contain.
"""

CLAUSE_TYPE_ENUM = ClauseType.values()
