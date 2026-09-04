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

PROMPT_VERSION = "v4-severity-anchors"

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

WHAT IT MEANS FOR YOU:
One sentence on the practical consequence. Not a restatement of the clause -
what actually happens to the reader. "If you are admitted for a knee
replacement in your first two years, you pay the entire bill yourself."

EXTRACTED DETAILS:
- triggers: what activates this clause - procedures, conditions, ages,
  circumstances. Short phrases. Empty list if it always applies.
- monetary_limits: every cap, percentage or amount, as written
  (e.g. "1% of Sum Insured per day", "20% co-payment"). Empty if none.
- time_windows: every duration or deadline, as written
  (e.g. "36 months", "24 hours of admission"). Empty if none.

LIKELIHOOD (1-5) - how many policyholders this clause will actually touch:
  1  Almost nobody. War, nuclear perils, adventure sports.
  2  A small minority. Infertility treatment, AYUSH.
  3  A sizeable minority. Maternity, cataract, senior-citizen co-payment.
  4  Most people, sooner or later. Pre-existing disease waiting periods.
  5  Nearly every claimant. Room rent limits, notice deadlines, document
     submission rules - things that apply to any hospitalisation at all.

SEVERITY (1-5) - the financial damage when it does apply:
  1  Negligible. Minor administrative inconvenience.
  2  Small reduction in the amount paid.
  3  Meaningful reduction - a co-payment, a sub-limit on one item.
  4  Large loss. Proportionate deduction across an entire bill.
  5  Total loss. The claim is refused outright, or the policy is void.

Rate these independently. A war exclusion is severity 5 but likelihood 1. A
24-hour notice requirement is likelihood 5 and severity 5 - it applies to
everyone and can void the whole claim. Do not average them; that is done later.

SEVERITY IS ABOUT THE POLICYHOLDER'S MONEY, NOT THE CLAUSE'S TONE.
Formal, official-sounding language does not make a clause severe. Ask only:
how much worse off is the policyholder if this clause applies?

- A power the Company may exercise AT ITS OWN EXPENSE - requiring a medical
  examination it pays for, appointing its own assessor - costs the
  policyholder nothing. Severity 1, no matter how formally it is written.
- A right the Company already has by law, or a consumer protection such as a
  free-look or grievance process, is severity 1. Some of these HELP the reader.
- A cap that CASCADES is severity 4-5, not 3. If exceeding a room-rent limit
  also reduces surgeon, anaesthetist and theatre charges in proportion, the
  loss is spread across the entire bill, not confined to the room.
- A co-payment or deductible that applies to EVERY claim for the rest of the
  policy's life is severity 4: small each time, unavoidable and permanent.

Before answering, check: does this clause take money from the policyholder, or
merely describe a process? Only the first earns a severity above 2.
"""

CLAUSE_TYPE_ENUM = ClauseType.values()


def render_clause_batch(segments) -> str:
    """Format a batch of clauses for the classifier.

    Three details matter here:

    1. **The id is stated explicitly** and repeated in the schema's enum, so the
       model has an unambiguous handle for each clause. Relying on "the first
       one", "the second one" invites misattribution.

    2. **The section path is included.** "SECTION 4 - PERMANENT EXCLUSIONS" is
       strong evidence about a clause's type, and it is evidence a human reader
       has too. Withholding it would make the task artificially harder.

    3. **Clauses are separated by a clear delimiter.** Policy text contains
       colons, numbers and newlines, so the boundary between clauses has to be
       something that cannot occur inside one.
    """
    parts = []
    for seg in segments:
        header = f"### CLAUSE id={seg.order_idx}"
        if seg.number:
            header += f" number={seg.number}"
        if seg.section_path:
            header += f"\nSection: {seg.section_path}"
        parts.append(f"{header}\n{seg.text.strip()}")

    return (
        "Analyse each clause below. Return one entry per clause, using the id "
        "given for each.\n\n" + "\n\n".join(parts)
    )
