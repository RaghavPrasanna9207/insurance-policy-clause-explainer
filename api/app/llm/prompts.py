"""Prompt templates and their version.

PROMPT_VERSION is a LABEL. Bump it whenever a prompt's wording changes, so eval
reports, the run history and stored analyses record which instructions produced
them. It is not part of the cache key - the cache hashes the prompt text
itself, so a reworded prompt can never be served an answer to the old wording,
and an unchanged prompt keeps its stored answer. See app/llm/cache.py for why
it was taken out of the key.

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

from app.grounding import normalize
from app.taxonomy import ClauseType

PROMPT_VERSION = "v41-two-period-waits-capped-answers"

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
- waiting_period_value / waiting_period_unit: if this clause bars cover for a
  period, the number and its unit exactly as the clause states them. "the first
  thirty days" is 30 + days. "thirty six months" is 36 + months. Do NOT convert
  between units. Both null for any clause that is not a waiting period.
- exceptions: the cases where this clause does NOT apply. Look for "unless",
  "except", "other than", "save for", "provided that", "shall not apply".
  An exclusion with a carve-out is not an absolute bar, and the carve-out is
  the part a policyholder most needs to see:
    "cosmetic surgery ... unless necessitated by an Accident, Burn or Cancer"
       -> ["necessitated by an Accident, Burn or Cancer"]
    "no claim ... except claims arising out of an Accident"
       -> ["claims arising out of an Accident"]
  Empty list only when the clause genuinely admits no exception.

THE NUMBERS BEHIND A REDUCTION. Null on every clause that reduces nothing,
which is most of them. Report what the clause SAYS; the comparison against the
person's actual figures is done afterwards, in code.
- copay_percent: the co-payment percentage the insured must bear.
    "a co-payment of twenty percent of the admissible claim amount" -> 20
- copay_min_age_at_inception: the age that triggers that co-payment, where the
  clause sets one. Read the wording carefully - it usually keys on age when the
  policy STARTED, not age at the time of claim.
    "completed sixty years of age at the time of first inception" -> 60
    a co-payment with no age condition at all -> null
- cap_percent_of_sum_insured: an ordinary per-day accommodation cap expressed
  as a percentage of the sum insured.
    "room rent ... limited to one percent of the Sum Insured per day" -> 1
- icu_cap_percent_of_sum_insured: the same, for intensive care, where the
  clause states a separate rate.
    "Intensive Care Unit charges ... limited to two percent" -> 2
  A clause that states one rate for a room and a different rate for ICU fills
  in BOTH fields; one that mentions no ICU rate leaves this null.
- cap_max_inr_per_day / icu_cap_max_inr_per_day: a rupee ceiling on that same
  percentage cap, where the clause sets one. The limit is whichever is lower.
    "room rent up to 1% of the Sum Insured subject to a maximum of Rs.4,000 per day"
       -> cap_percent_of_sum_insured 1, cap_max_inr_per_day 4000
  A percentage cap with no rupee ceiling leaves these null.

A cap expressed ONLY in rupees, with no percentage of the sum insured
("twenty five thousand rupees per eye"), leaves all six fields null and belongs
in monetary_limits, where it already goes.

THE WINDOW AROUND A HOSPITAL STAY. Null on every clause that states none, which
is almost all of them.
- cover_window_value / cover_window_unit / cover_window_anchor: a clause that
  pays for expenses only within a period before admission or after discharge.
    "the sixty days immediately preceding the date of admission" -> 60, days, before_admission
    "the ninety days immediately following the date of discharge" -> 90, days, after_discharge
  That window counts from a hospital stay, not from when the policy began, so
  it is never a waiting period and never goes in waiting_period_value.

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


# ---------------------------------------------------------------------------
# Scenario simulator (stage 5)
# ---------------------------------------------------------------------------

FACTS_SYSTEM = """\
You extract the facts from a description of a medical situation, so that an
insurance policy can be checked against it.

Extract ONLY what the description actually says. This is the whole job.

If the person did not mention something, the field is null. Do not guess, do
not infer a typical value, and do not fill a gap with something plausible. A
missing fact must stay missing, because the next stage decides whether the
policy can answer at all - and it can only do that if it knows what was never
said.

Examples of the distinction:
- "I had knee surgery"                -> time_since_policy_start_value: null, unit: null
- "8 months after buying the policy"  -> time_since_policy_start_value: 8, unit: months
- "two weeks after my policy started" -> time_since_policy_start_value: 2, unit: weeks
- "I have held it five years"         -> time_since_policy_start_value: 5, unit: years
- "admitted a year after I took out cover" -> time_since_policy_start_value: 1, unit: years
- "six months into the policy"        -> time_since_policy_start_value: 6, unit: months

An AGE is never how long the policy has been held. "I was 61 when the cover
began and I claimed four years later" is age_at_policy_start: 61 AND
time_since_policy_start_value: 4, unit: years - two different numbers in two different
fields. A time_since_policy_start_value always comes with its unit; if there is no length
of time with a unit, it is null.

NEVER convert between units. "Two weeks" is 2 + weeks, not 2 + months and
not 14 + days. Report the number and the word the person actually used;
converting is done afterwards, in code.
TWO DIFFERENT AGES, AND THEY ARE NOT INTERCHANGEABLE.
`age` is how old the person is NOW. `age_at_policy_start` is how old they were
when the policy began. A senior-citizen co-payment keys on the second one, so
guessing it from the first would impose a 20% cut on someone who does not owe
it. Fill in only what was actually said; the arithmetic linking them is done
afterwards, in code.

- "I'm 67"                                    -> age: 67, age_at_policy_start: null
- "I bought this policy at 67"                -> age: null, age_at_policy_start: 67
- "I took it out at 70 and I'm 72 now"        -> age: 72, age_at_policy_start: 70
- no age mentioned                            -> both null

MONEY, THE SAME WAY: a value and the word that went with it.
Indian policies are written in lakh and crore, and converting them is
arithmetic done later in code - exactly like the units above.

- "my sum insured is 5 lakh"       -> sum_insured_value: 5, sum_insured_unit: lakh
- "sum insured of 75,000 rupees"   -> sum_insured_value: 75000, unit: rupees
- "a room costing 9,000 a night"   -> room_rent_per_day_inr: 9000, room_is_icu: false
- "four days in ICU at 12,000/day" -> room_rent_per_day_inr: 12000, room_is_icu: true
- no room or ICU mentioned         -> room_rent_per_day_inr: null, room_is_icu: null

room_rent_per_day_inr is the PER-DAY charge in plain rupees, never the total
bill. "Four days at 12,000 a day" is 12000, not 48000.

EXPENSES BEFORE OR AFTER A HOSPITAL STAY: how far from the stay, and which side.
This counts from a HOSPITAL STAY, not from when the policy began - it is never
the same number as time_since_policy_start.

- "tests fifty days before I was admitted"  -> expense_timing_value: 50, expense_timing_unit: days, expense_timing_anchor: before_admission
- "a scan 120 days after I was discharged"  -> 120, days, after_discharge
- "physio for two months after I went home from hospital" -> 2, months, after_discharge
- "a follow-up after I was discharged" (no number) -> null, null, after_discharge
- nothing said about expenses before admission or after discharge -> all three null

For pre_existing_condition, answer "unknown" unless the description makes it
clear either way. "Unknown" is the honest answer far more often than not.
"""


def render_scenario(scenario: str) -> str:
    return f"Extract the facts from this description:\n\n{scenario.strip()}"


REASON_SYSTEM = """\
You decide what an Indian health insurance policy says about one person's
situation, using ONLY the clauses given to you.

VERDICTS - choose exactly one:

- not_covered: a clause clearly excludes this, or a waiting period has not yet
  been served. The person will not be paid for this.
- covered: the policy pays, and no exclusion, waiting period or cap applies to
  what was described.
- conditional: the answer depends on something the clauses require and the
  description does not settle - a notice deadline, a document, a room category,
  a co-payment. Cover is possible but not assured.
- insufficient_information: the clauses provided do not address this situation
  at all, OR a fact needed to decide is missing from the description.

TELLING THE VERDICTS APART - these were measured getting confused:

- A waiting period that has NOT yet elapsed is **not_covered**, not
  conditional. If someone claims 8 months into a 36-month pre-existing disease
  waiting period, the claim is refused today. "Covered in another 28 months" is
  worth saying in the reasoning, but the answer to "will you pay for this" is
  no. Never soften a refusal into "conditional" because cover arrives later.

- A clause that clearly applies but leaves the Company a discretion
  ("may repudiate", "at its sole discretion") is **conditional**, not
  insufficient_information. The document DOES address the situation; the
  outcome is simply not automatic.

- Use **insufficient_information** when the clauses genuinely do not speak to
  the situation, or when a fact you would need is missing from the description.
  Not when the answer is merely unwelcome.

BEING PAID LESS IS NOT BEING REFUSED.
This is the most common mistake. A co-payment, a room-rent cap, a
proportionate deduction or a disease sub-limit means the claim IS paid - just
reduced. That is **conditional**, never not_covered.

  "20% co-payment applies"        -> conditional. You are paid 80%.
  "room rent capped at 1% a day"  -> conditional. You are paid, with a deduction.
  "cosmetic surgery is excluded"  -> not_covered. Nothing is paid.

Ask: does the policy pay ZERO for this, or does it pay a smaller amount?
Only zero is not_covered.

DO NOT DO THE WAITING-PERIOD ARITHMETIC YOURSELF.
Every waiting period has already been compared against how long the policy has
been held, and the results are given to you above under "WAITING PERIODS,
ALREADY CALCULATED FOR YOU". Those results are computed and correct.

  - A period marked "no longer applies" removes that ONE clause from
    consideration. It is not a reason to refuse, and it is not a reason to
    answer "covered" either - exclusions, payout caps, co-payments and notice
    conditions are all still live and must be checked separately.
  - A period marked "still applies" blocks treatment covered by that clause.
  - A period that cannot be determined means the person did not say how long
    they have held the policy. If the answer turns on that, the verdict is
    insufficient_information.

Waiting periods are ONE of five things that can decide a claim. Before
answering, ask which of these five actually governs the situation described:

  waiting_period  cover has not started yet
  exclusion       never paid, permanently
  condition       a duty on the policyholder - notice deadlines, documents
  sub_limit       a cap or co-payment that reduces the amount paid
  coverage        the benefit itself

A question about a notice deadline is decided by a condition. A question about
a room-rent cap or a co-payment is decided by a sub_limit. A question about a
hazardous sport is decided by an exclusion. In none of those does a waiting
period decide anything, and citing one instead of the governing clause is a
wrong answer even when the verdict happens to land correctly.

Never contradict those results and never recompute them.

DO NOT DO THE MONEY ARITHMETIC YOURSELF EITHER.
Every cap expressed as a percentage - a room-rent limit, an ICU limit, a
senior-citizen co-payment - has already been worked out against the person's
own figures, under "WHAT REDUCES THE PAYOUT". Those results are computed and
correct. Do not multiply percentages, and do not check them.

  - A reduction marked APPLIES means the claim is paid at less than the full
    amount. If nothing refuses the claim, that makes the verdict
    **conditional**, and the amount must be named in your reasoning.
  - A reduction marked CANNOT TELL means a figure needed for the comparison was
    never stated. If the answer turns on it, say so.
  - A reduction marked YOUR CALL has no arithmetic to it: whether a cataract
    sub-limit or a modern-treatment restriction covers this treatment is a
    question about what the treatment WAS, and that one is yours to decide from
    the clause text.
  - Reductions listed as ruled out decide nothing. Do not cite them.

ASK BOTH QUESTIONS, NOT ONE.
The failure this section exists to prevent: having established that nothing
BLOCKS a claim, stopping there and answering "covered". Those are two separate
questions and both must be answered.

  1. Is this claim refused?          exclusions, unserved waiting periods
  2. Is it paid IN FULL?             co-payments, room caps, sub-limits

Someone told "covered" who is actually paid 80% has been given a wrong answer,
and they find out when the money arrives.

READ THE EXCEPTIONS.
Where a clause has carve-outs they are listed under it as "EXCEPTIONS - this
clause does NOT apply when: ...". If the situation falls inside one, the clause
does not apply and must not be cited as a denial. A burn treated with
reconstructive surgery falls inside "unless necessitated by an Accident, Burn
or Cancer"; that claim is covered, not refused.

A FACT BEING UNSTATED ONLY MATTERS IF YOU ACTUALLY NEED IT.
The situation will always leave things unsaid - an age, a cost, an exact hour.
That is normal and is not by itself a reason to abstain. Ask only: do I need
THIS fact to answer THIS question?

If a clause plainly excludes what happened, the verdict is **not_covered**,
however much else went unmentioned. Cosmetic surgery for appearance is excluded
whether or not the person gave their age. Abstaining there is not caution, it
is a non-answer to a question the document plainly settles.

CHOOSING insufficient_information IS OFTEN THE CORRECT ANSWER.
It is not a failure and it is not a fallback. If the person did not say how
long they have held the policy, you cannot know whether a 36-month waiting
period has been served, and saying "covered" would be a guess presented as a
fact. Someone may make a financial decision on this. Say what the document
supports and nothing more.

CITING CLAUSES:
- Every clause you rely on MUST appear in deciding_clauses. Naming a clause in
  your reasoning while leaving deciding_clauses empty makes the answer
  unusable, because then nothing can be checked against the document.
- Cite every clause that decides the answer, and no others.
- For each, quote the words from that clause that do the deciding. Copy them
  EXACTLY from the clause text as given. Do not paraphrase, tidy, shorten or
  correct the wording. The quote is checked character by character against the
  policy, and an inexact quote is discarded.
- Quote the phrase that actually decides it: one sentence, not the whole
  clause. Two or three words prove nothing, and reprinting an entire clause is
  not a citation - it pushes the real reason back onto the reader to find.
- Put ONLY the clause's own words inside the quote. Do not append a clause
  number, a bracketed reference or any note of your own - the quote is compared
  against the policy character by character, and anything you add to it is a
  difference.
- A waiting period that has been SERVED is not a reason to deny. Check the
  elapsed time before citing one.

REASONING:
Two or three sentences, addressed to the person as "you". State the outcome
first, then why. Name the deadline or amount that decides it. Never soften a
refusal, and never promise cover the clauses do not give.
"""


# Phrases that state what a policy pays. The fact extractor is never shown the
# policy, so when one of these turns up in its free-text notes and the person
# did not say it, the extractor has answered the question instead of reading
# it. Measured: "This policy does not cover car theft" in the notes turned an
# honest insufficient_information into not_covered, six samples of six.
_COVERAGE_CLAIMS = (
    "does not cover", "doesn't cover", "not cover", "not covered", "is covered",
    "are covered", "will be covered", "excluded", "not payable", "is payable",
    "will not pay", "will pay",
)


def _invents_coverage(note: str, scenario: str) -> bool:
    """True if the note claims coverage in words the person did not use."""
    note, said = normalize(note), normalize(scenario)
    return any(p in note and p not in said for p in _COVERAGE_CLAIMS)


def render_reasoning_request(
    scenario: str, facts: dict, clauses,
    waiting_block: str = "", reduction_block: str = "",
    window_block: str = "",
) -> str:
    """Build the reasoning prompt.

    The clause ids embedded here are the same ids used to build the schema's
    `clause_id` enum, so the model can only cite something present in this
    text. That correspondence is the grounding guarantee, and it breaks
    silently if the two are ever built from different lists - which is why they
    are built from one list, in one place, in `pipeline/scenario.py`.
    """
    known = {k: v for k, v in facts.items() if v not in (None, "", [], "unknown")}
    if isinstance(known.get("notes"), str) and _invents_coverage(known["notes"], scenario):
        del known["notes"]
    # Imported rather than redeclared: the model's list and the user's list are
    # the same list. See DECISIVE_FACTS in pipeline/scenario.py for why.
    from app.pipeline.scenario import DECISIVE_FACTS, shared_words

    missing = [k for k in DECISIVE_FACTS if facts.get(k) in (None, "", "unknown")]

    lines = [
        "SITUATION (in the person's own words):",
        scenario.strip(),
        "",
        "FACTS UNDERSTOOD:",
        "\n".join(f"- {k}: {v}" for k, v in known.items()) or "- (none stated)",
    ]
    if missing:
        # Stated explicitly rather than left implicit. The model has to be able
        # to tell "the policy is silent" apart from "the person didn't say",
        # and those lead to different verdicts.
        lines += [
            "",
            "NOT STATED by the person (do not assume values for these):",
            "\n".join(f"- {k}" for k in missing),
        ]

    if waiting_block:
        lines += ["", waiting_block]
    # Beside the waiting periods, because both answer "is this expense payable
    # at all" - which has to be settled before "is it payable in full".
    if window_block:
        lines += ["", window_block]
    # After the waiting periods, because the two answer questions that come in
    # that order: first "is this claim payable at all", then "is it payable in
    # full". Reversing them puts a co-payment in front of a bar that means
    # nothing is paid at all.
    if reduction_block:
        lines += ["", reduction_block]

    # Last before the clauses, and silent unless something is shared: most
    # questions share no unusual words with any clause and see nothing here.
    shared = shared_words(scenario, clauses, facts)
    if shared:
        lines += ["", "WORDS THIS QUESTION SHARES WITH A CLAUSE (a lookup, not a judgement)."]
        lines += [
            f"- clause {clause_id} uses these words from the question, or forms of them: "
            f"{', '.join(words)}"
            for clause_id, words in shared.items()
        ]
        lines.append(
            "Read these clauses closely before choosing what to cite. Sharing words "
            "does not by itself mean a clause applies."
        )

    lines += ["", "POLICY CLAUSES AVAILABLE TO YOU:", ""]
    for clause in clauses:
        header = f"### clause_id={clause.clause_id}"
        if clause.number:
            header += f"  ({clause.number})"
        header += f"  [{clause.clause_type}]"
        lines.append(header)
        lines.append(clause.text.strip())
        # Surfaced separately from the body text. An exclusion carrying a
        # carve-out was being read as an unconditional bar, because the escape
        # hatch sits at the end of a long sentence.
        #
        # Removing this line was measured, and reverted. It had been copied into
        # quotations and was suspected of drawing citations to the one exclusion
        # carrying it. Without it, citation recall rose (0.742 -> 0.806), but
        # certified burn surgery was refused as "still cosmetic" and an accident
        # ten days into a policy came back conditional - three samples of three
        # each - while the two cases it was meant to fix stayed wrong. It is
        # safe to show now because every exception is verified against the
        # clause text first (app/grounding.py, verify_exception).
        if getattr(clause, "exceptions", None):
            lines.append(
                "EXCEPTIONS - this clause does NOT apply when: "
                + "; ".join(clause.exceptions)
            )
        lines.append("")

    return "\n".join(lines)
