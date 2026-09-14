# Documentation

## Start here

**[LEARNING-LOG.md](LEARNING-LOG.md)** — the main document. It explains what this system does, why each piece is built the way it is, and every non-obvious failure encountered along the way, in the order things were built.

It is written to be read cold, by someone who was not present while the code was written. You do not need any other context.

---

## If you only have twenty minutes

Read these three sections of the log, in order:

1. **Concept 1 — Constrained decoding** *(M0)*. The idea the whole system rests on: how a JSON Schema constrains token sampling itself, so invalid output is never generated rather than rejected afterwards.
2. **Concept 2 — What constrained decoding does NOT give you** *(M0)*. The failure that shaped the project: every answer was structurally valid and two were badly wrong, both in the dangerous direction.
3. **Failure 1 — The circular test** *(M1)*. A test suite passing 14/14 while proving nearly nothing, and what fixed it.

Together they cover the reasoning that most of the rest of the codebase follows from.

---

## What each entry covers

### M0 — Foundations: talking to a local model reliably
Whether a 7B model on a laptop GPU can be trusted to produce structured output.

- **Constrained decoding** — JSON Schema as a grammar constraint on sampling, not a prompt instruction. Why a hallucinated citation can be made *unrepresentable* rather than merely unlikely.
- **What it does not give you** — the enum guarantees the *shape* of an answer, never the *judgment* behind it. A measured 3-of-4 wrong, fixed to 8-of-8 purely by defining the labels in the prompt.
- **Content-addressed caching** — why the cache key must include the prompt version, and why `sort_keys=True` is correctness rather than tidiness.
- Measured hardware facts that drive the architecture.

### M1 — Reading the PDF: text, offsets, and clause boundaries
Turning a policy PDF into ~40 individually addressable clauses.

- **The offset invariant** — `raw_text[start:end] == text`, and why a slice is a fundamentally stronger claim than a copy.
- **Why segmentation must not use an LLM** — reproducibility, verifiability, speed, debuggability.
- **Failure 1: the circular test** — tuning thresholds on a document you generated, then testing on that same document.
- **Failure 2: a data model that hid the fix** — a clause's number is its identity; the heading is optional.
- **Failure 3: a bug you cannot see** — a literal backspace byte inside a regex, and a passing test suite concealing entirely dead code.

### M2 - Analysis and scoring: measuring instead of guessing
Classifying every clause, then ranking them by how much they could cost.

- **Splitting judgment from arithmetic** - the model rates likelihood and severity; buriedness is computed. Why arithmetic done by a language model cannot be trusted or reproduced.
- **Measuring "obscuring"** - position, Flesch-Kincaid grade, cross-references and defined-term dependence, and why buriedness multiplies rather than adds.
- **Failure 4: a silent empty array** - `{"clauses": []}` is valid under a schema that only constrains element shape. Constrained decoding guarantees what you encode, and nothing you forgot to encode.
- **Failure 5: an unmeasured optimisation** - batching was assumed faster, measured slower AND less accurate.
- **Failure 6: unbounded arithmetic** - a bug only reachable through a test that used the code differently than production does.
- **Measuring the right thing** - macro-F1 1.000 on classification, while the ranking buried the room-rent cap at 14. Two different metrics; only one is the product.
- **Knowing when to stop tuning** - why 6/9 with a reason beats 9/9 by fitting.

### M3 - The API, and a bug tests could not see
An HTTP surface over the pipeline: upload, poll, read ranked results.

- **Why upload returns 202, not 200** - the work is queued, not completed; and why progress is weighted by time rather than by stage count.
- **Committing status writes** - the poller reads from a different connection and can only see committed changes.
- **Failure 7: the bug tests could not catch** - `create_all` never alters an existing table, so the dev database went stale. Every fixture starts with `drop_all`, so the suite was structurally blind to it. A suite that rebuilds the world before each test cannot see bugs that need a world which has been running for a while.
- **Two kinds of test** - fast tests against a seeded database vs. end-to-end runs through the model. Tests check that it works; evals check how well.

### M4 - The interface, and designing against a default
Four screens, built to a written design system rather than to defaults.

- **Find the metaphor already in your data** - this product is a document arguing with another document, which is a critical edition. The API already pairs verbatim source with character offsets; the design makes that visible.
- **Typography can carry meaning** - a document serif for the policy's words, a sans for the app's, so the side-by-side needs no labels.
- **Colour must never be the only signal** - two semantic hues rather than a traffic light, always paired with a numeral and a text label.
- **Failure 8: a rule enforced everywhere except where it mattered** - the severity numeral vanished below the `sm` breakpoint, breaking a rule from this project's own DESIGN.md.
- **Failure 9: module resolution follows the file, not the shell** - the same bug as M0's `ModuleNotFoundError`, in a different language.
- **You cannot review a design you have not looked at** - a committed Playwright script drives the real UI and captures every screen; it found bugs no typecheck or unit test could.

### Interlude - Why a local model, when hosted models are better
Answers the obvious question this project invites, without being defensive about it.

- **What production systems actually use** - hosted frontier APIs, and by how much they beat a 7B model on hard reasoning.
- **Cost is not the reason** - a whole policy is ~28k tokens, a few cents at API prices. Do the arithmetic rather than assuming.
- **The reason that holds up** - a policy is a personal health and financial document. Local is a product property, not a budget choice.
- **The counterintuitive part** - the strongest correctness guarantee exists partly BECAUSE of local grammar-level token masking. A bigger model is not automatically a free upgrade for it.
- **Model choice is a parameter; the system around it is the work.**

### M5 - The scenario simulator, and making a citation impossible to fake
"I had knee surgery 8 months after buying this" -> a verdict, with the clauses that decide it.

- **Two mechanisms, not one** - an enum locks the citation's ADDRESS to a real clause; a verbatim quote check proves its CONTENT. The first alone permits a real clause id attached to invented wording, which was observed happening.
- **Why there is no retrieval** - the whole policy is ~3,100 tokens and the context holds far more. Top-k could only drop the clause that decides the case.
- **Abstention has to be built into the vocabulary** - constrained decoding cannot say "I don't know" unless that is an enum member.
- **Failure 10: a silent context-window default** - the design assumed 32k; Ollama was applying 4,096, and a larger policy would have been truncated with no error.
- **Failure 11: over-provisioning has a cost** - raising the window to 16k to "be safe" left 2GB free of 15.7GB and got the eval killed by the OS.
- **Failure 12: two numbers that must agree should not be two numbers** - a clause budget set independently of the context window can exceed it.
- **A check that cries wolf gets ignored** - a quotation 233 of 245 characters perfect was reported as if fabricated.

### M6 - One report, and proving the repo works for someone else
Consolidating the evals, and the fresh-clone gate.

- **Failure 15: a report that could not say when it was written** - two report files describing runs of different code, one saying "see PROMPT_VERSION in prompts.py" instead of recording it. A report is a record of a run; a pointer is not a record.
- **Separating what is measured from what is promised** - the report's `Kind` column, and why the eval exits non-zero on detection integrity but never on verdict accuracy.
- **Failure 16: misreading my own test failure** - "it failed" and "it failed for the reason I assumed" are different claims. A collection error that looked like a broken repo was a race with a background `pip install`.
- **What a fresh clone actually gets** - 881KB, no PDF ever committed, all golden fixtures rebuilt from their generator.

### M7 - Fixing 0.688, and four bugs hiding behind each other
Verdict accuracy was not acceptable. Four of the five failures turned out to be this project's bugs, not the model's.

- **I broke this project's own rule** - stage 5 was asking a 7B model whether 5 years exceeds 36 months, while stages 2 and 4 carefully kept arithmetic away from it. A capability limit and an architectural mistake look identical from outside.
- **Failure 17: a green test that passed for the wrong reason** - "thirty days" read as 30 MONTHS, and the eval case still passed because a fortnight is short of both.
- **Failure 18: the same bug on the other operand** - "two weeks" read as 2 months. A comparison has two sides; fixing one is not fixing it.
- **Failure 19: a field that existed but was never populated** - adding a field is not adding a feature.
- **Correct facts can still mislead** - four true statements that a waiting period "does not block the claim" made the model answer covered to questions about co-payments. Accuracy fell to 0.625 while the arithmetic worked perfectly.
- **Knowing when the number stops being a signal** - five runs, 0.688 / 0.625 / 0.812 / 0.688 / 0.812, on a set where one case is 0.0625.

### M8 - Widening the ruler, and the second family of comparisons
Rebuilding the scenario eval set, and a correct fix that measured worse.

- **A ruler that cannot resolve your changes** - on 16 cases one case is 0.0625, and five consecutive runs moved within that. Below its granularity a metric does not merely fail to help, it misleads, because a number that moved feels like evidence.
- **0.812 was an easier exam, not a better system** - the same unchanged code scores 0.725 on 40 cases. The smaller set had not been asking the questions it was bad at.
- **Recall only sees one of the two ways to be wrong** - it goes UP when a system cites more, so a fix trading under-citing for over-citing scores as a win. Hence `must_not_cite` and a separate false-citation rate.
- **The same mistake in a family nobody had noticed was the same** - an earlier fix took DURATION comparisons away from the model and left MONEY and AGE. "Is 8,000 more than 1% of 10 lakh" is not a language question.
- **The operand that was secretly two operands** - a co-payment keys on age AT INCEPTION, not age now. The third time this project has hit a comparison whose operands were not what they appeared to be.
- **Failure 20: the fix made it worse by 0.125** - correct arithmetic, injected into the prompt, sent every new failure to `conditional` or `insufficient_information`.
- **Concept 24: silence is not uncertainty** - "cannot be determined" about a room nobody mentioned manufactures doubt on nearly every question.
- **Concept 25: the honest read on a fix that did not pay for itself** - net -2 cases, diagnosed to one specific line that carried no information and cost three cases.

### M9 - The ruler was made of rubber, and two requests at once was why
The eval harness had never been tested for reproducibility. It was not reproducible.

- **The same code, measured three times, gave three numbers** - 0.725, 0.700, 0.750 on identical code and byte-identical prompts. Five of forty cases flip between runs; 27 always pass, 8 always fail.
- **Concept 26: `temperature = 0` is not reproducibility** - it makes the SAMPLER deterministic, which is only half the sentence. The other half is "given the same logits", and a GPU does not reliably produce the same logits twice.
- **Failure 22: the fix I was certain of, which changed nothing** - pinning the seed. At temperature 0 nothing draws from the random generator, so seeding it is seeding a die that is never rolled.
- **Failure 23: claiming the cause after one observation each way** - one failure at concurrency 2 and one pass at concurrency 1 is two coin flips, not a diagnosis. Six repeats each settled it.
- **Concept 27: float addition is not associative, and that is a product bug** - batching two requests together changes the order the matmul partial sums are combined, which changes the last bits of a logit, which flips a near-tie. The same policy was being rewritten differently on each upload.
- **Concept 28: measure the cost of the safe choice first** - serialising "obviously" halves throughput. Measured, it cost 9%: one request already saturates an 8GB card.
- **Failure 24: the residual, and a second fix that measured nothing** - serialising did not finish the job. Two fresh runs still score 0.650 and 0.675. The first substantive generation of a process differs from every one after it; a real warm-up call was tried, measured to change nothing, and reverted. Recorded as open.

### M10 - Measuring only what changed, and a third family of arithmetic
Making the eval measure a change despite a model that never answers twice the same way, and fixing a case that failed in every run.

- **Repeat is not sequence** - asking one question twice gave one answer; asking forty questions twice changed 7 of 10 at the level the metrics read. Test the property your conclusion depends on.
- **Concept 29: the cache as a controlled experiment** - a content-addressed cache replays the stored bytes of any prompt that did not change, so those cases cannot have moved. Only the cases a change reached are regenerated. A one-word change reaching 2 cases: 38 replayed, 2 regenerated, exactly as predicted.
- **Failure 25: the cache's row count lies** - a truncated answer retried with a larger limit is stored under a key nothing looks up, so "did the table grow" misses regenerations. Count model calls instead.
- **Failure 26: counting the wrong calls** - counting fact extraction made every case look regenerated. Only the reasoning call decides a verdict.
- **Failure 27: advice that confirmed a result with a copy of itself** - "re-run with --only" replayed the answer being doubted. Hence `--resample`.
- **Concept 30: a version label is not a cache key** - hashing PROMPT_VERSION was redundant for safety and forced every case to regenerate on every change. Now a label only.
- **The fix: a cover window, decided in code** - "is a scan 120 days after discharge inside a 90-day window" taken away from the model, with an anchor so before-admission and after-discharge are never compared.
- **Failure 28: one paragraph reclassified clauses it never mentioned** - a note and a field description dropped classification from 1.000 to 0.919. The same text in a different position broke nothing. Check a prompt edit against everything the prompt does.
- **Failure 29: a correct fix that measured net negative** - the fact extractor leaked a verdict through a free-text `notes` field. Removing the field fixed one case, broke three, and cost citations. Reverted; recorded as open.
- **Failure 30: a warning that blamed a cause that did not exist** - reverting a change replays older answers, which the comparison had mistaken for an unrecorded run.

### M11 - Reading the inputs: two bugs upstream of every verdict
Two errors that happened before the model ever reasoned, found by reading the failing cases' own explanations rather than their scores. Verdict accuracy 0.725 to 0.800; failed quotations 5 to 1.

- **Checking what a fix would buy, before building it** - the planned fix (code-added citations) could reach one case of 31 and would have broken another. Set aside.
- **Failure 31: every check did its job, on a fact that was wrong** - 9 of 40 questions had the policy's age missing or replaced by the person's age. Every downstream stage was correct given its input, and the scenario eval, which scores only verdicts, could not see it.
- **Concept 31: a schema key is part of the prompt** - under constrained decoding the key is the last thing written before the value. `policy_age` pulled in ages; `policy_held_for` fixed that and lost "after my policy started"; `time_since_policy_start` read both. Measured on held-out sentences, with a second held-out batch written after the first had been tuned against.
- **Failure 32: eight fabricated quotes, and half were our own words** - the model copied an annotation the pipeline prints beneath each clause.
- **Failure 33: the pipeline was telling the model things the policy does not say** - 5 of 8 extracted exceptions were wrong, one copied from the analysis prompt's own example. "The alcohol exclusion does not apply to accidents" explained a case that had failed in every run. The first diagnosis of its cause was wrong too.
- **Concept 32: checking an extraction by its grammar, not only its text** - a substring check cannot catch real text in the wrong role. An exception must be preceded by an exception word in its own sentence.
- **A verdict-neutral change, kept, and why** - it broke two cases and fixed two, and it stopped the system asserting falsehoods about the policy. Weighed against a structurally correct M10 fix that was reverted.

### M12 - Seven attempts, three kept, and a number that means what it says
Working through the remaining failures one measured change at a time. Verdict accuracy reaches 0.925 as a majority of three samples, the first score in the log that is not a single sample.

- **Step 0: measure a rule before writing it** - replaying stored answers showed which cases each proposed rule would catch, and changed one rule before it existed.
- **Fix 1 and Fix 2** - a waiting-period line that names the clause's own carve-out, and fact-extractor notes that claim coverage the person never mentioned. Each reached exactly the 2 and 1 cases predicted.
- **Failure 34: removing the emphasis, which was carrying information** - the line suspected of distorting citations was the one making burn surgery covered. Reverted.
- **Concept 33: checking the answer against the arithmetic** - "never contradict these results" is a request; detecting a citation of a clause the arithmetic cleared is mechanical. One retry, provably wrong citations removed, verdicts never set by code.
- **Failure 35: a true fact that fixed three other cases and not its own** - why a change is not kept for effects it was not designed to cause.
- **Failure 36: a second turn that chose the same wrong clauses** - a clean negative result about where citation errors come from.
- **Failure 37: two attempts at third-person ages, both worse** - a key name is an instruction, but a guessed rename is only an experiment.
- **Concept 34: majority of three** - a strict majority per case, run in chunks, with a replay check that proved three reverts were complete.

### M13 - The last failures, and the answer key itself
Majority-of-three verdict accuracy 0.925 to 0.975, failed quotations to zero, and an honest qualification of that number.

- **Failure 38: computed lines that said more than was computed** - "the co-payment applies, so this claim is paid at 80%" for a refused claim, and a room cap ruled out without its reason.
- **Fix: a misfiled age, checked against its source words**, and **Failure 39: a prediction made from stale data** - the check found two "I am 55" misreadings in the eval nobody had noticed.
- **When the answer key is wrong** - two expected answers that contradicted the file's own rule, changed only by the project owner and recorded inside the cases.
- **Concept 35: a lookup reported as a lookup** - shared unusual words as a hint, not retrieval: every clause stays in the prompt. A noisy first version hinted at 38 of 40 questions; the final one at 8, all correct, with the fitting caveat stated.
- **A missing annexure, checked only where an answer leans on it** - and the refinement after it broke a correct answer.
- **Failure 40: a design promise the code never kept** - the documented quote retry did not exist; built, it fixed 0 of 2; replaced by trimming appended words, with a guard against rescuing inventions.
- **Failure 41: the test question inside the prompt** - the reasoning prompt's example is an eval case. Replacing it broke 8 cases, so the headline score depends on it. Open.

### M14 - A score on questions it has never seen
Held-out scenario batches, and what they said: about 0.83-0.85 on unseen questions against 0.95 on the tuned set. **If you only read one entry about evaluation, read Concept 36.**

- **Concept 36: a score on the cases you tuned against** - hundreds of reasonable keep-or-revert decisions fit a system to its test set. The first held-out batch scored 0.750 against 0.975. Once held-out failures are read, the batch is no longer held out; a second batch was written before it was run.
- **Failure 42: "unknown is the honest answer", and it answered nothing else** - one sentence stopped the fact extractor reading. The fix improved the extractor and cost six main-set cases, so it was reverted: the end-to-end measurement decides.
- **A missing list, finished** - an effect label that slipped past a check, a downgrade when the retry rests on nothing checkable, and a refusal that names no refusing clause.
- **Failure 43: a retry that talked right answers out of themselves** - turned into a filter that removes an unsupported citation and can never change a verdict.
- **Failure 44: a fallback that destroyed a corrected answer** - the retry fixed the verdict and not the label, and the safety fallback threw the answer away.
- **Failure 45: blaming the last change for a drop that was sampling** - check a change can reach the cases that moved before attributing the movement to it.
- **Failure 46: a report that misdescribed its own run** - a commit stamp that could not say the tree was dirty, and a generated sentence describing a cache key M10 had removed. Plus a check of the fix that ran the old code: setting up a clean tree had removed the change under test.

---

## Related documents

- **[../DESIGN.md](../DESIGN.md)** - the design system: aesthetic direction, the four-face typography with its semantic rule, the colour tokens, and the list of forbidden anti-patterns.
- **[../CLAUDE.md](../CLAUDE.md)** — the architecture, the deliberate omissions (no vector DB, no task queue, no LLM in the scoring path), and the conventions this project follows.
- **[../evals/REPORT.md](../evals/REPORT.md)** - the consolidated results, stamped with the model, prompt version, decoding settings and commit that produced them. Regenerate with `python evals/run_all.py`.
- **[../evals/](../evals/)** — the evaluation harness. Model and prompt changes in this project are justified by measurements, not by impressions.

---

## Reading the code alongside the log

The pipeline runs in stages, and each file's module docstring explains why that stage exists in the form it does:

| Stage | File | LLM? |
|---|---|---|
| 1. Ingest | `api/app/pipeline/ingest.py` | No |
| 2. Segment | `api/app/pipeline/segment.py` | No |
| 3. Analyze | `api/app/pipeline/analyze.py` | Yes |
| 4. Score | `api/app/pipeline/score.py` | No |
| 5. Scenario | `api/app/pipeline/scenario.py` | Yes |

The governing principle: **deterministic wherever possible, a model only where language understanding is genuinely required.** That is what makes a small local model reliable enough to build on.

Supporting modules: `app/llm/client.py` (constrained JSON calls), `app/llm/cache.py`, `app/llm/prompts.py`, `app/taxonomy.py` (the shared vocabulary), `app/models.py` (database tables).
