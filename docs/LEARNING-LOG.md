# Learning Log

A running record of what was built, why, and what broke along the way. Written so that reading it top to bottom teaches this codebase from scratch.

---

# M0 — Foundations: talking to a local model reliably

## What M0 had to prove

Before any PDF parsing or UI, one question had to be answered: **can a 7-billion parameter model running on a laptop GPU be trusted to produce structured output we can build a pipeline on?** If not, the whole architecture changes.

Answer: yes, but only with two mechanisms in place — and the second one was not obvious until it failed.

---

## Concept 1: Constrained decoding

**The problem it solves.** The usual way to get JSON from a model is to ask nicely — "respond only with valid JSON" — then defend yourself: strip markdown fences, catch parse errors, retry when it apologises instead of answering. You are *hoping*, and hope fails a few percent of the time. At 200 clauses per document, a few percent is several failures per run.

**How it actually works.** A language model generates one token at a time. At each step it produces a probability distribution over the whole vocabulary and samples one token. Constrained decoding inserts a filter into that loop: given the JSON Schema and the partial output so far, it computes which tokens *could* legally come next, and sets every other token's probability to zero before sampling.

Say the schema is `{"clause_type": {"enum": ["coverage", "exclusion", ...]}}`. After emitting `{"clause_type": "`, the only tokens with non-zero probability are those that begin one of your seven values. The model cannot write `"Limitation"` — not "is discouraged from", *cannot*. There is no path through the sampler that produces it.

**Why this is a big deal.** It converts a prompt-engineering problem into a type system. `json.loads()` on the result cannot fail on malformed syntax. In `app/llm/client.py` this is the entire `format` parameter:

```python
"format": schema,   # a grammar constraint, not a hint
```

**Where we cash this in later.** For scenario citations, the schema's clause-ID enum will be built at runtime from exactly the clauses in that prompt. A hallucinated citation becomes *unrepresentable*. Most RAG systems detect bad citations after generation and retry; this makes them impossible to generate in the first place.

**The proof, run before any code was written:**

| Schema | Model output |
|---|---|
| `clause_type` as plain string | `"Limitation"` — invented, not in our taxonomy |
| `clause_type` with `enum` | `"exclusion"` — forced into our vocabulary |

---

## Concept 2: What constrained decoding does NOT give you

**This is the real lesson of M0, and it arrived as a failure.**

With the enum in place, I ran four real clauses through a bare prompt — *"You classify Indian health insurance policy clauses"* — and nothing else.

| Clause | Model said | Correct answer |
|---|---|---|
| "No claim payable for pre-existing disease during the first 36 months" | `exclusion` | `waiting_period` |
| "Room rent limited to 1% of Sum Insured per day" | **`coverage`** | `sub_limit` |
| "Notice must be given within 24 hours, failing which the claim may be repudiated" | **`coverage`** | `condition` |

**Every answer was a valid member of the enum. Two were badly wrong.**

Worse, look at the *direction* of the errors. A clause that quietly caps your payout, and a clause that can void your claim entirely, were both filed under `coverage` — the reassuring bucket, the one a user scrolls past. Our own classifier reproduced the exact harm this project exists to prevent.

**Root cause.** The enum constrains the *shape* of the answer. It says nothing about the *meaning* of the labels. The model saw seven plausible English words and picked by vibe. Nothing had told it that in this system `sub_limit` means "caps a payout" and `condition` means "a duty whose breach voids the claim". And these categories genuinely overlap — a room-rent cap *is* describing coverage; it just happens to be describing its ceiling.

> **Constrained decoding guarantees the shape of an answer, never the judgment behind it.** This generalises far beyond this project.

**The fix** — entirely in the prompt, no code change. `app/llm/prompts.py` now defines each label in the system's own terms, and adds ordered tie-breakers for the overlaps:

```
1. Does it cap or reduce an amount?             -> sub_limit
2. Does coverage start after a stated period?   -> waiting_period
3. Does it impose a duty that can void a claim? -> condition
4. Is it a permanent bar on payment?            -> exclusion
...
```

"The more specific label always wins" is what resolves the overlap: the cap beats the coverage it is capping.

**Result on 8 clauses spanning all 7 types: 8/8 correct.** Same model, same enum, same code — the entire difference was telling it what our words mean.

---

## Concept 3: Content-addressed caching

Analysing a 200-clause policy takes 2–5 minutes. Without a cache, every prompt tweak costs a full re-run just to see the effect on the few clauses you changed.

`app/llm/cache.py` keys each response on a SHA-256 of **everything that could change the answer**: model, prompt version, messages, and schema.

**Why hash all four.** The nastiest bug in an LLM pipeline is editing a prompt, seeing no change, and concluding the model ignored you — when in fact a stale cache entry was served. Making the prompt version part of the key makes that impossible by construction: change the prompt, and it is a different key. There is no "remember to clear the cache" step to forget. `PROMPT_VERSION` is already at `v2-typed-definitions` because of the fix above.

One subtlety worth internalising, from `make_key`:

```python
json.dumps({...}, sort_keys=True)
```

Python dicts preserve insertion order, so two logically identical schemas built with their fields in a different order would serialise differently, hash differently, and silently miss the cache forever. `sort_keys=True` is not tidiness — it is correctness.

The cache lives in its own SQLite file so `rm data/llm_cache.db` resets model output without touching uploaded documents.

---

## Measured facts about this machine

Everything in the architecture follows from these, gathered before writing any code:

| | |
|---|---|
| Hardware | RTX 4060 Laptop, 8GB VRAM, 15.7GB RAM |
| Model | `qwen2.5:7b-instruct-q4_K_M`, 4.7GB — fits fully in VRAM |
| Cold start | ~27s to load into VRAM |
| Warm throughput | ~50 tokens/sec |
| Batched | 3 clauses in 4.4s ≈ 1.5s/clause |

The cold start is why `client.warm()` runs in the FastAPI `lifespan` hook: pay the 27 seconds at startup, not during a user's first upload where it looks like a crash.

`temperature=0.0` throughout is not decoration. Extraction must be reproducible, or both the cache and the eval numbers become meaningless.

---

## What broke, beyond the classification failure

**PowerShell mangled a here-string.** Piping a multi-line Python script into `python -c` via a PowerShell `@'...'@` here-string produced `SyntaxError: '{' was never closed` — the quoting was destroyed in transit. Fix: write the script to a file and run the file. Faster than debugging shell quoting, and the script sticks around to re-run.

**`ModuleNotFoundError: No module named 'app'`.** That script then lived in a temp directory, so Python's import path had no idea where `app/` was. Python resolves imports from `sys.path`, which is seeded from *the script's own directory* — not your shell's working directory. Fix: `PYTHONPATH=.../api`. The same issue in tests is handled by `pythonpath = .` in `pytest.ini`.

**A bash heredoc choked on this very file.** Writing long markdown through `cat <<'EOF'` failed with `unexpected EOF while looking for matching quote`. Prose full of backticks, quotes and apostrophes is worth writing with a real file-writing tool rather than fighting shell quoting.

---

## Files M0 produced

| File | Role |
|---|---|
| `app/config.py` | Every tunable in one place, so evals can swap models without touching logic |
| `app/taxonomy.py` | The 7 clause types + verdicts. One definition shared by DB, LLM schemas, and UI |
| `app/llm/client.py` | Constrained-JSON calls, retries, caching |
| `app/llm/cache.py` | Content-addressed response cache |
| `app/llm/prompts.py` | Prompts + `PROMPT_VERSION` |
| `app/models.py` | Four SQLModel tables |
| `app/db.py` | Engine and per-request sessions |
| `app/main.py` | FastAPI app, CORS, `/health`, model warming |
| `tests/test_llm.py` | The M0 gate — 5 tests |

**Why `Clause` and `ClauseAnalysis` are separate tables:** parsing is deterministic and done once; analysis is LLM-driven and re-run constantly while tuning prompts. Separating them means re-analysis can never corrupt parsed structure, and one document can hold results from two prompt versions side by side for comparison.

**Why `/health` reports more than `{"ok": true}`:** the most common failure in this stack is Ollama running but the expected model not pulled. `/health` surfaces that explicitly rather than letting it fail deep inside the pipeline where the error is unrecognisable.

---

## Check it yourself

```bash
cd api

# The M0 gate. Note it hits real Ollama, not a mock: the claim being tested
# is a claim about Ollama's behaviour, so a mock would only test our belief.
.venv/Scripts/python.exe -m pytest -v

# Only the tests that need no model
.venv/Scripts/python.exe -m pytest -m "not llm"

# Boot the API, then open http://localhost:8000/health and /docs
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

**Question to sit with before M1:** `test_schema_forces_a_value_even_for_an_unrelated_clause` feeds the classifier *"The quick brown fox jumps over the lazy dog"* and asserts it still returns a valid clause type. It returned `procedural`.

Why is that the *correct* behaviour to assert — and what does it tell you about why `Verdict` in `taxonomy.py` needs an explicit `insufficient_information` member?

<details>
<summary>Answer</summary>

Constrained decoding **cannot abstain**. The grammar permits only the seven values, so "I don't know" and "this isn't an insurance clause" are literally unsayable. The model must pick one, and it will.

So abstention only exists if you *build it into the vocabulary*. That is why `Verdict` carries `insufficient_information` as a first-class member: without it, the scenario simulator would be structurally incapable of saying "your policy doesn't address this", and would be forced to guess `covered` or `not_covered` on a question the document never answers.

In this domain that distinction is the whole ballgame — a confident wrong answer costs someone a real claim.
</details>

---

# M1 — Reading the PDF: text, offsets, and clause boundaries

## What M1 had to produce

Two things, from a policy PDF:

1. **The document's text**, with a guarantee strong enough that any later citation can be proven rather than trusted.
2. **Clause boundaries** — the document split into the ~40 individual provisions that a policy is actually made of.

Neither stage uses a language model. That is a deliberate design decision, explained below.

---

## Concept 4: The offset invariant

`api/app/pipeline/ingest.py` produces one `raw_text` string for the whole document, plus a list of `Line` objects each carrying `char_start` and `char_end` offsets into it. The guarantee is:

```python
raw_text[line.char_start : line.char_end] == line.text
```

for every line, always. Clauses inherit those offsets in stage 2.

**Why bother, instead of just storing each clause's text on its own?**

Because of what the app ultimately claims to a user. "The model said this, and here is our copy of some text" is weak — a copy can drift from its source after any edit, and nothing detects it. "The model said this, and the same characters sit at offset 14,203 of the document we actually parsed" is a different kind of claim. A slice cannot drift from what it is a slice of.

This is the foundation the entire grounding story rests on. `tests/test_ingest.py::test_every_line_slices_back_byte_identical` asserts it for every line of the test policy, and the docstring there says why it matters more than it might look:

> If this ever fails, the app can still render confident explanations that point at the wrong part of the document — which is worse than not working at all, because it looks like it works.

There is a second, subtler test — `test_offsets_are_ordered_and_non_overlapping`. Offsets could each slice back correctly while still being out of order. Then "clause A comes before clause B" would be wrong, and with it the document-position component of the buriedness score in M2.

The invariant is upheld by one easily-missed detail in `ingest.py`:

```python
cursor += len(text) + 1   # +1 for the "\n" that joins lines in raw_text
```

That `+ 1` must stay in lockstep with the `"\n".join(chunks)` that builds `raw_text`. Change one without the other and every offset after the first line is wrong.

---

## Concept 5: Why segmentation must not use an LLM

It is tempting to hand the whole document to a model: *"split this into clauses."* That fails on four counts:

| | |
|---|---|
| **Reproducible** | Two runs give different boundaries, so every downstream number — clause counts, scores, eval metrics — becomes noise. |
| **Verifiable** | A model that paraphrases while splitting destroys the character offsets above. |
| **Fast** | A 12,000-character policy would be re-read on every single run. |
| **Debuggable** | When a boundary is wrong you can step through a rule. You cannot step through a forward pass. |

And it is unnecessary, because **the layout already encodes the structure**. A policy is typeset with headings in larger, bolder type, and clauses under numbers like "4.2", precisely so a human can navigate it. Reading that is a parsing problem, not a language problem.

`api/app/pipeline/segment.py` classifies each line as a SECTION heading, a CLAUSE heading, or BODY, using font size relative to the document's own body size, clause numbering, and known section names.

**"Relative to the document's own body size" is the important part.** `IngestResult.body_size()` measures the most common font size in *that* document, so a policy set in 8pt and one set in 11pt both work with the same thresholds. It is a mode, not a maximum, and weighted by character count rather than line count — headings are numerous but short, so counting lines could elect a heading size as the "body" size and invert every threshold in the file.

---

## Getting a document to test against

Real insurer policy wordings are unsuitable for testing: redistributing copyrighted wording would make the repo non-self-contained, and nobody has labelled them.

So `evals/golden/build_synthetic_policy.py` **generates** a 39-clause IRDAI-style policy, spanning all seven clause types, and emits a label file alongside the PDF. Because the PDF and the labels come from the same `CLAUSES` list, they can never disagree — there is no separate labelling step to drift out of sync.

The clauses deliberately include the cases that break naive classifiers: a waiting period phrased to sound like an exclusion, a sub-limit phrased to sound like a benefit, and a `"Hospital"` definition whose narrowness quietly guts coverage elsewhere.

---

## Failure 1: the circular test

**Everything above passed on the first run — 14/14 tests, in 0.19 seconds.**

That result was close to worthless, and it is worth understanding why.

The segmenter's font-size thresholds were *tuned against the very PDF the tests run on*, and that PDF is generated by this repo. Passing proved the segmenter handles documents we designed to be easy. It said nothing about real ones.

So `build_hostile()` was added: **the same 39 clauses, typeset with one font, one size, no bold, and the clause number run inline with the body** — the way many machine-generated policy PDFs actually arrive. Every styling signal removed.

Result on the hostile document:

```
segments: 8      expected: 39
MISSING: 1.1, 1.2, 1.3, ... 7.5     (all 39)
```

**Total collapse.** Only the 7 section headings and one front-matter block survived. Every clause had been silently swallowed into its section.

**Root cause.** The boundary rule read:

```python
is_clause_start = number is not None and (
    line.bold                                        # False - nothing is bold
    or line.size >= body_size * CLAUSE_SIZE_RATIO    # False - one uniform size
    or len(line.text) <= HEADING_MAX_CHARS           # False - number is inline
)
```

The numbering was detected perfectly. But I had written styling as a **requirement** when it should only ever have been **corroboration**. A line beginning "4.2" in a policy *is* a clause boundary, whatever font it is in.

**Why I couldn't simply delete the guard.** The numbering regex `^(\d+(?:\.\d+)*)\.?\s+` also matches ordinary prose: insurance text is full of lines starting `"24 hours of hospitalisation"`, `"15 days from the date of discharge"`, `"1.5 lakhs and fifteen in-patient beds"`. Treating those as boundaries would shatter clauses into fragments.

So the guard belonged on the **numbering pattern**, not on the font. `_numbering()` now returns `(number, is_strong)`:

- **Strong** — believed with no styling support: `(a)`, `(iv)`, `Clause 4.2`, or a dotted number followed by a **capital letter**.
- **Weak** — needs styling to corroborate: a bare integer, or a dotted number followed by lower case.

The capital-letter test is what separates `"4.2 Room Rent Limit"` from `"1.5 lakhs and fifteen"`. Clause numbers are followed by a heading or a new sentence, so the next character is upper case; a measurement is followed by its unit in lower case.

**The lesson:** *a test suite that only runs on data you designed proves your code handles data you designed.* The hostile document is now a permanent fixture, and `test_hostile_document_really_has_no_styling` guards the guard — if the generator ever started emitting bold text, the regression tests would keep passing for the wrong reason.

---

## Failure 2: a data model that hid the fix

After the fix, the hostile document produced 41 segments — the boundaries were being found. But the check still reported all 39 clauses missing.

The check identified clauses by parsing the first word of `Segment.heading`. In the hostile document there *are* no headings: the number runs inline with the body, so `heading` is empty by design.

The segmentation was correct; the **data model** was wrong. A clause's number is its stable identifier — the heading is optional decoration that many policies simply do not have. So `number` became a first-class field on `Segment`, rather than something re-parsed out of a heading string.

Worth noticing: this bug was only visible *because* of the hostile document. On the styled PDF, heading-parsing worked fine and the design flaw was invisible.

---

## Failure 3: a bug you cannot see

With `number` in place, one test still failed: only 6 of 7 sections were found. `SECTION 7 - GENERAL PROVISIONS` was missing.

The unstyled-document fallback matched ALL-CAPS lines against a vocabulary list, and that list contained `"general condition"` but not `"provisions"`. The heading fell through, was parsed as a *clause*, and every clause beneath it was filed under Section 6.

**The real problem was the approach, not the missing word.** A vocabulary list can only ever recognise headings someone remembered to add. So the primary signal became **structural** — `SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE` followed by a number — which recognises the *shape* of a heading. The word list stayed as a secondary fallback.

Then the tests passed, and the fix was still broken.

```
regex compiled OK: ^(?:SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE)\s+[\dIVXL]+
  SECTION 7 - GENERAL PROVISIONS   -> False
```

The pattern printed perfectly and matched nothing. The tests only passed because I had *also* added `"provision"` to the word list, so the fallback caught it — **the new code was dead, and a passing suite was hiding that.**

A hex dump of the line found it:

```
457c 414e 4e45 5855 5245 295c 732b 5b5c   E|ANNEXURE)\s+[\
6449 5658 4c5d 2b08 2229 0a               dIVXL]+.").
                    ^^
```

Byte `08` — a literal **backspace character**, where `\b` (regex word boundary) was intended.

**Root cause.** I had written that line through a bash heredoc into a `python -c` script. One level of backslash escaping was consumed in transit, so Python received `\b` inside a *non-raw* string and parsed it as the backspace escape, embedding control character 0x08 into the pattern. The regex then required a literal backspace in its input, which no text contains.

The giveaway had been there all along and I skimmed past it: `SyntaxWarning: invalid escape sequence '\s'`. Python was reporting that it had seen single backslashes where I intended doubles.

**Three lessons, all of which generalise:**

1. **Don't push regexes through nested shell + Python string layers.** Every layer silently consumes escaping. Write the file directly.
2. **A passing test suite is not proof your new code runs.** Here an old code path masked a completely inert new one. If you add a mechanism, check the *mechanism* works — not just that the outcome is right.
3. **Read the warnings.** `invalid escape sequence` named the exact problem before I went looking for it.

`test_no_regex_contains_a_control_character` now scans every pattern in the module, because this failure class is genuinely invisible in a diff, in a terminal, and in a code review.

---

## Where M1 ended up

| | Styled PDF | Unstyled PDF |
|---|---|---|
| Clauses found | 39/39 | 39/39 |
| Sections found | 8 | 7 |
| Offset mismatches | 0 | 0 |

**26 tests passing** (21 need no model; 5 need Ollama).

Files added: `app/pipeline/ingest.py`, `app/pipeline/segment.py`, `evals/golden/build_synthetic_policy.py`, `tests/conftest.py`, `tests/test_ingest.py`, `tests/test_segment.py`.

---

## Check it yourself

```bash
cd api

# Everything, including the hostile-document regression tests
.venv/Scripts/python.exe -m pytest -v

# Regenerate both test PDFs from source
.venv/Scripts/python.exe ../evals/golden/build_synthetic_policy.py
```

Open `evals/golden/synthetic-hostile-policy.pdf` next to `synthetic-health-policy.pdf`. They contain identical text; only the typesetting differs. The segmenter now gets the same 39 clauses out of both.

**Question to sit with before M2:** `test_finds_every_numbered_clause` checks that no clause was *missed*. `test_does_not_over_split` checks that clauses were not *shattered into fragments*. Both matter — but if you had to weaken one, which should it be?

<details>
<summary>Answer</summary>

Weaken **over-splitting** (precision). Prefer to keep recall.

An over-split clause is still analysed, still scored, and still shown to the user. It appears as two entries instead of one — untidy, mildly confusing, but everything is on screen.

A **missed** clause is invisible for the rest of the pipeline's life. It is never classified, never scored, never displayed, and never cited. Nothing downstream can recover it, and nothing signals its absence.

Now consider *which* clause. If the one lost happens to be an exclusion or a notice condition, the app confidently shows a user a policy that looks safer than it is — the precise harm this project exists to prevent, delivered with a clean UI and full confidence.

That asymmetry is why `_numbering()` leans toward accepting a boundary, and why the hostile-document tests are all about recall.
</details>

---

# M2 — Analysis and scoring: measuring instead of guessing

## What M2 had to produce

Two stages, and one habit.

**Stage 3 (`api/app/pipeline/analyze.py`)** asks the model, for each clause: what type is it, what does it mean in plain English, and two judgments — how **likely** is it to affect a typical policyholder, and how **severe** is the consequence.

**Stage 4 (`api/app/pipeline/score.py`)** combines those two judgments with signals it computes itself into a single impact score, and ranks the clauses.

The habit is the important part: **from here on, decisions in this project are settled by measurement.** `evals/run_eval.py` exists so that a prompt edit, a model swap, or a batch-size change is judged by a number rather than by looking at a few outputs and forming an impression. Impressions are formed on exactly the sample where a plausible-sounding wrong answer is most convincing.

---

## Concept 6: Splitting judgment from arithmetic

The impact score has two halves, and which half does what is the whole design:

| | Source | Why there |
|---|---|---|
| `likelihood`, `severity` | The model | Judging that a 20% senior co-payment is financially significant needs language understanding |
| `buriedness` | Computed | Knowing a clause sits 78% through a document, references three others, and reads at grade 17 is arithmetic |

A 7B model asked to do the second would produce numbers that look plausible, vary between runs, and cannot be checked. Arithmetic done by a language model is arithmetic you cannot trust or reproduce.

The formula:

```
impact = 100 * type_weight * blend * (1 + BURIEDNESS_BOOST * buriedness)
                                   / (1 + BURIEDNESS_BOOST)

blend = 0.5*norm(likelihood) + 0.5*norm(severity),  norm(x) = (x-1)/4
```

Two deliberate choices:

**Buriedness multiplies, never adds.** A clearly written, prominently placed exclusion is still an exclusion. Being buried should make a *consequential* clause worse; it must not make a trivial clause important. Multiplication preserves that ordering, addition would not.

**Dividing by `(1 + BURIEDNESS_BOOST)` rather than clamping at 100.** Clamping would collapse several genuinely different clauses onto exactly 100, destroying the ordering at the very top of the list — which is the only part anyone reads.

### Measuring "obscuring"

The problem statement says policies *obscure* the clauses that decide claims. `buriedness` makes that a number, from four computed signals:

| Signal | Weight | What it captures |
|---|---|---|
| Position in document | 0.20 | Readers give up; drafters put unwelcome parts after appealing ones |
| Flesch–Kincaid grade | 0.35 | Long sentences built from Latinate words — "indemnify", "repudiate" |
| Cross-references | 0.25 | Every "as defined in Annexure III" is a hop, and a chance to give up |
| Defined-term dependence | 0.20 | A clause you cannot understand where it sits |

Flesch–Kincaid is `0.39*(words/sentence) + 11.8*(syllables/word) - 15.59`. Legal drafting scores high on *both* terms at once, which is why difficulty carries the heaviest weight. Syllables are counted by a vowel-group heuristic — wrong on individual words, but averaged across a clause, and used only to rank clauses against each other, so a consistent small bias changes no ordering.

The defined-term signal is derived **per document**, from that policy's own definition clauses, not from a fixed word list. Which terms are load-bearing depends on what *this* policy chose to define.

---

## Failure 4: a silent empty array

The first full eval run reported `39/40 clauses analysed` and logged `UNANALYSED CLAUSES: [0]`.

Clause 0 is the document's front matter — title, UIN, disclaimer. Investigating, the individual retry did not throw an error. It **succeeded**, returning `{}`.

The model had replied `{"clauses": []}` — an empty array. Faced with text that is not a clause, it declined to produce an entry.

**That response is completely valid under the schema.** The schema constrained the *shape of each element*; it said nothing about *how many elements* the array must contain. Constrained decoding had done exactly what was asked, and a clause vanished from the pipeline with no error anywhere.

The fix is to make cardinality part of the grammar too:

```python
"clauses": {
    "type": "array",
    "minItems": len(ids),
    "maxItems": len(ids),
    ...
}
```

Verified enforced: given a prompt insisting there were no clauses at all, the model still emitted exactly three entries.

Forcing an entry for genuinely non-clause text is the right trade. It lands as `procedural` with low ratings and scores near zero — harmless. A silently dropped clause is invisible for the rest of the pipeline's life, which is precisely the failure this project exists to prevent.

> **The refinement of Concept 1:** constrained decoding guarantees exactly what you encode in the schema, and *nothing you forgot to encode*. "Every element is well-formed" and "the right number of elements exist" are separate guarantees, and you get only the ones you ask for.

---

## Failure 5: a performance optimisation that was never measured

The module was written to send clauses in **batches of 5**, with a confident docstring explaining that batching amortises the long system prompt across several clauses and is therefore faster.

That reasoning sounded obvious. It was wrong.

| Batch size | Macro-F1 | Wall time |
|---|---|---|
| 5 | 0.973 | 121.9s |
| **1** | **1.000** | 128.5s |

Batching bought about 5% wall time and cost a clause.

**Why the speed argument failed:** Ollama already caches the repeated system-prompt prefix between calls, so re-sending it is nearly free. The saving being optimised for did not exist.

**Why the accuracy loss is the more interesting half:** the clause batching got wrong — 5.2 "Proportionate Deduction", a `sub_limit` read as a `condition` — classifies **correctly when sent on its own**. Nothing about the clause was hard. It was the company it kept. Sharing a generation lets the model's reading of one clause bleed into the next, and clauses in a policy sit next to each other precisely *because* they are related, which makes the interference worse rather than better.

The default is now 1. Batching is still supported, because the wall-time picture changes for a much larger document.

> **A performance argument that has not been measured is a guess, however obvious it sounds.** This is the entire reason `evals/` exists.

---

## Failure 6: the test that caught unbounded arithmetic

A unit test asserting scores stay within 0–100 failed with `impact_score=462.44` and `position_signal=99.0`.

Position was computed as `seg.order_idx / len(segments)`. That silently assumed `order_idx` values always span `0..len-1`. Scoring any *subset* — as a unit test does, or as a re-scoring pass would — produced ratios above 1 and impact scores in the hundreds.

The fix computes position from the clause's **rank within the list actually passed in**, which is bounded by construction:

```python
for rank, seg in enumerate(segments):
    position = rank / max(total - 1, 1)
```

Worth noticing: the full pipeline would never have exposed this, because there `order_idx` and list position always coincide. The bug was only reachable through a test that used the function differently than production does — which is a large part of what unit tests are *for*.

---

## Concept 7: Measuring the right thing

The classification eval came out at **macro-F1 1.000** — every one of the 39 clauses correctly typed, across all seven types.

Macro-F1 rather than accuracy, deliberately: the golden policy has 8 exclusions but only 4 waiting periods, and accuracy lets a model coast on the common types. Macro-F1 weights every type equally, because a missed waiting period misleads a policyholder just as badly as a missed exclusion.

A perfect score is a good moment to be suspicious. So: **what does this number not cover?**

It measures *labelling*. The product is a *ranking*. Those are different, and only the first was being measured.

Inspecting the actual top 10 confirmed the gap. The room-rent cap sat at rank 14 and the senior-citizen co-payment at rank 22 — the two most notorious causes of unexpected shortfalls in Indian health insurance, both below the fold. Meanwhile rank 3 was "Medical Examination": a clause letting the insurer require an examination **at its own expense**, which costs the policyholder nothing.

Classification F1 reported all of this as flawless, because every one of those clauses was labelled correctly.

### Measuring it

`evals/golden/ranking-expectations.json` names five clauses that must appear in the top 10 and four that must not, each with a written justification, authored from how claims actually go wrong — and written *before* looking at any ranking, so the metric tests the system rather than rationalising its output.

Baseline: **5/9**.

### Two fixes, both principled

**1. Severity anchors (prompt).** The model had rated "Medical Examination" severity **5** — the same as a claim being refused outright. The severity scale was being read as a measure of a clause's *tone* rather than its cost. Added explicit anchors: *a power the Company exercises at its own expense is severity 1, however formally written*; *a cap that cascades across the whole bill is 4–5, not 3*.

→ 5/9 to **6/9**. Classification held at 1.000.

**2. Sub-limit weight (formula).** `TYPE_WEIGHT[sub_limit]` was 0.70 against `condition` at 0.95, so a sub-limit could never outrank a condition with equal ratings.

The original weighting reflected the *worst case*. But conditions and exclusions are **contingent** — a notice deadline costs everything, but only if you miss it. A sub-limit is **certain**: a room-rent cap applies to every claim, automatically, forever. On expected loss rather than worst case, sub-limits were systematically understated. Raised to 0.85, staying below `condition` because total repudiation is still the worse tail, and tail risk is what people are least able to absorb.

→ still **6/9**, but a *different* 6/9. Room rent and proportionate deduction moved to ranks 7 and 5; the pre-existing disease waiting period slipped to 11.

### Knowing when to stop

That last result is the useful one. The score held while the composition shifted, meaning changes had started **trading one expectation against another**. That is the signature of overfitting to 9 hand-authored expectations on a single synthetic document.

So tuning stopped. The ranking is materially better than at baseline — both room-rent clauses now surface, which is what a real policyholder most needs — and the honest position is that further progress needs a **larger, more diverse eval set with real policies**, not more knob-turning against this one.

> Ending at 6/9 with a documented reason is worth more than 9/9 reached by fitting the metric. The second number would not survive contact with a real policy, and there would be no way to know.

---

## Where M2 ended up

| Metric | Result |
|---|---|
| Classification macro-F1 | **1.000** (39/39, all 7 types) |
| Ranking expectations | **6/9** (from 5/9 baseline) |
| Clauses analysed | 40/40 |
| Analysis time | ~128s for 40 clauses, local 7B |
| Tests | **38 passing** |

Files added: `app/pipeline/analyze.py`, `app/pipeline/score.py`, `evals/run_eval.py`, `evals/golden/ranking-expectations.json`, `tests/test_score.py`.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -v

# Full eval; writes evals/report.md
python evals/run_eval.py

# Reproduce the batching finding for yourself
python evals/run_eval.py --batch-size 5 --no-report
python evals/run_eval.py --batch-size 1 --no-report
```

The second run of any eval takes almost no time — the content-addressed cache from M0 serves every response. Edit a prompt and bump `PROMPT_VERSION`, and it recomputes; forget to bump it, and you would silently measure the old prompt. That is why the version is part of the cache key.

**Question to sit with before M3:** the classification eval scores 1.000 on a 39-clause policy that this repository generated. Name two distinct reasons that number will not hold on a real insurer's policy wording.

<details>
<summary>Answer</summary>

**1. The clauses were authored to be classifiable.** Each one was written as a clean example of a single type, with the boundaries the taxonomy assumes. Real policy clauses are frequently hybrids — a coverage grant with an embedded cap, a proviso, and a cross-reference to an annexure, all in one sentence. The tie-breaker list decides those cases, but "which label is *most* right" is a genuinely harder judgment than the golden set ever asks for.

**2. The labels and the document share an author.** Both come from the same `CLAUSES` list, so the label reflects the intent the text was written to express. On a real policy, someone must read ambiguous legal language and decide what it *means* — and two insurance lawyers would not always agree. A benchmark where the ground truth was inferred, rather than declared, is inherently noisier.

A third, worth knowing: **39 clauses is a small sample.** A single misclassification moves macro-F1 by roughly 0.03, so the difference between 1.000 and 0.97 is one clause and well inside run-to-run noise.

This is why M2 ends with "the ranking needs a real eval set" rather than with a victory lap.
</details>

---

# M3 — The API, and a bug tests could not see

## What M3 had to produce

An HTTP surface over the pipeline: upload a PDF, watch it being processed, read the ranked results.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a policy PDF; returns `202 Accepted` |
| `GET` | `/documents/{id}/status` | Stage and progress, for polling |
| `GET` | `/documents/{id}/summary` | The risk dashboard |
| `GET` | `/documents/{id}/clauses` | All clauses, filterable and sortable |
| `GET` | `/clauses/{id}` | One clause, with verbatim source text |

---

## Concept 8: Why the upload returns 202, not 200

Analysis takes about two minutes on a local model. An HTTP request cannot politely stay open that long — browsers, proxies and load balancers all time out — and even if it could, the user would be staring at a blank tab.

So `POST /documents` returns **`202 Accepted`**, whose specific meaning is *"the work has been queued, not completed."* Its body deliberately contains no results. The client then polls `/status`.

The work runs in FastAPI's `BackgroundTasks`, which the plan chose over Celery or a job queue: this is a single-user local app, and a queue would be infrastructure with no reader.

**Progress is weighted by time, not by stage count.** Ingest and segment finish in milliseconds; analysis is essentially all of the wall time. Giving each of four stages 25% would produce a bar that leaps to 50% instantly and then appears frozen for two minutes. Instead analysis owns 10%→95%, and the live run reports genuinely smooth movement:

```
[10.0%] analyzing   Reading 40 clauses
[12.1%] analyzing   Read 1 of 40 clauses
...
[92.9%] analyzing   Read 39 of 40 clauses
[100.0%] ready      Analysed 40 of 40 clauses
```

### One detail that is easy to get wrong

`_update()` in `api/app/pipeline/run.py` opens its **own short-lived session** for each status write, rather than reusing one session held across the whole run:

```python
def _update(doc_id: str, **fields) -> None:
    with Session(engine) as session:
        ...
        session.commit()
```

The polling endpoint reads that row from a *different* connection, and it can only observe changes that have actually been **committed**. Held open in one long transaction, the writes would be invisible to the poller and the progress bar would sit at 0% until the very end.

### Failing without a listener

`process_document` never raises. It runs as a background task with nobody awaiting it, so an escaping exception would vanish into the event loop and leave the document reporting "analyzing" forever. Failures are written to the row instead, where the polling UI can surface them — verified by `test_a_corrupt_pdf_fails_the_document_not_the_server`.

---

## Failure 7: the bug the test suite could not have caught

All 57 tests passed. The API worked under `TestClient`. So as a last check I ran a **live uvicorn server** and drove it over real HTTP.

```
POST /documents -> 202 in 0.23s  id=c04963489b6f
polling /status:
  [ 0.0%] failed      Processing failed
```

```
OperationalError: table clause has no column named number
```

**Root cause.** `init_db()` calls SQLAlchemy's `create_all`, which creates tables that do not exist. **It never alters one that does.** The development database had been created back in M0, before `Clause.number` was added in M3. `create_all` looked at it, saw the `clause` table already present, and did nothing.

**Why no test caught it.** The test fixture starts like this:

```python
SQLModel.metadata.drop_all(engine)
init_db()
```

Every test runs against a schema built moments earlier from the current models. **They never run against a schema that has aged.** The tests were not merely failing to catch this bug — they were structurally incapable of it.

> A test suite that rebuilds the world before every test is blind to every bug that only appears in a world which has been running for a while. Schema drift, stale caches, accumulated data, migrations — none of it is reachable from a fixture that starts by deleting everything.

**The fix, and what it deliberately does not do.** `check_schema()` in `api/app/db.py` now compares the live database's columns against the model metadata at startup and raises with the specifics:

```
The database schema is out of date:
  table 'clause' is missing: number
  table 'clauseanalysis' is missing: reading_grade

There are no migrations in v1. Delete data/app.db and re-upload your documents.
The LLM cache is a separate file, so re-analysis will still be served from cache
and take seconds.
```

It **raises rather than dropping the tables itself.** There is no migration path in v1 — that was a deliberate scope decision — but silently discarding someone's uploaded policies is not this function's call to make. It reports exactly what is wrong and exactly how to fix it, and leaves the decision with the person who owns the data.

Note the last line of that message. Because the LLM cache lives in a *separate* SQLite file (a decision made back in M0 for a different reason), deleting the application database costs seconds rather than minutes — every model response is still cached. A small early decision paying off somewhere unrelated.

`test_schema_drift_is_detected` builds a deliberately stale database and asserts the error names both the missing column and the remedy.

---

## Concept 9: Two kinds of test, and why they are separated

`tests/test_api.py` splits along a hard line:

| Kind | What it covers | Cost |
|---|---|---|
| **Fast** (18 tests) | Routing, filtering, sorting, validation, error handling — against a directly seeded database | ~3 seconds |
| **End-to-end** (2 tests, marked `llm`) | One real upload through all four stages, on a 4-clause policy | ~23 seconds |

The seeded fixture inserts a finished document straight into the database. That is not a shortcut — it is the correct scope. A test asserting *"filtering by `clause_type=exclusion` returns only exclusions"* is a test about the API. Routing it through a language model would make it slow, non-deterministic, and liable to fail because the model's opinion of a clause changed.

The end-to-end tests use a purpose-built **4-clause** policy rather than the 39-clause golden set, because analysis costs roughly three seconds per clause and a two-minute test is a test people stop running.

> **Tests check that it works. Evals check how well.** Keeping those separate is what lets both be honest: the tests stay fast and deterministic, and the eval is free to be slow and to report an uncomfortable number.

The end-to-end test also re-verifies the **M1 offset invariant through the entire stack** — for every clause returned over HTTP, `raw_text[char_start:char_end] == source_text`. The grounding guarantee is not something the pipeline has internally and then loses at the API boundary.

---

## Where M3 ended up

Live run against a real server, full 39-clause policy:

```
POST /documents -> 202 in 0.23s
pages=5 clauses=40 unanalysed=0
types: exclusion 8, procedural 6, coverage 6, condition 6,
       definition 5, sub_limit 5, waiting_period 4

top risks:
  1. [85.0] exclusion  4.7 Non-Medical Expenses
     "The policy will never pay for things like phone, TV, internet..."
  2. [84.7] condition  6.1 Notice of Claim
     "If you do not notify the company within 24 hours for emergencies..."
  5. [65.3] sub_limit  5.2 Proportionate Deduction
     "If you are admitted to a room that costs more than the allowed amount,
      the company will reduce the cost of your whole treatment..."
```

**58 tests passing.** Files added: `app/schemas.py`, `app/pipeline/run.py`, `app/routers/documents.py`, `tests/test_api.py`.

---

## Check it yourself

```bash
cd api
.venv/Scripts/python.exe -m pytest -q                  # everything
.venv/Scripts/python.exe -m pytest -q -m "not llm"     # fast only, ~3s

.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs> — FastAPI generates interactive documentation from the route signatures and response models, so you can upload a policy and watch the status endpoint from a browser with no frontend at all.

**Question to sit with before M4:** the `/clauses/{id}` endpoint returns `source_text`, `char_start` and `char_end` — the verbatim wording and its exact position in the document. Serving the offsets costs bytes and the UI does not currently draw anything with them. Why send them at all?

<details>
<summary>Answer</summary>

Two reasons, one immediate and one structural.

**They make the claim checkable by a third party.** `source_text` alone is *our copy* of the wording — a reader has to trust that it matches the PDF. The offsets say precisely where in the extracted document those characters live, so anyone can verify the quote against the document independently of anything this app says about it. That is the difference between showing evidence and asking to be believed.

**They are what the PDF highlighter will need.** M4 renders the plain-language and original text side by side, but the roadmap has a real PDF viewer that highlights the exact clause. Offsets and the stored bounding boxes are what make that a later drop-in rather than a data migration — the decision made in M1 to capture `bboxes` before anything used them, for the same reason.

There is a third, quieter reason: they make the invariant **testable at the boundary**. Because the API exposes the offsets, the end-to-end test can assert `raw_text[char_start:char_end] == source_text` on data that has travelled through the database and out over HTTP. An invariant you cannot observe from outside is one you cannot verify has survived the trip.
</details>

---

# M4 — The interface, and designing against a default

## What M4 had to produce

A web UI over the API: upload a policy, watch it process, read the ranked results, open any clause beside the policy's own wording.

The explicit brief was that it **must not look AI-generated**. That is a real and specific failure mode, not a vague aesthetic worry, and it is worth naming precisely because it has a recognisable signature: purple or violet gradients, glassmorphism, a centered hero with a big heading and a subtitle beneath, three-column feature grids with icons in coloured circles, uniform bubbly border-radius, gradient buttons, and Inter (or its stand-in, Space Grotesk) doing all the typographic work.

Those choices are what a model reaches for by default. Avoiding them requires having an actual idea instead.

---

## Concept 10: Find the metaphor already in your data

The idea came from asking what this product structurally *is*, rather than what category it belongs to.

It is not a dashboard, and it is not a marketing site. **It is a document that argues with another document.** That has an established form: the **critical edition** — a source text on one side, an editor's gloss on the other, with an apparatus of notes and references.

That metaphor was already latent in the data model. The API returns, for every clause, `source_text` alongside `char_start` and `char_end` — the original wording and its exact position. The design's job was to make that structural fact visible, rather than inventing a decorative theme and laying it on top.

Everything else followed:

| Decision | Because |
|---|---|
| Warm archival paper `#F7F4EE`, not white | Documents are printed on paper; clinical white is the SaaS default |
| **2px** border radius | Documents have square corners, and uniform bubble-radius is the loudest generated-design tell |
| Left-aligned masthead with a double rule | A report leads with its identity and metadata; a centered hero is a pitch |
| Borders, never decorative shadows | Structure in a document is drawn with rules |
| Minimal-functional motion only | This is an anxiety product; playful easing would be actively wrong |

The full system was written down before any screen was built, including a list of forbidden anti-patterns: gradients, glassmorphism, centered heroes, uniform bubble radius, and Inter as a typeface.

---

## Concept 11: Typography can carry meaning, not just style

This is the part of the system worth stealing for other projects.

The app has two voices: **the policy's** and **its own**. Rather than labelling them, the design gives them different typefaces:

```css
--font-sans:   "Instrument Sans"  /* everything the app says */
--font-source: "Source Serif 4"   /* the policy's own words, and nothing else */
```

A document serif for the source, a neutral sans for the translation. In `ClausePanel.tsx` the verbatim quotation carries a single `.verbatim` class, and the result is that the side-by-side needs **no "original wording" caption** — a reader can see which is which before reading a word.

That is typography doing semantic work instead of decorating. Two more faces complete the set, each with a job rather than a mood: Instrument Serif for display, JetBrains Mono for anything numeric (clause numbers, page references, impact scores, with tabular figures so columns of numbers line up for comparison).

The rule is written down as a hard constraint because it is the sort of thing that erodes silently: *the policy's words are always Source Serif 4; everything the app says is always Instrument Sans; never mix them.*

---

## Concept 12: Colour must never be the only signal

Severity uses exactly two hues:

- **Oxide red `#A8321E`** — can deny or void your claim
- **Ochre `#946A22`** — reduces what you are paid

Deliberately **not** a red/amber/green traffic light. A traffic light implies a scale from bad to good, but "reduces your payout" is not a midpoint between "denied" and "covered" — it is a **different kind of harm**. Encoding it as a middle state would misrepresent it.

And the hard rule, from the design system:

> **Severity is never carried by colour alone.** Every severity indicator pairs its hue with a numeral (the impact score) and a text label ("Can void your claim").

A risk product that fails a colourblind reader is a broken risk product. This rule caught a real bug in my own implementation, below.

---

## The design decision that changed the product

Asked what one thing a person should remember after using this, the answer chosen was:

> **"I can see what they were hiding."**

That is a design answer with an engineering consequence. `buriedness` had been a number computed in the backend, used to sort a list and otherwise invisible. If the whole promise is showing the reader what was obscured, then **hiding the reasons wastes the project's most distinctive idea.**

So M4 changed the backend. `ClauseAnalysis` now stores the four buriedness components separately rather than only their blended total, and the API exposes them, because `buriedness: 0.47` tells a reader nothing while its parts tell them exactly what happened. `clauseMeta.ts` turns them into sentences:

```
WHY YOU'D MISS THIS  buried on page 3, deep in the document ·
                     written at grade 16 reading level ·
                     sends you to other clauses to understand it
```

Thresholds are set so a typical clause produces one or two reasons rather than four. **A note that fires on everything stops being information.**

---

## Failure 8: enforcing a rule everywhere except where it mattered

The impact score lived in a right-hand rail, which was hidden below the `sm` breakpoint to make room on narrow screens:

```tsx
<div className="hidden w-[132px] shrink-0 pt-0.5 sm:block">
```

Reasonable-looking, and wrong. On a phone the severity colour and its text label survived, but **the numeral disappeared entirely** — which is precisely the rule the design system states as non-negotiable, violated by the same person who wrote it, in the same week.

It was invisible in code review and obvious in a screenshot at 430px wide.

The fix keeps the rail hidden but moves the number inline at that width, so the numeral always travels with the hue:

```tsx
<span className="tnum font-mono opacity-70 sm:hidden">
  · impact {clause.impact_score.toFixed(0)}
</span>
```

> A design rule you have written down is not a design rule you have enforced. Responsive breakpoints are where they quietly break, because a rule holds at the width you happen to be developing at.

**A second bug from the same screenshot pass:** the impact bar used `bg-current` inside a container whose text colour was never set, so it inherited plain ink instead of the severity hue, and its track was nearly invisible against the page. The bar was drawn but carried no information. Moving the band colour onto the rail container fixed both — the bar now reads as a proportional oxide-red measure, and 85 versus 82 is visibly different.

---

## Concept 13: You cannot review a design you have not looked at

Both bugs above were found the same way: by driving a real browser against a real policy and **looking at the result**.

`web/scripts/screenshots.mjs` is committed rather than thrown away. It launches Playwright, uploads the 39-clause golden policy through the actual UI, waits for the pipeline, and captures every screen in light and dark at retina density:

```bash
npm run shots
```

This matters more than it sounds. The design system forbids a specific list of visual patterns, and there is no way to check a built interface against that list except by seeing it. A typecheck passes on a page with an invisible progress bar. A unit test passes on a layout with 500px of dead space. Neither can tell you the severity numeral vanished on a phone.

Three things the screenshots caught that no other check could:
1. The score disappearing at narrow widths (a stated rule, broken).
2. The impact bar carrying no colour and no visible track.
3. A large empty lower half on the upload page, which reads as unfinished. Filled with a "What it looks for" strip that names the four costly clause types — content that sets expectations rather than padding that fills space.

---

## Failure 9: module resolution follows the file, not the shell

The screenshot script was first written to a scratch directory and run from the project:

```
Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'playwright'
```

Node resolves imports by walking up from **the importing file's own location** looking for `node_modules`, not from the shell's working directory. A script in a temp folder has no `node_modules` above it, so nothing resolves however you invoke it.

This is the same failure that appeared in M0 with `ModuleNotFoundError: No module named 'app'` for a Python script in a temp directory. Two languages, one rule: **module resolution follows the file, not the shell.**

The fix was also the better outcome — the script moved into `web/scripts/` and became a committed, repeatable tool instead of a throwaway.

---

## A smaller failure worth knowing

The Vite React-TS template enables `erasableSyntaxOnly` in `tsconfig`, which forbids TypeScript syntax with no plain-JavaScript equivalent. Constructor parameter properties are one:

```ts
constructor(message: string, readonly status: number) {}   // rejected
```

`tsc --noEmit` passed; `npm run build` failed. They run different configurations, and the build is the one that tells the truth. Declaring and assigning the field explicitly fixes it.

---

## Where M4 ended up

Four screens, in light and dark: upload, processing, the ranked report, and the clause reading panel. Verified against a real 39-clause policy through a real browser.

```
23 clauses in this policy could reduce or deny a claim.

01  4.7  Non-Medical Expenses          Will not be paid        IMPACT 85
    WHY YOU'D MISS THIS  buried on page 3, deep in the document ·
                         written at grade 16 reading level ·
                         sends you to other clauses to understand it
```

Files added: `web/src/index.css` (design tokens), `clauseMeta.ts`, `api.ts`, `types.ts`, `components/Chrome.tsx`, `components/RiskCard.tsx`, `components/ClausePanel.tsx`, `routes/Upload.tsx`, `routes/Policy.tsx`, `scripts/screenshots.mjs`.

---

## Check it yourself

```bash
# terminal 1
cd api && .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
# terminal 2
cd web && npm run dev          # http://localhost:5173

# with both running, recapture every screen
cd web && npm run shots        # writes web/.screenshots/
```

Toggle the theme with the button marked **Ink** / **Paper** in the masthead.

**Question to sit with before M5:** the clause panel shows the policy's exact wording and, beneath it, a line in mono reading `characters 6,757–7,013 of the extracted document`. Nobody asked for a character range, and no ordinary reader will ever use it. Why is it on screen?

<details>
<summary>Answer</summary>

Because it is the difference between **showing evidence and asking to be believed.**

Everything else on that panel is the app's claim about the policy: a plain-language rewrite, a clause type, an impact score. All of it is generated, and all of it could be wrong. The quotation plus its exact position is the one element a reader can check *independently of anything the app says* — that text is at that offset in the document, or it is not.

Most readers will never verify it. That is not the point. The point is that verification is **possible and visible**, which is what separates a tool that shows its working from one that asks for trust. In a domain where a confident wrong answer costs someone a real claim, a system that cannot be checked should not be believed — including by the person who built it.

There is a practical reason too: those offsets are what the future PDF highlighter needs, and displaying them keeps them honest. A number on screen that stopped matching the document would be noticed. A number only used internally would not.
</details>

---

# Interlude — Why a local model, when hosted models are better

## The question this answers

This project runs `qwen2.5:7b-instruct-q4_K_M` through Ollama, on a laptop GPU. Most production LLM products do not do that — they call a hosted frontier API. So the obvious question, and one worth answering honestly rather than defensively: **was local a compromise?**

Partly yes, and partly no. The distinction matters more than the answer.

## What production systems actually use

| Tier | Examples | Typically used for |
|---|---|---|
| Frontier hosted | Claude (`claude-opus-5`, `claude-sonnet-5`), GPT, Gemini | Most production reasoning work |
| Small hosted | `claude-haiku-4-5-20251001`, mini / flash tiers | High-volume classification and routing |
| Local, open-weight | Qwen, Llama, Mistral, via Ollama or vLLM | Privacy, offline use, regulated data, very high volume |

On hard reasoning, a frontier hosted model beats a 7B local model **badly**, not marginally. Nobody builds a serious legal-reasoning product on a 7B model because it is the strongest thing available. Any claim otherwise should be treated with suspicion.

## Cost is not the reason here

It is worth doing the arithmetic rather than assuming, because the assumption is usually wrong in both directions.

One policy in this project is about 40 clauses. Each clause costs roughly 500 input tokens (the long system prompt with the taxonomy, plus the clause) and about 200 output tokens. That is **~28,000 tokens per document**.

At frontier API prices, that is a few cents per policy. For a personal tool analysing one document at a time, **API cost is not a meaningful constraint.** Anyone claiming a local model here saved them money is optimising something that was never expensive.

## The reason that does hold up

The upload screen says it plainly: *"Your policy is never uploaded to anyone."*

An insurance policy is a personal health and financial document. It names conditions, ages, sums insured. For this specific product, running locally is not the budget option — it is a **product property that cannot be bought back later by switching to an API.** Once the document leaves the machine, that promise is gone.

That is the honest justification. Not cost, not capability. Privacy.

## The counterintuitive part

The project's strongest correctness claim exists **partly because** it is local.

`llama.cpp`, which runs under Ollama, does grammar-level token masking: at each decoding step the sampler is prevented from emitting any token that could not continue a valid document under the supplied JSON Schema. That is what makes a fabricated citation *unrepresentable* rather than merely unlikely — the guarantee this whole project's grounding design rests on.

Hosted providers differ here, and the difference is worth checking rather than assuming:

- OpenAI's structured outputs with `strict: true` are also grammar-constrained, giving a comparable guarantee.
- Anthropic's tool `input_schema` is validated — highly reliable, but validation after generation is a different mechanism from masking during it.

So "switch to a bigger model" is **not automatically a free upgrade** for the citation guarantee. A larger model reasons better; whether it can still make an invalid citation impossible depends on the provider's mechanism, not on its size. That is exactly the kind of thing to verify before assuming bigger is strictly better.

## Why this is a decision and not a lock-in

Nothing here has to be guessed at, which is the real payoff of the eval harness:

- `settings.model` in `api/app/config.py` is a one-line change.
- Every model call funnels through a single `_post_chat` in `api/app/llm/client.py`, so another provider is a small adapter, not a rewrite.
- `evals/run_eval.py` already reports classification macro-F1 and ranking quality, so a swap can be **measured** rather than argued about.

## What separates a real project from a wrapper

Not the model. The model is the most replaceable part of the system — a config line.

What is not replaceable is the surrounding engineering: deterministic stages where determinism is possible (segmentation, scoring), verifiable grounding (character offsets, enum-constrained citation IDs), and an eval harness that turns "which model" into a measurement instead of an opinion.

That is worth internalising beyond this project. **Model choice is a parameter. The system around it is the work.**

## The plan from here

Keep local as the default, for the privacy reason above.

Then, once the scenario simulator exists, run the same eval against a hosted frontier model and publish the difference. The scenario simulator is precisely where a 7B model is expected to struggle: multi-hop reasoning where a waiting period, an exclusion and a notice condition all bear on a single question. Single-clause classification already scores macro-F1 1.000 locally; combining three interacting rules is a different task.

If a hosted model scores materially higher there, the defensible architecture is a **hybrid**: local for the 40 bulk classifications, which are private, effectively free, and already perfect on the golden set; hosted for the one hard reasoning call, where capability actually shows up.

A table reading *"local 7B: 0.72 · frontier: 0.91 · here is the trade we chose and why"* is worth more than either number alone, because it shows the choice was made rather than defaulted into.

---

# M5 — The scenario simulator, and making a fabricated citation impossible

## What M5 had to produce

Ask a question in your own words — *"I had knee surgery 8 months after buying this"* — and get back a verdict, the clauses that decide it, and the policy's own wording quoted from each.

Four stages:

```
scenario ──> [5a extract facts] ──> [5b shortlist] ──> [5c reason] ──> [5d verify]
             LLM, schema             pure filter        LLM, id-enum     substring
```

---

## Concept 14: Two grounding mechanisms, because one is not enough

This is where the enum result from M0 finally earns its keep, and where its limit becomes visible.

**Layer one: the citation's address.** The reasoning schema's `clause_id` field is an enum built at request time from exactly the clause ids placed in that prompt:

```python
# api/app/pipeline/scenario.py
"clause_id": {"type": "string", "enum": clause_ids},
```

Ollama enforces JSON-Schema enums during sampling, so while a citation is being generated, every token that would spell a different id has probability zero. A citation pointing at a clause the model was never shown is **unrepresentable**, not merely unlikely. Most systems detect bad citations after generation and retry; this makes them impossible to produce.

**Layer two: the citation's content.** The enum guarantees the address exists. It says nothing about whether the claim attached to that address is true — and that gap is not hypothetical. It was observed:

> The model cited clause `1.4` while quoting text belonging to clause `2.4`. The enum accepted it, because `1.4` was a real id.

So `api/app/grounding.py` checks that any quoted text actually appears inside the clause it was attributed to. That is what catches invented wording bolted onto a valid id.

> **The first layer makes the address real. The second makes the content real. Neither alone is grounding.**

### Why the ids are the policy's own clause numbers

The misattribution above had a cause. The ids were internal (`c14`) while the clause text itself began "2.4 Day Care Procedures", so the model was translating between two numbering systems mid-sentence — and got it wrong.

Citation ids are now the policy's own numbers, so the id being cited and the number printed inside the clause are the same string. The translation step is gone because there is nothing left to translate.

---

## Concept 15: Why there is still no retrieval

The standard architecture for "answer a question about a document" is retrieval: embed the clauses, embed the question, fetch the top k. This project does not, and the reason is arithmetic.

The whole policy is **~3,100 tokens** against an 8,192-token context. Every clause that could possibly matter fits in one prompt with room to spare.

Given that, retrieval could only make the answer worse. Top-k means choosing a k, and any k below "all of them" can drop the single clause that decides the case — which here means confidently telling someone they are covered because the exclusion did not make the cut. **Retrieval would solve a problem this document does not have, and introduce a failure mode it did not previously have.**

`shortlist()` therefore sorts by impact and keeps everything that fits. On a normal policy nothing is dropped at all; the ordering only matters for a document large enough to overflow, and then what survives is the clauses most likely to cost the reader money.

---

## Failure 10: a silent context-window default

The architecture above depends entirely on "the whole policy fits in the context". That claim was false.

`qwen2.5` supports 32,768 tokens, but this model's Ollama definition sets no `num_ctx`, so **Ollama was applying its own 4,096-token default.** The test policy fits under that by luck. A slightly larger one would have had its opening clauses silently dropped, and the answer would have been reasoned over a truncated policy with nothing to indicate it.

The fix is one line of options, but the lesson is bigger: **a guarantee that depends on a runtime default you never set is not a guarantee.** The design said 32k; the runtime said 4k; nothing in between checked.

---

## Failure 11: over-provisioning is not free

Having found that, the obvious move was to set the window generously. 16,384 seemed safely large.

The KV cache lives in VRAM beside the 4.7GB of weights. At 16,384 the runtime held 5.46GB, the machine was left with **2.0GB free of 15.7GB**, and the operating system killed the eval run for memory pressure.

Sized from the actual requirement instead — a 40-clause policy is ~3,100 tokens, the system prompt ~1,200, the response ~500, so ~4,800 — the window is now 8,192. Comfortable headroom, half the cache.

> "To be safe" is not a number. Caution that is not measured is just a different guess, and this one was paid for in RAM the rest of the system needed.

---

## Failure 12: two numbers that must agree should not be two numbers

While fixing the above, a latent bug surfaced: `scenario_token_budget` was **12,000** while `num_ctx` was **8,192**. Two independently-set values that must agree.

A 12k clause budget against an 8k window builds a prompt larger than the context, and Ollama truncates it silently — dropping exactly the clauses the shortlist had just been careful to include. The guarantee and the config value that could break it lived twenty lines apart and were never checked against each other.

The budget is now derived:

```python
# api/app/config.py
@property
def scenario_token_budget(self) -> int:
    return max(self.num_ctx - self.scenario_reserved_tokens, 1_000)
```

The same mistake appeared a second time in M5, in a different place. The list of facts the model is told are missing had been narrowed to the five that can decide an Indian health claim, but the list shown to the *user* still included every null field — so the interface asked for the reader's "body system" and "estimated cost inr" before it could answer. Both now read from one `DECISIVE_FACTS` tuple.

---

## Failure 13: an error message that said nothing

One scenario ran past a 420-second timeout three times and failed with:

```
LlmError: qwen2.5:7b-instruct-q4_K_M failed after 3 attempts:
```

Nothing after the colon. Twenty-one minutes of failure, reported as no information at all.

The cause is a detail worth carrying to other projects: **`httpx.ReadTimeout` stringifies to an empty string.** The handler formatted the exception with `{last_error}` and trusted every exception class to have a useful `__str__`. Some do not. Error messages now always include the exception *type*.

---

## Failure 14: retrying a deterministic failure

With the message fixed, the real cause appeared: **generation was never capped.** `llama.cpp` generates until a stop token or the context fills, so a model that begins repeating itself does the latter. Capping it at `num_predict` took the same case from three consecutive timeouts to **9.9 seconds**.

Then the cap was too tight, and the JSON came back truncated mid-quote. But the more interesting failure was how the client responded to that:

```
JSONDecodeError: Unterminated string starting at: line 6 column 16
```

...three times, identically.

> **temperature is 0.** Re-sending an identical request produces identical tokens. Retrying a deterministic failure is not a retry — it is the same failure, three times, at three times the cost.

So the two failure kinds are now handled as the different things they are:

| Failure | Correct response |
|---|---|
| Truncated JSON | **Change something** — double `num_predict`, and log that the configured ceiling is too low |
| Transport error | **Repeat unchanged** — a timeout is not a property of the prompt |

Both behaviours are pinned by tests, including one asserting the ceiling must *not* creep upward on transport failures.

The cap itself was also resized from what the schema can hold rather than from a round number that felt safe — worst case is 4 citations of ~200 characters plus reasoning, about 500 tokens, so 1,600 with triple headroom. **"Cap runaway generation" and "cap useful output" are the same knob.**

---

## Concept 16: measuring the right thing, again

The eval reports four numbers over 16 hand-authored scenarios:

| Metric | Score |
|---|---|
| Verdict accuracy | **0.688** (11/16) |
| Citation recall | **0.750** (12 cases require a citation) |
| Fabrication rate | **0.125** (3 invented quotes) |
| **Detection integrity** | **1.000** |

The first version of this eval had a single metric called "groundedness" with a hard 100% target. Then a case failed it: the model cited a waiting-period clause for a co-payment question and quoted words that were not in it. The verbatim check caught it and flagged the answer unverified.

Reporting that as a system failure would have been **exactly backwards.** It is the check doing its job. The metric was conflating two different questions:

- **Fabrication rate** is about *the model*. A 7B model will sometimes attach an invented quote to a real clause id. Driving this to zero is a quality goal.
- **Detection integrity** is about *the system*. Every fabricated quote must be caught and shown as unverified. **This** is the property the design promises, and the only one with a hard target.

A real failure would be an unverifiable quote reaching a reader unflagged. That number is 1.000.

One further detail: detection integrity is computed by **independently re-verifying every quotation** and comparing against the flag the pipeline set — not by reading the flag. A self-reported guarantee is not a measurement; if the verifier were silently broken, its own flag would cheerfully report everything fine.

---

## Knowing when to stop, again

The first run scored **0.562** with an obvious systematic bias: six of seven misses were `not_covered` for cases that were not denials. That was an over-correction of *my own* earlier fix — having pushed hard on "do not soften a refusal", the model began refusing everything.

One principled fix addressed a genuine definitional gap rather than fitting the metric:

```
"20% co-payment applies"       -> conditional. You are paid 80%.
"room rent capped at 1% a day" -> conditional. You are paid, with a deduction.
"cosmetic surgery is excluded" -> not_covered. Nothing is paid.
```

**Being paid less is not being refused.** That took accuracy 0.562 → 0.688 and fixed all three `conditional` cases.

It also broke one that had been right: `cataract-served` regressed from `covered` to `not_covered`. That is the signal to stop — the same signature as M2's ranking work, where changes began trading cases against each other rather than improving the system.

### What the remaining misses actually are

Worth reading, because they are not random:

| Case | Expected | Got | Needs |
|---|---|---|---|
| `ped-waiting-served` | covered | not_covered | 5 years > 36 months |
| `cosmetic-after-accident` | covered | not_covered | reading "unless necessitated by an Accident" |
| `initial-waiting-period` | not_covered | covered | 2 weeks < 30 days |
| `cataract-served` | covered | not_covered | 4 years > 24 months |

**Four of five require date arithmetic or following an exception clause.** Not vocabulary, not classification — multi-hop reasoning over interacting rules. That is precisely where a 7B model is weakest, and precisely what the earlier interlude on local versus hosted models predicted would need measuring against a frontier model.

Single-clause classification scores macro-F1 1.000 locally. Combining three interacting rules scores 0.688. The gap between those two numbers *is* the finding.

---

## Where M5 ended up

**91 tests passing.** Files added: `app/grounding.py`, `app/pipeline/scenario.py`, `app/routers/scenarios.py`, `web/src/components/ScenarioPanel.tsx`, `evals/run_scenario_eval.py`, `evals/golden/scenarios.json`.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -v
python evals/run_scenario_eval.py        # writes evals/scenario-report.md
```

Then run both servers and ask the app something it cannot answer — *"does this cover my car being stolen?"* It should say your policy does not settle it, rather than guessing.

**Question to sit with:** the eval's `detection_integrity` is 1.000, computed by re-verifying every quotation independently rather than by reading the `verified` flag the pipeline already set. Reading the flag would have been one line instead of ten, and would have produced the same number.

Why is the ten-line version the only one worth having?

<details>
<summary>Answer</summary>

Because reading the flag would measure **nothing**.

`ScenarioResult.verified` is set by the very code the metric is supposed to be testing. If `verify_quote` had a bug — a normalisation step that was too generous, a regex that silently matched nothing, an `in` test on the wrong variable — then every citation would be marked verified, the flag would report a clean run, and the metric would print **1.000**.

A perfect score produced by a broken verifier is indistinguishable from a perfect score produced by a working one, if the score comes from the verifier.

Re-verifying independently means the eval computes the answer a second time and compares. It can now catch the case the design fears most: a quotation that genuinely is not in the policy, which the pipeline nevertheless waved through.

The general form: **a test that asks the system under test whether it worked is not a test.** It is the same reason the M1 tests slice `raw_text[start:end]` and compare, rather than asking the parser whether its offsets are correct — and the same reason the M0 LLM tests hit a real Ollama rather than a mock.
</details>

---

# M6 — One report, and proving the repo works for someone else

## What M6 had to produce

Two things, both about a reader who is not the author: a single evaluation report that states its own conditions, and proof that the project actually runs from a fresh clone.

---

## Failure 15: a report that could not say when it was written

The project had two report files, `evals/report.md` and `evals/scenario-report.md`, each written by its own script. Reading them together turned out to be misleading, and the reason is worth spelling out.

The classification report had been generated when `PROMPT_VERSION` was `v4`. By the time the scenario report existed, prompts were at `v6`, and the LLM cache key had also grown to include decoding settings that had changed. **The two files described different runs of different code, and nothing in either said so.**

Worse, the classification report contained this line:

```
- **Prompt version**: see `PROMPT_VERSION` in `api/app/llm/prompts.py`
```

A pointer, not a record. It tells a reader where to look *today*, which is precisely useless for judging whether a number generated months ago still describes the code in front of them.

> **A report is a record of a run. If it cannot state the conditions of that run, it is not evidence of anything.**

This is the same class of bug as a stale LLM cache — results that no longer correspond to the code that appears to have produced them — reproduced one layer up, in the reporting rather than the caching.

`evals/run_all.py` now runs both evals against the same code in one pass and stamps the output with the model, prompt version, decoding settings, batch size, git commit and timestamp. There is exactly one report, `evals/REPORT.md`, and the two per-eval files are gitignored intermediates.

Re-running everything at `v6` also answered a question that had been quietly open: classification was still **macro-F1 1.000**. The prompt version and the context window had both changed since it was last measured, so that was worth confirming rather than assuming.

### A smaller honesty fix

The consolidated report first printed `Total wall time | 0s`, because every response came from cache. True, and misleading — it invites a reader to think the eval is free. It now says:

```
| Total wall time | 0s (served from cache; a cold run takes several minutes) |
```

---

## Concept 17: separating what is measured from what is promised

The report's headline table has a `Kind` column, and it carries more weight than the numbers:

| What is measured | Score | Kind |
|---|---:|---|
| Clause classification (macro-F1) | 1.000 | quality |
| Risk ranking expectations | 6/9 | quality |
| Scenario verdict accuracy | 0.688 | quality |
| Scenario citation recall | 0.750 | quality |
| Quote fabrication rate | 0.125 | quality |
| **Citation detection integrity** | **1.000** | **guarantee** |

Every *quality* row measures how well a 7B model on a laptop performs a hard task. Those are published exactly as they came out, including 0.688 and 6/9.

**One row is different in kind.** Detection integrity asks: of the quotations the model invented, how many were caught and shown to the reader as unverified rather than presented as evidence? That is the property the system actually promises. It is the only number that would count as a *bug* if it moved.

That distinction is enforced in code, not just in prose. `run_all.py` exits non-zero if detection integrity drops below 1.000, so it can gate a pipeline — and the quality numbers deliberately do **not** gate:

```python
if scenario["detection_integrity"] < 1.0:
    sys.exit(1)
```

> A threshold on a quality metric invites tuning to the threshold. A threshold on a guarantee catches a broken guarantee. They should not be treated the same way, and only one of them belongs in CI.

---

## Failure 16: misreading my own test failure

The M6 gate was to clone the repository somewhere clean and follow the documented steps. On the first attempt:

```
ERROR tests/test_llm.py
ERROR tests/test_scenario.py
ERROR tests/test_score.py
!!! Interrupted: 3 errors during collection !!!
```

The immediate conclusion — *the repo does not work from a fresh clone* — was wrong, and stated out loud before checking.

The tests had been launched while `pip install` was still running in the background. The three modules that failed were the three importing packages that were not on disk *yet*. Running the same command after the install finished gave **85 passed**.

Two things worth keeping from that:

1. **"It failed" and "it failed for the reason I assumed" are different claims.** A collection error naming three files looks like a repo problem and was a race with my own setup. One `--tb=line` would have shown an ordinary `ModuleNotFoundError` and settled it in seconds.
2. It is still the right gate. Running it is what surfaced the genuinely stale report files that a fresh clone was shipping.

---

## What a fresh clone actually gets

Verified rather than assumed:

- **72 tracked files, 881KB.** No `.env`, no database, no `node_modules`, no virtualenv, and **no PDF has ever been committed** — checked across the entire history, not just the current tree.
- **All three golden PDFs rebuild themselves.** They are gitignored build artefacts; `tests/conftest.py` regenerates them from `build_synthetic_policy.py` when missing, so a clone with no binary fixtures still runs the whole suite.
- **85 tests pass** with no model running; 91 with Ollama up.

The licensing position holds as a consequence: evals run on a policy this repository authors, so the repo is self-contained and redistributable, and no insurer's copyrighted wording is anywhere in it.

---

## The project, end to end

| Stage | LLM? | What guarantees it |
|---|---|---|
| 1. Ingest | No | Offsets slice back byte-identical, asserted per clause |
| 2. Segment | No | Rules over layout; recall verified on a deliberately unstyled PDF |
| 3. Analyze | Yes | Every categorical field enum-constrained; responses cached by content |
| 4. Score | No | Pure arithmetic, unit-tested, reproducible |
| 5. Scenario | Yes | Citation ids enum-locked, quotations verified against source |

The through-line, stated once: **deterministic wherever possible, a model only where language understanding is genuinely required, and every model output constrained by a grammar rather than a request.**

That is what makes a 7B model on a laptop worth building on. It is not that the model is good enough to trust — it is that the system is arranged so that the places it can go wrong are either impossible or caught.

---

## Check it yourself

```bash
git clone <repo> && cd insurance-policy-clause-explainer
cd api && python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -m "not llm"    # 85, no model needed
cd .. && python evals/run_all.py                    # writes evals/REPORT.md
```

**A last question, and the one most worth sitting with.** `evals/run_all.py` exits non-zero when detection integrity falls below 1.000, and never fails on verdict accuracy — not even if it dropped from 0.688 to 0.2.

That looks backwards. A system answering four in five questions wrongly is obviously broken. Why is that not the failing condition?

<details>
<summary>Answer</summary>

Because the two numbers make different promises to the person using this, and only one of them is a promise this system can keep.

**Verdict accuracy is a capability measure.** It moves with the model, the prompt, the policy, and the difficulty of the questions asked. A drop from 0.688 to 0.2 is bad news, but it is *information* — and the honest response is to publish it, diagnose it, and decide whether to change models. Turning it into a build failure creates pressure to make the number go up, and the cheapest way to make an eval number go up is to tune against the eval. That is how you get a system that scores well on sixteen cases and worse on a real policy. The M2 ranking work already hit exactly that wall, and stopped at 6/9 for the same reason.

**Detection integrity is a safety property.** It does not say the answers are good; it says that when an answer is *not* supported by the document, the reader is told. A user can work with a tool that is often uncertain and always honest about it. A tool that presents an invented quotation as evidence is worse than no tool at all, because it is wrong in the specific way that looks most like being right — and in this domain, that costs someone a real claim.

So a low verdict accuracy means "this model is not good enough at this task yet". A detection integrity below 1.000 means "this system will lie to someone". Only the second is a defect, and only defects should break a build.

The general principle: **gate on the promises you make, measure everything else.** Confusing the two either blocks work over numbers that were never guaranteed, or ships silently past the one thing that was.
</details>

---

# M7 — Fixing 0.688, and four bugs hiding behind each other

## The starting position

The scenario simulator answered 11 of 16 questions correctly — **0.688**. That was published honestly and framed as a limitation of a 7B model on multi-hop reasoning.

That framing was wrong. Four of the five failures were bugs in this project, not weakness in the model, and finding them took five measured runs.

---

## Concept 18: I broke this project's own rule

The failures were specific:

```
5 years held vs a 36-month bar    -> said NOT covered
2 weeks held vs a 30-day bar      -> said covered
4 years held vs a 24-month bar    -> said NOT covered
```

None of those is a language problem. Each is a comparison of two numbers.

The project's governing principle is: *deterministic where possible, LLM only where language understanding is genuinely required.* Stage 2 (segmentation) and stage 4 (scoring) follow it — I specifically refused to let the model compute buriedness because that would be arithmetic. **Stage 5 had quietly broken the same rule**, and I had been calling the result a model limitation.

`api/app/pipeline/waiting.py` restores the split:

| Task | Who |
|---|---|
| Reading "thirty six months" out of legal prose | the model — that is language |
| Deciding whether `60 >= 36` | Python — that is arithmetic |

The comparison result is handed to the reasoning step as settled fact. `UNKNOWN` is a first-class outcome: if the person never said how long they have held the policy, no comparison is possible, and that is what lets the verdict be `insufficient_information` rather than a guess.

> A capability limit and an architectural mistake look identical from the outside. Both show up as a wrong answer. The difference is that one of them is your fault, and it is worth checking which before blaming the model.

---

## Failure 17: a green test that passed for the wrong reason

The first version asked the model for `waiting_period_months`. Clause 3.1 reads *"the first thirty days"*, and came back as **30** — read as thirty *months*.

The eval case still passed. A two-week-old policy is short of both 30 days and 30 months, so the wrong unit produced the right answer.

> A test that passes for the wrong reason is worse than one that fails. It reports that the thing works *and* removes your reason to look at it.

The schema now asks for a value and a unit separately — `(30, "days")`, `(36, "months")` — because normalising units is arithmetic, while reporting what the clause *says* is reading. Python converts.

---

## Failure 18: the same bug, on the other operand

Fixing the clause side surfaced its mirror image. The scenario extractor was asked for `months_since_policy_start`, and *"two weeks after my policy started"* came back as **2**.

The number read correctly; the unit was dropped. A fortnight-old policy then counted as two months old and cleared the 30-day bar it should have failed. The computed block said, correctly and uselessly:

```
clause 3.1: requires 1 month, policy held 2 months -> no longer applies
```

Arithmetic flawless, input wrong.

> **A comparison has two operands.** Fixing one and leaving the other is not fixing the comparison.

Both sides now report value plus unit and Python converts. The prompt says it in as many words: *"Two weeks" is 2 + weeks, not 2 + months and not 14 + days.*

---

## Failure 19: a field that existed but was never populated

Clause 4.1 excludes cosmetic surgery *"unless such surgery is necessitated by an Accident, Burn or Cancer"*. An `exceptions` field had been added to catch exactly this. It was empty.

The field existed; nothing had told the model to look for carve-outs. Adding the signals explicitly — `unless`, `except`, `other than`, `provided that` — with worked examples populated it.

> Adding a field is not adding a feature. Nothing fills it because it exists.

Worth noting how this was found: not by reasoning about it, but by printing what the analyzer actually extracted. The fix had been written, committed in spirit, and was inert.

---

## Concept 19: correct facts can still mislead

The deterministic block worked, and **accuracy fell from 0.688 to 0.625.**

Each served waiting period was stated as:

```
clause 3.2 ... HAS BEEN SERVED - this clause does NOT block the claim
```

Every word true. But "does not block the claim" reads as a verdict on the *question*, not on the *clause*. With four served periods listed, the model saw four statements that nothing blocked the claim and began answering `covered` to questions about co-payments and room-rent caps — citing `3.1, 3.2, 3.3, 3.4` while missing clause 5.3 entirely.

Two fixes:

**Scope.** The block now says out loud what it does *not* settle: exclusions, payout caps, co-payments, notice conditions. Its per-clause wording narrowed to *"this waiting period no longer applies (it says nothing about any other clause)"*.

**Prominence proportional to decisiveness.** A served waiting period is a non-event. Four of them each getting an emphatic line outweighed the one clause that mattered. Served periods now collapse to a single line naming them as irrelevant; only periods that actually block something keep individual treatment.

> Injecting correct computed facts into a prompt is not free. Their *weight* is part of the message, and four true irrelevant statements can drown one decisive clause.

---

## Concept 20: knowing when the number stops being a signal

Across five measured runs:

| Run | Change | Verdict accuracy |
|---|---|---|
| 1 | baseline | 0.688 |
| 2 | deterministic arithmetic | 0.625 |
| 3 | scoped the block | 0.812 |
| 4 | units on both sides | 0.688 |
| 5 | prominence + clause-type routing | **0.812** |

Every change fixed its target case and disturbed a neighbour. On 16 cases **one case is 0.0625**, so 0.688 versus 0.812 is two cases wide — inside the range this set can resolve.

That matters for how the result is held:

- The **unit bugs and the missing carve-out extraction are correctness fixes.** They are right whatever the aggregate does. A 30-day bar and a 30-month bar must not collapse together.
- The **prompt-shaped changes are held loosely.** They are the ones the metric cannot cleanly separate.

Final state: verdict accuracy **0.812**, citation recall **0.750 → 0.917**, detection integrity **1.000** throughout.

---

## The three that remain, and why they are different

| Case | Expected | Got | What happened |
|---|---|---|---|
| `ped-waiting-served` | covered | not_covered | Told the 36-month bar was satisfied, went looking for another reason and cited the **cosmetic surgery** clause for a blood-pressure question |
| `senior-copay` | conditional | covered | Confirmed nothing *blocks* the claim, never asked whether anything *reduces* it. Missed the 20% co-payment |
| `accident-in-initial-period` | covered | not_covered | The carve-out *"except claims arising out of an Accident"* was correctly extracted and shown. Cited the clause and refused anyway |

**None is a computation error.** The model has the served waiting period, the extracted carve-out, and the co-payment clause in front of it, and does not act on them.

`senior-copay` is the most consequential: a real user would be told they are covered and never learn they are paying a fifth of the bill. It suggests a missing structural step rather than a prompt weakness — nothing forces the question *"does anything reduce this?"* once the model is satisfied nothing blocks it.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q     # 107 tests
python evals/run_all.py                             # writes evals/REPORT.md
```

**Question to sit with:** the fix that moved accuracy most was moving a comparison out of the model and into ten lines of Python. Everything else — five prompt revisions across five runs — moved the number around by two cases and settled roughly where it started.

What does that suggest about where to look first, the next time a model gives a wrong answer?

<details>
<summary>Answer</summary>

**Look for the thing you are asking it to do that is not a language task.**

A language model is being asked to do many things at once in any real prompt: read, classify, compare, count, convert units, apply rules in order, remember a constraint from four paragraphs earlier. Some of those are language. Most of the rest have exact, testable implementations that are ten lines long and never wrong.

Every failure fixed here followed that shape. Comparing 60 against 36 is not language. Converting weeks to days is not language. Deciding that a satisfied waiting period should be mentioned once rather than four times is not language. The model was failing at tasks it should never have been handed.

The tell is that prompt engineering plateaus. Five revisions moved this number by two cases and put it back where it started, because the prompt was not the problem — the division of labour was. When more careful wording keeps producing the same class of error, that is evidence the task is misassigned, not that the wording needs another pass.

The residual, once the misassignment is fixed, is the real capability limit — and it is worth measuring against a larger model rather than guessing at. But you cannot see that residual until you stop asking a language model to do arithmetic, because its arithmetic errors look exactly like reasoning errors from the outside.
</details>

---

# M8 — Widening the ruler, and the second family of comparisons

## The starting position

The scenario simulator answers a what-if question about a health policy —
*"I had knee surgery 8 months after buying this"* — with one of four verdicts
(`covered`, `not_covered`, `conditional`, `insufficient_information`) and the
clauses that decide it.

Its quality was measured against a hand-written set of 16 cases in
`evals/golden/scenarios.json`, and that set reported **verdict accuracy 0.812**.

Two things were wrong with that number, and neither was visible from inside it.

---

## Concept 21: a ruler that cannot resolve what you are measuring

On a 16-case set, one case is worth 0.0625. Five consecutive measured runs of
this system, each following a real code or prompt change, produced:

```
0.688  ->  0.625  ->  0.812  ->  0.688  ->  0.812
```

Every one of those gaps is one or two cases wide. The set could not tell a
genuine improvement from the model happening to land differently on two
borderline questions, which means five runs of careful work produced almost no
information about which changes had helped.

This is a general trap and it is worth naming precisely. **The granularity of
your measurement sets a floor on the size of the effect you can detect.** Below
that floor, a metric does not merely fail to help — it actively misleads,
because a number that moved feels like evidence. Tuning against a ruler too
coarse for the change you are making is how you end up confidently shipping a
regression, or reverting an improvement.

The fix is not cleverness, it is more cases. At 40 cases one case is 0.025, and
a genuine two-case improvement clears the noise instead of drowning in it.

So the set was rewritten to 40 cases, all hand-authored against the same
synthetic policy, and all written *before* anything was measured — the
discipline that keeps a test set testing the system rather than rationalising
whatever the system already did.

The new cases are deliberately not spread evenly. The old set had **three**
cases out of sixteen that turned on a *reduction* — a co-payment, a room-rent
cap, a disease sub-limit — while the system's single worst known failure was in
exactly that gap. A set that under-samples the failure mode cannot measure a
fix to it. The widened set has **thirteen**.

---

## Concept 22: recall only measures one of the two ways to be wrong

The set records, for each case, the clauses the answer cannot be right without:

```json
{
  "id": "senior-copay",
  "scenario": "I bought this policy at 67 and I am claiming for a
               hospitalisation three years later for something new.",
  "expected_verdict": "conditional",
  "must_cite": ["5.3"],
  "why": "Over 60 at inception, so a 20% co-payment applies to every
          admissible claim."
}
```

Citation recall asks: did the answer cite what it had to? That measures one
direction of failure — citing too little.

It cannot see the other direction, and the other direction has a specific way
of appearing. A system pushed to hunt harder for co-payments will start finding
them for people who do not owe one. **Recall goes UP when a system cites more,
so a fix that trades under-citing for over-citing scores as an improvement.**

Hence a second field, and a second metric:

```json
{
  "id": "copay-just-under-sixty",
  "scenario": "I bought this policy when I was 58 and I am claiming now,
               four years later, for a gallbladder operation.",
  "expected_verdict": "covered",
  "must_cite": [],
  "must_not_cite": ["5.3"],
  "why": "The co-payment requires having COMPLETED sixty years at first
          inception. 58 is not 60, so it does not apply."
}
```

`evals/run_scenario_eval.py` scores it separately rather than blending it into
recall, because the two fail for opposite reasons and one number would let each
hide the other:

```python
# The false-positive direction of citation quality. Reported alongside
# recall rather than folded into it, because the two fail for opposite
# reasons and a single blended number would let one hide the other.
"false_citation_rate": (
    sum(bool(r["wrongly_cited"]) for r in guarded) / len(guarded)
    if guarded
    else 0.0
),
```

---

## What the wider ruler found the moment it was used

Running the **unchanged** system against the 40-case set:

| | 16 cases | 40 cases |
|---|---:|---:|
| Verdict accuracy | 0.812 | **0.725** |
| Citation recall | 0.917 | **0.710** |
| False citation rate | not measured | **0.400** |

The system had not got worse. It had always been this good; the smaller set
simply had not asked the questions it was bad at. **0.812 was not a wrong
measurement, it was a measurement of an easier exam.**

And the new failures fell into a pattern:

```
room-rent-within-cap   expected covered      got conditional
icu-rate-breach        expected conditional  got covered
oral-chemo-limit       expected conditional  got covered
post-hosp-too-late     expected not_covered  got covered
senior-copay           expected conditional  got covered
```

`room-rent-within-cap` describes a room costing 8,000 rupees a night against a
sum insured of 10 lakh. The policy caps room rent at 1% of the sum insured per
day — 10,000 — so the room is comfortably *within* the cap. The model read it
as a breach. `icu-rate-breach` describes 12,000 a day in intensive care against
a 2% cap on a 5 lakh sum insured — 10,000 — an actual breach, which the model
read as fine.

Both are the comparison of two numbers, in opposite directions.

---

## Concept 23: the same mistake, in a family nobody had noticed was the same

This project's governing rule is: *deterministic where possible, LLM only where
language understanding is genuinely required.*

An earlier milestone had already found the scenario simulator breaking that
rule, and fixed it. The module `api/app/pipeline/waiting.py` exists because the
model was being asked whether five years exceeds thirty-six months, and getting
it wrong about half the time. Its docstring states the split:

```
    reading "thirty six months" out of legal prose   -> the model's job
    deciding whether 60 >= 36                        -> this module's job
```

That fix was correct and it was narrow. It took **durations** away from the
model. It left **money** and **age** exactly where they were, and nobody
noticed, because the 16-case set contained no case that required either
comparison.

The failures above are all the same shape as the ones `waiting.py` was built to
kill:

```
deciding whether 8,000 > 1% of 10,00,000    is not language
deciding whether 12,000 > 2% of 5,00,000    is not language
deciding whether 58 >= 60                   is not language
```

`api/app/pipeline/reduction.py` restores the split for them. The analyzer is
asked for the numbers the clause *states* — never for a comparison — in the
same value-and-unit shape that `waiting.py` established:

```python
"copay_percent": {"type": ["integer", "null"]},
"copay_min_age_at_inception": {"type": ["integer", "null"]},
"cap_percent_of_sum_insured": {"type": ["integer", "null"]},
"icu_cap_percent_of_sum_insured": {"type": ["integer", "null"]},
```

and Python does the arithmetic.

### The operand that was secretly two operands

The co-payment clause in the golden policy reads:

> All admissible claims in respect of an Insured Person who has **completed
> sixty years of age at the time of first inception of the policy** shall be
> subject to a co-payment of twenty percent of the admissible claim amount.

It keys on age **at inception** — which is not the person's age now. Someone who
is 68 today may have bought the policy at 50 and owes nothing. The fact
extractor had only a single `age` field, so the two were the same number, and
the comparison had a 50% chance of being made against the wrong operand.

`api/app/pipeline/reduction.py` separates them and derives one from the other
where it can, in Python:

```python
def age_at_inception(facts, policy_age_days):
    stated = facts.get("age_at_policy_start")
    if stated and stated > 0:
        return stated

    age_now = facts.get("age")
    if age_now and age_now > 0 and policy_age_days is not None:
        derived = age_now - policy_age_days // 365
        if 0 < derived <= age_now:
            return derived
    return None
```

This is the third time this project has hit the same bug. A waiting period read
`"thirty days"` as 30 *months*. A scenario read `"two weeks"` as 2 *months*. Now
an age at claim was read as an age at inception. **Every one of them was a
comparison whose two operands were not the things they appeared to be.**

The same lesson was applied pre-emptively to money. Indian policy schedules say
*"5 lakh"*, never 500000, so the extractor reports `(5, "lakh")` and the
multiplication happens in code — because converting units is exactly the
arithmetic that produced the two bugs above.

---

## Failure 20: the fix made it worse, by 0.125

With the reduction sweep wired in, the measured result went **backwards**:

```
0.725  ->  0.600
```

It fixed what it was built to fix. `icu-rate-breach`, `oral-chemo-limit` and
`copay-applies-emergency` all became correct, and the arithmetic behind them was
flawless. But fourteen other cases broke, and they broke in a way that named the
cause immediately:

```
every new failure landed on either `conditional` or `insufficient_information`
```

Nothing else. Not a spread of wrong answers — two specific wrong answers, over
and over.

Here is the block the prompt was receiving for `intoxication-injury`, a case
whose scenario is *"I fell down the stairs after drinking heavily and fractured
my hip. I have held the policy five years"* — a question decided entirely by the
policy's alcohol exclusion, mentioning no money, no room and no age:

```
- CANNOT TELL: clause 5.1: room rent are capped at 1% of the sum insured
  per day, but the sum insured and the per-day charge was not stated, so
  whether the cap is exceeded cannot be determined
- CANNOT TELL: clause 5.3: a 20% co-payment applies if the person had
  reached 60 at inception, but their age when the policy STARTED was not
  stated, so this cannot be determined
- YOUR CALL: clause 5.2: caps or reduces what is paid ...
- YOUR CALL: clause 5.4: caps or reduces what is paid ...
- YOUR CALL: clause 5.5: caps or reduces what is paid ...
```

Five statements. Every one of them true. Every one of them irrelevant to
whether a drunken fall is covered. Two of them say a figure is missing — which
reads as *insufficient information* — and three say a cap might apply — which
reads as *conditional*.

**The block that was supposed to add a finding added only doubt, and the model
answered the doubt.**

---

## Concept 24: silence is not the same as uncertainty

What makes this failure worth recording is that the lesson needed to avoid it
had already been learned, written down, and was quoted in the new module's own
docstring.

The earlier lesson, from `api/app/pipeline/waiting.py`, is that prominence must
be proportional to decisiveness. Served waiting periods had each been given
their own emphatic line saying the clause *"does NOT block the claim"*; with
four of them listed, the model read four statements that nothing blocked the
claim and started answering `covered` to questions about co-payments. So
`waiting.py` collapses served periods to a single line.

The new module dutifully did the same thing — for the wrong status. It
collapsed `DOES_NOT_APPLY`, and gave every `UNKNOWN` and `JUDGEMENT` its own
line. `DOES_NOT_APPLY` is *rare*: it requires both operands to be present. The
common statuses were the ones left shouting.

The repair is a distinction the module did not originally have. Two situations
had been collapsed into `UNKNOWN`:

- someone gave a sum insured and no room rate — a **real open question**, which
  may decide the answer
- someone described a broken hip and mentioned neither — **not a question at
  all**, because the room-rent cap was never what they were asking about

Both mean "no comparison was made", which is why one status looked sufficient.
They differ entirely in what that means, so `reduction.py` now separates them,
and the second prints nothing:

```python
    # NEITHER figure given: the person never mentioned a room or a sum
    # insured, so this cap is simply not what their question is about.
    # Saying "cannot be determined" here is technically true and actively
    # harmful - it manufactures doubt about a clause nobody raised.
    if not sum_insured and not charged:
        return ReductionCheck(
            clause.clause_id, ReductionKind.ROOM_CAP,
            ReductionStatus.NOT_RAISED,
            "no room charge or sum insured was mentioned",
        )
```

The framing was made proportional too. Nine lines explaining how reductions
interact with a verdict earn their space when a co-payment has actually been
computed; on a question about a broken hip, where the only content is "some
caps exist, read them", they are pure noise. A block that says nothing can
still change the answer if it says nothing at length.

> **The generalisable point:** when you inject computed facts into a prompt,
> you are not only choosing what is true, you are choosing what is *salient*.
> A true statement about something nobody asked about is not neutral — it is
> evidence that the question is open. In a system whose whole purpose is to be
> able to say "I don't know", manufacturing doubt is not a small bug.

---

## Concept 25: the honest read on a fix that did not pay for itself

Three measured runs on the 40-case set:

| Run | State | Verdict accuracy |
|---|---|---:|
| 1 | baseline — no reduction sweep | **0.725** |
| 2 | sweep, every status given its own line | 0.600 |
| 3 | sweep, with `NOT_RAISED` silent and framing sized to stake | **0.675** |

Run 3 recovered most of the regression and **still did not reach the baseline.**
Case by case against run 1:

```
FIXED (3)                      BROKEN (5)
+ copay-applies-emergency      - initial-waiting-period
+ icu-rate-breach              - dental-no-accident
+ oral-chemo-limit             - cataract-served
                               - copay-just-under-sixty
                               - non-disclosure
                               net -2 cases
```

**So the reduction sweep, as integrated, is a net negative.** That is the
result, and it is recorded here as it came out.

It is worth being precise about *which* half failed, because the two halves have
very different standing:

- **The arithmetic is correct and is not in question.** `reduction.py` compares
  8,000 against 1% of 10 lakh, 12,000 against 2% of 5 lakh, and 58 against 60,
  and it gets all of them right every time. 22 unit tests in
  `api/tests/test_reduction.py` pin the boundaries, including that "completed
  sixty years" is satisfied at exactly 60 and not at 59. Those comparisons were
  previously being made by a 7B model inside a paragraph of legal prose, and it
  got them wrong in both directions. That is a genuine correctness fix and it
  stands whatever the aggregate does — the same distinction drawn in the
  previous milestone between correctness fixes and prompt-shaped changes held
  loosely.

- **The prompt integration is what costs more than it earns.** Feeding those
  correct results into the reasoning step still pushes the model toward
  `conditional` and `insufficient_information` on questions the reductions have
  nothing to do with.

And the evidence points at exactly one remaining culprit. Three of the five
broken cases — `initial-waiting-period` (a chest infection two weeks into a
30-day bar), `dental-no-accident`, `non-disclosure` — mention no money, no room
and no age. Every computed check on them returns `NOT_RAISED` and prints
nothing. The *entire* block those three received was one line:

```
- Note: clauses 5.2, 5.4, 5.5 cap what is paid for the treatments they
  name. Read them; if this treatment is not one of them, they decide
  nothing here.
```

All three had been confident, correct refusals at baseline. All three became
`insufficient_information`.

That line carries no computed finding. It is a reminder to read three clauses
whose full text is already in the prompt a few hundred tokens further down. It
buys nothing and, on this evidence, costs three cases.

> **The lesson, which is the same one this milestone keeps re-teaching in new
> costumes:** every line added to a prompt is a claim about what matters. A line
> that adds no information still adds emphasis, and emphasis is not free. "It
> can't hurt to remind it" is a hypothesis, and on this set it measured false.

---

## What this milestone actually delivered

Stated plainly, because two of the three are worth more than the headline
number suggests:

1. **A ruler that works.** 40 cases instead of 16, weighted toward the failure
   mode, with a false-citation metric that can see over-citing. The previously
   reported 0.812 was a measurement of an easier exam; the honest figure for
   the same code is 0.725.
2. **Two integer comparisons taken away from the model**, with tests, in the
   family that the earlier duration fix had missed.
3. **A prompt integration that does not yet pay for itself**, diagnosed to a
   specific line rather than left as a mystery.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 129 tests
python evals/run_scenario_eval.py                        # writes evals/scenario-report.md
python evals/run_all.py                                  # writes evals/REPORT.md
```

To watch the failure in this entry directly, render the block for a scenario
that mentions no money and no age:

```python
from app.pipeline.reduction import evaluate, render
# every check returns NOT_RAISED; only the judgement line survives
print(render(evaluate(policy_clauses, {}, policy_age_days=1825)))
```

**Question to sit with:** the arithmetic in this milestone is provably correct,
and feeding it to the model made the answers worse. Two of the three runs
above were spent discovering that being right is not the same as being useful
to say.

What would you check *before* adding your next correct fact to a prompt?

<details>
<summary>Answer</summary>

**Whether the fact is relevant to the question being asked — and if you cannot
determine that, whether saying nothing is the safer default.**

The failures in this entry are not failures of accuracy. Every statement the
block made was true. They are failures of *relevance*, and relevance is
something the injecting code has to decide, because the model cannot: a fact
placed in front of it has already been marked as worth its attention by the
act of placing it there.

That gives a concrete pre-flight check for any computed fact you are about to
inject:

1. **Can this fact change the answer to THIS question?** If the person never
   mentioned a room, the room-rent cap cannot change their answer. Print
   nothing. This is the difference between `NOT_RAISED` and `UNKNOWN`, and it
   was worth 0.075 of accuracy on its own.
2. **Is its prominence proportional to its decisiveness?** Four satisfied
   waiting periods are one non-event, not four findings. Three caps that might
   apply are not three results.
3. **Does it carry information the model does not already have?** The judgement
   line failed this test — it pointed at clauses whose full text was already in
   the same prompt.

The deeper point is that a prompt is not a database the model queries as
needed. It is a *briefing*, and everything in it is implicitly asserted to be
worth reading. Adding true-but-irrelevant material does not leave the answer
unchanged; it shifts what the model believes the question is about. In a system
designed to say "I don't know" when the document is silent, manufactured doubt
is not a neutral failure — it converts correct refusals into abstentions, which
is precisely what happened to three cases here.
</details>

---

# M9 — The ruler was made of rubber, and two requests at once was why

## The starting position

The scenario simulator answers a what-if question about a health insurance
policy — *"I had knee surgery 8 months after buying this"* — with one of four
verdicts (`covered`, `not_covered`, `conditional`, `insufficient_information`)
and the clauses that decide it. Its quality is measured by running 40
hand-written cases in `evals/golden/scenarios.json` and counting how many
verdicts match the expected one.

The previous milestone had ended on an unresolved note. It added a module that
does insurance arithmetic in code rather than asking the model — comparing a
room rate against 1% of the sum insured, an age against a co-payment threshold
— and then fed those computed results into the reasoning prompt. The
arithmetic was right. The aggregate got **worse**, from 0.725 to 0.675, and the
cause had been traced to one specific line of prompt text.

That line appeared only when every computed check came back "this question
doesn't raise this cap at all". With nothing computed to report, the block
still printed:

```python
# api/app/pipeline/reduction.py, before this milestone
computed = applies or unknown or ruled_out
if not computed:
    return (
        f"- Note: clauses {', '.join(c.clause_id for c in judgement)} cap "
        f"what is paid for the treatments they name. Read them; if this "
        f"treatment is not one of them, they decide nothing here."
    )
```

Three cases that had been confident, correct refusals — a chest infection two
weeks into a 30-day waiting period, a dental crown with no accident, an
undisclosed thyroid condition — became `insufficient_information` when that
line was present. None of them mentions money, a room, or an age. The line
told the model to go read three clauses whose full text was already sitting in
the same prompt a few hundred tokens below.

So this milestone began with an obvious task: delete the line, re-measure,
expect roughly +3 cases.

---

## What deleting the line did

The branch now returns nothing at all:

```python
# api/app/pipeline/reduction.py
computed = applies or unknown or ruled_out
if not computed:
    return ""
```

Re-running the scenario eval moved verdict accuracy from **0.675 to 0.725**,
citation recall from 0.774 to 0.806, and the quote fabrication rate from 0.125
to 0.050.

Two of the three predicted cases came back. `dental-no-accident` recovered.
`initial-waiting-period` and `non-disclosure` did not.

Before accepting "the diagnosis was two-thirds right", it is worth checking the
mechanism rather than the outcome. The claim was that those cases receive an
empty block now. That is checkable directly, by wrapping the function and
printing what it returns for those exact scenarios:

```
initial-waiting-period | I got a bad chest infection two weeks after my policy started...
statuses: [('5.1','not_raised'), ('5.2','judgement'), ('5.3','not_raised'), ('5.4','judgement'), ('5.5','judgement')]
block repr: ''
```

Empty, as designed — and the case still fails. So for those two the deleted
line was never the cause, and something else was making them abstain.

That was the correct conclusion from the evidence available. It was also built
on sand, for a reason that had nothing to do with insurance.

---

## Failure 21: the same code, measured three times, gave three different numbers

To publish the result properly, the full report needed regenerating. The
project's convention is that `PROMPT_VERSION` in `api/app/llm/prompts.py` is
bumped whenever the words sent to the model change, because that string is part
of every cache key — so bumping it discards every cached model response and
forces a genuinely fresh run:

```
v12-reductions  ->  v13-silent-reductions
```

The fresh run scored **0.700**, not the 0.725 measured minutes earlier on the
same code.

A third run, with the cache cleared again, scored **0.750**.

| Run | Cache state | Verdict accuracy |
|---|---|---:|
| 1 | warm (only changed prompts regenerated) | 0.725 |
| 2 | cold (everything regenerated) | 0.700 |
| 3 | cold (everything regenerated) | 0.750 |

Identical code. Identical prompt text — `PROMPT_VERSION` is a cache key and is
never rendered into a prompt, which is worth verifying rather than assuming:

```bash
grep -rn "PROMPT_VERSION" api/app/llm/ api/app/pipeline/
# every hit passes it to cache.make_key(); none concatenates it into a message
```

And `temperature` is 0.0.

Comparing the three runs case by case, **five of the forty cases flip between
runs**:

```
ALWAYS WRONG (8)              FLIPS BETWEEN RUNS (5)     ALWAYS RIGHT (27)
ped-waiting-served            senior-copay
initial-waiting-period        cataract-served
accident-in-initial-period    copay-just-under-sixty
room-rent-within-cap          senior-but-excluded
intoxication-injury           breach-of-law
non-disclosure
post-hospitalisation-too-late
day-care-not-listed
```

So the system's real score is *27 guaranteed passes plus up to five coin
flips*: somewhere in 0.675–0.800, and which end gets reported is luck.

**This invalidates more than one milestone's conclusions.** The previous
milestone compared 0.725 against 0.675 and wrote a careful diagnosis of why a
change had cost two cases. Two cases is inside the noise. The diagnosis may
still be right — its mechanism was verified independently — but the *number*
never supported it. The milestone before that had already noticed five runs
landing on 0.688 / 0.625 / 0.812 / 0.688 / 0.812 and concluded the 16-case set
was too small to resolve the changes being made against it. That conclusion was
half right in an expensive way: the set was too small, but enlarging it to 40
cases did not fix the problem, because the problem was never the set size.

> **The lesson:** when a measurement moves, there are always two explanations —
> the thing you changed, or the instrument. Nothing in this project had ever
> checked the second one. An eval you have not tested for reproducibility is
> not an instrument, it is an anecdote generator, and it will produce a
> confident story for any change you make.

---

## Concept 26: `temperature = 0` is not reproducibility

This is the assumption that went unexamined for eight milestones, and it is
worth taking apart properly because almost everyone building on a local model
holds it.

**What temperature actually does.** A language model does not emit a token. It
emits a *score for every token in its vocabulary* — roughly 150,000 numbers for
qwen2.5. Those scores are called logits. Turning them into a choice needs a
sampling rule, and temperature is the knob on that rule:

- **temperature = 1.0** — convert the logits to probabilities and draw from
  that distribution. A token the model rates at 30% is chosen about 30% of the
  time. Genuinely random, differently each run.
- **temperature = 0.7** — sharpen the distribution first, so likely tokens get
  likelier and unlikely ones nearly vanish. Still random, less adventurous.
- **temperature = 0.0** — the limit of that sharpening: *always take the
  highest-scoring token*. No draw, no randomness. Also called greedy decoding.

So at temperature 0 the sampler is deterministic. **Given the same logits, it
returns the same token, every time.** That much is true, and it is where the
reasoning usually stops.

**The hidden clause is "given the same logits."** The sampler is deterministic;
producing the logits is a separate question. And on a GPU, the same prompt
through the same weights does not reliably produce bit-identical logits — which
means the argmax can land on a different token, and from there the two
generations diverge completely, because every subsequent token is conditioned
on a text that now differs.

That is why the divergence, when it appears, is never subtle. Two runs of one
scenario:

```
run A: {"deciding_clauses": [{"clause_id": "3.1", "effect": "reduces", ...
run B: {"deciding_clauses": [{"clause_id": "4.3", "effect": "denies",  ...
```

One says a waiting period reduces the claim; the other says an exclusion denies
it. A single flipped token at position 37, and the answers have nothing in
common. Near-ties are not rare events at the margins — they are the normal
condition of a 150,000-way choice, and there is one at every token.

---

## Failure 22: the fix I was certain of, which changed nothing

The first hypothesis was the seed. Ollama accepts a `seed` option and the
project was not sending one, so Ollama drew a random one per request. The
decoding options are built in one place:

```python
# api/app/llm/client.py, before this milestone
return {
    "temperature": settings.temperature,
    "num_ctx": settings.num_ctx,
    "num_predict": settings.num_predict,
}
```

No seed. Pinning it looked like a one-line fix to a whole milestone's worth of
confusion.

It made no difference. With `"seed": 0` sent explicitly, three sequential calls
with identical messages still returned two different answers.

**Why it could never have worked, in hindsight:** a seed feeds a random number
generator, and at temperature 0 *nothing draws from that generator*. The
sampler takes an argmax. Seeding it is seeding a die that is never rolled. The
hypothesis contradicted the very explanation of temperature that justified it,
and it survived precisely because the fix felt cheap enough not to think hard
about first.

The seed is still pinned in `api/app/config.py`, because a system whose premise
is reproducibility should not leave an input unspecified. But it is documented
there for what it is: worth stating, and measured to fix nothing.

---

## The tool that settles it, rather than another opinion

The right response to "I do not know whether my measurements are real" is not a
better guess. It is an instrument. `evals/check_determinism.py` sends the same
input N times with the cache bypassed and compares the results byte for byte,
at two layers that fail for different reasons:

```
MODEL     the same messages, sent N times with the cache bypassed, must come
          back byte-identical. This is llama.cpp's decoding, with no pipeline
          involvement at all.

PIPELINE  the analysis stage run N times must produce identical structured
          output. This is the same model plus THIS PROJECT'S concurrency.
```

Separating the layers is the whole design. A single pass/fail would have said
"not reproducible" and left the cause open. Two layers turn the result into a
diagnosis, because **the model layer passing while the pipeline layer fails
points at the project's own code**, not at llama.cpp.

That is exactly what happened. Eighteen sequential model calls, byte-identical
every time. The pipeline layer, diverging — and always inside `plain_language`,
the free-text rewrite, where near-ties are densest.

Bypassing the cache without destroying it needed one small change to production
code: threading the flag that `complete_json` already had through the stage
that did not expose it.

```python
# api/app/pipeline/analyze.py
async def analyze(
    segments: list[Segment],
    *,
    batch_size: int | None = None,
    concurrency: int | None = None,
    progress=None,
    use_cache: bool = True,
) -> dict[str, ClauseAnalysis]:
```

The eval harness's existing `--no-cache` calls `cache.clear()`, which deletes
every cached response in the database. That is right for a full eval run and
useless for a check that wants to re-run twelve clauses without throwing away
twenty minutes of unrelated work.

---

## Failure 23: claiming the cause after one observation each way

With the layers separated, the suspect was `analyze_concurrency`, which was 2 —
the analysis stage keeps two requests in flight at once. One run at
concurrency 2 diverged; one run at concurrency 1 was identical.

That looked like the answer, and it was written up as the answer.

Then concurrency 2 was run again and **passed**. One failure and one pass at
the same setting is not evidence of anything — it is two coin flips, and it is
the same error as reading a two-case difference on a noisy eval. Intermittent
faults need repeats, and the check already took `--repeats`:

```
concurrency=1, 6 repeats:  6/6 identical                        OK
concurrency=2, 6 repeats:  3 of 6 diverge, alternating exactly  FAIL
concurrency=4, 6 repeats:  1 of 6 diverges                      FAIL
```

The alternation at concurrency 2 — `9046 / 9080 / 9046 / 9080 / 9046 / 9080`
characters, perfectly regular — is the tell. Random corruption does not
alternate. Request interleaving does.

---

## Concept 27: float addition is not associative, and that is a product bug

School arithmetic says `(a + b) + c` equals `a + (b + c)`. Floating-point
arithmetic says no. Each intermediate result is rounded to fit 32 or 16 bits,
and rounding at a different point loses a different crumb:

```
(1e16 + 1.0) - 1e16   ->  0.0     the 1.0 is rounded away, then subtracted
1e16 + (1.0 - 1e16)   ->  2.0     ...depending entirely on the grouping
```

A transformer layer is millions of such additions. The GPU splits them across
thousands of threads, and **how it splits them depends on the shape of the work
it is given**. One request alone is one shape. Two requests batched together is
another: the kernel processes both sequences in one matmul, different threads
take different slices, and the partial sums are combined in a different order.

The result differs in the last bits. That is invisible almost everywhere — and
decisive at a near-tie, where two tokens are separated by less than the error.
The tie breaks the other way, and the two generations part company for good:

```
run A: "A hospital is any place that provides inpatient or day care treatment..."
run B: "A hospital is any place that takes sick people in for treatment..."
```

Nothing is corrupted here. Both readings are correct. **But they are different
readings of the same clause, produced by the same code from the same PDF**, and
that is the part that matters beyond the eval: this was never only a
measurement problem. A user who uploads their policy, closes the tab, and
uploads it again gets a different plain-English rewrite of their own document.
The cache hides it — identical input, cached response — but a cache is an
optimisation, not a guarantee, and it is empty the first time anything runs.

---

## Concept 28: measure the cost of the safe choice before assuming you cannot afford it

Serialising the analysis stage was obviously correct for reproducibility, and
obviously expensive. Two requests in flight, so removing one should halve
throughput — 200-clause policies going from minutes to many minutes, a real
cost to a real user, for a property only the eval harness cares about.

That is the sort of tradeoff worth agonising over, and it is worth ten minutes
to check the number first. Same 40-clause policy, cache off, both settings:

```
concurrency=1: 40 clauses in 203.0s
concurrency=2: 40 clauses in 184.2s
```

**Nine percent. Nineteen seconds.**

One request already saturates an 8GB card — the GPU has no idle capacity for a
second one to use, so the second mostly queues. The original comment in the
config had recorded that "2 measured better than 4 on an 8GB card" without
following the trend one step further, to 1.

The agonising tradeoff did not exist. The concurrency was buying nineteen
seconds and costing the ability to measure anything at all.

> **The lesson:** an unmeasured cost will be estimated, and estimates of
> performance are reliably wrong in the direction that justifies keeping what
> you already have. The expensive-sounding safe choice is often nearly free,
> and finding that out is usually cheaper than the deliberation it replaces.

---

## Failure 24: the residual, and a second fix that measured nothing

Serialising the analysis stage removed a proven source of nondeterminism. It
did not make the eval reproducible.

Two full runs on the serialised code, both with every model response generated
fresh, scored **0.650** and **0.675**. Twelve of the failing cases were common
to both; three differed - `breach-of-law` and `cataract-served` failed in the
first and passed in the second, `other-policy-contribution` the reverse.

So there is a second source, and the search for it produced one more refuted
hypothesis worth recording.

**The observation.** Sending the same reasoning prompt three times back to back,
with the cache bypassed, gives:

```
call 1: 1015 chars
call 2: 1384 chars
call 3: 1384 chars
```

Only the first differs. Interleaving a second, unrelated scenario between the
repeats does not disturb it - calls 2 through 6 all agree. Whatever is
different about the first call, it is not the identity of the request before
it.

**The hypothesis.** Something about a process's first generation is unlike the
ones after it, and the project already had a function whose job was to get that
out of the way early:

```python
# api/app/llm/client.py
async def warm() -> None:
    """Preload the model into VRAM."""
    ...
    # An empty message list asks Ollama to load the model and stop.
    json={"model": settings.model, "messages": [], "stream": False},
```

An empty message list loads the weights and returns **without decoding a single
token**, so the first real request of the process was still the first
generation. The app calls `warm()` at startup; the eval harness never called it
at all. Making it run one throwaway one-token generation looked like the fix,
and it would have fixed a product bug too: the first policy uploaded after
server startup would have had one clause read differently from the way it would
be read a minute later.

**It changed nothing.** With a real warm-up generation first, the same test:

```
call 1: 1961 chars
call 2: 1364 chars
call 3: 1364 chars
```

Still the odd one out. The change was reverted rather than kept, because
keeping an unmeasured improvement on the grounds that it cannot hurt is the
exact reasoning this project has already been burned by twice - once with four
true statements about waiting periods, once with a one-line note about
sub-limits.

**The likely mechanism, stated as a hypothesis rather than a conclusion:**
llama.cpp reuses the cached KV of a matching prompt prefix. The first time a
6,000-token prompt arrives, all 6,000 tokens are prefilled in batches. The
second time the identical prompt arrives, the prefix is already in the cache
and almost nothing is prefilled. Those are different computations over the same
numbers, and by Concept 27 they need not agree in the last bits. A one-token
warm-up on the word "ok" shares no prefix with a 6,000-token policy prompt, so
it cannot prevent the full prefill that follows.

If that is right, the fix is not a warm-up but pinning whatever varies in the
prefill path - and confirming it needs a test that isolates prefill batching,
which does not exist yet. **It is recorded here as open, because the honest
state of this milestone is one source found and removed, one source identified
and not yet removed.**

---

## What this milestone delivered

1. **A one-line prompt deletion**, correct on its own terms: it removed text
   carrying no information the model could not already read, and two tests pin
   that the block is empty when nothing was computed. Its measured effect is
   **within the noise and cannot be claimed** - the case its recovery was
   attributed to, `dental-no-accident`, failed again on a later run.
2. **The discovery that every eval number this project has compared was read
   through ±2-3 cases of noise**, and that a third of the failing set are coin
   flips rather than verdicts.
3. **`evals/check_determinism.py`**, which turns "is this reproducible" from an
   assumption into a command, at two layers that fail for different reasons.
4. **One proven source removed**: concurrent requests, 6/6 identical at
   concurrency 1 against 3-of-6 diverging at 2, for a measured 9% of analysis
   wall time.
5. **A product bug fixed as a side effect**: the same policy read twice at
   concurrency 1 now gives the same reading, where before it did not.
6. **An open residual**, characterised to a single reproducible observation -
   the first substantive generation of a process differs from every one after
   it - with the obvious fix tried, measured, and rejected.

The headline number is **0.650**, from the run that produced
`evals/REPORT.md`. It is lower than the 0.675-0.750 this project has been
reporting, and none of that drop is a regression: it is the first number taken
on a pipeline where the largest source of variance had been removed, and the
range it replaces was never a measurement of one thing.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q     # 130 tests
python evals/check_determinism.py                   # MODEL and PIPELINE layers pass;
                                                    # the SEQUENCE layer fails - see M10
```

To watch the failure this milestone is about, ask for the old setting back:

```bash
python evals/check_determinism.py --repeats 6 --concurrency 2
```

The model layer passes and the pipeline layer fails, which is the shape of the
whole diagnosis in one screen.

**Question to sit with:** the eval harness in this project was written to stop
model and prompt changes being justified by impressions instead of numbers. It
did that job for eight milestones while quietly producing numbers that moved on
their own.

What makes a measurement trustworthy, if not the fact that it is a number?

<details>
<summary>Answer</summary>

**That you have measured the measurement.** A number earns trust by having a
known error bar, and nothing about being a number supplies one.

The specific trap here is that noise is invisible from inside a single run. One
run of this eval produces `0.700` - not "0.700 plus or minus 0.05", not a
distribution, just a decimal with three digits of implied precision. Every
property that would warn you is missing from the output: the spread, the number
of times it was measured, which cases are unstable. Print a mean without a
variance and readers supply a variance of zero, because that is what the format
implies.

Concretely, three habits separate an instrument from an anecdote generator:

1. **Run it twice before you trust it once.** The cheapest possible experiment,
   and this project went eight milestones without it. Two identical runs that
   disagree tell you more than a hundred single runs that agree with your
   hopes.
2. **Report instability as a first-class result.** "27 always right, 8 always
   wrong, 5 flip" is a far more useful sentence than "0.725". It says exactly
   where the system is solid, where it is broken, and where it has no
   conviction at all - and only the middle number is a quality problem.
3. **Never let a delta smaller than the noise floor justify a decision.** The
   previous milestone's two-case regression, agonised over and diagnosed in
   detail, was smaller than the instrument's own error.

The deeper point is that an eval harness is itself a piece of software, and it
is the one piece nobody tests, because its output *is* the test. That makes its
failure mode uniquely quiet: a broken test suite goes red, while a broken
measurement just keeps producing plausible decimals that send you looking for
explanations in the wrong place. The unstable cases here had been read as
evidence about prompt wording across two milestones. At least some of them were
evidence about batch scheduling on a GPU.
</details>

---

# M10 — Measuring only what changed, and a third family of arithmetic

## The starting position

The scenario simulator answers a what-if question about an Indian health
policy — *"I had a follow-up scan 120 days after I was discharged"* — with one
of four verdicts (`covered`, `not_covered`, `conditional`,
`insufficient_information`) and the clauses that decide it. Its quality is
measured by 40 hand-written cases in `evals/golden/scenarios.json`.

The previous milestone established two uncomfortable facts about that
measurement:

1. **The model does not answer identically twice**, even at temperature 0. One
   source (concurrent requests) was found and removed; a residual remains,
   inside the inference runtime, and could not be switched off.
2. **So the aggregate score moves on its own.** Runs of identical code scored
   anywhere from 0.650 to 0.750. A prompt change worth two or three cases was
   indistinguishable from doing nothing.

It also left eight cases that failed in every run — real bugs rather than
noise. This milestone set out to do two things: make the eval able to measure
a change despite the wobble, and fix the clearest of the eight.

---

## One more measurement before anything else: repeat is not sequence

The determinism check in `evals/check_determinism.py` had two layers. Both sent
the **same** input several times and compared the outputs. Both passed.

An eval never does that. It sends forty **different** prompts, in order, once
each. So a third layer was added that does exactly that — runs real eval
scenarios in order, twice, and compares pass one against pass two case by case:

```
SEQUENCE (10 cases, 2 passes): 7 case(s) differ  FAIL
```

Those are two different properties, and a system can have one without the
other:

    repeat stability    asking the same question twice gives one answer
    sequence stability  asking forty questions gives the same forty answers

The first comparison was byte for byte, and reported 9 of 10 differing — true
and nearly useless, because most of the difference was in the free-text
`reasoning` paragraph that no metric reads. Comparing only what the metrics
consume (the verdict, the cited clauses, the quotes) still gave **7 of 10**.
Verdicts turned out fairly steady; *citations* did not, which is why citation
recall had been wandering between runs nobody had changed anything in.

> **The lesson:** test the property your conclusion depends on, not a
> neighbouring one that is easier to check. Two passing layers had said "this
> is reproducible". The property the eval rests on was never among them.

---

## Concept 29: the cache as a controlled experiment

If the model's answers cannot be made identical, the noise has to be kept out of
the *comparison* instead. The tool for that already existed, and had been
treated as a mere speed-up.

**What a content-addressed cache is.** Every model request is hashed — the
model name, the exact messages, the JSON schema, the decoding options — and the
hash is the key under which the answer is stored:

```python
# api/app/llm/cache.py
def make_key(model, messages, schema, options=None) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": schema,
         "options": options or {}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

"Content-addressed" means the address *is* the content: change one character of
the prompt and you are asking for a different key.

**Why that makes an experiment.** Suppose you edit a prompt that only three of
the forty eval cases actually send. The other thirty-seven send byte-identical
text, find their key, and get back the *stored bytes* of their earlier answer.
Those answers cannot have moved — not "are unlikely to have", **cannot**. Only
the three cases whose text changed reach the model and produce new samples.

In a lab, a control group is the part of the experiment you deliberately leave
untouched, so that any difference in the treated part can be attributed to the
treatment. The cache hands you a control group for free, and it is a better one
than a lab gets: the untreated cases are not merely similar to before, they are
identical.

So instead of comparing two blended scores, `evals/run_scenario_eval.py` now
reports which cases a change could have reached. Here is a real run, made to
test exactly this: one word changed in the heading of the cover-window block
(introduced later in this entry), which only two of the forty questions
receive. The prediction was 38 replayed and 2 regenerated:

```
Compared with the previous run (2026-09-13 17:31 UTC, v15-cover-window-section):

   38 replayed from cache   same text sent, stored answer returned - cannot have moved
    2 regenerated         the only cases this run could have changed
        fixed          0
        broken         0
        still right    2
        still wrong    0
```

Thirty-eight cases contributed no noise at all, and the two that could have
moved did not.

To know which cases were regenerated, the model client counts requests that
actually reach the model:

```python
# api/app/llm/client.py
model_calls = 0
...
    global model_calls
    model_calls += 1
```

and the eval reads that counter before and after each case. Every run's
per-case verdicts, and whether each was fresh, are appended to
`evals/run-history.json`, which is what the comparison, the stability report and
the `--watchlist` flag (re-run only cases that failed or wobbled) are computed
from.

---

## Failure 25: the cache's row count lies in exactly one case

The obvious way to detect "was this answer regenerated" is to count the cache's
rows before and after: a new answer adds a row.

It does not always. When a response is cut off mid-JSON, the client retries
with a larger output limit, and stores the answer under a key built from the
*new* options:

```python
# api/app/llm/client.py
options = {**options, "num_predict": options["num_predict"] * 2}
...
cache.put(cache.make_key(model, messages, schema, options), model, parsed)
```

The next run looks the request up under the *original* options, misses,
regenerates, retries again, and overwrites its own row. The table does not
grow, so a regenerated answer would have been reported as replayed — the one
error the whole comparison cannot afford. Hence counting model calls directly.

---

## Failure 26: counting the wrong calls

The first version marked a case "fresh" if **any** model request happened while
answering it. But every case first extracts facts from the question, and the
very next change in this milestone edited the fact extractor's prompt. Every
case would have shown as regenerated — including cases whose facts came out the
same, whose reasoning prompt was therefore byte-identical, and whose verdict
was the stored answer.

The verdict comes from the reasoning step alone. So the eval now extracts facts
*first*, outside the count, and counts only what happens after:

```python
# evals/run_scenario_eval.py
await extract_facts(case["scenario"])
calls_before = client.model_calls
result = await run_scenario(case["scenario"], clauses, resample=resample)
fresh = client.model_calls > calls_before
```

`run_scenario`'s own fact extraction is then a cache hit, and "fresh" means
exactly "the verdict could have changed".

---

## Failure 27: advice that would have confirmed a result with a copy of itself

When a regenerated case flips, it is one sample from a model that does not
answer identically twice. The comparison's output originally said:

```
not evidence on its own - re-run it with --only before believing it.
```

`--only` runs a chosen subset. But re-running a case replays its **stored**
answer — the very sample being doubted. Following that advice would have
"confirmed" every fix, always, by reading it back.

A second opinion needs a genuinely new sample of the *same* prompt, so the
reasoning step gained a way to bypass the cache without writing to it:

```python
# api/app/pipeline/scenario.py
async def run_scenario(scenario, clauses, *, resample: bool = False):
    ...
    payload = await reason(scenario, facts, considered, use_cache=not resample)
```

```bash
python evals/run_scenario_eval.py --only post-hospitalisation-too-late --resample
```

Nothing resampled is stored, so the original answer stays the baseline. Every
result in this entry marked "×3" was checked this way.

---

## Concept 30: a version label is not a cache key

The project's convention had been to bump `PROMPT_VERSION` whenever a prompt
changed, *because the version was part of every cache key* — so bumping it
guaranteed a reworded prompt never got a stale answer.

Two things were wrong with that, and the second only became visible once the
cache was being used as a control group.

**It was redundant.** The key already hashes the message text. A reworded prompt
is different text and gets a different key whether or not anyone remembers to
bump anything.

**It destroyed the control group.** A version bump changes *every* key, so a
change to one prompt discarded the stored answer to every prompt. All forty
cases regenerated, each a fresh chance to wobble, and a three-case change was
buried under forty cases of noise — exactly the situation Concept 29 exists to
avoid.

So the version was taken out of the key and kept as a label:

```python
# api/app/llm/prompts.py
PROMPT_VERSION is a LABEL. Bump it whenever a prompt's wording changes, so eval
reports, the run history and stored analyses record which instructions produced
them. It is not part of the cache key - the cache hashes the prompt text
itself...
```

The guarantee the old test protected — editing a prompt must never serve a
stale answer — is still tested, now against the thing that actually provides
it:

```python
# api/tests/test_llm.py
def test_changing_the_prompt_text_changes_the_cache_key():
    reworded[-1]["content"] = reworded[-1]["content"] + " "
    ...
    assert len({k1, k2, k3, k4}) == 4
```

A bonus appeared the first time a change was reverted: going back to an earlier
prompt replayed that prompt's stored answers exactly, instead of regenerating a
new and different sample of what used to be measured.

---

## The fix: a cover window, decided in code

Indian health policies pay for treatment around a hospital stay only within a
window. The synthetic policy says:

```
2.2 ... Medical Expenses incurred during the sixty days immediately preceding
    the date of admission ...
2.3 ... Medical Expenses incurred during the ninety days immediately following
    the date of discharge ...
```

The case `post-hospitalisation-too-late` — *"a follow-up scan a hundred and
twenty days after I was discharged"* — failed in every run on record. The model
answered `covered` or `conditional`. Is 120 more than 90?

That is the third time this project has found the same mistake. `waiting.py`
took duration comparisons away from the model; `reduction.py` took money and
age comparisons away. This is a third family, and it gets the same split:

    reading "the ninety days immediately following discharge"  -> the model
    reading "a scan 120 days after I was discharged"           -> the model
    deciding whether 120 <= 90                                 -> window.py

The clause reader records each window as a value, a unit, and an **anchor**;
the fact extractor records the person's expense the same way; and
`api/app/pipeline/window.py` compares them:

```python
# api/app/pipeline/window.py
if anchor != clause_anchor:
    status = WindowStatus.NOT_RAISED
elif offset_days is None:
    status = WindowStatus.UNKNOWN
elif offset_days <= window:
    status = WindowStatus.WITHIN
else:
    status = WindowStatus.OUTSIDE
```

**Why the anchor exists.** A window has a side. Fifty days *before admission*
says nothing about the window *after discharge*; comparing them compares numbers
that measure different things. It is the same operand trap an earlier milestone
hit, when a senior co-payment turned out to key on age *at inception* rather
than age now.

**Why NOT_RAISED prints nothing.** Most questions never mention a stay's before
or after. An earlier milestone measured what happens when a prompt carries a
note about something the question never touched: confident, correct answers
turn into `insufficient_information`. So an unraised window renders to an empty
string, and ten tests in `api/tests/test_window.py` pin the boundaries,
including that day 90 of a 90-day window counts as inside it.

It also fixed a misreading nobody had noticed. The clause reader had been
putting clause 2.2's "sixty days before admission" into the *waiting period*
field, and `waiting.py` then told every scenario that 2.2 was a 60-day waiting
period. With a field of its own, the window stopped leaking into the wrong one.

---

## Failure 28: one paragraph of instructions reclassified clauses it never mentioned

The first version of the clause reader's new instructions did two things: added
a note to the definition of `waiting_period` saying a hospital-stay window is
not one, and described the three new fields in the middle of the existing
field list.

Clause 2.2 was fixed. Classification accuracy across all clauses fell from
**1.000 to 0.919**:

```
waiting_period -> exclusion   (3.2)
sub_limit      -> condition   (5.3, the senior co-payment)
condition      -> procedural  (6.5, medical examination)
```

None of those clauses has anything to do with a hospital stay. Removing one
piece of the edit at a time isolated the cause:

| Instructions | 2.2 | 5.3 | 6.5 |
|---|---|---|---|
| before this milestone | waiting_period ✗ | sub_limit ✓ | condition ✓ |
| + field description only | coverage ✓ | condition ✗ | condition ✓ |
| + field description + type note | coverage ✓ | condition ✗ | procedural ✗ |
| field description **moved to its own section** | coverage ✓ | sub_limit ✓ | condition ✓ |

The type note did nothing but harm; the field description alone fixed 2.2. And
the field description broke the co-payment clause purely by **where it sat** —
inserted just above the co-payment fields. Moved to its own short section after
them, it broke nothing: 39 of 39 clauses correct, in four separate samples.

(Clause 3.2's misreading was different: resampled, it came back correct every
time. The eval run had simply drawn an unlucky reading — and every one of that
run's forty scenarios was then shown a waiting period labelled as a permanent
exclusion. One bad sample upstream contaminates everything downstream of it.)

> **The lesson:** a small model does not read instructions the way a person
> does, one relevant rule at a time. Every line shifts the context for every
> other line, and the effects are not local. A prompt edit has to be checked
> against everything the prompt does — here, all 40 clauses — not only against
> the one thing it was written to fix. Checking three clauses would have shipped
> the version that broke two others.

---

## Failure 29: a correct fix that measured net negative, and was reverted

The window change introduced one stable regression. Asked *"Does this policy
cover my car being stolen from the hospital car park?"*, the correct answer is
`insufficient_information` — a health policy is silent on car theft. After the
change it answered `not_covered`, six samples out of six, citing the cosmetic
surgery exclusion with a quotation that does not exist.

The cause was in the facts. The extractor has a free-text `notes` field, and
for this question it wrote:

```
notes: This policy does not cover car theft from the hospital car park.
```

That is not something the person said. It is the extractor *answering the
question* — and the note reaches the reasoning step under "FACTS UNDERSTOOD",
presented as a stated fact.

The structural fix was obvious and principled: never show `notes` to the
reasoner. Its content is a paraphrase of the question, which the reasoner
already receives verbatim. One line, one test, and a leaked conclusion becomes
impossible rather than discouraged.

Measured three times each:

| Case | Before | Notes removed |
|---|---|---|
| `not-in-document` | ✗ ✗ ✗ | ✓ ✓ ✓ |
| `no-preauth-cashless` | ✓ in every earlier sample | ✗ ✗ ✗ |
| `senior-but-excluded` | ✓ ✓ ✓ ✓ | ✗ ✗ ✗ |
| `other-policy-contribution` | ✓ in 4 of 5 | ✗ ✗ ✗ |

One fix, three breaks, and citation recall down from 0.774 to 0.677. Replaying
the old answers from the cache showed exactly which citations were lost — the
alcohol exclusion for an intoxication injury, the disclosure condition for an
undisclosed thyroid condition. The notes had been restating situations in
something close to policy vocabulary, and that paraphrase was helping a 7B
model find the right clause. Removing the field removed the leak and the help
together.

So it was reverted. This is the same judgement an earlier milestone reached
about a correct piece of arithmetic fed into a prompt (see *Concept 25: the
honest read on a fix that did not pay for itself*): being right in principle is
not the same as helping, and only the measurement can say which. The leak is
recorded as open.

---

## Failure 30: a warning that blamed a cause that did not exist

Reverting the notes change replayed the earlier answers — and the comparison
printed:

```
WARNING: 9 replayed case(s) differ from the last recorded run ...
The cache was filled by a run that was never recorded
```

No such run existed. Those answers came from a recorded run; they differed from
the *immediately previous* run because that run used the notes-removed prompt,
and reverting brings back an earlier prompt's answers. The logic was right to
surface the difference and wrong about why. It now names both causes — a
reverted change, or an unrecorded run — and counts neither as a fix or a break.

---

## Where this leaves the numbers

Stable per-case results, each checked over at least three samples:

| Case | Before this milestone | Now |
|---|---|---|
| `post-hospitalisation-too-late` | ✗ ✗ | ✓ ✓ ✓ ✓ **(the target)** |
| `initial-waiting-period` | ✗ ✗ | ✓ ✓ ✓ ✓ |
| `dental-no-accident` | ✗ ✗ | ✓ ✓ ✓ ✓ |
| `senior-copay` | ✗ ✗ | ✓ ✓ ✓ ✓ |
| `not-in-document` | ✓ ✓ | ✗ ✗ ✗ ✗ (notes leak — open) |
| `copay-unknown-inception-age` | ✓ ✓ | ✗ ✗ ✗ ✗ (cause not found — open) |
| `icu-rate-breach` | ✗ ✗ | ✓ ✗ ✗ ✗ (wobbles) |

Verdict accuracy over two full runs of the final code: **0.750 and 0.725**,
against 0.650 and 0.675 for the code this milestone started from. Clause
classification: **1.000** in four samples.

Why three unrelated cases improved is not proven. The likeliest explanation is
the misreading fixed as a side effect: clause 2.2 had been presented to every
scenario as a 60-day waiting period, and for a chest infection two weeks into a
policy that is a spurious extra bar to reason about. It is recorded as a
hypothesis, not a finding.

**Still open:**

- the `notes` leak, which needs a fix that keeps the paraphrase's help without
  its conclusions;
- `copay-unknown-inception-age`, which started failing with the corrected
  clause instructions and whose cause is not yet found;
- a quotation for clause 2.3 that splices the last three words of clause 2.2
  ("...accepted **by the Company**") onto 2.3's text. The verbatim check catches
  it every time, so the guarantee holds, but the deciding citation is shown as
  unverified.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 148 tests
python evals/run_scenario_eval.py                        # full run, compared with the last one
python evals/run_scenario_eval.py --watchlist            # only cases that failed or wobbled
python evals/run_scenario_eval.py --only post-hospitalisation-too-late --resample
```

Run the plain eval twice in a row and read the second comparison: every case
should say *replayed*.

**Question to sit with:** you change one word in the heading that
`api/app/pipeline/window.py` puts above its block, and run the eval. Before
running it, predict how many of the 40 cases regenerate. Then predict the same
for a one-word change to the waiting-period block in `waiting.py`.

<details>
<summary>Answer</summary>

**Two, and forty** — and the reason is decided by what each person said, not
by the policy.

The window block prints only when a question places an expense before
admission or after discharge. Two of the forty do: `pre-hospitalisation-window`
and `post-hospitalisation-too-late`. Every other question's reasoning prompt
contains no window block at all, so its text is unchanged and its stored answer
replays. That experiment was run while writing this entry: 38 replayed, 2
regenerated.

The waiting-period block reaches all forty, because it prints even when the
person never says how long they have held the policy — every waiting period
then comes back "cannot be determined", and that line is printed. So a change
to its wording regenerates every case and cannot be isolated, however
carefully the cache is used.

That is the general skill Concept 29 depends on: a change's reach is a property
of the data flow — which inputs end up inside which prompt — and it can be
predicted by tracing that flow. The same tracing found Failure 26 (a fact
extractor change that looked as if it touched every verdict) and Concept 30 (a
version label that made every change touch everything). The cache does not
decide a change's reach; it just makes it visible, case by case.
</details>

---

# M11 — Reading the inputs: two bugs upstream of every verdict

## The starting position

The scenario simulator answers a what-if question about an Indian health
policy with one of four verdicts (`covered`, `not_covered`, `conditional`,
`insufficient_information`) and the clauses that decide it. It runs in steps:

```
question ──> extract facts ──> arithmetic in code ──> reasoning ──> quote check
             (LLM)             waiting.py, reduction.py,  (LLM)
                               window.py
```

Its quality is measured by 40 hand-written cases in
`evals/golden/scenarios.json`. At the start of this milestone it scored 0.725
on verdicts and 0.774 on citation recall, and 5 of its 40 answers contained a
quotation that failed the verbatim check.

Two things from earlier milestones matter here. The model does not answer
identically twice, so one changed case is one sample, not a result; everything
claimed below as fixed or broken was asked at least three times (`--resample`).
And the planned next step was **code-added citations**: the arithmetic modules
already know which clause decides some cases, so code, rather than the model,
could add that citation.

---

## Checking what a fix would buy, before building it

Code-added citations were checked against the latest report before any code was
written. Seven cases were missing a citation they needed. Only one of them, the
co-payment clause 5.3 in `senior-copay`, is decided by the arithmetic modules.
The other six (the cosmetic exclusion's accident carve-out, the IVF exclusion,
the breach-of-law exclusion, the contribution clause, the AYUSH clause, the
ambulance clause) turn on reading language, which code cannot do.

Worse, adding citations blindly would break a passing case. In
`senior-but-excluded`, someone who bought the policy at 70 has a nose job. The
co-payment arithmetically applies, but the claim is refused outright, and the
case lists 5.3 under `must_not_cite`: there is nothing to take 20% of. Code
knows the co-payment applies. It does not know the claim is refused.

So the fix would have bought one case of 31. It was set aside, and the same
report was read for something else: the *reasons* the failing cases gave. Four
of them said a version of this:

```
ped-waiting-served:         "The policy does not specify how long the policy has been held."
accident-in-initial-period: "However, the policy age was not stated"
copay-just-under-sixty:     "the duration for which you have held the policy was not stated"
non-disclosure:             "The policy does not specify how long the policy has been held."
```

Every one of those questions states it: *"5 years after taking the policy
out"*, *"ten days after my policy started"*, *"four years later"*, *"two years
into the policy"*.

> **The lesson:** a score says how many cases failed. The failures' own
> explanations say why, and here they pointed where no metric was looking.

---

## Failure 31: every check did its job, on a fact that was wrong

The first step of a scenario asks the model to extract facts from the question
into a schema. Two of the fields held how long the policy has been held, as a
number and a unit:

```python
# api/app/pipeline/scenario.py, before this milestone
"policy_age_value": {"type": ["integer", "null"]},
"policy_age_unit": {"type": ["string", "null"], "enum": ["days", "weeks", "months", "years", None]},
```

Extracted facts are cached, so what every eval case had actually received could
be replayed without calling the model. Nine of forty were wrong or missing:

| What came back | Cases |
|---|---|
| Stated, returned null | `ped-waiting-served` ("5 years after taking the policy out"), `accident-in-initial-period` ("ten days after my policy started"), `non-disclosure` ("two years into the policy"), `copay-and-room-breach` ("have held it five years") |
| The person's **age** in the duration field, with no unit | `senior-copay` (67), `copay-just-under-sixty` (58), `senior-but-excluded` (70) |
| Not stated, only derivable (72 now, 70 at the start) | `copay-applies-emergency`, where null is correct |
| No number given ("for years") | `dental-no-accident`, where null is correct |

Follow one through. For *"I was hospitalised for it 5 years after taking the
policy out"*, the extractor returned null. `policy_age_days()` correctly turned
null into None. `waiting.py` correctly reported every waiting period as
undeterminable:

```
clause 3.2: requires 3 years, but how long the policy has been held was NOT STATED,
so whether this bar has lifted cannot be determined
```

And the reasoning step correctly concluded that a missing fact means
`insufficient_information`. Every stage did exactly what it was built to do.
The answer was wrong because the first input was.

The three age cases were half-protected by an earlier fix: `policy_age_days()`
refuses a value with no unit, so 67 never became 67 days. But the reasoning
prompt lists every extracted fact, so the model was still shown
`policy_age_value: 67`.

**Why nothing caught this for several milestones:** the scenario eval scores
only the final verdict and citations. A wrong fact is invisible to it, except
as a wrong verdict three steps later with a plausible reason attached. The
extraction step had no measurement of its own.

---

## Measuring the extractor on its own

The fix was developed against the extraction step alone. A script called the
extractor directly, bypassed the cache (a cached answer re-read is a copy, not
a sample), and checked the duration and age fields against expected values, in
three groups:

- **broken**: the eval questions above whose facts came back wrong;
- **control**: eval questions whose facts came back right, which must stay right;
- **held-out**: sentences in no eval case and no prompt example, to tell a fix
  that generalises from one that memorises.

The first held-out run showed the bug was worse than a missing value:

```
"I was 62 when I bought the policy, and I needed a bypass six years later."
    policy_age_value: 62, unit: null, age_at_policy_start: 56
```

The age went into the duration field, and the six years went into a
subtraction nobody asked for: 62 − 6 = 56. Given to the co-payment check, 56 at
inception clears a senior-citizen co-payment this person owes.

---

## Concept 31: a schema key is part of the prompt

Constrained decoding (Concept 1) forces the model's output to follow a JSON
Schema. What is easy to miss is what that means token by token. The grammar
built from the schema lays the properties out in the order they are declared,
so when the model reaches the duration field there is exactly one legal
continuation, the key itself, spelled out. The output so far ends:

```
..."condition": "high blood pressure", "body_system": null, "policy_age_value":
```

The next token, the value, is predicted from everything before it. The last
thing the model has "read" before writing that number is not the system
prompt, hundreds of tokens back. It is the key it has just written.

So a key name is an instruction, and the nearest one. `policy_age` sits very
close in meaning to "age at the policy", and the model filled it with an age.

That is a hypothesis, and a cheap one to test, because a key can be renamed
without touching anything else. Four variants on 25 sentences, one fresh call
each:

| Variant | Broken (8) | Control (10) | Held-out (7) | Total |
|---|---|---|---|---|
| Current prompt, `policy_age_value` | 2 | 10 | 5 | 17 |
| Renamed to `policy_held_for_value` | 4 | 8 | 5 | 17 |
| Two new examples, old name | 7 | 10 | 5 | 22 |
| Both | 6 | 9 | 6 | 21 |

The rename-only row is the informative one. It ended the age confusion
completely: no age landed in the duration field and no subtraction appeared.
And it broke sentences that used to work. *"Two weeks after my policy
started"*, *"8 months after buying this policy"*, *"three weeks into my new
policy"* all came back null. "Held for" matched "I have held it for" and not
much else. The name moved which phrasings the model recognised, in both
directions, which is strong evidence that the name was doing the work.

A name had to fit both kinds of phrasing: `time_since_policy_start`. Before
trying it, **eight more held-out sentences were written**. The first seven had
now been looked at while choosing between variants, and a set you have tuned
against is no longer held out. Then the best candidates ran on all 33
sentences, twice each, the second time in reverse order. The inference server
reuses the previous request's cached prefix, so request order is a genuine
source of variation (M10: repeat is not sequence).

| Variant | Forward | Reverse |
|---|---|---|
| New examples, `policy_age_value` | 29 / 33 | 30 / 33 |
| New examples, `time_since_policy_start_value` | **31 / 33** | **31 / 33** |

Most of the old name's misses were a subtraction between two ages (72 now and
70 at the start became "2 years"), three per run. The new name's two misses were
one subtraction (*"I am 70. I bought this policy at 55"* became 15 years: the
right number, done at the wrong stage) and *"My mother, who is 75"*, recorded
as 75 **at policy start**. That would impose a co-payment nobody has
established, and it is recorded as open.

The committed change:

```python
# api/app/pipeline/scenario.py
"time_since_policy_start_value": {"type": ["integer", "null"]},
"time_since_policy_start_unit": {
    "type": ["string", "null"],
    "enum": ["days", "weeks", "months", "years", None],
},
```

```
# api/app/llm/prompts.py, FACTS_SYSTEM (the new lines)
- "admitted a year after I took out cover" -> time_since_policy_start_value: 1, unit: years
- "six months into the policy"        -> time_since_policy_start_value: 6, unit: months

An AGE is never how long the policy has been held. "I was 61 when the cover
began and I claimed four years later" is age_at_policy_start: 61 AND
time_since_policy_start_value: 4, unit: years - two different numbers in two different
fields. A time_since_policy_start_value always comes with its unit; if there is no length
of time with a unit, it is null.
```

Two details of discipline. First, the committed prompt was compared **byte for
byte** with the text that was measured. An edit had re-wrapped two of those
lines, and on a 7B model a line break is a different prompt (Failure 28).
Second, a test now forbids the word `age` in any duration key, so a
tidy-minded rename cannot bring the bug back:

```python
# api/tests/test_scenario.py
durations = [k for k in FACTS_SCHEMA["properties"] if k.endswith(("_value", "_unit"))]
assert not [k for k in durations if "age" in k.split("_")]
```

The script became `evals/check_fact_extraction.py`: 18 eval sentences and 15
held-out ones, uncached, about three minutes. Its first run on the committed
code gave eval 18/18 and held-out 14/15. The mother-75 sentence came out right
that time, so it is a wobble: wrong in two runs of three.

---

## What the verdicts did

The key name appears in every reasoning prompt, under "FACTS UNDERSTOOD" when
stated and under "NOT STATED" when not. So this change reached all 40 cases,
and the cache could isolate nothing (Concept 29): every case was a fresh
sample. Verdict accuracy was **0.800**, against 0.725. The cases that moved,
each asked three times:

| Case | Samples | Reading |
|---|---|---|
| `non-disclosure` | ✓ ✓ ✓ | fixed; had never passed |
| `copay-just-under-sixty` | ✓ ✓ ✓ | fixed; the right verdict had never appeared in any earlier sample |
| `ped-waiting-served` | ✗ ✗ ✗ | right facts, still wrong, now `not_covered` |
| `accident-in-initial-period` | ✗ ✗ ✗ | right facts, still wrong, now `not_covered` |
| `icu-rate-breach` | ✓ ✓ ✓ | better, but its facts did not change and it has wobbled before, so not claimed |

The two that stayed wrong changed *how* they were wrong, and each now fails at
a later step. `ped-waiting-served` is told "Already satisfied: 3.1, 3.2, 3.3,
3.4", and replies that five years has not served a 36-month wait.
`accident-in-initial-period` is told "clause 3.1 ... still applies and blocks
treatment covered by THIS clause", and follows that line over the clause's own
"except claims arising out of an Accident". Both are recorded as open.

---

## Failure 32: eight fabricated quotes, and half were our own words

The same run reported 8 answers with a quotation that failed the verbatim
check, up from 5. Replaying those answers from the cache and printing each
failed quote showed that none had invented policy content. Four looked like
this:

```
ped-waiting-served, clause 3.2
  QUOTED : '... after the date of inception of the first policy with the Company.
            EXCEPTIONS - this clause does NOT apply when: direct complications of a pre-existing disease'
  CLAUSE : '... after the date of inception of the first policy with the Company.'
```

That is the clause, correctly copied, followed by a line the policy never
contained. The line is written by the pipeline: the reasoning prompt prints
each clause's extracted exceptions directly beneath its text.

```python
# api/app/llm/prompts.py, render_reasoning_request()
lines.append(clause.text.strip())
if getattr(clause, "exceptions", None):
    lines.append(
        "EXCEPTIONS - this clause does NOT apply when: "
        + "; ".join(clause.exceptions)
    )
```

The model could not tell where the clause stopped and the annotation began.
The verbatim check caught every one, and detection integrity stayed at 1.000,
so the guarantee never failed. But the annotation it had copied raised a much
worse question. "Direct complications of a pre-existing disease" is not an
exception to clause 3.2. The clause *excludes* "a Pre-existing Disease and its
direct complications". The annotation says the opposite of the policy.

---

## Failure 33: the pipeline was telling the model things the policy does not say

At analysis time (stage 3), each clause's carve-outs are extracted into an
`exceptions` list, and the scenario step prints them as above. Every extracted
exception on the synthetic policy, beside the clause it came from:

| Clause | Extracted exception | The policy |
|---|---|---|
| 3.1 initial waiting | claims arising out of an Accident | ✓ "except claims arising out of an Accident" |
| 4.1 cosmetic | necessitated by an Accident, Burn or Cancer; certified … medically necessary | ✓ "unless such surgery is necessitated by …" |
| 4.6 dental | necessitated by an Accident and requiring hospitalisation | ✓ "unless necessitated by …" |
| 3.2 pre-existing | direct complications of a pre-existing disease | ✗ excluded, not excepted |
| 4.2 self-injury, alcohol | necessitated by an Accident, Burn or Cancer | ✗ 4.2 has no exception at all |
| 4.4 breach of law | … committing a breach of law with criminal intent | ✗ that *is* the exclusion |
| 4.5 infertility | reversal of sterilisation | ✗ on its list of things excluded |
| 4.7 non-medical | items listed in Annexure III | ✗ "any other item listed in Annexure III" is excluded |

**Five of eight were wrong.** Each was presented to the reasoning step as a
statement of when an exclusion does *not* apply, in exactly the situation where
it does.

That explains a case that had failed in every run on record.
`intoxication-injury`, *"I fell down the stairs after drinking heavily and
fractured my hip"*, must be `not_covered` under 4.2's alcohol exclusion. The
prompt told the model that 4.2 does not apply when the injury was
"necessitated by an Accident". Falling down the stairs is an accident.

Where did 4.2's exception come from? The first explanation offered was bleed
from clause 4.1, the neighbour with exactly that wording, echoing the batching
interference recorded in M2. **That was wrong, and the configuration showed
why.** Analysis sends one clause per call (`analyze_batch_size = 1`, itself an
M2 finding), so 4.2 never shared a generation with 4.1. The wording is the
analysis prompt's own worked example:

```
# api/app/llm/prompts.py, CLASSIFY_SYSTEM
- exceptions: the cases where this clause does NOT apply. Look for "unless",
  "except", "other than", "save for", "provided that", "shall not apply".
  ...
    "cosmetic surgery ... unless necessitated by an Accident, Burn or Cancer"
       -> ["necessitated by an Accident, Burn or Cancer"]
```

For a clause with no carve-out, the model returned the example's answer. A
mechanism remembered from an earlier milestone is a hypothesis about this one,
not a diagnosis of it.

---

## Concept 32: checking an extraction by its grammar, not only its text

This project already has a tool for "the model claims the document says X":
the verbatim check (M5). Applied to exceptions, it catches 4.2 (that text is
not in the clause) and, by luck of paraphrase, 3.2 and 4.7. It cannot catch 4.4
or 4.5, because those spans *are* in their clauses, word for word. They are
real text playing the wrong role.

The role has a visible signature. In a policy wording, an exception is
introduced by a word that says so, the same words the analysis prompt lists,
and that word comes before the exception in the same sentence:

```
"... first policy with the Company, except claims arising out of an Accident."
                                    ^^^^^^ exception word, same sentence

"... including assisted reproduction services, gestational surrogacy,
 reversal of sterilisation and any form of contraception, are excluded ..."
     (no exception word anywhere before the span)
```

So an exception is kept only if the words are in the clause and an exception
word precedes them in their sentence:

```python
# api/app/grounding.py
EXCEPTION_MARKERS = (
    "unless", "except", "other than", "save for", "provided that", "shall not apply",
)
_MARKER = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in EXCEPTION_MARKERS) + r")\b")
_SENTENCE_END = re.compile(r"[.!?]\s")


def verify_exception(span: str, source_text: str) -> bool:
    needle = normalize(span).strip(" .,;:")
    if not needle:
        return False
    haystack = normalize(source_text)
    start = haystack.find(needle)
    while start != -1:
        sentence_so_far = _SENTENCE_END.split(haystack[:start])[-1]
        if _MARKER.search(sentence_so_far):
            return True
        start = haystack.find(needle, start + 1)
    return False
```

Three details:
- A sentence ends at a full stop *followed by a space*, so "3.1" and "1.5 lakh"
  do not end one.
- Unlike a quotation, there is no minimum length. The exception word is what
  stops a short span from matching by accident.
- A test holds the code's list of words and the prompt's list together, because
  two lists that must agree should not be trusted to.

The check runs in the analysis step, straight after the model's answer is
parsed:

```python
# api/app/pipeline/analyze.py, _analyze_batch()
analyses = _parse(payload)
text_by_id = {str(seg.order_idx): seg.text for seg in batch}
for key, analysis in analyses.items():
    kept = [e for e in analysis.exceptions if verify_exception(e, text_by_id[key])]
    ...
    analysis.exceptions = kept
```

It sits after the cache, so the model call and its stored answer are untouched
and nothing had to be re-analysed. Only what the system believes about that
answer changed. On the synthetic policy it keeps 3.1, both conditions of 4.1
(the second is still governed by the "unless" earlier in its sentence) and 4.6,
and drops all five wrong ones.

**Why dropping is the safe direction.** A dropped exception leaves the clause's
full text in the prompt, so the model loses emphasis, not a fact. A wrong
exception kept is the pipeline asserting something false.

**What it does not catch.** A wrong span that happens to follow an exception
word in the same sentence passes, for instance the words just after "unless"
taken from the wrong part of a long sentence. This checks the common shape of
the mistake. It does not prove correctness.

---

## What the verdicts did, again

Removing five annotations changed every reasoning prompt, so again every case
regenerated. Verdict accuracy stayed at **0.800**. Each case that moved, asked
three times:

| Case | Samples | Reading |
|---|---|---|
| `intoxication-injury` | ✓ ✓ ✓ | fixed; had never passed, and its cause is gone |
| `copay-unknown-inception-age` | ✓ ✓ ✓ | steadier than before (✓ ✓ ✗) |
| `non-disclosure`, `breach-of-law`, `infertility-ivf` | ✓ ✓ ✓ | verdicts still right |
| `oral-chemo-limit` | ✗ ✗ ✗ | **broken**; passed in every earlier run |
| `senior-but-excluded` | ✗ ✗ ✗ | was ✗ ✓ ✓, now consistently wrong |

Answers with a failed quotation fell from 8 to **1** on the full run, and to 0
in both resamples of those seven cases. Citation recall fell from 0.806 to
0.742, because `oral-chemo-limit` and `non-disclosure` lost their deciding
citation.

Two predictions made before the run were wrong. `breach-of-law` and
`infertility-ivf` were expected to start citing 4.4 and 4.5 once those clauses
stopped claiming not to apply. They did not. Whatever makes the model cite the
wrong clause there, it was not the exception line.

`oral-chemo-limit` turns on clause 5.5, whose prompt text did not change. The
lines removed were on 3.2 and section 4. That is the non-local effect of
Failure 28 again. One unproven observation: four wrong answers now cite 4.1,
the only exclusion still carrying an exceptions line. Emphasis may be drawing
citations.

**Why this was kept, when Failure 29's fix was reverted.** Both were
structurally correct, and neither improved verdicts. The notes fix cost three
cases to fix one. This change is verdict-neutral, removes 7 of 8 failed
quotations, and does something no metric scores: it stops the system asserting
falsehoods about the policy in its own voice. "The alcohol exclusion does not
apply to accidents" was untrue in every answer whose prompt printed it. The
trade is written down here, with its cost, rather than hidden behind an
unchanged headline number.

---

## Where this leaves the numbers

| | Start (v15) | Facts fixed (v17) | Exceptions checked (v18) |
|---|---|---|---|
| Verdict accuracy | 0.725 | 0.800 | 0.800 |
| Citation recall | 0.774 | 0.806 | 0.742 |
| False citation rate | 0.400 | 0.400 | 0.400 |
| Answers with a failed quotation | 5 | 8 | 1 |
| Detection integrity | 1.000 | 1.000 | 1.000 |

Each column is one full run; the per-case claims above rest on three samples
each. Clause classification and ranking read neither the facts nor the
exceptions, so neither could move. (There is no v16 in this table: that label
belongs to the reverted experiment in Failure 29.)

**Still open:**

- `ped-waiting-served`: given correct facts and a block saying every waiting
  period is satisfied, the model still answers that five years has not served
  36 months;
- `accident-in-initial-period`: the waiting-period block says 3.1 "blocks" the
  claim without mentioning 3.1's own accident carve-out, and the model follows
  the block;
- `oral-chemo-limit` and `senior-but-excluded`, broken by the exceptions change,
  cause not found;
- from before: the `notes` leak in `not-in-document` (M10), `cataract-served`,
  `room-rent-within-cap`, `day-care-not-listed`;
- the fact extractor sometimes records a third person's current age as age at
  policy start (*"My mother, who is 75"*), and sometimes subtracts two ages
  itself;
- quote copying itself: the three correct exceptions are still printed beneath
  their clauses, and can still be copied into a quotation.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 156 tests
python evals/check_fact_extraction.py                    # the extractor alone, ~3 minutes
python evals/run_scenario_eval.py --only intoxication-injury,oral-chemo-limit --resample
```

**Question to sit with:** a clause reads

> *Hearing aids are excluded unless prescribed by an ENT specialist. Spectacles
> are not covered under any circumstances.*

The analysis returns `exceptions: ["prescribed by an ENT specialist",
"Spectacles are not covered"]`. Before opening the answer, decide which of the
two `verify_exception` keeps, and then which a plain substring check would
keep.

<details>
<summary>Answer</summary>

**`verify_exception` keeps the first and drops the second. A substring check
keeps both.** (Both results were checked by running the function on this
exact text.)

Both spans appear in the clause word for word, so a substring check has nothing
to object to. The difference is the exception word. "prescribed by an ENT
specialist" is preceded by "unless" in its own sentence. "Spectacles are not
covered" also has an "unless" before it, but in the previous sentence: the full
stop and space after "specialist" end that sentence, and nothing in the new
sentence before the span marks it as an exception, because it is not one. It is
a second exclusion.

That is the distinction this milestone turned on. A quotation check asks whether
the words exist. An extraction also claims what the words are *doing*, and when
that claim has a visible grammatical signature, it can be checked too.
</details>

---

# M12 — Seven attempts, three kept, and a number that means what it says

## The starting position

The scenario simulator answers a what-if question about an Indian health
policy with one of four verdicts (`covered`, `not_covered`, `conditional`,
`insufficient_information`) and the clauses that decide it:

```
question ──> extract facts ──> arithmetic in code ──> reasoning ──> quote check
             (LLM)             waiting.py, reduction.py,  (LLM)
                               window.py
```

Its quality is measured by 40 hand-written cases in
`evals/golden/scenarios.json`. At the end of M11 one full run scored 0.800 on
verdicts and 0.742 on citation recall, two of the five cases with a
"must not cite" clause cited it, and one answer had a quotation that failed the
verbatim check. The open problems fell into four groups:

1. eight wrong verdicts;
2. eight cases missing the clause that decides them;
3. two cases citing a clause they must not rely on;
4. smaller issues: third-person ages misread by the fact extractor, the
   pipeline's own annotations copied into quotations, and a score that moves
   between runs of identical code.

Reading the model's own explanations for the eight wrong verdicts grouped them
into five causes:

| Cause | Cases |
|---|---|
| A. The model overrides arithmetic it was given | `ped-waiting-served` (told a 36-month wait was satisfied at five years, refused anyway), `room-rent-within-cap` (told 8,000 is within a 10,000 cap, said "paid at 80%") |
| B. A computed line omits the clause's own carve-out | `accident-in-initial-period` ("3.1 still applies and blocks", though 3.1 excepts accidents) |
| C. A reduction softening a refusal | `senior-but-excluded` ("not covered … however the 20% co-payment applies" → `conditional`) |
| D. Suspected: one clause over-emphasised | `cataract-served`, `oral-chemo-limit` and three missing citations all leaned on the cosmetic exclusion 4.1, the only clause still printed with an "EXCEPTIONS" line |
| E. Facts or document gaps | `not-in-document` (the fact extractor wrote "This policy does not cover car theft"), `day-care-not-listed` (day care depends on an annexure the document does not contain) |

---

## How every step was run

The previous milestones produced a working method, and this one used it for
every change:

- **One change at a time**, with a written prediction of which cases it reaches.
  The LLM cache replays the stored answer for any prompt whose text did not
  change (Concept 29), so a change's reach is visible as the count of
  regenerated cases. A wrong count means the change touched something it
  should not have.
- **Three samples** before calling anything fixed or broken: one run plus
  `--only <ids> --resample` twice.
- **Revert** a change that is net negative over three samples, unless it removes
  something false the pipeline was asserting. A revert is checked by running
  the eval again: every case must replay.

Four of the seven attempts below were reverted. Each revert was checked rather
than assumed: the eval returned to exactly its previous score with 40 of 40
answers replayed from the cache, which is only possible if every prompt is back
to its previous text byte for byte.

---

## Step 0: measure a rule before writing it

Two of the planned fixes were rules that fire on some answers and not others.
Before any code existed, a script replayed every stored answer from the cache
(no model calls) and counted which cases each rule would catch:

| Rule | Would catch | Of which already right |
|---|---|---|
| (a) a clause the arithmetic cleared, cited as refusing, delaying or reducing | `ped-waiting-served`, `room-rent-within-cap`, `ambulance-admissible` | `ambulance-admissible` (right verdict, but citing three satisfied waiting periods as "reduces", which is itself wrong) |
| (b) one clause cited as refusing AND another as reducing, verdict not `not_covered` | `senior-but-excluded` | none |
| notes claiming coverage the person never mentioned | `not-in-document` | none |

The measurement changed a rule before it was written. Rule (b) was first framed
as "a refusing citation with a `conditional` verdict". The replay showed
`late-notice` and `no-preauth-cashless` are correctly `conditional` while citing
a clause as refusing, because the Company *may* repudiate. Only the pairing of a
refusal with a reduction is a contradiction.

---

## Fix 1: the waiting-period line names the clause's own carve-out

Asked about being hit by a car ten days into a policy, the system refused three
times out of three. Clause 3.1 is a 30-day initial waiting period "except claims
arising out of an Accident". The code that compares waiting periods told the
model:

```
clause 3.1: requires 1 month, policy held 10 days -> this waiting period still
applies and blocks treatment covered by THIS clause
```

True, and incomplete. The model followed "blocks" over the clause's carve-out.
The line now carries the clause's exceptions, which M11 made trustworthy by
verifying each one against the clause text:

```python
# api/app/pipeline/waiting.py, WaitingCheck.describe()
if not self.exceptions:
    return blocked
carve_outs = "; ".join(f'"{e}"' for e in self.exceptions)
return (
    f"{blocked}, EXCEPT where the situation falls within this clause's "
    f"own exception: {carve_outs}. Decide from the description whether it does"
)
```

Whether a situation *is* an accident is language, so the line names the
exception and leaves that decision to the model. Only an unserved period gets
this; a satisfied or undeterminable one is not deciding the case, and extra
emphasis on it has cost cases before (M7).

**Predicted reach:** 2 cases, the only two questions under 30 days into a
policy. **Measured:** 38 replayed, 2 regenerated. `accident-in-initial-period`
✓ ✓ ✓; the guard `initial-waiting-period` stayed ✓ ✓ ✓.

---

## Fix 2: notes that answer the question are not shown

The fact extractor has a free-text `notes` field. For *"Does this policy cover my
car being stolen from the hospital car park?"* it wrote:

```
This policy does not cover car theft from the hospital car park.
```

The extractor is never shown the policy, so that sentence is invented, and it
reached the reasoning step under "FACTS UNDERSTOOD". M10 tried removing notes
entirely and measured it net negative (Failure 29): notes usually restate the
situation in wording that helps find the right clause. So only the harmful kind
is removed:

```python
# api/app/llm/prompts.py
_COVERAGE_CLAIMS = (
    "does not cover", "doesn't cover", "not cover", "not covered", "is covered",
    "are covered", "will be covered", "excluded", "not payable", "is payable",
    "will not pay", "will pay",
)

def _invents_coverage(note: str, scenario: str) -> bool:
    note, said = normalize(note), normalize(scenario)
    return any(p in note and p not in said for p in _COVERAGE_CLAIMS)
```

The comparison with the question is what keeps the rule narrow. Two eval cases
say "the treatment itself was covered", and their notes repeat it. The person
said it, so it stays.

**Predicted reach:** 1 case. **Measured:** 39 replayed, 1 regenerated.
`not-in-document` ✓ ✓ ✓.

---

## Failure 34: removing the emphasis, which was carrying information

Cause D was a hypothesis. The cosmetic exclusion 4.1 was the only exclusion
still printed with an "EXCEPTIONS - this clause does NOT apply when: …" line,
and the reasoning prompt's section on exceptions used burn surgery under a
cosmetic exclusion as its example. The model was citing 4.1 for cataracts, chemotherapy and IVF. Removing
both should have reduced that clause's pull.

It reached all 40 cases, and three samples of the cases that moved said:

| Case | Samples | |
|---|---|---|
| `senior-but-excluded` | ✓ ✓ ✓ | fixed |
| `cosmetic-after-accident` | ✗ ✗ ✗ | **broken**: certified burn surgery refused as "still cosmetic" |
| `accident-in-initial-period` | ✗ ✗ ✗ | **broken**, three samples after Fix 1 had it ✓ ✓ ✓ |
| `cataract-served`, `oral-chemo-limit` | ✗ ✗ ✗ | the two targets, **not fixed** |

Citation recall rose (0.742 → 0.806), but verdicts were net −1, and the break
was exactly the case the line was originally added for. And the targets failed
without the emphasis just as they had with it, so emphasis was not their cause.
Reverted.

> **The lesson:** a line added for a reason carries that reason even after the
> reason is forgotten. Before removing prompt text, find out what it was put in
> to fix, and include that case in the guards.

---

## Concept 33: checking the answer against the arithmetic

The reasoning prompt already said "Never contradict those results and never
recompute them." For `ped-waiting-served` it was told
"Already satisfied, so IRRELEVANT here: 3.1, 3.2, 3.3, 3.4" and answered that
five years has not served clause 3.2's 36 months, citing 3.2 as the refusal.

An instruction is a request. The contradiction itself is mechanical to detect:
the answer names a clause and an effect, and the arithmetic has already said
whether that clause can have that effect.

```python
# api/app/pipeline/scenario.py, find_contradictions()
for citation in citations:
    if citation.effect not in _ADVERSE_EFFECTS:        # denies, delays, reduces
        continue
    cleared = served.get(citation.clause_id) or ruled_out.get(citation.clause_id)
    if cleared is not None:
        problems.append(
            f"- You cited clause {citation.clause_id} as a reason this claim is "
            f"{_ADVERSE_EFFECTS[citation.effect]}, but it was calculated: "
            f"{cleared.describe()}."
        )
        irrelevant.append(citation)

denying = [c.clause_id for c in citations if c.effect == "denies"]
reducing = [c.clause_id for c in citations if c.effect == "reduces"]
if denying and reducing and verdict != Verdict.NOT_COVERED:
    problems.append(...)   # "a refused claim has nothing to reduce"
```

What happens next follows the shape of the existing retry for an answer that
arrives with no citations:

```python
# api/app/pipeline/scenario.py, run_scenario()
problems, _ = find_contradictions(citations, verdict, computed)
if problems:
    payload = await reason(
        scenario, facts, considered, computed,
        nudge=CONSISTENCY_NUDGE.format(problems="\n".join(problems)),
        use_cache=not resample,
    )
    citations, verdict = _read_answer(payload, source_by_id)
    _, irrelevant = find_contradictions(citations, verdict, computed)
    if irrelevant:
        citations = [c for c in citations if c not in irrelevant]
        if verdict != Verdict.INSUFFICIENT_INFORMATION and not citations:
            verdict = Verdict.INSUFFICIENT_INFORMATION
```

Three design decisions:

1. **One retry, with the calculation quoted back.** The retry is a second user
   turn, so it has a different cache key and cannot be served the answer that
   failed.
2. **Code may remove a citation it can prove wrong, and never sets a verdict.**
   A clause the arithmetic cleared cannot be refusing the claim; that is proof.
   What the right verdict *is* still needs reading the situation.
3. **The arithmetic is computed once** (`compute()`), then used both to render
   the prompt and to check the answer. Two separate computations would be two
   lists that must agree.

That refactor touched the code that renders every prompt, so it carried a risk
of changing the text sent for all 40 cases. **Predicted reach:** the 4 cases
Step 0 found. **Measured:** 36 replayed, 4 regenerated, which also proves the
refactor changed no prompt by a single byte.

| Case | Samples | |
|---|---|---|
| `ped-waiting-served` | ✓ ✓ ✓ | fixed; had never passed |
| `ambulance-admissible` | ✓ ✓ ✓ | verdict held; the wrong waiting-period citations are gone |
| `room-rent-within-cap` | ✗ ✓ ✓ | better than ✗ ✗ ✗; the miss kept citing 5.1 after the retry, so the citation was dropped and the verdict downgraded |
| `senior-but-excluded` | ✗ ✗ ✗ | the retry did not persuade the model |

Rule (b) shows the limit. Its answer contradicts itself, and there is no
arithmetic to say which half is wrong, so code has nothing to remove.

---

## Failure 35: a true fact that fixed three other cases and not its own

`day-care-not-listed`: six hours on a drip, same-day discharge. Day care is paid
only for treatments "listed in Annexure II", and the synthetic policy contains
no Annexure II. The honest answer is that the document cannot settle it. The
model answered `covered`.

A missing section is an absence, so nothing on the page shows it, and whether a
section exists is a lookup rather than a judgement. Code found every
`Annexure <n>` a clause relies on that no segment was filed under (2.4 → II,
4.7 → III, confirmed on the real policy), and printed one line under each
clause: "NOT IN THIS DOCUMENT - this clause relies on Annexure II, which is not
included, so what it lists cannot be checked."

| Case | Samples | |
|---|---|---|
| `day-care-not-listed` (the target) | ✗ ✗ ✗ | **not fixed** |
| `cataract-served`, `senior-but-excluded`, `oral-chemo-limit` | ✓ ✓ ✓ | fixed |
| `copay-unknown-inception-age` | ✗ ✗ ✗ | broken |
| `breach-of-law` | ✗ ✗ ✗ | broken: now tells someone injured while being arrested for shoplifting that they are covered |
| `ped-waiting-served`, `initial-waiting-period` | ✗ ✓ ✗, ✗ ✗ ✓ | wobbling, from ✓ ✓ ✓ |

By count that is about net −0.3 of a case. The deciding point was not the
count. The mechanism did nothing for the case it was built for, and the three
cases it fixed are about cataracts, chemotherapy and cosmetic surgery, none of
which touches an annexure. They moved because an extra line changed every
prompt, the non-local effect M10 recorded as Failure 28. Keeping a change for
effects it was not designed to cause, and cannot explain, is tuning against
noise that happens to be favourable today. Reverted, including the plumbing it
needed.

---

## Failure 36: a second turn that chose the same wrong clauses

Five cases reached the right verdict and cited the wrong clause in every run on
record: a hospital definition instead of the AYUSH clause, basic in-patient
cover instead of the ambulance clause, the adventure-sports exclusion instead of
the breach-of-law exclusion beside it. One hypothesis: a single generation was
deciding the verdict, writing the reasoning and choosing citations, and the last
job was being done badly. It is the same argument the pipeline's design already
makes for extracting facts in a separate call from reasoning.

The test was a follow-up turn in the same conversation: the reasoning prompt,
then the model's own verdict and reasoning (not its earlier citations, so they
could not anchor it), then "keep that verdict, now choose the clauses that
decide it". The prompt was an exact prefix of the reasoning call, so the
inference server could reuse its work.

Measured on 12 cases twice (the 5 misses, 5 already-right citations, 2 guards):
all verdicts right, no new false citations, and **the same wrong clause in four
of the five misses, both times**. The fifth now cited 4.1, but its quotation
failed verification in both samples. The cost was about five seconds on every
question.

That is a clean negative result. Given the verdict and asked only which clause
decides it, the model still picks the adventure-sports exclusion for
shoplifting. The single generation was not the problem: the model's reading of
which clause is specific to a situation is. Reverted.

---

## Failure 37: two attempts at third-person ages, both worse

The fact extractor sometimes records "My mother, who is 75" as 75 **at policy
start**, which would impose a senior-citizen co-payment nobody established.
First, the held-out check (`evals/check_fact_extraction.py`) gained five
third-person sentences written before any change. One of them ("My mother took
this policy out at 61") *does* state an age at the start, so a fix could not
pass by simply never filling that field for someone else. On the unchanged
prompt, "My father, 82" failed the same way. The bug was real on a sentence
nobody had looked at.

- **A one-line example** ("my husband is 64" → `age`: 64,
  `age_at_policy_start`: null): `father-82` still wrong in both runs, and
  `mother-75`, right at baseline, now wrong in both. Reverted.
- **A rename.** The wrong answers had a pattern: `age` came back empty and the
  number landed in the next age field, as if `age` meant "the speaker's age".
  Concept 31 says a key name is an instruction, so `age` was renamed
  `patient_age_now` in a scratch copy of the prompt. First-person ages still
  worked, and *every* third-person sentence failed, in both runs.

Concept 31 is a mechanism, not a recipe. A name does steer the value written
after it, but that does not mean the name you guess will steer it the right way.
Each rename is an experiment. Recorded as open.

---

## Concept 34: majority of three, and a number that means what it says

Every verdict accuracy in this log before now is one sample. The model does not
answer identically twice (M9), so a single full run's score moves by two or
three cases on its own. A reader of "0.875" assumes it describes what the system
usually answers. One run cannot say that.

`--repeats N` asks each case N times: the first may replay the cache, the rest
are asked afresh. It scores the majority verdict.

```python
# evals/run_scenario_eval.py, majority()
top, count = Counter(got).most_common(1)[0]
winner = top if count * 2 > len(got) else None
```

A strict majority is required. With three samples and four verdicts, a
three-way split is possible, and it counts as wrong: picking one of three
disagreeing answers would be choosing a result, not measuring one.

Three samples of 40 cases take longer than the ten-minute limit on one
command, so they ran in three chunks with `--only`. Majority counts add across
chunks. One built-in check: sample 1 of each chunk replays the stored answers,
so it must reproduce the last single run. It did, 35 of 40, confirming that
three reverts had left the prompt exactly as it was.

---

## Where this leaves the numbers

| | End of M11 | Now, sample 1 | Now, sample 2 | Now, sample 3 | **Now, majority of 3** |
|---|---|---|---|---|---|
| Verdict accuracy | 0.800 | 0.875 | 0.900 | 0.925 | **0.925** (37/40) |
| Citation recall | 0.742 | 0.742 | 0.806 | 0.774 | |
| False citation rate | 0.400 | 0.200 | 0.400 | 0.000 | |
| Answers with a failed quotation | 1 | 1 | 1 | 0 | |

37 of 40 cases gave the same verdict in all three samples. Detection integrity
was 1.000 in the consolidated report (`evals/REPORT.md`), which replays sample
1's answers. The majority summary did not print it for any sample: that column
was added only after this run, so samples 2 and 3 have no recorded figure. Clause
classification (macro-F1 1.000) and ranking (8 of 9) did not move.

Against the bar set before starting: verdict accuracy of at least 0.850 was met
(0.925), and at most one failed quotation was met. Zero false citations was not
met (0.200 on average), and neither was citation recall of at least 0.806
(0.774 on average).

**Kept:** Fix 1 (carve-outs in the waiting-period line), Fix 2 (notes that claim
coverage), Concept 33 (the consistency check), Concept 34 (`--repeats`).
**Reverted:** removing the exceptions line, the missing-annexure line, the
separate citation turn, the third-person age example.

**Still open:**

- `senior-but-excluded`: refuses and applies a co-payment in the same answer,
  3/3, and the consistency retry does not change it;
- `room-rent-within-cap`: in the final run the retry kept citing the ruled-out
  cap 3/3, so the citation was dropped and the answer downgraded to
  `insufficient_information`;
- `day-care-not-listed`: stating the missing annexure did not help;
- 2-of-3 wobbles: `cataract-served`, `oral-chemo-limit`,
  `copay-unknown-inception-age`;
- citations the model gets wrong even in a dedicated turn: 4.4, 6.6, 2.6, 2.5;
- third-person ages in the fact extractor, after two failed attempts; and the
  extractor sometimes subtracts two stated ages itself, which is accepted (the
  number is right).

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 172 tests
python evals/run_scenario_eval.py                        # one run; after no change, 40 replayed
python evals/run_scenario_eval.py --repeats 3 --only ped-waiting-served,room-rent-within-cap
python evals/check_fact_extraction.py                    # the extractor alone
```

**Question to sit with:** the consistency check fired on `ambulance-admissible`,
whose verdict was already right. Before opening the answer: was that a false
trigger that should have been prevented, and what did the retry risk?

<details>
<summary>Answer</summary>

**Not a false trigger.** The check is about the answer's consistency, not its
verdict. The answer cited clauses 3.1, 3.2 and 3.3 as *reducing* the ambulance
claim, and all three are waiting periods the arithmetic showed were satisfied
years earlier. Those citations were provably wrong, and a reader would have been
shown three irrelevant clauses as the reasons their claim is cut.

The risk was real: a retry is a fresh sample, and a fresh sample of a right
answer can come back wrong. That is why the case was listed as a guard before
the change was made, and checked three times after. The verdict held, and the
wrong citations were gone.

The general point: a check that fires only on wrong verdicts can only be built
by knowing the right verdict, and that is the one thing code does not know here.
What code does know is which statements inside an answer contradict arithmetic
it has already done, so that is what it checks.
</details>

---

# M13 — The last failures, and the answer key itself

## The starting position

The scenario simulator answers a what-if question about an Indian health
policy with one of four verdicts and the clauses that decide it. It is measured
by 40 hand-written cases in `evals/golden/scenarios.json`. Since M12 the
headline figure is a **majority of three samples per case**, because the model
does not answer identically twice and a single run moves by two or three cases
on its own.

At the end of M12 that figure was 0.925 (37 of 40). Still open:

- wrong in every sample: `senior-but-excluded`, `room-rent-within-cap`,
  `day-care-not-listed`;
- right in two samples of three: `cataract-served`, `oral-chemo-limit`,
  `copay-unknown-inception-age`;
- seven cases missing the clause that decides them, and one citing a clause it
  must not rely on;
- the fact extractor filing "my father, 82" as his age when the policy began.

The method was the same as M12's: diagnose from evidence, one change at a
time, predict how many cases the change reaches (the cache replays every case
whose prompt did not change), three samples before calling anything fixed, and
revert what is net negative.

---

## A theory disproven in seconds

The deciding clauses the model kept failing to cite (ambulance, AYUSH,
contribution) are low-impact clauses. The prompt includes only as many clauses
as fit a size budget, ranked by impact, and the citation id list is built from
that same set. If they were being cut, the model could not cite them at all.

Checked without a model call: the budget keeps all 40 clauses, including every
one it was failing to cite. The misses are the model's reading, not missing
text. That took one command, and it was worth running before building anything
on the assumption.

---

## Failure 38: computed lines that said more than was computed

Two of the three always-wrong cases failed for the same reason, and the reason
was in text this project's own code writes. Printing exactly what the model was
shown made both visible.

**`senior-but-excluded`**: bought the policy at 70, then a nose job for
appearance, refused outright by the cosmetic exclusion. The reduction block
said:

```
- APPLIES: clause 5.3: aged 70 at inception against a threshold of 60 -> the
  20% co-payment DOES apply, so this claim is paid at 80% of the admissible amount
```

The arithmetic established one thing: the person was over 60 when the policy
began. "So this claim is paid" was never computed, and here it is false. The
model took the line at its word and answered `conditional`. It is the same
mistake M7 recorded ("correct facts can still mislead"), where a line said a
waiting period "does NOT block the claim". The fix makes the line say only what
was computed:

```python
# api/app/pipeline/reduction.py, _copay_check()
f"aged {inception_age} at inception against a threshold of "
f"{threshold} -> the {percent}% co-payment DOES apply to this "
f"person. IF this claim is payable at all, it is paid at "
f"{100 - percent}% of the admissible amount",
```

**`room-rent-within-cap`**: sum insured 10 lakh, a room at 8,000 a night.
Code had computed that 1% of 10 lakh is 10,000, so the room is within the cap,
but printed only this:

```
- Ruled out by the numbers, so IRRELEVANT here and not worth citing: 5.1
```

Without the reason, the model redid the comparison itself and answered "paid at
80% of 8,000". That is the proportionate-deduction clause applied backwards.
Clause 5.2 applies only "where … room rent **exceeds the limit specified in
Clause 5.1**". The ruled-out line had been collapsed to an id on purpose (M8: a
line per non-event adds emphasis that costs cases), but a room cap is only
computed when the person named their room rate, so it is the subject of their
question. It now gets its own line, in the words clause 5.2 is written in:

```
- WITHIN THE LIMIT: clause 5.1: 1% of 10 lakh is 10,000 rupees per day; room
  rent of 8,000 rupees per day does NOT exceed that limit, so this cap costs nothing here
```

**Predicted reach:** 5 cases (the four with an applying co-payment, plus this
one). **Measured:** 35 replayed, 5 regenerated. Both targets ✓ ✓ ✓; the three
co-payment cases that must stay `conditional` stayed ✓ ✓ ✓.

Two existing tests failed after the change, and both were pinning the old
wording or layout. They were rewritten to keep guarding their original point
(non-events still collapse, just not a room within its cap), not deleted.

---

## Fix: a misfiled age, checked against the words it came from

M12 tried two prompt fixes for "My father, 82" being filed as 82 **at policy
start** (which fires a senior-citizen co-payment nobody established). Both made
it worse. This time the extraction is checked against its source text, the way
M11 checks extracted exceptions: an age at policy start is only something a
person can have said if they mentioned the policy starting.

```python
# api/app/pipeline/scenario.py
_POLICY_START = re.compile(
    r"\b(bought|buy|purchased?|took|taken|take|got|started?|began|begin|"
    r"signed|joined|inception|enrolled|insured me)\b",
    re.IGNORECASE,
)

def correct_misfiled_age(facts, scenario):
    start_age = facts.get("age_at_policy_start")
    if start_age and not facts.get("age") and not _POLICY_START.search(scenario):
        return facts | {"age": start_age, "age_at_policy_start": None}
    return facts
```

It only moves a value, never invents one, and it leaves both ages alone when
both were given. `evals/check_fact_extraction.py` was changed to measure the
facts the pipeline actually uses, after this correction, and scored eval
sentences 18/18 and held-out 18/19. The held-out set includes five third-person
sentences written before the change, one of which does state an age at the
start ("My mother took this policy out at 61"), so the correction cannot pass by
never filling that field.

## Failure 39: a prediction made from stale data

The prediction for the eval was 0 regenerated cases: no eval question has an
age at policy start without words about buying or starting. **Measured: 2.**
Both were questions that say "I am 55" and "I am 50", and the stored
extraction had filed those as **age at policy start** too: first person,
present tense, no third person involved.

The prediction was wrong because it was checked against a printout of the facts
taken before M11 changed the fact prompt. Those two misreadings had been
invisible: both ages are under 60, so the co-payment check gave the same answer
either way. The correction fixed two real errors nobody had spotted.

It also moved one answer the wrong way. `oral-chemo-limit` went from two
samples of three right to zero of three, with the correct fact in place. It
was kept anyway, because it removes a false statement the pipeline had been
making ("age at policy start: 50" for someone who said "I am 50"), and it
fixed third-person ages. `oral-chemo-limit` then needed its own fix.

---

## When the answer key is wrong

Fixing `oral-chemo-limit` exposed an inconsistency in the golden answers
themselves. The file states its rule for sub-limits: a reduction that
definitely applies on the facts described makes the verdict `conditional`.
`oral-chemo-limit` and `robotic-surgery-limit` follow it, since clause 5.5 caps
those treatments by name. `cataract-served` has the same shape (clause 5.4 caps
"treatment of cataract") and expected `covered`. Any fix that pointed the model
at a sub-limit naming the treatment would fix one and break the other.

Expected answers were written before any measuring, which is what stops them
being rewritten to match whatever the system outputs. So the question went to
the project owner rather than being decided here, with the evidence. Two
changes were approved, and each is recorded inside the case itself:

- `cataract-served`: `covered` → `conditional`, with clause 5.4 added as its
  deciding clause, matching how `oral-chemo-limit` lists 5.5;
- `room-rent-within-cap`: citing 5.1 is no longer forbidden, because citing the
  limit the room stays within is a fair reason for `covered`, while relying on
  the proportionate deduction (5.2) is still wrong.

A label change moves the score without any code changing, so the log says
exactly which answers moved and why.

---

## Fix: a sub-limit that names the described treatment

`oral-chemo-limit`: someone on oral chemotherapy, and clause 5.5 restricts "oral
chemotherapy" by name. The reduction block told the model only "Also capped by
5.2, 5.4, 5.5, but only for the treatments those clauses name. Read them", and
the model answered that no sub-limit applied. Whether a clause's text contains
a word the person used is a lookup, so code does it:

```python
# api/app/pipeline/reduction.py
def _named_in(clause_text, facts):
    described = " ".join(str(facts.get(k) or "") for k in ("procedure", "condition"))
    mine = {w for w in re.findall(r"[a-z]{4,}", described.lower()) if w not in _GENERIC_WORDS}
    return sorted(mine & set(re.findall(r"[a-z]{4,}", clause_text.lower())))
```

`_GENERIC_WORDS` holds words like "operation", "surgery" and "treatment", so a
gallbladder operation does not "name" the clause about "operation theatre
charges". The match uses the extracted procedure and condition, not the whole
question, so "sum insured" can never match anything. A matched clause gets a
line of its own ("NAMES THIS TREATMENT: clause 5.5 mentions 'chemotherapy' …
whether it caps this claim is your call"). The line "Nothing here reduces this
claim" is suppressed beside it, since the two would contradict each other.

**Predicted reach**, checked against the current stored facts this time: 3
cases. **Measured:** 3 regenerated. `oral-chemo-limit`, `cataract-served` and
`cataract-sublimit-amount` ✓ ✓ ✓, each citing its sub-limit.

---

## Concept 35: a lookup reported as a lookup

Seven cases reached the right verdict with the wrong citation, and M12 showed
the model chose the same wrong clause even when asked for citations separately.
One of them, Ayurvedic treatment at an unaccredited private clinic, cited the
definition of a hospital. The AYUSH clause shares five unusual words with that
question.

This project deliberately has no retrieval (no embeddings, no BM25; Concept 15),
because ranking clauses and dropping the low-ranked ones could drop the one
that decides the case. A hint is a different thing: every clause stays in the
prompt, nothing is ranked, and code reports only a fact it can check ("these
words appear in both"). What a shared word means stays with the model.

The first version shows why the details matter. A word counted if at most two
of the 40 clauses used it. That hinted at **38 of 40** questions, because
everyday words like "years", "after" and "rupees" are rare inside a policy. It
even pointed two questions at the co-payment clause they must not cite
("years"). The final version drops standard English function words and names a
clause only when it shares **at least two** such words:

```python
# api/app/pipeline/scenario.py, shared_words()
rare_mine = {s for s in mine if 0 < used_by[s] <= _RARE_IN_POLICY}   # used by <= 2 clauses
for clause_id, stems in per_clause.items():
    common = rare_mine & stems.keys()
    if len(common) >= _MIN_SHARED:                                    # at least 2 words
        shared[clause_id] = sorted(mine[s] for s in common)
```

Words are compared by their first six letters, so "Ayurvedic" meets "Ayurveda".
That reached **8 of 40** questions, and in all 8 named exactly the clause the
case requires. **Caveat, stated in the code too:** both thresholds were chosen
after seeing the noisy version on the same 40 questions, so 8 of 8 is partly
fitted. That is why the hint only names clauses and never sets anything.

**Measured:** 8 regenerated. Citation recall 0.781 → 0.844 (`ayush-private-clinic`
now cites 2.6, `cosmetic-after-accident` cites 4.1), verdicts unchanged, both
resamples 8/8 verdicts and 8/8 required citations. It cannot help
`breach-of-law` ("shoplifting" shares no word with "breach of law") or
`ambulance-admissible` (only one shared word).

---

## Fix: a missing annexure, checked only where an answer leans on it

`day-care-not-listed`: six hours on a drip. Day care is paid for treatment
"listed in Annexure II", and the document has no Annexure II, so the honest
answer is that it cannot be checked. M12 printed a note under that clause in
every prompt and reverted it: it did not fix this case and moved four unrelated
ones.

This time the note is not in any prompt. It is a fourth rule in the
consistency check (Concept 33), which fires only on an answer that relies on
such a clause to pay:

```python
# api/app/pipeline/scenario.py, find_contradictions()
names_treatment = any(c.named for c in computed.reductions)
if verdict in (Verdict.COVERED, Verdict.CONDITIONAL) and not names_treatment:
    for citation in citations:
        names = computed.absent.get(citation.clause_id)
        if citation.effect == "permits" and names:
            problems.append(...)   # "refers to Annexure II, which is not included ..."
```

Predicted reach was 2 answers: the target, and `cataract-served`, which also
cites the day-care clause. The first version fixed the target ✓ ✓ ✓ and broke
`cataract-served` in both resamples: the retry turned it into a cosmetic-surgery
refusal. The refinement is the `names_treatment` condition. "Treatment of
cataract shall be limited to 25,000" presupposes cataract treatment is paid, so
the missing list is not what that answer rests on. Nothing in the policy names
a six-hour drip. After the refinement both were right in both resamples.

In the final majority run, `day-care-not-listed` was right in the stored sample
and wrong in both fresh ones. Across every sample since the fix it is about 5 of
7. It is fixed most of the time, not reliably.

---

## Failure 40: a design promise the code never kept

The project's written grounding design says: a quotation that fails the
verbatim check gets "one retry → otherwise shown as unverified". Reading the
code for the two quotations that failed in every run showed that the retry had
never been built. A failed quote was simply flagged.

The retry was built, and measured: **it fixed 0 of 2.** Printing both attempts
side by side showed why. The model repeated each failed quotation almost byte
for byte. Both had the same shape: the clause's own words, exactly, with
something appended:

```
clause 2.3: "... and the in-patient claim has been accepted by the Company."
            (clause 2.3 ends at "accepted"; "by the Company" is from clause 2.2)

clause 4.1: "4.1 Cosmetic and Plastic Surgery ... to be medically necessary.
             EXCEPTIONS - this clause does NOT apply when: ..."
            (the whole clause, then the pipeline's own annotation line)
```

Asking again cannot fix a mistake that is repeated deterministically. But the
true part is already there to keep. The retry was removed (code that measurably
does not help is not left in) and replaced by a trim:

```python
# api/app/grounding.py
def verified_prefix(quote, source_text):
    tokens = quote.split()
    haystack = normalize(source_text)
    whole = len(normalize(quote))
    for k in range(len(tokens) - 1, 0, -1):
        kept = " ".join(tokens[:k]).rstrip(" ,;:")
        needle = normalize(kept)
        if len(needle) < MIN_QUOTE_CHARS or len(needle) < MIN_KEPT_SHARE * whole:
            return None
        if needle in haystack:
            return kept
    return None
```

It keeps the longest leading part of the quotation that appears word for word
in the clause, only if that part is at least 60% of the quotation and at least
25 characters long. The first condition is the guard. Without it, a true opening
followed by a longer invention would become a verified quotation once the
invention was thrown away. It generalises a tolerance the verifier already had
for a trailing "[3.2]" tag, under the same rule: what is kept must match exactly.

**Measured:** failed quotations 2 → 0 with no model call, and detection
integrity 1.000. That figure comes from the eval re-verifying every displayed
quotation independently of the pipeline's own flag, so the trimmed quotations
were checked, not trusted.

---

## Failure 41: the test question inside the prompt

Two of the remaining citation misses cite the cosmetic exclusion, 4.1, for IVF
and for an undisclosed thyroid condition. Looking for why 4.1 stands out
turned up something that should have been caught long ago. The reasoning
prompt's only example of a carve-out is:

```
A burn treated with reconstructive surgery falls inside "unless necessitated
by an Accident, Burn or Cancer"; that claim is covered, not refused.
```

That is the eval case `cosmetic-after-accident`, written into the instructions.
A test case inside the prompt inflates that case's score and draws attention to
one clause in every question.

It was replaced with an example matching no clause in the policy ("an exclusion
of hearing aids 'unless prescribed after an injury' …"). One sample: verdict
accuracy **1.000 → 0.800**, with 8 cases broken. Seven of the eight had nothing
to do with cosmetic surgery, and `ped-waiting-served` started citing 4.1. Eight
simultaneous breaks is far beyond the two or three cases a run moves on its
own, so it was reverted without spending more samples.

Two readings fit and this run cannot separate them. Either the prompt depends
on that example much more than one test case would explain, or the system
prompt is fragile enough that any new example in that section disrupts it, the
non-local effect seen in Failure 28. Either way, the current score depends on
an instruction that copies a test question. That is recorded as an open problem
rather than a solved one, and it qualifies the headline number below.

---

## Where this leaves the numbers

| | End of M12 (majority) | Now, sample 1 | Sample 2 | Sample 3 | **Now, majority of 3** |
|---|---|---|---|---|---|
| Verdict accuracy | 0.925 (37/40) | 1.000 | 0.975 | 0.925 | **0.975** (39/40) |
| Citation recall | 0.774 average, of 31 | 0.844 | 0.813 | 0.844 | of 32 |
| False citations | 0.200 average | 0 of 5 | 1 of 5 | 1 of 5 | |
| Failed quotations | 0–1 | 0 | 0 | 0 | |
| Detection integrity | 1.000 | 1.000 | 1.000 | 1.000 | |

37 of 40 cases gave the same verdict in all three samples. Sample 1 replays the
stored answers and reproduced the last single run exactly (40 of 40), which
confirms the reverts were complete. Clause classification (macro-F1 1.000) and
ranking (8 of 9) did not move.

The two label changes affect how these compare with M12: `cataract-served` now
needs a citation, so 32 cases require one instead of 31. The two verdict gains
(`senior-but-excluded`, `room-rent-within-cap`) are fixes, not relabels.

**Kept:** the co-payment and room-cap wording, the misfiled-age correction,
named sub-limits, shared-word hints, the missing-annexure check with its
refinement, quote trimming. **Reverted:** the quote retry (replaced by
trimming), replacing the prompt's test-case example.

**Still open:**

- `day-care-not-listed`: right about 5 samples of 7, and wrong in the final
  majority;
- five citations the model gets wrong after every approach tried: 4.4, 6.3,
  6.6, 2.5, 4.5;
- two samples of three: `dental-no-accident`, `copay-unknown-inception-age`;
- in samples 2 and 3, one of the four guarded cases in the second chunk cited a
  clause it must not rely on. The majority summary prints only the rate, so
  which case it was is not on record;
- the reasoning prompt's carve-out example is a copy of an eval question, and
  removing it cost 8 cases;
- `documents-late` also cites 5.1 and 5.3 beside its correct 6.2: not
  forbidden, but noise;
- the fact extractor sometimes subtracts two stated ages itself (accepted).

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 191 tests
python evals/run_scenario_eval.py                        # one run
python evals/run_scenario_eval.py --repeats 3 --only day-care-not-listed,ayush-private-clinic
python evals/check_fact_extraction.py                    # the fact extractor alone
```

**Question to sit with:** clause 3.2 ends "…excluded until the expiry of thirty
six months of continuous coverage…". The model quotes it as:

> *"Expenses related to the treatment of a Pre-existing Disease and its direct
> complications shall be excluded until the expiry of thirty six months, except
> for diabetes"*

Before opening the answer: does `verified_prefix` trim this into a verified
quotation? And if it does, what has it fixed, and what has it not?

<details>
<summary>Answer</summary>

**It trims it.** The leading part up to "thirty six months" appears in the
clause word for word, and it is well over 60% of the quotation, so it is kept
and marked verified. ", except for diabetes" is gone. (Checked by running the
function on this exact text: it returns the quotation up to "thirty six
months", while the untrimmed quotation fails verification.)

**What it fixed:** the reader no longer sees "except for diabetes" presented as
the policy's words. Every quotation shown is text the policy contains, which is
the grounding guarantee.

**What it did not fix:** if the model's *reasoning* relied on that invented
exception, for instance to answer `covered` for a diabetic, the verdict is
still built on it. Trimming makes a quotation true; it cannot make the
reasoning behind it true. That is why the consistency check and the rest of the
arithmetic exist beside the quote check rather than instead of it, and why a
short true opening attached to a long invention is left unverified: there, what
was invented is most of what was said.
</details>

---

# M14 — A score on questions it has never seen

## The starting position

The scenario simulator answers a what-if question about an Indian health
policy with one of four verdicts and the clauses that decide it. It is measured
on 40 hand-written cases in `evals/golden/scenarios.json`, and since M12 the
headline is a majority of three samples per case, because the model does not
answer identically twice.

At the end of M13 that figure was 0.975 (39 of 40). Two problems were judged
major:

1. **The headline could not be fully trusted.** The reasoning prompt's only
   example of a carve-out is one of the 40 eval questions, written into the
   instructions, and replacing it broke eight cases.
2. **`day-care-not-listed`** — six hours on a drip, where day care is paid only
   for treatments "listed in Annexure II" and the document has no Annexure II —
   still answered "covered" in about two samples of seven. A confident wrong
   answer where the honest one is "the document cannot tell".

Also open: five cases citing the wrong clause, two wobbling cases, and one
wrong citation the eval summary could not name.

The work was committed first (`2da3dd6`), so that every experiment below started
from a recorded state.

---

## Naming the case behind a rate

The majority-of-three summary printed a false-citation *rate* per sample, so
"one guarded case in five cited a forbidden clause" was on record and which case
it was was not. A small function now lists, per case, each missing or forbidden
citation and in how many samples it happened:

```
citation problems:
  copay-unknown-inception-age: cited forbidden 5.3 in 1/3
  breach-of-law: missing 4.4 in 3/3
```

A measurement you cannot act on is only half a measurement. Every later step in
this milestone used this output.

---

## Concept 36: a score on the cases you tuned against

Every decision in M10–M13 — keep this change, revert that one — was made by
looking at how the same 40 cases moved. Each decision was reasonable. But a
system chosen by hundreds of small decisions against one set of questions will
fit that set, the same way a model trained and tested on the same data scores
well on it. The prompt containing one of the test questions is the visible
instance of a much more general effect.

The only way to know how much of 0.975 was fit rather than skill is to ask
questions the system was never shaped around. So a **held-out batch** of 16 new
cases was written against the same policy and conventions — cancer
reconstruction (testing the same carve-out as the prompt's burn example, without
being it), war, a hysterectomy too soon, scuba diving, surrogacy, an ICU rate
within its cap, a co-payment where the age at inception has to be derived, stem
cell therapy, late notice of planned surgery, tests too early before admission,
a dental accident, lost luggage, a short procedure — and run with the current
system before anything else changed.

**Result: 12 of 16 = 0.750**, every case unanimous across three samples, against
0.975 on the main set.

That gap is the finding of this milestone. Twelve of the sixteen answers were
right; four were confident and wrong, and no main-set case would have shown any
of them.

**The discipline that follows** is easy to state and easy to break. The moment
held-out failures are read to decide what to fix, that batch is no longer held
out: any fix designed while looking at it will tend to fix it. So:

- the first batch became a *diagnostic* set once its failures were read;
- a **second batch** of 13 cases (`scenarios-heldout-2.json`) was written after
  the first round of fixes existed and before any run of it — mountaineering,
  contraception, a cataract too soon, maternity after the wait, drink-driving,
  an ICU rate over its cap, a co-payment that does not apply, notice given in
  time, physiotherapy inside the post-discharge window, deep brain stimulation,
  vet bills, loose-skin removal, gallbladder removal with no timing;
- each file's header records what it has been exposed to, and
  `evals/REPORT.md` now prints both held-out scores beside the main one, with
  that history next to each.

The batches are written by the same author as the fixes, so they are not fully
independent. Cases written by someone who has never seen the prompt would be a
stronger test, and that is recorded as a limitation rather than hidden.

---

## What the held-out failures had in common

Reading the four failures' own reasoning found general causes rather than
four accidents:

| Case | What happened | Cause |
|---|---|---|
| knee replacement 30 months in, "the knee trouble only started after I took the policy out" | refused under the 36-month pre-existing-disease wait | the fact extractor recorded pre-existing as "unknown" |
| kidney stone "which I never had before taking the policy" | the same refusal | the same extraction |
| minor procedure, home after three hours | "conditional on the procedure being listed in Annexure II" | the missing-annexure check only looked for citations labelled "permits"; this one was labelled "delays" |
| shelling during a war | "covered … subject to the room rent limit and the senior co-payment" | the war exclusion was never read, and two caps the person never raised were cited as reducing the claim |

---

## Failure 42: "unknown is the honest answer", and it answered nothing else

A check of the fact extractor alone (`evals/check_fact_extraction.py`) was given
new sentences about whether a condition was pre-existing, worded unlike any
eval question. On the current prompt all five came back "unknown" — including
*"I was diagnosed with asthma long before I bought this policy"*.

The cause was one sentence in the fact prompt, written earlier to stop the
extractor guessing:

```
For pre_existing_condition, answer "unknown" unless the description makes it
clear either way. "Unknown" is the honest answer far more often than not.
```

It stopped guessing, and it stopped reading. Replaced with a direct question
and two examples (worded unlike any eval sentence), the extractor went from five
failures on those sentences to two.

The main set then lost **six cases, each three samples of three** — in places
with nothing to do with pre-existing conditions (an ICU rate, a breach of law,
pre-hospitalisation expenses). The fact prompt feeds every reasoning prompt;
changing what it says about one fact changed many answers. Reverted, and
recorded as open: the extractor still never answers "no", and the two held-out
refusals remain.

> **The lesson:** a correct fix to one stage can be a net loss for the system.
> The extractor check said better; the end-to-end measurement said worse; the
> end-to-end measurement is the one that decides.

---

## A missing list, finished

M13's missing-annexure check fired on an answer that paid on a clause referring
to an absent annexure. Three gaps were found and closed:

**1. The effect label.** The held-out short procedure cited the day-care clause
as "delays", so a check for "permits" never fired. It now fires on any label
but "denies".

**2. What happens after the retry.** When the retry still leans on nothing but
that clause — plus definitions, which explain a word and pay nothing — the
answer has nothing checkable behind it. That is the same position as an answer
with no citations, and it goes the same way:

```python
# api/app/pipeline/scenario.py
def _rests_only_on_a_missing_list(citations, verdict, computed, clauses):
    ...
    definitions = {c.clause_id for c in clauses if c.clause_type == "definition"}
    noise = unraised_reductions(citations, computed)
    basis = [c for c in citations if c.effect != "denies" and c not in noise]
    return any(c.clause_id in computed.absent for c in basis) and all(
        c.clause_id in computed.absent or c.clause_id in definitions for c in basis
    )
```

**3. A refusal that answers a different question.** Four fresh samples of
`day-care-not-listed`, printed side by side, all did this after being told the
list was missing:

```
first  conditional  [2.4 permits, 5.1 reduces]
retry  not_covered  [2.1 permits]
       "not covered ... Clause 2.1 ... hospitalisation exceeding 24 consecutive hours"
```

A refusal that names no clause refusing it has abandoned the question rather than
answered it. For answers that leaned on a missing list before the retry, a
refusal that cites no exclusion, waiting period or condition as denying is also
downgraded. It is scoped that narrowly on purpose: elsewhere, *correct* refusals
carry sloppy labels too — a refusal on a cover window cites the window clause as
"permits" — and a general rule would catch them.

Result: `day-care-not-listed` three samples of three in the final run, including
both fresh ones; the held-out short procedure two of three.

---

## Failure 43: a retry that talked right answers out of themselves

Two caps the person never raised — no room, no age — were being cited as
reducing claims: a late-paperwork claim cited the room cap and the co-payment
beside the documents clause that decided it, and the held-out war injury cited
both. The arithmetic already says NOT_RAISED for exactly these.

First version: treat it as a contradiction and retry, telling the model
"nothing the person described bears on it". It reached 4 main-set cases,
exactly as predicted from the stored answers. Measured:

- the late-paperwork citations were cleaned up;
- **a held-out case whose right answer was `conditional` became "not enough
  information"** — its only citation had been the unraised co-payment, and
  without it the model lost its way;
- two main cases dropped to two samples of three.

Telling a model its reason is unsupported cannot tell it whether the verdict was
right for some *other* reason, and code cannot know that either. So the retry
became a **filter**: the unsupported citation is removed silently, never
retried, and only when another citation still stands — so it can never leave a
verdict with nothing behind it and trigger a downgrade.

```python
# api/app/pipeline/scenario.py, run_scenario()
noise = unraised_reductions(citations, computed)
if noise and len(noise) < len(citations):
    citations = [c for c in citations if c not in noise]
```

Every stored main-set answer then replayed exactly (0 of 40 fresh), the
paperwork case kept its single correct citation, and the held-out late-notice
case was right again.

---

## Failure 44: a fallback that destroyed a corrected answer

In the final main run `room-rent-within-cap` — a room at 8,000 within a 10,000
cap — answered "insufficient information" in both fresh samples. Printing four
fresh first answers and retries side by side:

```
first  conditional  [5.1 reduces]
retry  covered      [5.1 reduces]
       "... 8,000 rupees per night does not exceed ... (10,000 rupees per day).
        Therefore, the claim is paid in full."
```

Four of four. The retry *fixed the verdict*, with exactly the right reason — and
kept the label "reduces" on the room cap. The code then saw a cleared clause
cited as reducing, dropped it (Concept 33), found no citations left, and
downgraded the corrected answer.

The label was wrong, not the clause. Under a `covered` verdict, the answer and
the arithmetic agree the clause cuts nothing, so it permits. The fallback now
relabels in that case and still drops under any other verdict:

```python
if irrelevant and verdict == Verdict.COVERED:
    citations = [replace(c, effect="permits") if c in irrelevant else c for c in citations]
elif irrelevant:
    ...  # dropped, as before
```

Three fresh samples of three afterwards, for this case and the two others that
pass through the same fallback.

> **The lesson:** a safety fallback is code too, and it can be the bug. This one
> was measured working (M12) on answers where the model repeated its mistake; it
> had never met an answer where the model fixed the verdict and not the label.

---

## One more contradiction, and what finding it cost

A held-out batch 2 case — intensive care at 9,000 a day against a 6,000 cap —
answered `covered` while citing the ICU cap as *reducing* the claim, three
samples of three. The arithmetic had already computed that the cap applies. The
citation and the calculation agree the claim is cut; only the verdict disagrees.
That is now a fourth consistency rule, with one retry that shows the
calculation and ends "a claim paid less than in full is conditional, not
covered". It reached no stored main-set answer, and the case went to three
samples of three.

It was found by reading a batch 2 failure, so that one case is no longer clean
evidence of generalisation, and the report says so.

---

## Failure 45: blaming the last change for a drop that was sampling

Batch 2 scored 12 of 13 when first run and 10 of 13 on the version after three
more changes. The obvious story: those changes hurt generalisation — and one of
them had just replaced a retry with a filter, which would fit.

Before believing it, the two cases that moved were probed directly:

- one's fresh first answers cited nothing any of the three changes acts on — no
  code path had changed for it at all;
- the other went through a rule present, unchanged, in both versions.

So the drop was the model's variance, not the changes: thirteen cases at three
samples each moves by one or two on its own. The obvious story was wrong, and
it would have led to reverting a change that was not responsible.

> **The lesson:** before attributing a movement to a change, check the change
> can even reach the cases that moved. It is usually a two-minute probe.

---

## Where this leaves the numbers

Majority of three samples per case:

| Set | End of M13 | End of M14 |
|---|---|---|
| Main 40 (tuned on) | 0.975 (39/40) | **0.950** (38/40) |
| Held-out batch 1 (16, read while fixing) | 0.750 before any M14 change | **0.813** (13/16) |
| Held-out batch 2 (13, written before running) | — | **0.846** (11/13) |

Detection integrity was 1.000 in every sample of every set. Clause
classification (macro-F1 1.000) and ranking (8 of 9) did not move.

Two notes on how these were run, because they bound what the numbers can claim.
The main chunks ran across the last three versions; each later change was shown
to reach no stored main-set answer, and the only fresh-sample effect of the last
one could be to turn a wrongly `covered` capped claim into `conditional`.
Batch 2's second chunk ran one version before the last, which has no case the
final rule can reach.

**The honest headline** is therefore not 0.950. On questions this system was not
tuned on it answers about **83–85%** correctly, and every quotation it shows is
verified. That is the number to quote.

**Still open:**

- the fact extractor never records a condition as *not* pre-existing, so two
  held-out claims for new conditions are refused under the pre-existing wait;
  the prompt fix for it cost six main-set cases;
- the war exclusion is never read (held-out);
- `breach-of-law` answers `covered` in some fresh samples;
- held-out misses `gallbladder-no-timing` and `emergency-notice-on-time`;
- citations the model gets wrong after every approach: five on the main set,
  several on the held-out batches (4.1, 4.8, 4.6, 5.3, 6.1);
- the reasoning prompt still contains an eval question as its example — now
  measured around rather than removed;
- the batches were written by the same author as the fixes.

---

## Closing out M14: Failure 46, a report that misdescribed its own run

`evals/REPORT.md` is this project's record of a measurement. It is generated by
`evals/run_all.py`, which stamps the report with the model, prompt version,
decoding settings and git commit that produced it. That stamp exists because of
Failure 15 (M6): two reports describing different runs, with nothing in either
saying so. **A report is a record of a run; a record that is wrong about the
run is worse than none, because it is believed.**

Before merging M8–M14, the documents were read against the code, and the report
was wrong about itself in two ways.

**1. The commit stamp could not say the code was uncommitted.** The stamp was
the bare hash of HEAD:

```python
# evals/run_all.py, before
["git", "rev-parse", "--short", "HEAD"]
```

The M14 report said `2da3dd6` — the M13 commit — while every number in it
measured M14 changes that existed only in the working tree. Anyone checking out
`2da3dd6` to reproduce it would have got different code. The stamp now appends
`-dirty` when `git status --porcelain` reports any change, with one exclusion:

```python
# evals/run_all.py, _git_commit()
["git", "status", "--porcelain", "--", ".", ":(exclude)evals/REPORT.md"]
```

REPORT.md is excluded because this script rewrites it. Without the exclusion,
running the eval twice on a clean commit would call the second run dirty — a
warning that fires on every run is a warning nobody reads (the same lesson as
the quotation check that "cried wolf" in M5).

**2. The report described a cache key that no longer existed.** Its
"Reproducing this" section said responses are "cached by a hash of model,
prompt version, messages, schema and decoding settings". Since Concept 30 (M10)
the prompt version is a label only; `cache.make_key()` in `api/app/llm/cache.py`
hashes `model`, `messages`, `schema` and `options`. The sentence was generated
by code, so it was reprinted, wrong, into every report after M10. A reader
trusting it would expect a version bump to regenerate every answer — the exact
opposite of what the cache does and the property the M10–M14 experiments relied
on.

The same pass found the README still reporting M8's 0.675 and 129 tests, and
the scenario module's docstring claiming a 32,000-token context against a
configured 8,192. All corrected.

### The check of the fix was circular, the first time

To verify the new stamp, the edit to `run_all.py` was stashed to get a clean
tree, and the stamp was printed: `7379563`, as expected. Then REPORT.md alone
was modified: `7379563` again, as expected.

Both results were worthless. Stashing the edit also removed the new
`_git_commit()` — those two runs executed the **old** function, which prints a
bare hash no matter what. They would have "passed" with the exclusion broken or
missing. Only the `-dirty` result from the edited tree had tested the new code.

The check was redone by copying the new `run_all.py` outside the repository,
importing it from there, and pointing it at the repository:

| Tree state | Stamp from the NEW code |
|---|---|
| clean | `7379563` |
| only REPORT.md modified | `7379563` |
| another tracked file modified | `7379563-dirty` |

> **The lesson:** "the expected output appeared" is only evidence if the code
> under test is the code that ran. Setting up the conditions for a test can
> quietly change what is being tested — here, getting a clean tree removed the
> change being verified. It is the circular test of Failure 1 in a smaller form.

---

## Check it yourself

```bash
cd api && .venv/Scripts/python.exe -m pytest -q          # 201 tests
python evals/run_all.py                                  # main + both held-out batches
python evals/run_scenario_eval.py --heldout 2 --repeats 3 --only ho2-icu-over-cap,ho2-vet-bills
```

**Question to sit with:** you change the fact prompt, and the main set loses six
cases three times out of three — but the fact extractor's own check improves from
five failures to two. Before opening the answer: which result decides whether
the change stays, and what would you need to see to change that decision?

<details>
<summary>Answer</summary>

**The end-to-end result decides, and the change goes.** The extractor check
measures one stage; the product is the verdict a person reads. A stage getting
more accurate while the verdicts get worse means the stage's output is used in
ways the stage check cannot see — here, a fact that appears in every reasoning
prompt shifted answers to questions that had nothing to do with it.

What could change the decision is not a better extractor score. It would be
end-to-end evidence that the six breaks are recovered without losing the gain —
for example, the same fact given to the reasoning step in a way that does not
alter every prompt (only when a pre-existing wait is actually at issue), measured
on the main set *and* on a held-out batch, three samples each.

The general shape is worth keeping: a component metric is a diagnostic. It can
tell you where to look and whether a stage improved. It cannot tell you whether
the system did.
</details>

---

# M15 — First contact with real policies

## The starting position

This system reads an Indian health insurance policy PDF in five stages: ingest
the text with character offsets, segment it into clauses by rules, classify each
clause with a local 7B model, score each clause's risk by arithmetic, and answer
what-if questions ("I had knee surgery 8 months in") with cited clauses.

At the end of M14, every number in this project described **one document**: a
synthetic 39-clause policy, 5 pages, about 12,000 characters or 3,400 tokens,
written for this repository by the same author as the code. On it, scenario
verdicts scored 0.950 on the tuned cases and 0.81–0.85 on held-out ones.

The design rests on a premise about size. There is deliberately no retrieval
(Concept 15): instead of searching for relevant clauses, the scenario step is
shown every clause, because a whole policy was believed to fit in the model's
context. The clause budget is derived in `api/app/config.py`:

```python
num_ctx: int = 8_192
scenario_reserved_tokens: int = 2_500   # system prompt, facts and the answer

@property
def scenario_token_budget(self) -> int:
    return max(self.num_ctx - self.scenario_reserved_tokens, 1_000)   # 5,692
```

and clause text is converted to tokens by an estimate,
`CHARS_PER_TOKEN = 3.5` in `api/app/pipeline/scenario.py`. None of this had
ever met a real insurer's policy wording.

**The rule for this milestone: measure, do not fix.** A fix chosen before the
measurement is a guess about which problem is the biggest, and every earlier
milestone that guessed paid for it.

---

## Getting real documents without committing them

A policy wording belongs to the insurer that wrote it, so no PDF is committed.
`evals/real/sources.json` records where each is published and a SHA-256
fingerprint of the exact bytes measured, and `evals/real/fetch_policies.py`
downloads them into `samples/real/`, which is git-ignored.

Three wordings, chosen to differ:

| Policy | Why |
|---|---|
| Star Health, **Arogya Sanjeevani** | IRDAI's standard product: every insurer must sell it with the same terms |
| Niva Bupa, **ReAssure 2.0** | a retail product from a standalone health insurer |
| HDFC ERGO, **my: Optima Secure** | a large retail product from a general insurer; the longest |

The fingerprint exists because insurers revise wordings and re-upload them to
the same address. Without it, a later run would silently measure a different
document and compare the numbers as if nothing had changed. A mismatch stops
the run:

```python
# evals/real/fetch_policies.py
if digest != p["sha256"]:
    changed = target.with_suffix(".changed.pdf")
    changed.write_bytes(response.content)
    ...
    failed.append(p["id"])
```

That branch was tested by corrupting one fingerprint: the run exited 1, named
the policy, and kept the new bytes aside for inspection.

---

## Concept 37: a probe is not an eval

An **eval** compares answers against a key and produces a score. There is no
key for these documents yet, and writing one now would be premature: a key
names clauses by the ids the segmenter gives them, and if the segmentation is
wrong, the key names artefacts.

A **probe** asks a different question: *do the assumptions the design rests on
hold for this input?* It needs no labels, only the ability to count.
`evals/real/probe_real.py` asks, for each document:

- does the segmenter find clause boundaries, or cut blindly at its size cap?
- is a clause's number still a unique identifier?
- how much of the policy fits in the scenario step's budget, and what is dropped?

Probe first, eval second. It is the difference between checking a thermometer
reads 100 in boiling water and using it to diagnose a fever.

---

## The size of a real policy

| Policy | Pages | Characters | ≈ Tokens |
|---|---:|---:|---:|
| Synthetic golden policy | 5 | 12,022 | 3,400 |
| Star Health, Arogya Sanjeevani | 26 | 75,870 | 21,700 |
| Niva Bupa, ReAssure 2.0 | 24 | 79,552 | 22,700 |
| HDFC ERGO, Optima Secure | 53 | 150,538 | 43,000 |

Six to twelve times the synthetic document. Every page of all three had
extractable text — none was a scanned image.

---

## What the segmenter did to them

The segmenter (stage 2, no model) runs in seconds, so this was measured before
anything else:

| | Synthetic | Star | Niva Bupa | HDFC ERGO |
|---|---:|---:|---:|---:|
| Segments | 40 | 112 | 95 | 76 |
| Cut at the 3,000-character cap, not at a boundary | 0 | 5 | 7 | **44** |
| Median segment length (chars) | 286 | 178 | 503 | **2,930** |
| Segments sharing a number with another | 0 | **58** | 19 | 42 |
| Offset invariant `raw_text[start:end] == text` holds | yes | yes | yes | yes |

Listing Star Health's segments one per line showed three defects, none of which
the synthetic policy could have exposed.

### Failure 47: page furniture became clauses

Every Star page carries a running header, `7 / 25  Arogya Sanjeevani Policy,
Star Health And Allied Insurance Co Ltd. UIN : ...`, and each one was
segmented as a clause numbered with its page number: **26 of Star's 112
segments are page headers.**

How much of the number-sharing they explain took two wrong guesses to settle.
The first guess was section numbering restarting; the second, after seeing the
headers, was that headers were most of it. Counting found 19 of the 58
segments sharing a number are headers; the other 39 are restarted numbering
and list items. Both guesses were partly true and neither was the answer.

The most striking case is a phone number. The policy prints a toll-free number
`1800 425 2255.` that wraps, so a line begins `2255. Senior Citizens may call
at 044-40020888`. The segmenter's numbering rule accepts a dotted number
followed by a capital letter as a strong clause boundary — the test designed to
tell "4.2 Room Rent" from "1.5 lakhs" — and this line passes it. After
analysis and scoring, **clause "2255" ranked first by risk on the whole
policy**, at 82.2.

HDFC ERGO shows the same family in a table of sums insured: rows beginning
`75 Lakhs` and `100 & 200 Lakhs` became clauses numbered `75` and `100`.

### Failure 48: a defensive sort that interleaved two columns

Star's wording is set in two columns. Its segment `13#2` reads:

> 13. Treatments received in health hydros, *deliveries and caesarean sections*
> nature cure clinics, spas or similar establishments...

The italic words belong to the maternity exclusion in the other column. The
cause is one line in `api/app/pipeline/ingest.py`:

```python
blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]
# Sort top-to-bottom, then left-to-right. PyMuPDF usually returns
# blocks in reading order already, but real policy PDFs with tables
# and sidebars do not always cooperate.
blocks.sort(key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
```

Sorting every block on the page by its vertical position puts a line from the
left column next to the line level with it in the right column. PyMuPDF's own
order was already correct: extracting the same PDF with a plain `get_text()`
read cleanly, column by column. The sort was written as a precaution against
tables and sidebars that no test document contained, and on the first real
two-column document it destroyed the reading order.

**The offset invariant still held, and that is the lesson.** Every clause is an
exact slice of `raw_text` — but `raw_text` itself was assembled in the wrong
order. An invariant proves consistency with its source. It cannot prove the
source is right. A perfect slice of scrambled text is still scrambled.

### List items cut away from the heading that gives them meaning

Star's specified-disease waiting period is a heading, `24 Months waiting
period`, followed by a numbered list of twenty conditions. Each list item became
its own clause: `12. Hernia of all types` is a 24-character segment, and
`24 Months waiting period` is a separate one — placed after the list, because
of the sort above. A question about hernia surgery can find the word "hernia"
and no waiting period attached to it.

### HDFC ERGO: segmented by the size cap

44 of HDFC's 76 segments were cut at the 3,000-character cap. The segmenter's
own docstring describes the cap as "capping the damage done by a document with
no detectable structure at all", and that is what happened: most of the
53-page policy was divided into equal-sized pieces rather than clauses.

---

## Concept 38: the no-retrieval premise, measured

All 283 real segments were then analysed by the model (1,738 seconds; results
cached) and scored, and the scenario shortlist was applied exactly as a real
question would apply it. `shortlist()` sorts clauses by impact score and keeps
them until the token budget is spent:

| | Synthetic | Star | Niva Bupa | HDFC ERGO |
|---|---:|---:|---:|---:|
| Clause text, ≈ tokens | 4,128 | 22,757 | 24,306 | 43,273 |
| Clauses kept (budget 5,692) | 40 of 40 | 27 of 112 | 18 of 95 | **6 of 76** |
| **Coverage clauses kept** | 6 of 6 | **0 of 17** | **0 of 16** | **0 of 26** |

On every real policy, **every coverage clause was dropped**.

The reason is in what impact measures: how much a clause can cost the reader.
On these policies no clause that *grants* cover scored high enough to survive
the cut. Ranking by impact and cutting at a budget is top-k retrieval with a
ranking that prefers reasons to refuse. The reasoning step would see why a claim might
be denied and never the clause saying the treatment is covered.

Concept 15 rejected retrieval because "any k below all of them can drop the
single clause that decides the case". On real documents the system does exactly
that — silently. The only trace is a log line at `INFO` level:

```python
if len(kept) < len(clauses):
    log.info("shortlist dropped %d clause(s) for budget", len(clauses) - len(kept))
```

The design was right about the danger. It was wrong about the size, because the
size was measured on a document one-sixth as long as the smallest real one.

---

## The prompt, measured in tokens

Every size above is an estimate. Ollama reports the real one:
`prompt_eval_count`, the number of prompt tokens it evaluated. Nothing in the
codebase had read it. `evals/real/measure_prompt.py` sends the same scenario
prompt for each policy **twice**, because Ollama can reuse the start of a
prompt it has just processed, and if the count then covered only the new part a
single reading would understate the prompt. Both readings matched in every
case, so the count is trustworthy.

| | Prompt tokens | Left for the answer, of a 1,600 ceiling |
|---|---:|---:|
| Synthetic | 5,270 | 2,922 |
| Star Health | 7,326 | **866** |
| Niva Bupa | 7,105 | **1,087** |
| HDFC ERGO | 7,344 | **848** |

The reservation of 2,500 tokens has to hold everything that is not clause text
— the system prompt, the facts, the per-clause headers and computed lines — and
the answer, whose ceiling is `num_predict = 1,600`.

The synthetic policy's prompt is 22,711 characters and measured 5,270 tokens:
**4.3 characters per token**, averaged over the whole prompt, not the 3.5 the
budget assumes. At that rate its clause text (11,711 characters) is about 2,700
tokens, so everything else in the prompt is about 2,550 tokens — already more
than the whole 2,500 reservation, before any room for the answer. The system
prompt alone is 8,054 characters, about 1,900 tokens, and has grown a great
deal since the reservation was set in M5.

Two things hid this. The 3.5 estimate over-counts clause text, which errs
safe and quietly absorbed part of the shortfall. And the synthetic clauses
used only 3,346 of the 5,692-token budget, leaving slack. Real policies fill the
budget, and the answer's room falls to 848–1,087 tokens.

Nothing was truncated — every prompt is below 8,192 — and M5 estimated the
largest legitimate answer at about 500 tokens, so these answers still fit. What
is gone is the margin the ceiling exists to provide. This is Failure 12's
lesson again: the reservation and the things it must hold are numbers that
must agree, set independently.

The client now checks the reported count on every call:

```python
# api/app/llm/client.py, _check_context()
if prompt_tokens >= options["num_ctx"]:
    log.warning("prompt filled the %d-token context window - it was probably truncated", ...)
elif prompt_tokens + options["num_predict"] > options["num_ctx"]:
    log.warning("prompt used %d of %d context tokens, leaving %d for an answer capped at num_predict=%d", ...)
```

### Failure 49: a warning that claimed more than it measured

The first version had one branch, and its message said the prompt "may have been
truncated". On its first real run it fired for a 7,326-token prompt in an
8,192-token window — which cannot have been truncated. It had merged two
different conditions into the more alarming one. A warning is a claim; it should
say what was measured and nothing more, the same point Failure 38 made about
computed lines.

---

## Failure 50: the eval and the product gave clauses different ids

The scenario step's citations are locked by an `enum` to clause ids, and each
id is the clause's own number. Seeing "10" shared by four Star segments, the
first conclusion was that a citation of clause 10 had become ambiguous in the
product.

That was wrong, and reading the product's code showed it. The scenario endpoint
(`api/app/routers/scenarios.py`) already made repeated numbers unique — `10`,
`10#2`, `10#3`. It was the **eval's** copy of that logic
(`build_clauses()` in `evals/run_scenario_eval.py`) that did not:

```python
clause_id=seg.number or f"c{seg.order_idx}",     # the eval, before
```

The eval's own quote re-check then looked clauses up in a dict keyed by id, so a
repeated number silently kept only the last clause's text. On the synthetic
policy, which repeats no number, the two constructions could not disagree, so
nothing had ever shown they were two constructions. The logic now lives once:

```python
# api/app/pipeline/scenario.py
def citation_ids(numbered: list[tuple[str, int]]) -> list[str]:
    seen: dict[str, int] = {}
    ids = []
    for number, order_idx in numbered:
        base = number or f"c{order_idx}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return ids
```

and both the endpoint and the eval call it.

> **The lesson:** an eval that rebuilds part of the product measures its
> rebuild. Where the two must agree, share the code rather than keep two copies
> in step.

**A smaller failure while making that change.** The new function was inserted
after what looked like the last field of the `ShortlistClause` dataclass, having
read the file only that far. One more field followed. It ended up after the
function's `return`, where it is still valid Python — unreachable — so nothing
complained until 24 tests failed with `'ShortlistClause' object has no attribute
'section_path'`. Read to the end of the structure you are editing.

---

## Two smaller measurements

- **Analysis speed.** 6.0–6.3 seconds per clause on real wordings (112 clauses
  in 689 s). Older notes in this log and in `app/llm/cache.py` say a 200-clause
  policy takes 2–5 minutes; with one clause per call and one request at a time
  (M9), it is about 20 minutes. HDFC's 76 segments would take about 8 minutes in
  the app.
- **A truncated analysis.** One real clause's analysis ran past the 1,600-token
  ceiling mid-JSON. The M5 retry — double the ceiling rather than repeat an
  identical request — recovered it, as designed.

---

## Ten questions about a real policy, and a score that could not see its reasons

With the damage measured, the question is what it does to answers.
`evals/real/scenarios-star-arogya-sanjeevani.json` holds ten questions about the
Star Health policy, written in plain words and with every expected verdict
fixed from the policy's wording before any run: a hernia fourteen months in, a
hernia after the wait, an amateur's trekking injury, a caesarean, dengue in the
first month, a cataract bill over the limit, a room over the rent cap, treatment
in Dubai, teeth whitening, and gallstones with no timing given.

Two things about this policy shape the answer key. It applies a 5% co-payment to
**every** claim, so under this project's convention that a claim paid less than
in full is `conditional`, no payable claim here is ever `covered`. And
`must_cite` was left empty in every case: citation ids come from the
segmentation measured above, and a key naming `13#2` would name an artefact.

`run_scenario_eval.py` gained `--policy` and `--cases` for this, and refuses
`--policy` without `--cases`, because the synthetic cases' clause numbers mean
nothing in another document.

**Majority of three samples: 9 of 10 verdicts correct**, nine unanimous.
Detection integrity 1.000 in every sample: 11–14 quotations per sample failed
verification, and every one was flagged as unverified rather than shown as
evidence. The guarantee held on real input.

### Failure 51: nine right answers, mostly for wrong reasons

A verdict score is not an explanation, so the first sample's stored answers were
replayed — zero model calls — and read case by case:

| Case | Verdict | What the answer rested on |
|---|---|---|
| hernia at 14 months | ✓ not_covered | the 36-month **pre-existing disease** wait. The deciding clause is the 24-month specified-disease list. |
| dengue in the first month | ✓ not_covered | the 36-month pre-existing wait again. The deciding clause is the 30-day initial wait. |
| treatment in Dubai | ✓ not_covered | "pneumonia is a pre-existing condition". The overseas exclusion was not in the shortlist. |
| teeth whitening | ✓ not_covered | a fragment of a waiting-period definition, not the cosmetic exclusion |
| trekking, cataract, hernia after the wait | ✓ conditional | the 5% co-payment — true, but the adventure-sports exclusion and the cataract limit were never seen |
| room over the cap | ✓ conditional | the co-payment and a room cap of 6,000; the policy caps it at 5,000 (Failure 52) |
| caesarean | ✗ conditional | the maternity exclusion was dropped by the shortlist, and its words "caesarean sections" had been spliced into a different clause (Failure 48) |
| gallstones, no timing | ✓ insufficient_information (2 of 3) | the replayed sample said conditional, on the co-payment |

Four refusals landed on the right verdict through the wrong clause. Four
payable claims landed on `conditional` through the one clause — the universal
co-payment — that happens to make every payable claim on this policy
conditional. On this document, a system that answered "conditional, because of
the co-payment" to everything payable and "not covered, because of the
pre-existing wait" to every refusal would score about the same.

This is the M11 lesson (Failure 31: every check did its job, on a fact that was
wrong) at the level of the whole answer. A verdict-only score measures whether
the last word is right. It cannot see whether the reason is, and on a document
whose structure collapses the verdicts into two likely answers, the last word
carries very little information. **9 of 10 here is not evidence that the system
works on real policies.** It is a demonstration of why `must_cite` exists — and
it cannot be added until the segmentation gives clauses stable, meaningful ids.

The replay also showed where the failed quotations come from. The model quotes
the policy's real sentences, in their real order; the stored clause text is in
the interleaved order, so the sentence is not a contiguous substring of it. One
quotation was of a computed line — "room rent of 8,000 rupees per day EXCEED
that" — which is the pipeline's words, not the policy's (the same leak as
Failure 32). And one citation passed verification while being irrelevant: the
"clause 2255" segment happens to contain the fraud condition, whose text was
quoted, correctly, as the reason a hernia was refused. A verified quote proves
the words are in the clause. It never proved the clause is the reason.

### Failure 52: the arithmetic was right about a cap the policy does not have

The room-over-cap answer's "6,000" did not come from the model. It came from the
pipeline's own computed line, which the model then quoted:

```python
# api/app/pipeline/reduction.py, _room_cap_check()
limit = sum_insured * percent // 100
if charged > limit:
    return ReductionCheck(..., f"{percent}% of {_human_rupees(sum_insured)} is "
                               f"{_human_rupees(limit)} per day; {label} of "
                               f"{_human_rupees(charged)} per day EXCEED that, ...")
```

The synthetic policy caps room rent at a plain percentage, so a percentage is
all the analysis schema extracts (`cap_percent_of_sum_insured`). Star's wording
is "up to 2% of the Sum Insured subject to maximum of Rs.5000/-, per day" — the
lower of two limits, and the code models one. For a 3-lakh policy it stated a
6,000-rupee cap as calculated fact. Here the verdict survived, because 8,000
exceeds both. A 5,500-rupee room would have been declared within the cap — by
the part of the system that exists precisely so the model is never trusted with
arithmetic.

Deterministic code is only as correct as its model of the input, and that model
was built from one document. Moving arithmetic out of the model (M7, M8, M10)
made it reproducible; it did not make it complete.

---

## What M16 should fix, and in what order

Ordered by what everything downstream depends on:

1. **Reading order** (`ingest.py`, the block sort). Every later stage reads the
   spliced text.
2. **Page furniture and number-like prose** — running headers and footers, a
   phone number, table amounts — must not become clause boundaries.
3. **List items belong to their heading.**
4. **The shortlist** must not remove coverage wholesale, and the reservation
   must be derived from the measured system prompt and `num_predict` rather than
   set beside them.
5. **Caps with an absolute maximum** ("2% of sum insured subject to Rs.5,000")
   in the analysis schema and the room-cap arithmetic.

Only after those: an answer key with `must_cite` for real policies, and the
hosted-model comparison — which would otherwise be comparing models on
scrambled input.

Deliberately not done here: any of those fixes. Each is a design decision with
alternatives, and making them from a milestone whose purpose was measurement
would have been choosing before seeing.

---

## Where this leaves the numbers

The M14 numbers still describe the synthetic policy accurately; its 40 main
cases replay byte-identically after this milestone's changes. They say nothing
about real ones.

On real wordings, what is measured is structural: reading order broken on a
two-column document, page furniture as clauses, a scenario step that sees 6–27
clauses and no coverage clause, and an answer margin of 848–1,087 tokens against
1,600. The ten-question probe scored 9 of 10 verdicts and mostly for the wrong
reasons. **No accuracy figure for real policies should be quoted from this
milestone** — only that the system is not yet sound on them, and why.

---

## Check it yourself

```bash
python evals/real/fetch_policies.py                    # three wordings, fingerprint-checked
python evals/real/probe_real.py --structure-only       # seconds; the segmentation table
python evals/real/probe_real.py                        # adds analysis and the shortlist table
python evals/real/measure_prompt.py                    # real prompt tokens, with the GPU otherwise idle
python evals/run_scenario_eval.py --policy samples/real/star-arogya-sanjeevani.pdf \
    --cases evals/real/scenarios-star-arogya-sanjeevani.json --repeats 3 --no-report
```

**Question to sit with:** the offset invariant — every clause is an exact slice
of the extracted text — held on all three real policies, while one of them had
its two columns interleaved line by line. Before opening the answer: what would
a check need to compare against to catch that, and why can no check that only
looks at `raw_text` ever do it?

<details>
<summary>Answer</summary>

It needs a second, independent reading of the same page. The invariant compares
each clause with `raw_text`, and `raw_text` was built by the same code that
scrambled it, so the two agree perfectly while both are wrong. Any check whose
only reference is the pipeline's own output inherits the pipeline's mistakes.

A check that could catch it compares against something produced differently —
PyMuPDF's native block order, a text layer extracted by another library, or a
handful of sentences a person copied from the PDF by hand and asserted to appear
contiguously in `raw_text`. The last is the cheapest and the strongest: a
sentence that spans two lines of one column cannot appear contiguously if the
columns were interleaved.

It is the same shape as detection integrity (M6): a guarantee checked only
against the system's own record is a self-report.
</details>

---

# M16 — Reading a real policy correctly

## The starting position

This system reads an Indian health insurance policy PDF in five stages: extract
the text with character offsets (ingest), cut it into clauses by rules
(segment), classify each clause with a local 7B model (analyze), score its risk
by arithmetic (score), and answer what-if questions with cited clauses
(scenario).

Until M15 it had only ever read one document, a synthetic 39-clause policy
written for this repository. M15 ran it on three real insurer wordings and
measured, without fixing anything, what went wrong:

1. **Reading order.** Ingest sorted every text block on a page by its vertical
   position, which read a two-column page straight across and spliced the
   columns together (Failure 48).
2. **Page furniture.** Running headers, page numbers, a wrapped phone number
   (`2255. Senior Citizens may call…`) and table amounts became clauses
   (Failure 47). The phone number ranked first by risk on the whole policy.
3. **List items** were cut away from the heading that gives them meaning:
   `12. Hernia of all types` was a clause of its own, with no waiting period.
4. **The scenario step saw a fraction of the policy.** The context window was
   8,192 tokens, real wordings are several times that, and the shortlist that
   decides what fits ranked by impact — which dropped every coverage clause on
   all three policies.
5. **A room-rent cap with a rupee maximum** ("2% of the Sum Insured subject to
   maximum of Rs.5000/- per day") was computed from the percentage alone
   (Failure 52).

And ten questions about one real policy scored 9 of 10 verdicts, mostly for the
wrong reasons (Failure 51), because the answer key held verdicts only.

M16 fixes them in that order, because each stage reads what the one before it
produced. It is scoped to one policy, **Arogya Sanjeevani**: IRDAI requires every
health insurer in India to sell it with the same wording, so "the system reads
the standard IRDAI product correctly" is a claim that can be checked. Star
Health's copy is the one measured; the other two wordings are watched for
regressions, not tuned for.

---

## Step 1: reading order

### What was measured first

M15's diagnosis was that PyMuPDF's own block order was already correct and the
sort destroyed it. Before deleting the sort, that was checked on all three real
documents, not only the one that showed the bug. Printing each block of a
two-column Star Health page in the order PyMuPDF returns it:

```
    0 x  147- 552 y  813- 827   '9  /   25'                         <- footer
    2 x  100- 496 y   38-  55   'STAR HEALTH AND ALLIED INSURANCE…' <- header
    3 x   88- 291 y   96- 175   'basis, provided no claim has been' <- left column
   …
   10 x   88- 291 y  718- 782   'has been increased at the time of'
   11 x  330- 556 y   96- 114   'viii. If a claim is made in the…'  <- right column
```

Left column top to bottom, then right column top to bottom. Sorting by `y` puts
block 11 (y=96) straight after block 3 (y=96).

A crude check for disorder was then run over all three: count the places where
reading order jumps back *up* the page within one column. It found 99 on HDFC
ERGO's wording, 29 on Niva Bupa's, 24 on Star's — which looked like evidence
that native order was *also* broken. Printing every one of them showed none
were: they were table cells read row by row (`'75 Lakhs' -> '100 & 200'`, the
second cell's text sitting two points higher), and list numbers vertically
centred beside taller text. The metric flagged the right shape for the wrong
reason. A count that looks like evidence is only a list of places to look.

### The fix

```python
# api/app/pipeline/ingest.py, _page_lines()
# PyMuPDF's own block order, deliberately unsorted. It follows the order
# the typesetter wrote the text, which on all three real policies measured
# in M15 was reading order: a column at a time, tables row by row. Sorting
# by position, as this once did, read a two-column page straight across and
# spliced the columns together.
blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]
```

What was rejected: detecting columns by clustering blocks on their `x`
position. It is more code, it would have to guess where a full-width table
ends, and native order was measured correct on every page that exists. The
risk kept: a PDF whose content stream is written out of visual order. None has
been seen; if one is, this is where it will show.

### The test, and where its reference comes from

M15 closed on a question: the offset invariant (`raw_text[start:end] ==
clause.text`) held on all three policies while one had its columns interleaved,
so what could catch that? Answer: only a reference the pipeline did not
produce. A check against `raw_text` inherits every mistake in `raw_text`.

Real PDFs cannot be committed, so the test builds its own two-column page with
paragraphs of staggered height — the arrangement that interleaved on the real
document — and compares with the order the paragraphs were *written*:

```python
# api/tests/test_ingest.py
left = [(90, 150, "LEFT-ONE …"), (160, 260, "LEFT-TWO …")]
right = [(90, 120, "RIGHT-ONE …"), (130, 230, "RIGHT-TWO …")]
…
positions = [raw.index(label) for label in ("LEFT-ONE", "LEFT-TWO", "RIGHT-ONE", "RIGHT-TWO")]
assert positions == sorted(positions), f"reading order scrambled: {raw!r}"
```

It was run against the old code first and failed exactly as the real document
did — `RIGHT-ONE` between the two left paragraphs. A test never seen failing
has not been shown to test anything.

The synthetic policies extract byte-identically before and after, so none of
M14's numbers moved.

---

## Step 2: page furniture

### Measured before a rule was written

For every line in the top or bottom tenth of a page, how many pages carry the
same text, with digits masked so `7 / 25` matches `8 / 25`:

| | Repeated in the margin on | Most repeated real content |
|---|---|---|
| Star Health | 4 lines on **100%** of 26 pages | list markers (`#.`, `i.`) on ≤ 50%, all over the page |
| Niva Bupa | 1 line on **100%** of 24 | `#.#.#.` on 33% |
| HDFC ERGO | 6 lines on **100%** of 53 | `c.`, `i.` on 43% |
| Synthetic ×2 | nothing | — |

Two groups with a gap between them: furniture is on every page at the same
height, content repeats on at most half the pages and anywhere on them.

### The rule

```python
# api/app/pipeline/ingest.py
MARGIN_BAND = 0.12            # headers and footers sat within 10% of an edge
FURNITURE_PAGE_SHARE = 0.5    # furniture on 100% of pages; content at most 50%
```

A line in the top or bottom 12% of the page, whose digit-masked text is in that
same margin on at least half the pages (and at least 3), is not policy text.
Removing it from `raw_text` rather than skipping it later matters: a clause that
runs across a page break is otherwise sliced *with* the header inside it, and a
quotation of the sentence that crosses the break can never be found in it.

**Recognised by repetition, not by wording.** A list of header phrases can only
know the insurers someone remembered. Every insurer's header repeats.

Ingest became two passes, because a line can only be recognised as furniture by
comparing pages. Offsets are assigned in the second pass, after removal, so the
invariant holds by the same construction as before.

Three tests: headers and page numbers removed while a sentence broken across a
page reads unbroken; a body line repeated on every page but *not in a margin*
kept; and a two-page document keeping its header, because two pages are too few
to tell a running header from a title. The last one passes on the old code too,
by design — it guards the new rule against over-reaching, not the old bug.

### A character nobody could see

Star's clause numbers arrived as `13.\t \x07Treatments`: a tab and a bell
character (`\x07`) between the number and the word. The segmenter's evidence
that `13.` is a clause number rather than a quantity is that a capital letter
follows it, and the first character after the whitespace was `\x07`. Control
characters are now replaced with spaces at extraction. The test builds a PDF
containing `\x07` and was checked to fail on the old code — worth checking,
because had PyMuPDF stripped the character when writing the test PDF, the test
would have passed without testing anything.

---

## Step 3: what is a clause number

With order and furniture fixed, listing Star's segments one per line showed
exclusions 1–9 and conditions 2–9 were still missing — buried inside
3,000-character pieces cut by the size cap.

### Numbers on a line of their own

Printing the raw lines explained it:

```
p11 x  43- 54 y 147-165  bold  '2.'
p11 x  66-290 y 147-165  bold  'Speciﬁed disease / procedure waiting'
```

The number is a separate text run, a tab stop away from its words, and
extraction returns it as its own line. Every numbering pattern requires text
after the number, so `2.` alone matched none. HDFC ERGO does the same for every
clause (`1.2.` | `Home Health Care`), which is most of why 39 of its 70 segments
had been cut by the size cap.

A line holding only a marker is now read together with the next line, if that
line sits on the same row and to its right — the way a reader sees it:

```python
# api/app/pipeline/segment.py
def _as_read(lines: list[Line], i: int) -> str:
    line = lines[i]
    if i + 1 < len(lines) and _RE_MARKER_ONLY.match(line.text):
        mate = lines[i + 1]
        half_height = (line.bbox[3] - line.bbox[1]) / 2
        if (mate.page == line.page
                and abs(mate.bbox[1] - line.bbox[1]) < half_height
                and mate.bbox[0] > line.bbox[0]):
            return f"{line.text} {mate.text}"
    return line.text
```

### The same shape, two meanings

That fix alone would have made things worse, and the same page shows why:

```
p11 x 330  bold  '3.' | '30-day waiting period - Code Excl 03'     <- exclusion 3
p11 x 330  plain '2.' | 'Age-related Osteoarthritis & Osteoporosis' <- item 2 of a list
```

Same position, same form, both "a number, then a capital". Indentation cannot
separate them (both at x=330), and neither can sequence (3 follows 2 in both
the list and the exclusions). Listing every numbered line on Star found one
signal that did: **every real clause number was bold, and every list item was
plain** — exclusions 1–23, conditions 1–28, coverage 1–9 bold; `i.`–`xvi.`,
`A.`–`L.`, `01.`–`20.`, and `2255.` plain.

## Concept 39: evidence relative to the document

This project already refuses to use absolute font sizes: `body_size()` measures
each document's own body text, so "larger than body" means the same thing in a
policy set at 8pt and one set at 11pt. Bold is the same kind of signal. In a
policy that sets its clause numbers in bold, a number in plain type is telling
you something; in a policy with no bold anywhere, plain type says nothing.

So it is decided per document:

```python
# api/app/pipeline/segment.py, segment()
if (is_clause_start and bold_numbers and not line.bold
        and _RE_SINGLE_LEVEL.match(numbering[0])):
    is_clause_start = False
```

where `bold_numbers` is true when at least 10% of the document's numbered lines
are bold. Two restrictions make it safe:

- **Only single-level numbers** (`7.`, `01.`, `(a)`). A list inside a clause is
  numbered 1, 2, 3; no list is numbered 5.1.3. Niva Bupa sets some real
  sub-clauses (`4.15.1.`) in plain type, and the rule leaves them alone.
- **Only when the document has a bold convention at all.** The unstyled test
  policy, built to catch exactly the mistake of requiring styling, has no bold,
  and a test asserts plain list numbers still start clauses there.

The transferable idea: a styling signal is only evidence against the document's
own baseline. The same bold line means "clause" in one policy and nothing in the
next.

### Failure 53: a threshold written down before it was measured

The first version of that rule had this comment and constant:

```python
# Measured in M16 on Star Health's wording: every clause number bold, every
# list item plain, 62% of numbered lines bold overall.
BOLD_NUMBERING_SHARE = 0.25
```

"62%" was not measured. It came from glancing at a list and estimating. When it
was measured, over every numbered line:

| Document | Numbered lines | Bold | Share |
|---|---:|---:|---:|
| Synthetic, styled | 39 | 39 | 1.00 |
| Synthetic, unstyled | 39 | 0 | 0.00 |
| HDFC ERGO | 259 | 82 | 0.32 |
| Star Health | 260 | 65 | **0.25** |
| Niva Bupa | 354 | 84 | **0.24** |

The share is far lower than guessed, because many plain lines merely begin with
a number ("24 hours", table rows). And the threshold picked before measuring,
0.25, landed exactly on Star and one hundredth above Niva Bupa. Three similar
documents would have been split by rounding.

The fixed threshold sits in the widest gap in the data — between 0.00 and 0.24
— at 0.10, and its comment now quotes the table rather than an impression. The
lesson is Failure 11's again: "to be safe" is not a number, and neither is "about
62%". A number in a comment is a claim, and it was checked only because
the code was about to depend on it.

### Bare integers are never clause numbers

Two policies still produced clauses like `24 Months waiting period`, `1 year
Tenure` and, from a table header, `75 Lakhs`. The rule allowed a bare integer
(no dot) as weak evidence, confirmed by bold — and on real wordings sub-headings
and table headers are bold. Across all six documents measured, no clause is
numbered with a bare integer: real ones use `3.`, `1.2.` or `4.15`, the
synthetic one `1.1`. So a bare integer is no longer numbering at all. An existing
test asserted the old behaviour (`_numbering("24 hours …")[1] is False`) and was
changed deliberately to `is None`, with the reason recorded in it.

One cost, measured rather than assumed: Star's `1 - PREAMBLE` and
`2 - OPERATIVE CLAUSE` were clauses numbered 1 and 2; they are now part of the
front-matter segment. The operative clause grants cover, so it is still shown to
the scenario step — under the id `c0` instead of `2`.

### Section words, matched as whole words

Star's definitions were filed under the section `STAR HEALTH AND ALLIED
INSURANCE COMPANY LIMITED`, and a table of non-payable items under `AIR
CONDITIONER CHARGES`. The fallback for unstyled section headings looks for words
like "limit" and "condition" in an all-capitals line, and matched them as
substrings — `LIMIT` in `LIMITED`, `CONDITION` in `CONDITIONER`. They now match
whole words, optionally plural.

### A test expectation that was wrong

The test that the bold rule leaves unstyled documents alone first expected four
clauses, `["2", "01", "12", "3"]`, and got three. The missing `3` was not this
milestone's doing: `3. 30-day waiting period` has a digit, not a capital, after
the number, which the older rule treats as weak evidence, and with no styling
nothing confirms it. The expectation was corrected, and the limit it exposed is
written into the test: in an unstyled document, a clause whose title begins with
a digit is missed.

### What the segmenter now produces

| | Synthetic | Star Health | Niva Bupa | HDFC ERGO |
|---|---:|---:|---:|---:|
| Segments, M15 → M16 | 40 → 40 | 112 → 78 | 95 → 141 | 76 → 101 |
| Cut at the size cap | 0 → 0 | 5 → 10 | 7 → 3 | 44 → 20 |
| Median length (chars) | 286 → 286 | 178 → 408 | 503 → 312 | 2,930 → 861 |
| Page headers as clauses | 0 → 0 | 26 → 0 | | |

On Star: coverage 1–9, exclusions 1–23 and conditions 1–28 are one clause each
under their own numbers; `2255` is gone; exclusion 2 holds its whole
24-month and 36-month lists. Star's count of repeated numbers went *up*, and
correctly: coverage, exclusions and conditions each restart at 1, and
`citation_ids()` gives the repeats distinct ids (`2`, `2#2`, `2#3`).

The synthetic policies' segments are identical to M15's, field for field.

**Still wrong on Star, and left:** the definitions (about 16,000 characters)
have no numbers, only a bold `Term:` at the start of each, and are still cut at
the size cap into six pieces — and filed under the company letterhead, a
one-off 15-point line that the PDF's content stream places after page 2's
columns. The claim-settlement condition runs past the cap. The annexure tables
are cut at the cap. None of these decides a claim question on its own, and a
3,000-character piece can still be cited and have its quotations verified.

---

## Step 4: the context window

### Concept 40: a limit is a measurement, with a version

The window was 8,192 tokens because of Failure 11: at 16,384, on Ollama 0.33,
the runtime held 5.46GB and the operating system killed an eval for memory
pressure. That was a measurement — of one runtime version, on one day, with
whatever else the machine was running. By M16 the runtime was Ollama 0.34, and
the requirement had changed: the smallest real wording is about four times
what 8,192 can hold.

When the reason for a limit changes, the limit is a hypothesis again.
Re-measured, loading the model with a policy-sized prompt at each window and
reading Ollama's own placement report:

| num_ctx | GPU memory | Placement | Prompt speed |
|---:|---:|---|---:|
| 8,192 | 4.9 GB | 100% GPU | 2,149 tok/s |
| 16,384 | 5.3 GB | 100% GPU | 2,137 tok/s |
| 24,576 | 5.8 GB | 100% GPU | 2,036 tok/s |
| 28,672 | 6.2 GB | 100% GPU | |
| 32,768 | 6.8 GB | **8% CPU / 92% GPU** | generation 38.5 vs 48.3 tok/s |

Free system RAM did not change between sizes: the KV cache lives on the GPU.
32,768 is qwen2.5's maximum, but it no longer fits in 8GB of VRAM, spills onto
the CPU, and slows *every* call by a fifth — including the per-clause analyses
that need no long context. **28,672 is the largest window measured to run
entirely on the GPU**, and it is the new setting.

### Failure 54: a truncation check that could never fire

The measurement above produced something else first. The script sized its
prompts at 4.3 characters per token (M15's figure for the whole scenario prompt)
and Ollama reported:

```
num_ctx 16384: prompt tokens 8194
num_ctx 24576: prompt tokens 12290
```

Half the window plus two, both times. Plain policy text tokenises more densely
than 4.3, so both prompts had overflowed — and what Ollama did with the overflow
was not what M15's check assumed. Stepping one prompt across an 8,192-token
window:

```
33,000 chars -> prompt_eval_count 8115
34,000 chars -> prompt_eval_count 4098
40,000 chars -> prompt_eval_count 4098
60,000 chars -> prompt_eval_count 4098
```

**On overflow Ollama discards half the context**, and the reported count *falls*.
M15 had added this to the client:

```python
if prompt_tokens >= options["num_ctx"]:
    log.warning("prompt filled the %d-token context window - it was probably truncated", ...)
```

A truncated prompt reports about half the window, which that check reads as a
small, healthy prompt. The one case the warning existed for was the one it could
not see. It was written from a belief about what truncation looks like, and it
was tested against that belief (a unit test fed it 8,192) rather than against
Ollama.

### Failure 55: the second detector, disproven in both directions

The count alone cannot tell "short" from "cut in half". The first replacement
used characters per token: policy prompts measured 4.06–4.3, a prompt cut to half
reports about twice that, so anything over 6.0 was called truncated. Its own
live test then failed — a repeated English sentence measured 5.47 characters
per token, uncomfortably close to 6.0 — and measuring more kinds of text
disproved the rule outright:

| Text | Characters | Reported tokens | Chars/token | Actually |
|---|---:|---:|---:|---|
| policy | 20,000 | 4,710 | 4.25 | intact |
| prose | 20,000 | 3,666 | 5.46 | intact |
| long words | 20,000 | 2,529 | **7.91** | intact — the ratio rule refuses it |
| numbers | 20,000 | **4,098** | 4.88 | truncated — the ratio rule passes it |

A column of numbers tokenises at about two characters per token, so 20,000 of
them overflow, and halved they report a ratio inside the range of intact policy
text. The ratio's "gap" existed for one kind of text.

What held in every case — policy text, prose, numbers, a real system-plus-user
scenario prompt, windows of 8,192, 16,384 and 24,576 — was the exact count:

```python
# api/app/llm/client.py
def truncated_prompt_tokens(num_ctx: int) -> int:
    return num_ctx // 2 + 2

def _check_context(prompt_tokens, options):
    …
    if prompt_tokens == truncated_prompt_tokens(options["num_ctx"]):
        raise LlmError(f"Ollama reported {prompt_tokens} prompt tokens, half the …")
```

Three decisions in that:

- **An exact signature, not a statistic.** An intact prompt could by
  coincidence have exactly that many tokens; the cost is a loud error on that
  one prompt, never a silent wrong answer.
- **It raises.** A truncated scenario prompt means an answer reasoned over part
  of the policy with nothing to say so. Retrying cannot help — the same prompt
  truncates the same way — and because the error is raised before the cache is
  written, a truncated answer is never stored.
- **A test re-measures it against the live server.** The signature is a
  behaviour of Ollama 0.34, not a law. The unit test encodes the measurement; a
  model-backed test sends a prompt sized at seven characters per token of window
  and asserts it is refused, so a future Ollama that truncates differently fails
  a test instead of silently disabling the check. (Its first version used a
  fixed 60,000 characters, which overflowed 8,192 and fits in 28,672.)

### Why a wider window, and not retrieval

With the window measured, the options for "the policy does not fit":

| Option | Rejected because |
|---|---|
| **A window of 28,672** | — chosen |
| 32,768 | spills 8% onto the CPU; every call a fifth slower |
| A first pass where the model picks relevant clauses from a summary list | an extra model step whose misses are silent, solving a problem the window removes for this policy |
| Keyword search (BM25) | "gallstones" does not match "Calculi in … Gall Bladder"; the design rules it out |
| Embeddings and a vector store | the same top-k failure Concept 15 exists to avoid |

Star's clauses are 68,598 characters. At 4.25 characters per token that is
about 16,000 tokens; with per-clause headers, the system prompt, the question
and the answer ceiling, about 22,000. It fits. HDFC ERGO's 129,000 characters
do not, so the shortlist remains — as a fallback that has to fail better.

### The shortlist, rebuilt

Four changes, each tied to a measurement:

```python
# api/app/pipeline/scenario.py
DROPPED_FIRST = frozenset({"definition", "procedural"})

def clause_token_budget() -> int:
    system_prompt = int(len(REASON_SYSTEM) / CHARS_PER_TOKEN)
    return settings.num_ctx - settings.num_predict - system_prompt - QUESTION_TOKENS
```

1. **The budget is derived, not reserved.** The fixed 2,500-token reservation
   was set in M5; by M15 the system prompt alone was about 1,900 tokens and the
   answer ceiling is 1,600. Now the room for clauses is what the window has left
   after the system prompt (measured from its length), an allowance for the
   question and computed lines (`QUESTION_TOKENS = 1_000`, from M15's measured
   ~650), and the answer. A longer system prompt shrinks the budget by itself.
2. **Definitions and procedural clauses give way first.** Impact measures what a
   clause can cost you, so the clause *granting* cover ranks lowest, and M15's
   impact-only ranking dropped every coverage clause on every real policy.
3. **A clause that does not fit is skipped, not the end of the selection.** One
   3,000-character annexure no longer pushes out every short exclusion ranked
   after it.
4. **Dropping anything is a warning**, naming how many clauses of each type.

### Failure 56: document order was clause-number order, until numbers repeated

The shortlist returned its selection sorted by clause number, documented as
"document order, because a person reading the answer expects clause 3.2
discussed before 6.1". On the synthetic policy the two are the same. On Star,
whose numbering restarts in every section, sorting by number would present
coverage 1, exclusion 1 and condition 1 side by side — and put every unnumbered
clause at the end. This was found by reading the code while changing it, before
any run could show it.

The selection now keeps the order the clauses arrive in. That moved the
question to the caller, and the scenario endpoint turned out to have none: it
read clauses from a query with no `ORDER BY` (under a docstring saying "ordered
by impact"). Order there decides two things — the order the shortlist keeps,
and which repeat of a number becomes `10#2`, which the eval must assign
identically (Failure 50). The endpoint now sorts by `order_idx` explicitly.

---

## Step 5: a cap with a maximum

Failure 52: Star caps room rent at "up to 2% of the Sum Insured subject to
maximum of Rs.5000/-, per day", and ICU at 5% up to Rs.10,000. The analysis
schema had a field for the percentage only, so the arithmetic told a 3-lakh
policyholder their cap was 6,000 rupees.

Two schema fields, `cap_max_inr_per_day` and `icu_cap_max_inr_per_day`, and the
limit becomes the lower of the two:

```python
# api/app/pipeline/reduction.py, _room_cap_check()
limit = sum_insured * percent // 100
rule = f"{percent}% of {_human_rupees(sum_insured)} is {_human_rupees(limit)} per day"
if maximum and maximum < limit:
    rule += f", above the {_human_rupees(maximum)} maximum, so the limit is {_human_rupees(maximum)}"
    limit = maximum
```

The computed line shows its working — "2% of 3 lakh is 6,000 rupees per day,
above the 5,000 rupees maximum, so the limit is 5,000 rupees" — rather than a
figure the clause never prints, which the model might otherwise quote as policy
text (Failure 38). With no maximum the line is word for word what it was.

The test is Failure 52's exact case: a 5,500-rupee room on a 3-lakh policy,
within 2% and over the maximum. It failed on the old code with
`DOES_NOT_APPLY`.

The prompt's example uses different figures (1%, Rs.4,000) from the policy under
test. Failure 41 recorded what a test case inside a prompt does to a score: it
measures recall of the example, not reading.

---

## The answer key gains reasons

M15's ten Star questions had verdicts and no `must_cite`, because the clause ids
were then segmentation artefacts. They are now the policy's own numbers, so each
case's `must_cite` was filled in — mechanically, from the reason already written
in the case before any run: the hernia cases name exclusion 2 (`2#2`), the
caesarean the maternity exclusion (`18`), Dubai the overseas exclusion (`22`),
and so on.

One deliberate omission: **no case requires the 5% co-payment**, though every
payable claim on this policy is conditional because of it. M15 found that clause
carrying several right verdicts for wrong reasons; requiring it would reward the
shortcut the key exists to catch. And one known strictness: the room-rent case
requires coverage 1, while the table of benefits repeats the same limit — an
answer citing only the table has a right reason this key does not accept.

---

## Step 6: what the long window cost the machine

### Where the memory goes, read from the runtime's own log

Ollama writes a server log (on Windows, `%LOCALAPPDATA%\Ollama\server.log`;
older sessions are rotated to `server-1.log`, `server-2.log`, ...). Each time it
loads a model, the llama.cpp runner prints what it allocated. At a window of
28,672 tokens:

```
load_tensors:        CUDA0 model buffer size =  4168.09 MiB
load_tensors:    CUDA_Host model buffer size =   292.36 MiB
llama_kv_cache: size = 1568.00 MiB ( 28672 cells,  28 layers,  1/1 seqs), K (f16):  784.00 MiB, V (f16):  784.00 MiB
sched_reserve:      CUDA0 compute buffer size =   160.01 MiB
llama_context: flash_attn            = auto
resolve_fused_ops: Flash Attention enabled
```

Four allocations: the weights (4.2 GB on the GPU, a little on the CPU), the
KV cache, and a scratch area for the arithmetic.

**The KV cache is the part the window controls.** When a transformer reads a
token, every layer computes a *key* (what this token offers to later tokens) and
a *value* (what it passes on when a later token attends to it). Generating the
next token means comparing against every earlier key, so rather than recompute
them the runtime stores them. For qwen2.5-7B: 28 layers, each storing 4 key
heads and 4 value heads of 128 numbers — 1,024 numbers per layer, 28,672 per
token, at 2 bytes each in 16-bit floating point (`f16`): **56 KB per token**.
Times 28,672 slots is 1,568 MiB, exactly the logged figure.

The store is allocated for the whole window when the model loads, not for the
prompt that arrives. A 2,000-token clause-analysis call made at this setting
holds the same 1,568 MiB as a 20,000-token scenario call.

**Flash attention was already on.** Ollama's own setting read
`OLLAMA_FLASH_ATTENTION:false`, yet the runner was started with
`--flash-attn auto` and enabled it. Flash attention computes attention in blocks
small enough for the GPU's fast memory, instead of materialising the full table
of every token against every other token; that is why the scratch area is only
160 MiB at a 28,672-token window.

**One figure is not explained.** The runner process was measured once holding
about 14 GB of committed system memory — far more than the ~6 GB these lines add
up to. The log also records `disabling mmap for llama-server load by default ...
reason=windows_cuda`: on this platform the weights are read into memory rather
than mapped from the file. Whether that accounts for the gap was not measured.

### Failure 57: measuring memory while an eval was running

**Setup.** A script loaded the model at each of four windows (8,192 to 28,672)
with a two-token prompt, read the runner's committed memory, and unloaded the
model between sizes. It was started while the synthetic eval, three samples per
case, was still running against the same Ollama server.

**What happened.** Ollama keeps one loaded copy of the model, and a request
asking for a different `num_ctx` forces it to unload and reload. With two
clients asking for different windows, the log shows **18 runner starts in nine
minutes**, each re-reading the 4.7 GB model. The measurement came out
inconsistent: its "wait until unloaded" step kept timing out, because the eval
kept loading the model back. The eval spent its time waiting on reloads.

Soon after, three requests failed with HTTP 500 part-way through an answer. The
runner had been generating at 44 tokens per second, and then the process was
gone, with no error line. At the reloads around those failures, the log's
system-memory line shows Windows growing its page file: free swap 13.0 GiB, then
16.8, then 20.4. Pressure on committed memory is the likely cause; nothing in
the log proves it. The third failure ended the run.

**Why.** Evals already run one request at a time (M9), because concurrent
requests change what is measured. A measurement script that talks to the model
is one more client, and the same rule applies to it.

**What it does and does not affect.** The list of runner starts shows none
between 22:12 and 22:58. The full synthetic eval ran from 22:35 to 22:56, inside
that gap, so its drop in accuracy was not caused by this interference. The
repeated-sample runs that overlapped the interference are not used as evidence.
The only fix is procedural: a measurement that loads the model runs only when
nothing else is using the server.

What held the other 8 GB is the subject of Failure 58, below. A first guess,
that the 14 GB was two runners briefly alive at once during the reloads, was
written down and then disproven by the next run.

### Concept 41: rounding the KV cache

The KV cache (above) stores its numbers as 16-bit floats. Ollama can store them
at lower precision instead, a server setting read once at start-up:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

`q8_0` is one of llama.cpp's formats for storing model weights, applied here to
the cache (this model's weights use a 4-bit format, `q4_K_M`). The numbers
are split into blocks of 32. Each block stores one 16-bit scale, and each number
becomes a whole number from -127 to 127 that is multiplied by that scale when
read. That is 32 bytes plus 2 for the scale, 34 bytes where f16 needs 64: 53%
of the memory. The values that come back are close to the originals, not equal
to them. Ollama accepts a quantised cache only with flash attention on, which is
why both variables are set.

Measured on this machine with nothing else using the server, loading the model
at each window (Ollama's `/api/ps` for placement, the runner's private bytes for
system memory, the server log for the cache):

| num_ctx | KV cache | On GPU | Runner private memory |
|---:|---:|---|---:|
| 8,192 | — | 4.4 GB, 100% | 5,531 MB |
| 28,672 | **833 MiB** (f16: 1,568) | 5.0 GB, 100% | 6,176 MB |
| 32,768 | 952 MiB (f16: 1,792) | 5.2 GB, **100%** | 6,302 MB |

The 28,672 figure is the arithmetic above exactly: 1,568 × 34/64 = 833. And the
window that did not fit on the GPU at f16, qwen2.5's full 32,768, now does.

**What it costs: the answers can move.** Rounded keys and values produce
slightly different attention scores, and with them slightly different logits. A
near-tie between two tokens can break the other way. So every eval figure
measured at f16 has to be measured again at q8_0 before it is compared with
anything, and the synthetic suite is re-run first because it is the one with a
long f16 history.

### A setting the cache key could not see

The response cache (`api/app/llm/cache.py`) keys each stored answer by a hash of
everything that shapes it: model, messages, schema and the decoding options sent
with the request. That rule exists so that a stale answer cannot be replayed
without anyone remembering to clear anything.

The KV cache precision breaks it. It shapes the answer, but it is set on the
server and never sent with a request, so it is not in `options`. Switched to
q8_0 without a change to the key, every eval would have replayed the answers
stored at f16 and reported them as q8_0 results — a comparison of two identical
sets of bytes.

The setting is now mirrored in `api/app/config.py`, read from the same
environment variable the server reads:

```python
kv_cache_type: Literal["f16", "q8_0", "q4_0"] = Field(
    "f16", validation_alias="OLLAMA_KV_CACHE_TYPE"
)
```

and hashed into the key, but only when it is not the default:

```python
fields: dict[str, Any] = {
    "model": model,
    "messages": messages,
    "schema": schema,
    "options": options or {},
}
if kv_cache_type != "f16":
    fields["kv_cache_type"] = kv_cache_type
payload = json.dumps(fields, sort_keys=True)
```

The condition is there for the answers already stored. All of them were
generated at f16, before the field existed; adding `"kv_cache_type": "f16"` to
every key would have made every one of them unreachable. A test pins that
property by hashing the old payload by hand and requiring the f16 key to match
it. A second test makes sure the client actually passes the setting.

`Literal` refuses a typo such as `q8`. The mirror has one weakness it cannot
check: it agrees with the server only if both were started from the same
environment. That is why its default is `q8_0`, the project's setting (the
README lists the three Ollama variables), and not Ollama's own `f16`. A
program started before the variable was set, such as an editor or terminal
opened earlier, sees no variable at all. With an `f16` default it would store
q8_0 answers under f16 keys, which is the mistake the field exists to prevent. Ollama's server log states what it is really using, in its
`server config` line and in the `llama_kv_cache` line printed at every load.
Every eval report now records the precision beside `num_ctx`.

### Failure 58: a memory measurement that could not see the workload

**Setup.** The table above was measured by loading the model with a two-token
prompt at each window and reading the runner's private memory: about 6.2 GB at
28,672. On that evidence, the 14 GB from Failure 57 was put down to the
interference, and the full M16 run was started with the smaller cache. A
PowerShell script sampled the runner's private memory and the system's
committed memory every 30 seconds for the whole run, reading only operating
system counters so that it could not disturb the model.

**What happened.** Through the synthetic suite (prompts of 2,000–5,500 tokens)
the runner stayed between 6.2 and 6.5 GB, and it stayed there while the Star
policy's 78 clauses were analysed one at a time. When the Star questions began,
nine minutes into that step, it started to grow by about 650 MB every 30
seconds:

```
21:19:49  runner 6,312 MB   system commit 32,679 of 39,644 MB
21:20:49  runner 7,629 MB
21:22:20  runner 9,613 MB
21:24:21  runner 12,265 MB
21:26:22  runner 12,933 MB  system commit 38,584 of 39,644 MB
```

The tool supervising the run stopped it for low system memory, as it had in
Failure 57's run. The 14 GB was real, and a two-token prompt could never have shown
it.

**Why.** Ollama 0.34 serves the model with llama.cpp's `llama-server`, and that
server keeps a *prompt cache in system memory*. When a new request arrives,
the saved state of the previous prompt (its KV cache, the notes described in
Concept 41) is copied out of the GPU into RAM. A later prompt that begins with
the same text can then restore that state instead of recomputing it. The server
log shows the store and its limit:

```
srv    load_model: prompt cache is enabled, size limit: 8192 MiB
srv        update:    - prompt 000001FEE8CCC5F0:   22592 tokens, checkpoints:  0,   656.619 MiB
srv        update:  - cache state: 11 prompts, 6048.786 MiB (limits: 8192.000 MiB, 28672 tokens, 281858 est)
```

A Star question is about 22,600 tokens, so each saved state is 657 MiB even at
q8_0 (at f16 it would be about 1.2 GB). Eleven of them made 6 GB, on top of the
6.2 GB the runner needs anyway, and the limit would have allowed 8 GB. The
synthetic policy never showed this: its prompts are a quarter of the size, so
the same number of saved prompts fits in about 1.5 GB.

**The fix.** The server's option `--cache-ram N` sets the limit in MiB (0
disables the store). Ollama starts the server itself, but the server also reads
the option from an environment variable, `LLAMA_ARG_CACHE_RAM`, which it
inherits from Ollama. It is now 1024: enough for the one long prompt the next
question is likely to share, not enough to pile up eleven. Confirmed at the next
model load:

```
srv    load_model: prompt cache is enabled, size limit: 1024 MiB
```

Disabling the store outright was the simpler choice, and was rejected. In
`run_scenario` (`api/app/pipeline/scenario.py`), each question makes a short
`extract_facts` call and then the long `reason` call. The short call replaces
the long prompt in the GPU's working state, and the store is what brings it
back: the log of that run records 168 restores
(`found better prompt with f_keep = 0.825, f_sim = 0.980`). Without the store,
every reasoning call would recompute all 22,600 tokens.

**The lesson.** A measurement has to use the workload's own shape. A two-token
prompt measures the memory the runner reserves, not the memory it accumulates
while working.

### Failure 59: restarting Ollama left the old runner alive

**Setup.** To apply the new limit, Ollama was stopped (`Stop-Process` on
`ollama` and `ollama app`) and started again, and a two-token request confirmed
the limit in the log.

**What happened.** The process list afterwards showed two `llama-server`
processes: the new one (6.2 GB) and the one from the previous run, still alive
and still holding 13.4 GB. System commit stood at 45.5 of 46.0 GB. Stopping the
old runner brought it back to 32.1 GB.

**Why.** On Windows, ending a process does not end the processes it started.
`ollama.exe` launches `llama-server.exe` as a child. Forcibly terminating the
parent gives it no chance to stop the child, so the runner was orphaned: nothing
would ever send it another request, and nothing would stop it. The same was
true of the eval: when its supervising shell was stopped, the Python process it
had started kept running.

**The fix.** A restart now also stops any `llama-server` process and checks the
process list before continuing. After the restart, `ollama ps` and
`nvidia-smi` confirmed the new runner was entirely on the GPU. It had been
loaded while the orphan still held graphics memory, and could have been placed
partly on the CPU.

---

## Step 7: right answers for the wrong reasons

On the Star policy the scenario step gave 8 of 10 right verdicts, and in 9 of
the 10 it did not cite the clause that decides the question. The app's promise
is to show a person *which clause* decides their claim, so a right verdict
resting on the wrong clause is a failure the verdict score cannot see.

### Failure 60: a plausible cause, checked before it was built

**The hypothesis.** Star restarts its numbering in every section, so the model
is shown several different clauses numbered "8", told apart only by a suffix:

```
### clause_id=8  (8)  [coverage]
### clause_id=8#2  (8)  [exclusion]
```

The section name is not in that header. The plan was to add it, and to give
repeated numbers ids that say where they live (`Excl-8` rather than `8#2`).

**The check.** The answers were already in the response cache, so replaying
them cost no model calls. A replay script printed, for each question, the
clause the answer key requires and the clauses actually cited. The id scheme
was not the problem: the teeth-whitening answer wrote `8#2` correctly in every
sample, and the wrong citations were deliberate, not slips between two "8"s.
Two patterns remained:

| Question | Needed | Cited | What the reasoning said |
|---|---|---|---|
| Hernia, 14 months | specified-disease list | pre-existing diseases | "a 36-month waiting period as per clause 1#2" |
| Dengue, 20 days | 30-day waiting period | pre-existing diseases | "dengue fever, which is a pre-existing condition" |
| Treatment in Dubai | outside-India exclusion | pre-existing diseases | "a 36-month waiting period for pre-existing diseases" |
| Caesarean | maternity exclusion | co-payment | "a 5% co-payment applies" |
| Gallstones, no dates | specified-disease list | co-payment | "subject to a 5% co-payment" |

The label change would have been built, measured, and found to do nothing.

### Where the wrong reasons came from

The replay also printed the computed blocks each prompt carried (the waiting
periods, reductions and shared words described in earlier milestones). Most of
the misdirection was written by the pipeline itself, in code that was tuned on
the synthetic policy:

1. **The waiting-period block.** For the hernia question it said that both the
   pre-existing diseases period and the specified-disease period "still apply
   and block treatment covered by THIS clause", pre-existing first. Nothing in
   the question mentioned an illness from before the policy. The model took the
   first bar it was given.
2. **Clauses misread at analysis time.** Pre-hospitalisation ("30 days prior to
   admission") was stored as a 30-day waiting period. The cataract limit (25%
   of sum insured or Rs.40,000 per eye) became a per-day room-rent cap. The
   specified-disease list, whose hernia entry is under 24 months, was stored
   as 36. The list of items the policy never pays for was typed as a sub-limit,
   so the reductions block told the caesarean question that it "names this
   treatment", because the list contains "DELIVERY KIT".
3. **The co-payment line.** Star's 5% co-payment applies to every claim, so
   every question was told "APPLIES: clause 9". The model answered
   "conditional" before looking for an exclusion.
4. **The shared-words hint never fired.** It names a clause only when the
   question shares two unusual words with it. The Star clause that decides a
   question usually shares one: "hernia", "caesarean". Those thresholds were
   chosen on the 40 synthetic questions, and their docstring says so.

The answer key's pre-existing field did not help either: the fact extractor
returned `pre_existing_condition: unknown` for all 79 stored questions,
including one where the person says their diabetes was diagnosed before they
bought the policy.

### What was changed

Two of the four were fixed first, because both are plain code and could be
checked against the stored facts without a model:

- **A one-word rule, limited to what the question is about.**
  `named_in_question` (`api/app/pipeline/scenario.py`) names a clause that uses
  one unusual word (used by at most two clauses) from the procedure or
  condition the person named, skipping definitions and procedural clauses.
  Measured on the stored facts of all 79 questions before any model run, it
  names a clause for 18. For 13 of those, a clause the answer must cite is
  among the named. Of the 21 clauses named, 13 must be cited, 4 concern the same
  treatment without deciding it, and 4 are noise ("admitted", "removal",
  "existing", "anaesthesia").
- **The waiting-period block says whom a period concerns.** In
  `api/app/pipeline/waiting.py`, a period whose opening words mention
  pre-existing diseases gets its own sentence. A period naming the person's
  condition is listed first, the pre-existing one last. A served period that
  names the condition gets its own line instead of being filed as irrelevant.

### What the model did with it

Three samples per Star question (every question unanimous) and one sample
per synthetic question:

| | Before | First wording | Second wording |
|---|---|---|---|
| Star right verdicts | 8/10 | 7/10 | 7/10 |
| Star citing the deciding clause | 1/10 | 2/10 | 2/10 |
| Synthetic, main 40 | 34/40 | — | 34/40 |
| Synthetic held-out 1 / 2 | 11/16, 11/13 | — | 10/16, 12/13 |

The first wording appended a qualifier to the pre-existing line: "...still
applies and blocks treatment covered by THIS clause. It concerns only an illness
the person already had". Hernia at 14 months moved to the right clause. Dengue
and Dubai did not: the dengue answer quoted the opening words back. And the
served hernia line, "no longer stands in the way of this treatment", turned a
right "conditional" into "covered", read as permission.

The second wording opened the pre-existing line with whom it concerns, and
put the served line back to the narrow wording from M5. Dengue moved to the
right clause; hernia at 14 months moved back to the wrong one. On the synthetic
policy, the question that broke (`ho-short-procedure`) is one the new rule had
named a noise clause for, through "anaesthesia".

**Reading.** Each wording moved one question in and another out, and the
totals did not move. With ten questions, that is the pattern Concept 36
describes: tuning against a small set mostly redistributes its errors. The
changes are kept because each states something true that the block had been
leaving out, not because the numbers improved; they did not.

### A replay that disagreed with its own run

A replay of the second run's answers made two model calls when it should have
made none, and got a different verdict for `star-hernia-after-wait`. The run's
log explains the first half: that answer ran past the 1,600-token output limit,
was retried with 3,200, and was stored under a key built from the retry's
options (the client's own comments describe this). The replay looked it up
under the 1,600 key, missed, and asked again. The second half is not
explained: the same prompt, at temperature 0 with a fixed seed, ran away the
first time and answered normally the second. The server's prompt store
(Failure 58) restores a saved prompt state instead of recomputing it, which
could change the arithmetic slightly. That was not tested. (It was tested in
Step 9, Failure 65.)

---

## Step 8: four clauses the analysis misread

### The problem, restated

Step 7 found that most of the wrong reasons on the Star policy were written by
the pipeline itself: the computed blocks placed above the clauses in the
reasoning prompt (waiting periods, payout reductions, cover windows) repeated
four misreadings made at analysis time, when the model read each clause once
and filled in structured fields. The model reads a clause's
`waiting_period_value`, `cap_percent_of_sum_insured` and so on; Python then does
the arithmetic with them (the split described in the M5 and M7 entries). If the
reading is wrong, the arithmetic is right about the wrong thing, and the block
states it as settled fact.

The four, as the analysis stored them:

| Clause | What it says | What was stored |
|---|---|---|
| 4, Pre Hospitalization | "for a fixed period of 30 days prior to the date of admissible hospitalization" | a 30-day waiting period, type `waiting_period` |
| 3, Cataract Treatment | "25% of Sum Insured or Rs.40,000/-, whichever is lower, per each eye in one Policy Year" | a room-rent cap of 25% of the sum insured **per day** |
| Exclusion 2, specified diseases | a 24-month list (hernia, cataract, gallstones…) and a 36-month list (joint replacement, osteoarthritis) | one number: 36 months |
| c74, a 3,000-character piece of the annexure | the end of the list of items never paid, then the lists of items folded into room and procedure charges | type `sub_limit` |

These were found by replaying, not by running the model. The response cache
(Concept 29) holds every analysis, so a script that runs `analyze()` with the
model call replaced by an error prints exactly what was stored, and makes zero
model calls.

### The approach: correct the reading after the model, not the prompt

Two ways to fix a misreading were available. Change the analysis prompt, so the
model reads better; or check the model's reading against the clause's own
words, in code, before anything uses it.

The prompt route was rejected here for three reasons. The prompt already
separates a waiting period from a cover window, with an example, and the
synthetic policy's pre-hospitalisation clause ("immediately preceding the date
of admission") was read correctly; Star's phrasing ("prior to the date of
admissible hospitalization") was not, and adding phrasings one at a time does
not end. Any prompt change also changes the cache key of every clause on every
policy, so all of them are re-read, and the clause-type score on the synthetic
policy (macro-F1 0.973; 1.000 at the end of M14, see Failure 69) would have to be re-earned. And a reworded prompt moves
readings that were right as well as the ones that were wrong, which is the
opposite of a controlled change.

The code route is the pattern this project already uses twice: an extracted
exception is kept only if the clause states it with an exception word before it
(Concept 32, `verify_exception`), and an age filed as "at policy start" is moved
when the question never mentions the policy starting (`correct_misfiled_age`).
The corrections run in `_analyze_batch`, after the cache:

```python
# api/app/pipeline/analyze.py, _analyze_batch()
    text_by_id = {str(seg.order_idx): seg.text for seg in batch}
    for key, analysis in analyses.items():
        kept = [e for e in analysis.exceptions if verify_exception(e, text_by_id[key])]
        ...
        analysis.exceptions = kept
        correct_stay_period(analysis, text_by_id[key])
        correct_daily_cap(analysis, text_by_id[key])
    return analyses
```

Because the model call and its stored answer are untouched, every cached answer
stays valid, and each correction could be checked on all four policies with no
model at all.

## Concept 42: a correction needs positive evidence

A check that rewrites what a model extracted can itself be wrong, and the two
ways it can be wrong do not cost the same.

Take the waiting-period check. If it wrongly *keeps* a period that is really a
hospital window, a question is told a bar exists that does not: the answer
leans towards "not covered". If it wrongly *drops* a real waiting period, a
question is told nothing bars a claim that is barred: someone is told "covered"
and is refused later. This project treats the second as the worse error, the
same judgement that makes `insufficient_information` a first-class verdict.

So a correction is written to act only on **positive evidence for the other
reading**, never on the mere absence of evidence for this one. "The clause does
not mention the policy's start" is absence. "The clause counts its period from a
hospital stay, and does not mention the policy's start" is presence plus
absence. Only the second moves anything. A clause that mentions both is left
exactly as the model read it.

The general rule: before writing a check that overrides a model, ask which
direction its mistakes fall in, and make it silent whenever the evidence is
mixed.

### Failure 61: a hospital window read as a waiting period

**Setup.** Clause 4 of the Star policy pays pre-hospitalisation expenses "for a
fixed period of 30 days prior to the date of admissible hospitalization". A
waiting period counts from the day the policy began; this window counts back
from one hospital stay. The analysis schema has separate fields for the two
(`waiting_period_value` and `cover_window_value`, added in an earlier milestone
for exactly this confusion on the synthetic policy).

**What happened.** The model put 30 days in the waiting-period field and typed
the clause `waiting_period`. For a dengue fever 20 days into the policy, the
waiting-period block then led with:

```
- clause 4: requires 1 month, policy held 20 days -> this waiting period still
  applies and blocks treatment covered by THIS clause
```

above the real 30-day exclusion (exclusion 3, `3#2`). The model cites the first
bar it reads (Step 7).

**Why.** The prompt's example of a before-admission window uses the word
"preceding"; Star says "prior to". The model generalised the example's meaning
for one phrasing and not the other.

**The fix.** Measured first: every real waiting period on all four policies
(Star's three exclusions, HDFC's and Niva's, the synthetic 3.1–3.4) names the
policy's start in one of four ways. Clause 4 names none of them and names a
hospital stay instead.

```python
# api/app/pipeline/analyze.py
_POLICY_START = re.compile(
    r"waiting[\s-]+period|inception|commencement|continuous(?:ly)?\s+cover",
    re.IGNORECASE,
)
_BEFORE_STAY = re.compile(
    r"\b(?:prior\s+to|preceding|before)\s+(?:the\s+)?(?:date\s+of\s+)?(?:\w+\s+)?"
    r"(?:admission|hospitali[sz]ation)\b",
    re.IGNORECASE,
)
_AFTER_STAY = re.compile(
    r"\b(?:after|following|from)\s+(?:the\s+)?(?:date\s+of\s+)?discharge\b",
    re.IGNORECASE,
)

def correct_stay_period(analysis, text):
    if analysis.waiting_period_days is None or _POLICY_START.search(text):
        return
    before, after = _BEFORE_STAY.search(text), _AFTER_STAY.search(text)
    if not (before or after):
        return
    if analysis.cover_window_value is None and bool(before) != bool(after):
        analysis.cover_window_value = analysis.waiting_period_value
        analysis.cover_window_unit = analysis.waiting_period_unit
        analysis.cover_window_anchor = "before_admission" if before else "after_discharge"
    analysis.waiting_period_value = None
    analysis.waiting_period_unit = None
    if analysis.clause_type == ClauseType.WAITING_PERIOD:
        analysis.clause_type = ClauseType.COVERAGE.value
```

Three details. "waiting period" itself counts as naming the policy's start:
HDFC's benefit table says "post waiting period of 2 years" with no other
anchor, and dropping that would be the expensive error from Concept 42. A
clause naming both sides of a stay loses the misread period but gains no
window, because the side would be a guess. And the type is corrected too: the
reasoning prompt prints each clause's type in its header (`[waiting_period]`),
and a clause granting expenses within a window is coverage, as the synthetic
policy's own pre-hospitalisation clause is labelled.

Replayed from the cache: clause 4 is now `coverage` with a 30-day
before-admission window, and no waiting period.

### Failure 62: a per-eye cap read as a per-day room cap, on two policies

**Setup.** The field `cap_percent_of_sum_insured` means one thing: a per-day
accommodation cap as a percentage of the sum insured ("room rent … limited to
one percent of the Sum Insured per day"). The reductions module
(`app/pipeline/reduction.py`) multiplies it by the sum insured and compares the
result with the person's nightly room rate.

**What happened.** Star's cataract clause, "25% of Sum Insured or Rs.40,000/-,
whichever is lower, per each eye", was stored with
`cap_percent_of_sum_insured = 25`. Every question that gave a room rate was told:

```
- WITHIN THE LIMIT: clause 3: 25% of 3 lakh is 75,000 rupees per day; room rent
  of 8,000 rupees per day does NOT exceed that limit, so this cap costs nothing here
```

and the cataract question was told that "room rent are capped at 25% of the sum
insured per day". The same replay showed **the synthetic policy had the same
fault, unnoticed**: its modern-treatment limit, "fifty percent of the Sum
Insured per Policy Year", was stored as a 50% daily room cap. No synthetic
question had exposed it, because the line it produced was always "within the
limit".

**Why.** A number that is "a percentage of the sum insured" fits the field's
name; "per day" is in the field's description, not its name, and the model
matched on the name.

**The fix.** A daily cap is kept only where the clause says the rate is daily:

```python
# api/app/pipeline/analyze.py
_PER_DAY = re.compile(r"per\s+day|per\s+diem|/\s*day\b|\bdaily\b", re.IGNORECASE)

def correct_daily_cap(analysis, text):
    if _PER_DAY.search(text) or all(getattr(analysis, f) is None for f in _DAILY_CAPS):
        return
    for name in _DAILY_CAPS:
        setattr(analysis, name, None)
```

A cap that is not per day is still a cap, and the reductions module already
has a place for one: a `sub_limit` clause with no numbers is listed for the
model to read against the treatment (status JUDGEMENT), and named when its text
uses a word of the treatment. After the fix, the cataract question's block
says `NAMES THIS TREATMENT: clause 3 mentions "cataract"`, which is the clause
its answer key requires. Star's clause 1 and its benefits table, and the
synthetic 5.1, all say "per day" and keep their caps.

The lesson is the one M15 opened with, from the other side: a second document
does not only expose new faults, it exposes old ones the first document was
too kind to show.

### Failure 63: two waiting periods, one number

**Setup.** IRDAI's standard specified-disease exclusion (Code Excl 02) is
written with a blank for the period. Star fills it with "24/36 months" and two
lists: "24 Months waiting period" over twenty conditions including hernia,
cataract and gallstones, and "36 Months waiting period" over joint replacement
and age-related osteoarthritis. HDFC's wording puts its 36-month pre-existing
disease period and its 24-month list in one numbered block, which the
segmenter keeps as one clause.

**What happened.** The schema holds one `waiting_period_value`. Star's clause
was stored as 36 months. For a hernia 14 months in, the answer happened to be
right (short of both). For a hernia 30 months in, the block would have said the
bar "still applies" when the hernia list lifted six months earlier: a refusal
of a payable claim, and no question in the answer key was in that window to
show it. For the Dubai question, 24 months in, the block told a pneumonia
question that exclusion 2 "still applies and blocks treatment".

**Options considered.** Re-ask the model for a list of periods, each with the
conditions it covers: a new schema, so every clause on every policy re-read and
the clause-type score re-checked. Or keep what the model already read: its
`time_windows` field for this clause was `["24 months", "36 months"]`. The
second was chosen; it costs no model time.

**The fix, in two parts.** The analysis now carries every period the clause
sets:

```python
# api/app/pipeline/analyze.py, ClauseAnalysis
    @property
    def waiting_periods_days(self) -> list[int]:
        first = self.waiting_period_days
        if first is None:
            return []
        periods = {first}
        for window in self.time_windows:
            if m := _BARE_DURATION.fullmatch(window.strip()):   # "24 months"
                periods.add(int(m[1]) * _DAYS_PER[m[2].lower()])
        return sorted(p for p in periods if p > 0)
```

Only a `time_windows` entry that is nothing but a duration counts, and only on
a clause that has a waiting period at all: "within 30 days of discharge" is a
different kind of period, and reading it as a bar is Failure 61 again. The list
replaces the single number everywhere it travels: `ClauseAnalysis` in the
database (`waiting_periods_json`), the API's scenario router, the
`ShortlistClause` the scenario step reads, and the eval's copy of that path.

Then the comparison gains a fourth result:

```python
# api/app/pipeline/waiting.py, evaluate()
        if days_held is None:
            status = WaitingStatus.UNKNOWN
        elif days_held >= required[-1]:
            status = WaitingStatus.SERVED
        elif days_held < required[0]:
            status = WaitingStatus.NOT_SERVED
        else:
            status = WaitingStatus.PARTLY_SERVED
```

Past both periods, or short of both, the answer does not depend on which list a
treatment is on, and nothing changes. Between them it does, and deciding which
list names this treatment is reading, which is the model's job. So the line
gives both results and hands over the reading:

```
- clause 2#2: requires 24 months or 36 months (different periods for different
  treatments). Policy held 2 years, so the 24 months period is served and the
  36 months period is NOT. Which period this clause sets for this treatment
  decides whether it still blocks the claim: read the clause
```

A partly served period is listed with the periods that block, so the block
never also says "No waiting period blocks this claim" beside it; and it is not
in the set `find_contradictions` treats as cleared, so citing it as a refusal is
not flagged as a contradiction.

### Failure 64: the annexure lists, run together

**Setup.** IRDAI's standard annexure lists items a policy does not pay for:
List I, items never covered; Lists II–IV, items folded into room, procedure
and treatment charges. Star's copy is headed `LIST I - Items for which coverage
is not available in the policy`, in bold, smaller than body text.

**What happened.** No rule recognised the list headings, so all four lists and
the ombudsman addresses after them ran on as one clause under the previous
section, "TABLE OF BENEFITS", and the size cap cut them every 3,000 characters.
The piece `c74` began mid-list ("AMBULANCE EQUIPMENT 51 ABDOMINAL BINDER …"),
held the end of List I and the start of Lists II–IV, and was read as a
`sub_limit`. Because the reductions module looks up the treatment's words in
every sub-limit, the caesarean question was told:

```
- NAMES THIS TREATMENT: clause c74 mentions "delivery", which is in what the
  person described.
```

The word was List I's "DELIVERY KIT".

**The fix.** "List I" joins the labelled numbering forms the segmenter already
trusts on their own ("Clause 4.2", "Section 3"):

```python
# api/app/pipeline/segment.py
_RE_LABELLED = re.compile(r"^(?:Clause|Section|Article|Part|List)\s+[\dIVXLivxl]+\b", re.I)
```

The closing `\b` is new, and it is what keeps prose like "List Items continue"
from reading as "List I". The matched number has its whitespace collapsed
before it becomes an id, because Niva sets it as "List  I", with two spaces,
and the model cites that string.

Compared segment by segment with the previous version on all four policies:
the synthetic policy and HDFC are identical; Star's first 73 clauses are
identical, and the tail is now the benefits table, then `LIST I`, `LIST II`,
`LIST III`, and `LIST IV` (with the ombudsman addresses, cut into three pieces).
Seven model calls, 48 seconds, analysed only those. List I was read as an
`exclusion`, Lists II and III as `sub_limit`, List IV as `procedural` (its
piece is mostly addresses). Niva's lists split too, but its text arrives with
List II's rows under the List I heading, a reading-order problem with tables
that this rule does not fix; before, all of it was one block.

**A wrong turn, and a wrong claim.** The first version of this fix also taught
the segmenter to read letter-spaced headings: Star sets its annexure heading as
`A N N E X U R E  -  A`, which no word rule can match, at 1.18 times body size,
just under the 1.20 section threshold. The comparison showed what that did:
Star sets six other headings the same way (`S T A N D A R D  E X C L U S I O N S`
and similar), so most of the policy's clauses moved to new section names.
Section names are in the analysis prompt (`render_clause_batch`), so about 75
clauses would have been re-read, and their readings could have moved for
reasons unrelated to this fix. It was reverted; the list rule does the work on
its own. While reading that comparison, the section label "STAR HEALTH AND
ALLIED INSURANCE COMPANY LIMITED" on six definition clauses was first put down
to the letter-spacing rule. Printing the section paths without it showed the
label was already there: the company name is set large and bold, and passes
the section size test. That is a separate, older fault, still open.

### What the four fixes changed in the prompts

Replayed for all ten Star questions, with no model calls:

| Question | Before | After |
|---|---|---|
| dengue, 20 days | clause 4 listed first as a blocking waiting period | clause 4 gone |
| cataract cost | "room rent are capped at 25% of the sum insured per day" | clause 3 named for "cataract" |
| room over cap | "WITHIN THE LIMIT: clause 3: 25% of 3 lakh is 75,000 rupees per day" | line gone |
| Dubai, 2 years | exclusion 2 "still applies and blocks" | partly served, with both periods |
| hernia | "requires 36 months" | "requires 24 months or 36 months" |
| caesarean | c74 named for "delivery" | line gone |

One false lookup remains: the gallstones question's "gallbladder removal" now
finds "HAIR REMOVAL CREAM" in List III, where it used to find it in c74.
"removal" is one of the four noise words the one-word rule was measured to
produce (Step 7).

## Step 9: measuring it

### Failure 65: an answer that never ended took the eval with it

**Setup.** The Star questions were run three times each, in two halves of five
to stay under ten minutes each (the laptop's memory guard, Failure 57).

**What happened.** The first question, `star-hernia-too-soon`, never finished
its answer. The client retried with the output ceiling doubled, twice, as it is
built to; every attempt ran to the ceiling inside the `reasoning` string, and
the third raised an error that ended the whole run after six minutes, with no
results for any question:

```
attempt 1: response truncated mid-JSON; retrying with num_predict=3200.
attempt 2: response truncated mid-JSON; retrying with num_predict=6400.
prompt used 22578 of 28672 context tokens, leaving 6094 for an answer capped at num_predict=6400
app.llm.client.LlmError: ... JSONDecodeError: Unterminated string starting at: line 3 column 16 (char 45)
```

The same prompt, sent again by a probe script with the cache bypassed, came
back finished and correct, three times: `not_covered`, citing exclusion 2.

**Why.** The server's log records, for every request, how much of the prompt
it reused. The eval's first attempt restored the first 1,866 tokens (the shared
system prompt) from the server's prompt store (Failure 58) and computed the
other 20,700. The retries reused that state. The probe's first request found no
stored prompt to restore and computed all 22,578 tokens from the start; its
later requests reused *that* state.

```
eval,  attempt 1:  found better prompt with f_keep = 0.717   -> generated 1,599 tokens, cut off
eval,  attempt 2:  n_past was set to 22577                   -> generated 3,199 tokens, cut off
probe, request 1:  lcp = 4 on every stored prompt            -> 177 tokens, finished
probe, request 2:  n_past was set to 22577                   -> 283 tokens, finished
```

## Concept 43: the same prompt, computed two ways

Concepts 26 and 27 established that temperature 0 is not reproducibility: the
GPU sums floating-point numbers in an order that depends on the shape of the
work, and the last bits of a result can differ. There the shape changed because
two requests were batched together.

A long prompt is not computed in one piece. The server processes it in chunks
of a fixed size, and each chunk's arithmetic depends on where the chunk starts.
When the first 1,866 tokens are restored from memory, the remaining tokens are
chunked from position 1,866; when nothing is restored, they are chunked from
position 4. The notes the model keeps for each token (the KV cache, Concept 41)
therefore differ in their last bits, and so do the probabilities of the next
word. At a near-tie, the tie breaks the other way, and from there the two
answers are different texts: one finishes in 177 tokens, the other repeats
itself until it is cut off.

What decides how much is restored is **which prompts the server happened to
store before**, which is the history of earlier requests. So on this server,
with its prompt store on, a fresh answer to a long prompt depends on what was
asked before it. Only a replay from this project's own cache is exactly
repeatable. A sample of three fresh answers measures what the system usually
says under the history it had, which is still what the majority-of-three
method (Concept 34) was designed for, but "run it again and it will say the
same" is not a property this setup has.

The store is not the cause of the non-determinism; it moves where the chunks
start, and any reuse of a partly matching prompt does the same. Turning it off
would make every Star question recompute 22,600 tokens and still leave the
server's reuse of its own last prompt.

### The fix: a length the grammar enforces, and a run that survives

A runaway is a loop, and a larger ceiling only gives the loop more room. This
project's answer to a model producing something it must not is to make it
unrepresentable (the enum on citation ids is the first example). JSON Schema
has `maxLength` for strings, and it was checked before being relied on: a field
capped at 60 characters, given a prompt asking for a 300-word story, came back
at exactly 60 characters inside valid JSON. The grammar closes the string when
the cap is reached.

```python
# api/app/pipeline/scenario.py
MAX_REASONING_CHARS = 1_500
MAX_QUOTE_CHARS = 1_200
...
            "reasoning": {"type": "string", "maxLength": MAX_REASONING_CHARS},
...
                        "quote": {"type": "string", "maxLength": MAX_QUOTE_CHARS},
```

The sizes were read from the 229 answers in the cache, not chosen: the longest
reasoning was 974 characters and the longest quote 832 (Concept 40's rule, that
a limit is a measurement). A schema is part of the prompt (Concept 31), so this
changes the cache key of every reasoning answer, and `PROMPT_VERSION` became
`v41-two-period-waits-capped-answers`.

And the eval no longer dies with one answer: a case whose answer cannot be
produced is recorded as a row like any other, wrong and citing nothing
(`_unanswered` in `evals/run_scenario_eval.py`), and the run goes on.

### Results

Three samples per question, every question unanimous, no answer lost:

| | Before Step 8 | After |
|---|---|---|
| Star right verdicts | 7/10 | 7/10 |
| Star citing the deciding clause | 2/10 | 2/10 |
| Quotes that failed verification | 0 | 2 per sample, all flagged (detection 1.000) |

The totals did not move, and four questions did:

| Question | Before | After |
|---|---|---|
| hernia, 14 months | right verdict, wrong clause | right verdict, **right clause** (exclusion 2) |
| hernia, 3 years | covered (wrong) | **conditional** (right), citing only the co-payment |
| dengue, 20 days | right verdict, right clause | right verdict, cites the pre-existing diseases exclusion |
| Dubai | not covered (right) | **conditional** (wrong), citing only the co-payment |

Read the stored answers and the three wrong verdicts (caesarean, Dubai,
gallstones) have one reason between them: "The claim is conditional because a
5% co-payment applies." None mentions an exclusion or a waiting period. That is
the next fix, not this one. As in Step 7, this is Concept 36 in action: ten
questions, two moving each way. The fixes stay because each removes something
false the pipeline was telling the model.

**What the cap revealed.** Two answers that finished did so only because of the
cap: the hernia and teeth-whitening reasoning both repeat a sentence until the
1,500th character, and stop mid-word. The verdict and citation survive, but a
person would read that text. The loop was always there; before, it cost the
whole answer.

**Also seen, not fixed.** The room-over-cap answer cites "1.1" for the room cap
the block attributes to clause 1 (two nearly identical ids). The cataract answer
computes "95% of 70,000" itself instead of the 40,000 limit, because a cap of
"25% or Rs.40,000, whichever is lower, per eye" has no arithmetic in the
reductions module. The fact extractor stored the cataract question's sum
insured as 500000 lakh.

### Check it yourself

- `api/.venv/Scripts/python.exe -m pytest api/tests/test_analyze.py -v`: the
  corrections, each on a clause shaped like the one it was measured on.
- `api/.venv/Scripts/python.exe -m pytest api/tests/test_waiting.py -k two_periods`: the partly served
  status. Before running it, predict: a clause with periods of 24 and 36 months,
  a policy held exactly 24 months. Served, not served, or partly served? (The
  comparison is `>=`, so the 24-month period counts as served: partly.)
- In the server log (`%LOCALAPPDATA%\Ollama\server.log`), find a
  `found better prompt with f_keep` line and the `stop processing: n_tokens`
  line after it. The difference between that count and the prompt's length is
  how many tokens the answer took.

## Step 10: the co-payment line

### The problem, restated

Star's policy takes a 5% co-payment from each and every claim. The reductions
block (`app/pipeline/reduction.py`) printed it on every Star question, the same
each time:

```
- No waiting period blocks this claim. Some other clause decides it.

WHAT REDUCES THE PAYOUT (arithmetic, not opinion).
...
- APPLIES: clause 9: a 5% co-payment applies to the admissible claim amount
```

After Step 8, the three wrong verdicts (caesarean, Dubai, gallstones) had one
reason between them: "conditional because a 5% co-payment applies". Printing
the prompt showed the chain: the waiting-period block ends "Some other clause
decides it", and the next finding on the page is the only APPLIES line in the
prompt.

The age-based co-payment had met this before (Failure 38): "the 20% co-payment
DOES apply, so this claim is paid at 80%" turned a refused nose job into
`conditional`, and the fix made the line say only what was computed, "IF this
claim is payable at all, it is paid at 80%". The branch for a co-payment with
no condition never got that qualifier, because the synthetic policy has no
such co-payment.

### The change

```python
# api/app/pipeline/reduction.py, _copay_check(), no age condition
f"a {percent}% co-payment is taken from EVERY claim this policy pays, "
f"so it says nothing about whether THIS claim is paid - settle that "
f"from the exclusions and waiting periods first. IF this claim is "
f"payable at all, it is paid at {100 - percent}% of the admissible amount",
```

The synthetic policy's prompts do not change: its only co-payment has an age
condition. `PROMPT_VERSION` became `v42-universal-copay-if-payable`.

### What happened

Three samples per question, all unanimous: 7/10 right verdicts, as before.
The co-payment stopped being the only stated reason, and what replaced it was
worse reading, not better:

- **caesarean**: now `covered`, "under clause 18 of the policy, which states
  that the Company shall indemnify medical expenses incurred for inpatient
  care treatment under Ayurveda, Yoga …". That is coverage clause 2 (AYUSH).
  Clause 18 is the maternity exclusion the answer key requires. The quote
  failed verification, and the effect given was `permits`.
- **Dubai**: the reasoning says exclusion 2's "24-month waiting period still
  applies" after two years (the block said the 24-month period is served) and
  "therefore, the claim is not covered at this time", and the verdict is
  `conditional`.
- **weekend trek**: cites "clause 3.3", a 30-day exclusion, as delaying a
  claim four years in; the quote failed verification.
- **gallstones**: repeats itself to the reasoning cap.

**A measurement caveat this exposed.** Citation recall counted the caesarean
answer as citing its deciding clause (3/10, up from 2/10), because it checks
the cited id only. The quote attached to that id was another clause's text and
failed verification. A citation whose quote is unverified is not evidence that
the model read the right clause.

**And a regression that was not one.** Dubai was right before Step 8, and its
answers then never cited exclusion 22 (treatment outside India). The reason
given was exclusion 2 "still applies and blocks" at 24 months, the false 36-month
reading Step 8 removed. The verdict was right for a reason that was false. In
no run on record has the model found exclusion 22, and in none before this one
had it cited exclusion 18.

### Reading

The co-payment line was the excuse the model reached for, not the cause. With
22,600 tokens and 80 clauses in front of it, this 7B model attaches one
clause's text to another's id, contradicts its own reasoning in its verdict,
and loops. The line is kept because it states something true the old wording
left out (the Failure 38 standard). The next step was planned before M16's fixes
began, for exactly this case: if Star still cites the wrong clauses, reason in
two passes.

## Step 11: two passes, the first one tested alone

### The idea, and the risk it carries

Two-pass reasoning splits the scenario step: pass 1 shows the model the
question and every clause and asks only which clauses could decide it; pass 2
answers from those clauses, plus the ones code already flags, in a prompt a
quarter of the size. The risk is the reason this project has no retrieval
(Concept 38): if pass 1 leaves out the deciding clause, pass 2 cannot cite it.
Unlike retrieval, the model still reads every clause while choosing.

So pass 1 was built and measured on its own first, as a probe
(a throwaway script, not part of the app or this repository): pass 2 would be built only if the picks
held the deciding clause for most questions.

The pick is enum-locked to real ids, like every citation, and at most eight
long:

```python
"clause_ids": {
    "type": "array", "minItems": 1, "maxItems": 8,
    "items": {"type": "string", "enum": ids},
},
```

Its instructions list the kinds of clause that decide a claim and say that the
deciding clause "often uses different words from the question, and often
comes late". They contain no example, because the obvious examples (treatment
abroad and "the geographical limits of India") are eval questions, and an
instruction written from the answer key is tuning on it (Concept 36).

Measured against each question's `must_cite`, alone and together with the
clauses the computed blocks and lookups already name ("flagged").

### Failure 66: a pick that did not read

| | Synthetic policy | Star |
|---|---|---|
| clause text in the prompt | about 3,300 tokens | about 19,500 tokens |
| pick holds every must_cite | 28 of 32 | 2, 1 and 1 of 10 (three samples) |
| pick + flagged | 28 of 32 | 7 of 10, all from the flags |

On Star the picks do not look like choices. For the caesarean question, three
times: `c0, c1, c2, c3, c4, c5, c7, c8`, the first definition pieces in
document order. Most others are short runs of low numbers (`1, 2, 9, 19, 21,
22, 23`). No pick in thirty contained an id with a `#` suffix, and every
exclusion and condition that shares a number with a coverage clause has one.
For cosmetic teeth whitening the model picked `8` (Star's cumulative bonus)
three times, not `8#2` (the cosmetic exclusion).

On the synthetic policy the same instructions picked well, with a median of
four clauses, and one of its four misses was the same kind of run: `4.1, 4.2,
… 4.8` for a question decided by 6.4.

**Reading.** Picking works where the text is short and fails where it is long:
the same loss of track Step 10 showed in the answers, now without any answer
to hide behind. Two things are mixed in the Star result and not separated:
the length, and the `#` ids, where `8` is a complete id and also the start of
`8#2`, so the grammar lets the model stop after the `8`. Pass 2 was not built
on this pick.

### The pick, in groups

The synthetic result says the task is not beyond the model: at 3,300 tokens it
picked well. So the policy is given to it at that length. The clauses, in
document order, are cut into groups of about 3,000 tokens, and each group is
asked on its own, at most three from each:

```python
# api/app/pipeline/scenario.py
PICK_GROUP_TOKENS = 3_000
MAX_PICKS_PER_GROUP = 3
```

Two details are what make it work at all. The schema allows an empty answer
(`"minItems": 0`), because most groups of a policy have nothing to do with any
one question, and a schema that demands a pick turns every group into a false
positive. And each group's enum holds only that group's ids, so `8` and `8#2`
are rarely offered together.

Measured on Star, two samples:

| | whole policy | in groups |
|---|---|---|
| pick holds every must_cite | 1-2 of 10 | 7 of 10 |
| pick + flagged | 7 of 10 | 9 of 10 |
| calls per question | 1 | 8 (about 15 seconds) |
| clause text passed on | all 19,500 tokens | at most 6,800 |

The picks now find what no run had found: exclusion 22 (treatment outside
India), exclusion 18 (maternity), exclusion 8#2 (cosmetic). The one the pick
still misses is the hazardous-sports exclusion for an amateur's weekend trek.

## Concept 44: fitting in the window is not being read

The premise for having no retrieval was a measurement: every decision-relevant
clause of a policy fits in the context window (Concept 38), so nothing has to
be dropped, and dropping is what loses the clause that decides a case.

That premise is about the window. It says nothing about attention. M16
measured the other half on a real wording: with 80 clauses and 22,600 tokens in
front of it, this 7B model cited the deciding clause in 2 questions of 10,
attached one clause's text to another's id, and repeated itself until it was
cut off. The same model, given the same clauses in groups of a tenth the size,
found the deciding clause in 7 of 10.

So a context window has two limits, and only one of them is printed in the
model card. The second has to be measured on the shape of your own prompt, and
the measurement is cheap: ask the model to do a small, checkable part of the
task at different prompt lengths - here, name the clauses that matter - and
compare against the answer key you already have.

What this does NOT license is retrieval. Nothing here is ranked, embedded or
searched, and no clause is dropped before a model has read it: every clause is
read, in a group, by the same model that answers. The difference between that
and retrieval is who decides - a reader, or a similarity score - and it is the
difference between a clause left out because it uses unusual words and one left
out because the reader judged it irrelevant.

## Step 12: answering from the picked clauses

### What pass 2 is shown

The clauses the model picked, plus every clause code already names - and the
second half is not optional:

```python
# api/app/pipeline/scenario.py
def named_by_code(computed, scenario, clauses, facts) -> set[str]:
    named = {c.clause_id for c in computed.waiting}
    named |= {c.clause_id for c in computed.reductions
              if c.status is not reduction.ReductionStatus.NOT_RAISED}
    named |= {c.clause_id for c in computed.windows
              if c.status is not window.WindowStatus.NOT_RAISED}
    named |= set(shared_words(scenario, clauses, facts))
    named |= set(computed.absent)
    return named
```

The computed blocks say "clause 2#2: requires 24 months or 36 months…". If a
pick could drop 2#2, that line would point at a clause the model cannot read or
cite, and the bar it describes would go unanswered. The waiting periods,
reductions and windows are still computed over **every** clause, before any
picking: what the model picks decides what it READS, never what is checked.

A policy short enough to read whole is read whole:

```python
PICK_ABOVE_TOKENS = 6_000
```

The synthetic policy is 3,346 tokens, reads well whole, and picks at 28 of 32,
so picking it could only lose clauses. Star is 19,500 and does not. The
threshold sits between the two measured points, nearer the one that reads well.

### What it changed, and what it exposed

Three samples per question, all unanimous:

| | before M16 Step 8 | after Step 10 | with two passes |
|---|---|---|---|
| Star right verdicts | 7/10 | 7/10 | 6/10 |
| Star citing the deciding clause | 2/10 | 2-3/10 | 7/10 |

The answers stopped being about the wrong clauses and started being wrong
about the right ones:

- **hernia, three years**: "The claim is covered… the 5% co-payment applies,
  but this does not refuse the claim… Therefore, the claim is covered, but the
  payment will be conditional on the actual amount being 95%." One sentence
  saying both.
- **caesarean**: finds the maternity exclusion, quotes it correctly, and then:
  "Since your situation falls under the exception, the claim is covered in
  full." The exception is for an ectopic pregnancy.
- **gallstones, no dates**: "If the waiting period has been served, the claim
  will be covered… If it has not, the claim will be refused" - the definition
  of `insufficient_information` - and answers `conditional`.

### Failure 67: a contradiction the check could not see

**Setup.** `find_contradictions` compares an answer with what Python computed,
and one of its rules is that a claim which is paid less than in full is
`conditional`, never `covered`. It fired only when the answer CITED the
reduction as reducing:

```python
        for citation in citations:
            check = applies.get(citation.clause_id)
            if citation.effect == "reduces" and check is not None:
```

**What happened.** On the hernia question the rule did fire, the retry put the
calculation in front of the model, and the model answered `covered` again. On
the caesarean question it did not fire at all: that answer cited only the
maternity exclusion, so no cited clause matched a computed reduction, though
the policy takes 5% from every claim it pays.

**Why.** The contradiction is between the verdict and the ARITHMETIC, not
between the verdict and the citation. A policy with a co-payment on every claim
cannot pay one in full, whatever the answer happens to cite.

**The fix, in two parts.** The check now also reports an applying reduction the
answer never mentioned, in the words of what it does and does not settle:

```
- You answered covered, which means paid in full, but it was calculated that
  clause 9: a 5% co-payment is taken from EVERY claim this policy pays...
  If this claim is payable, it is paid at less than the full amount, which is
  conditional; if some clause refuses it outright, it is not_covered. It
  cannot be covered.
```

And when the retry answers `covered` anyway, the label is corrected rather than
published:

```python
    if verdict == Verdict.COVERED and any(
        c.status is reduction.ReductionStatus.APPLIES for c in computed.reductions
    ):
        log.warning("covered, though a reduction applies; correcting to conditional")
        verdict = Verdict.CONDITIONAL
```

This is a departure from `find_contradictions`' own rule that code never
changes a verdict, and the reason it is allowed here is that nothing about the
claim is being decided: "covered" states that the claim is paid IN FULL, and
the arithmetic - a percentage the policy takes from every claim it pays, or a
room rate over its cap - says it is not. The citations are left alone, because
which clause decides the case is the model's call and not arithmetic. It is the
safe direction, like the downgrade for an answer with no citations: it takes
away a promise of payment in full that no calculation supports.

### Results

| | before M16 Step 8 | now |
|---|---|---|
| Star right verdicts | 7/10 | 7/10 |
| Star citing the deciding clause | 2/10 | 7/10 |
| Synthetic main 40 | 34 | 34 |
| Synthetic held-out 1 / 2 (one sample) | 10/16, 12/13 | 9/16, 11/13 |

The verdict score has not moved all milestone. What has moved is what the
answers are ABOUT: seven questions of ten are now decided by the clause the
answer key names, against two before. For an app whose promise is to show a
person which clause decides their claim, that is the number that was failing.
(Step 13 measures it again the next day, and gets 5 of 10.)

The three questions still wrong are wrong in ways that are now visible and
separate: the caesarean answer applies an exception that does not fit, the
Dubai answer cites the co-payment although the pick found exclusion 22, and
the gallstones answer describes `insufficient_information` and labels it
`conditional`.

On the synthetic sets, one case each fell in the two held-out batches, both
single samples, and neither from the correction above (it never fired there).
`ho-icu-within-cap` is an answer whose own text says the claim is paid in full
and whose label says conditional; `ho-dental-accident` misreads an exclusion
this milestone did not touch.

### Check it yourself

- `api/.venv/Scripts/python.exe -m pytest api/tests/test_scenario.py -k "pick or picked or covered_"`:
  the grouping, the empty pick, what code always shows, and the correction.
- Predict before reading the code: a policy of 30 clauses, each about 250
  tokens, is asked one question. How many model calls does the scenario step
  make before it answers? (One per group of about 3,000 tokens - so three - plus
  the facts call and the answer.)

---

## Step 13: the closing measurement

### How it was run

Every eval set, three samples per question: the synthetic main set and both
held-out batches, recorded in the run history (`evals/run-history.json`, the
local file the comparison and stability sections are computed from), and the
ten Star questions, recorded in logs as in every Star run before, because the
history file does not say which policy a case belongs to. The runs went one at
a time, each under ten minutes, with nothing else using the model server.

`evals/run_all.py` went first. Every answer it needed was already in the
response cache (the first samples of the version under test, stored the day
before), so it took under a second and stamped `evals/REPORT.md` with the exact
commit. The main set's other two samples were then asked afresh with
`--resample`, and the held-out batches and Star with `--repeats 3`, whose first
sample replays the same stored answers and whose other two are fresh.

### Results

Majority of three per question. "After Step 5" is the first measurement of
this milestone, taken with the new window and before the switch to a `q8_0`
KV cache:

| | End of M14 | After Step 5 | End of M16 |
|---|---|---|---|
| Clause classification, macro-F1 | 1.000 | 0.973 | 0.973 |
| Synthetic main set (40) | 38/40 | 33/40 | 34/40 |
| Held-out batch 1 (16) | 13/16 | 13/16 | 10/16 |
| Held-out batch 2 (13) | 11/13 | 12/13 | 12/13 |
| Star, right verdicts | — | 8/10 | 7/10 |
| Star, citing the deciding clause | — | 1/10 | 5/10 |
| Detection integrity, every sample | 1.000 | 1.000 | 1.000 |

**Star.** Seven right verdicts, as in Step 12's final measurement, with every
sample agreeing on every verdict. The deciding clause is cited in 5 questions:
the same code, measured the day before in Step 12, cited it in 7. The two that
moved had already split that day, each missing in one sample of three: the
room-rent question cites `1.1` where the answer key requires coverage clause
`1`, and the gallstones question leaves out exclusion 2 (`2#2`). The honest
figure is 5 to 7 of 10, depending on the day (M17 found what the day changed:
the Ollama version), against 1 of 10 after Step 5.

**The synthetic sets.** The main set is 34/40 in every sample, and held-out
batch 2 is 12/13. Held-out batch 1 is 10/16, down from 13/16 at the start of
the milestone. Three questions account for the drop, each wrong in both fresh
samples: scuba diving, a pre-hospitalisation bill from before the window, and a
three-hour procedure the policy does not address. Failure 68 below explains
two of the three. The third is a misreading:

```
ho-scuba-diving  expected not_covered, got covered
  "... clause 4.3 explicitly excludes expenses related to hazardous
  activities, but scuba diving is not listed as a hazardous activity in this
  context."
```

Clause 4.3 names scuba diving. The same question was answered `not_covered` in
all three samples after Step 5. It was not investigated further: batch 1 is a
held-out set, and working through its failures to fix them is how a held-out
set stops being one (Concept 36).

### Failure 68: three samples, two of them the same one

**Setup.** Concept 34 introduced majority of three because this model does not
answer identically twice. One sample is a coin flip; three show which way the
coin usually lands. Every "three samples, all unanimous" in this milestone
depends on the three being different draws.

**What happened.** The two fresh samples of the closing measurement agreed on
all 79 verdicts: 40 main, 16 and 13 held-out, 10 Star. They differed on a few
citations. The stored first sample, generated the day before by the same code,
differed from both on 10 of the 69 synthetic verdicts. Then a script asked three
held-out questions afresh, in a different order from the eval's, and two came
back right that both fresh samples had got wrong:

| Question | Day before | Fresh sample 2 | Fresh sample 3 | Asked in another order |
|---|---|---|---|---|
| `ho-pre-hospitalisation-too-early` | not_covered (right) | conditional | conditional | not_covered (right) |
| `ho-short-procedure` | covered | conditional | conditional | insufficient_information (right) |
| `ho-scuba-diving` | not_covered (right) | covered | covered | covered |

**Why, as first understood.** Concept 43 established that, on this server, a
fresh answer depends on the prompts the server stored before it. The history decides how much of the
prompt is restored rather than recomputed, so it decides where the arithmetic's
chunks start, and so the last bits of every probability. `--repeats` asks the
questions in the same order in every sample. Each question is preceded by the
same questions as last time, so it finds nearly the same stored states and
computes nearly the same numbers. A second sample in the same order is not a
second draw: it is mostly the first draw again. Across days the history
differs, because the server restarted and other requests came first, and that
is where the answers moved.

M10 separated two properties: asking one question twice (repeat stability) and
asking forty questions twice (sequence stability). This is a third: whether a
sample is independent of the one before it at all. It is M10's lesson again:
test the property your conclusion depends on. "Unanimous" was read as
"stable", and what it measured was "reproducible under the same history".

**Measured afterwards, that explanation was mostly wrong** (M17, Failure 70).
Asked in shuffled orders, two samples still agreed on 76 of 79 verdicts, and a
freshly loaded server gave the same answers as a warm one. What changed between
the two days was Ollama itself: it updated from 0.34.1 to 0.34.2 overnight, and
the stored first sample came from the old version.

**What it means for the numbers above.** They are two days, not three
samples. Main set: 34/40 on both. Held-out batch 1: 9/16, then 10/16. Held-out
batch 2: 11/13, then 12/13. Star, deciding clause: 7/10, then 5/10.

**The fix is not in M16.** Each sample should ask the questions in a different
order, so each draws on a different history. That changes the harness and needs
its own measurement, so it opens the next milestone.

### Failure 69: a drop noticed and never explained

**Setup.** M14 closed with the main set at 38/40 and classification at macro-F1
1.000. The first measurement after Step 5 read 33/40 and 0.973: one clause of
the synthetic policy that M14 read as a `sub_limit` was now read as a
`condition`.

**What happened.** The drop was seen: Failure 57 checks that the memory
measurement running alongside did not cause it. Then nothing more was done with
it. Every later table in this milestone starts from 33 or 34, and Step 8 as
first written quoted the classification score as 1.000 (it is corrected
there).

**Why the runs on record cannot explain it.** Steps 1 to 5 changed three things
that each regenerate every answer on the synthetic policy:

- the context window, from 8,192 to 28,672 tokens (`num_ctx` is a decoding
  option, and the options are part of every cache key);
- the analysis schema, which gained two cap fields (a schema is part of the
  prompt, Concept 31);
- the reasoning prompt's example (Step 5).

Any one of them moves every near-tie. With all three in one run, that run
cannot say which one moved the main set. Nor can it say whether the move is a
real loss or the redistribution Concept 36 describes: 38/40 was reached by keep
or revert decisions, each judged on answers computed under the old window.

**What it would take.** Rerun the main set at M14's commit with only the window
changed: one variable per run, which is how the cache experiment of Concept 29
isolates a change. Not done here; recorded as open.

### Check it yourself

- `api/.venv/Scripts/python.exe -c "import json; h = json.load(open('evals/run-history.json', encoding='utf-8')); print([(r['at'], r['verdict_accuracy'], sum(c['fresh'] for c in r['cases'].values())) for r in h])"`:
  the recorded samples, with how many of each were fresh. The history is a
  local file, not in the repository; this works after any eval run.
- Predict before running: if `--repeats 3` shuffled the questions into a new
  order for each sample, would you expect more questions or fewer to be
  unanimous? (Fewer, if Failure 68's explanation is right; that prediction is
  what the fix will be measured against.)

---

## Closing out M16

### How it connects to what was already there

M15 measured a system built on one synthetic document and found it unsound on
real ones at every stage. M16 fixed the stages in the order they feed each
other:

- **Reading the page** (Steps 1 to 3). Ingest keeps the PDF's own reading order
  and removes page furniture recognised by repetition. The segmenter reads a
  document's own conventions, such as bold clause numbers, against its own
  baseline (Concept 39). Everything downstream reads what these produce, which
  is why they came first. On Star, coverage 1 to 9, exclusions 1 to 23 and
  conditions 1 to 28 are one clause each, under the policy's own numbers.
- **Fitting the policy** (Step 4). The window was re-measured rather than
  inherited (Concept 40), so the whole Star policy fits in one prompt, as the
  no-retrieval premise requires (Concept 38).
- **Paying for the window** (Step 6). A rounded KV cache (Concept 41) and a cap
  on the server's prompt store keep the long window inside an 8 GB laptop. The
  prompt store is also why fresh answers depend on history (Concept 43,
  Failure 68).
- **Checking the model's readings** (Step 8). The computed blocks from M7 to
  M13 are only as right as the fields they compute from. Four misreadings were
  corrected in code, against each clause's own words, the pattern Concept 32
  started, with Concept 42's rule for which way a correction may err.
- **Being read, not only fitting** (Steps 11 and 12). Concept 44 found that a
  prompt the window holds can still be one this model cannot keep track of.
  Picking clauses in groups keeps what the no-retrieval premise was for: every
  clause is read by the model that answers, and none is dropped by a
  similarity score.

### What it made possible

An answer key with reasons for a real policy (`must_cite` for all ten Star
questions), and with it a measurement of the app's actual promise on a real
wording: showing a person which clause decides their claim. That went from
1 question of 10 after Step 5 to 5 to 7 of 10. The verdict score moved the
other way, from 8 of 10 to 7, and the two are related: before Step 7 most right
verdicts on Star rested on the wrong clauses, and a right verdict for a wrong
reason is a coin that happened to land well.

### Where this leaves the numbers

On the synthetic policy the main set is 34/40, down from M14's 38/40
(Failure 69, unexplained), and the held-out batches together are 22 of 29 on
the latest day: about 76%, against the 83 to 85% M14 recorded. On the Star
wording, 7 of 10 verdicts are right, and 5 to 7 of 10 cite the deciding clause.
Ten questions on one policy is a probe (Concept 37), not a benchmark. The Star
figures describe this system on this policy, and are not an accuracy to quote
for real policies in general.

Detection integrity was 1.000 in every sample of every set. Every quotation
the model invented was caught and shown as unverified.

**Still open:**

- the samples in one run are not independent (Failure 68), which is the first
  thing the next milestone fixes;
- the drop from 38/40 at the start of the milestone (Failure 69);
- held-out: the scuba answer says the hazardous-sports exclusion does not name
  scuba diving, and it does;
- Star: the caesarean answer applies the maternity exclusion's
  ectopic-pregnancy carve-out to itself; the Dubai answer cites the
  co-payment although the pick found exclusion 22; the gallstones answer
  describes `insufficient_information` and labels it `conditional`; the pick
  misses the hazardous-sports exclusion (`9#2`) for the weekend trek;
- answers that repeat a sentence until the 1,500-character cap cuts them
  mid-word;
- the room-rent answer cites `1.1` for coverage clause 1;
- a cap of "25% or Rs.40,000, whichever is lower, per eye" has no arithmetic;
- the fact extractor stored one sum insured as "500000 lakh";
- citation recall counts a citation whose quotation failed verification;
- Star's company name and letter-spaced headings are read as section labels,
  and "removal" finds "HAIR REMOVAL CREAM" in List III.

---

## Check it yourself

```bash
api/.venv/Scripts/python.exe -m pytest api/tests -m "not llm"     # no model needed
python evals/run_all.py                                           # the report; replays from cache
python evals/run_scenario_eval.py --heldout 2 --repeats 3         # three samples, one batch
python evals/run_scenario_eval.py --policy samples/real/star-arogya-sanjeevani.pdf \
    --cases evals/real/scenarios-star-arogya-sanjeevani.json --repeats 3 --no-report
```

**Question to sit with:** the grouped pick (Step 11) finds the deciding clause
for 7 of 10 Star questions, and pass 2 is also shown every clause the computed
blocks name. Suppose a policy's deciding clause is one the pick misses and
that no computed block names, as the hazardous-sports exclusion is for the
weekend-trek question. What does pass 2 do, and why is that failure harder to see than
the same failure in retrieval?

<details>
<summary>Answer</summary>

Pass 2 cannot cite it: the citation enum holds only the clauses in its prompt,
so the deciding clause is unrepresentable. The model answers from what it was
shown, and nothing in its prompt contradicts that answer.

It is harder to see because it looks like a reading. A retrieval system's miss
has a visible cause: a similarity score below a cut-off, which can be logged
and inspected. Here a model read the clause and judged it irrelevant, and that
judgement leaves no trace except the list of picks. The mitigation this system
has is structural: every clause that code can connect to the question is added
back whatever the pick says. The measurement that watches the rest is the one
M16 built: `must_cite` in the answer key, checked against the picks and against
the citations.
</details>

---

# M17 — What a sample is

## The starting position

Since M12, every headline number in this project has been a **majority of
three samples** (Concept 34): the model does not answer identically twice, even
at temperature 0 (Concepts 26 and 27), so one run is one coin flip, and three
show which way the coin usually lands.

M16 closed on a finding about that method (Failure 68). In its closing
measurement, the two fresh samples of every question set agreed on all 79
verdicts, while the first sample, an answer stored the day before and replayed
from the response cache, differed from them on 10 of 69 synthetic verdicts. If
two of three samples are copies, a majority of three is one sample counted
twice.

M16's explanation was the order of the questions. On this server a fresh answer
depends on the prompts the server stored before it (Concept 43), and the eval
asked the questions in the same order in every sample. M17 tests that
explanation before building on it, and fixes what the test finds.

---

## Step 1: a different order for every sample

The change is small. Each sample after the first asks its questions in its own
order, shuffled with the sample's number as the seed, so a run can be repeated
in exactly the same orders:

```python
# evals/run_scenario_eval.py
def asking_order(cases: list[dict], sample: int) -> list[dict]:
    if sample <= 1:
        return list(cases)
    shuffled = list(cases)
    random.Random(sample).shuffle(shuffled)
    return shuffled
```

Sample 1 keeps the file's order, so a single run is asked exactly as it always
was. The rows are sorted back into the file's order before anything is scored
or printed, so every sample's table reads the same way. And `--sample N` runs
one sample in the order `--repeats` would give it, so a set too long for one
sitting can be run a sample at a time and still be recorded as whole runs.

The test drives the real `run_repeats` with the model replaced, records the
order the questions arrive in, and asserts that three samples use three
different orders of the same questions and report in the file's order. It was
run against the unchanged code first and failed there: all three samples asked
in the same order.

### What it changed

The same questions, on the same day and the same server, asked first in one
order and then in shuffled orders:

| | One order | Shuffled orders |
|---|---|---|
| Synthetic main set, majority of three | 34/40 | 32/40 |
| Held-out batch 1 | 10/16 | 11/16 |
| Held-out batch 2 | 12/13 | 12/13 |
| Star, verdicts / deciding clause cited | 7/10 / 5/10 | 7/10 / 6/10 |
| Verdicts that differ between the two fresh samples | 0 of 79 | 3 of 79 |

Order matters, a little: 3 verdicts in 79 moved when it changed. The difference
between the two days had been 10 in 69. Whatever changed between the days, it
was mostly not the order.

---

## Step 2: the reload test that had already run

The next candidate was the server's state. A server that has just loaded the
model has an empty prompt store; one that has answered a hundred questions has
a full one. If that decided the answers, reloading the model before each
sample would make every sample start the same way, and each day's first run
would differ from a warm server's.

The test turned out to have run twice already. Ollama's server log records
every time it starts a model runner, and the model unloads itself after five
idle minutes:

```
time=2026-09-19T17:43:57 ... msg="starting llama-server"    <- one-order sample 2 begins
time=2026-09-19T18:31:30 ... msg="starting llama-server"    <- shuffled sample 2 begins
```

Both second samples began on a freshly loaded server, and both third samples
ran on the warm one. In the one-order run, the fresh and the warm sample agreed
on all 40 main verdicts. A fresh server does not change the answers.

### Failure 70: an explanation that fitted, and the cause nobody had looked at

**Setup.** Failure 68 put the difference between two days of samples down to
the history of requests the server had seen, because that was the mechanism
this project already knew (Concept 43), and it fitted the evidence.

**What happened.** Tested directly, the two parts of that history moved 3
verdicts in 79 (the order) and none (a fresh server). The ten verdicts that
differed between the days had another cause.

**Why.** Ollama's logs name the version at every start:

| Log | Covers | Ollama |
|---|---|---|
| `server-3.log` | 17 September 15:53 to 18 September 00:24 | 0.34.1 |
| `server-2.log` | 18 September 18:10 to 22:49 | 0.34.1 |
| `server-1.log` | 19 September, 10:46 to 10:49 | 0.34.1 |
| `server.log` | 19 September, from 10:49 | **0.34.2** |

and `upgrade.log` records the installer running at 10:49:02. Ollama had started
with the laptop at 10:46, found an update, installed it and restarted itself. It had done the same two days earlier, from
0.34.0 to 0.34.1, in the middle of M16's measurements. The stored first sample
was answered by 0.34.1, the two fresh ones by 0.34.2.

Ollama ships its own copy of the inference engine (llama.cpp), and a new
version can compute the same prompt through different code. Concept 27 showed
what that does: floating-point addition gives slightly different results in a
different order, a slightly different probability breaks a near-tie the other
way, and from there the answer is a different text.

**What is and is not shown.** Proving it would mean reinstalling 0.34.1 and
asking again. What is shown: the update is the only difference found between
the two days, and the two differences that were tested moved 3 verdicts and 0.

**The lesson.** An explanation that fits is a hypothesis until it is tested,
and the one this project already knew was the one it reached for. The log that
held the answer was read in M16 for other reasons, in Failures 57 to 59, and
it names the version every time the server starts.

## Concept 45: the runtime is part of the model

A language model's answer is usually described as a function of three things:
the weights, the prompt, and the decoding settings. This project versions all
three: the model name, the exact prompt text, and the options are all in the
cache key.

There is a fourth. The weights are numbers; something has to do the arithmetic
with them, and that program has a version too. On a GPU the result depends on
the order the program adds numbers in (Concept 27), so two versions of the same
runtime, given the same weights, prompt and settings, can return different
answers. Here the runtime changed twice in one milestone, silently, because the
application updates itself.

Two consequences. A cache keyed on the first three things will replay an answer
from one runtime as if the current one had given it. And any comparison that
spans a runtime change measures the change and the runtime together, with no
way to separate them afterwards. Concept 41's KV cache precision was the first
instance of this in the project: a setting no request carries, which changes
the answer.

The transferable practice, for any evaluation of a model: record the runtime's
version with every number, hold it fixed for the length of a comparison, and
when it changes, measure the baseline again before comparing anything with it.

---

## Step 3: the version in the cache key

The fix follows Concept 41's: the version becomes part of every stored answer's
key, read from the server itself.

```python
# api/app/llm/client.py, runtime_version(), as first written
    async with httpx.AsyncClient(timeout=5.0) as http:
        resp = await http.get(f"{settings.ollama_url}/api/version")
        resp.raise_for_status()
        return resp.json()["version"]
```

```python
# api/app/llm/client.py, complete_json()
    if use_cache:
        version = await runtime_version()
        key = cache.make_key(model, messages, schema, options, settings.kv_cache_type, version)
```

```python
# api/app/llm/cache.py, make_key()
    if runtime_version is not None:
        fields["runtime_version"] = runtime_version
```

Three decisions in that:

- **Asked on every cached call, not once per process.** The app runs for hours,
  and Ollama updates when it restarts, which can happen while the app is
  running. The reasoning was that one local request costs little beside a
  model call of seconds. That was not measured, and it was wrong (Failure 71).
- **No exception for old entries.** The KV precision was added to the key only
  when it was not f16, because every older entry was known to be f16. No
  version can be assumed for the older entries here, so every one of them is
  unreachable, and the first measurement after the change is a full fresh run.
  So is the first after every Ollama update: that is the cost, measured in
  Step 4.
- **If the server cannot say its version, the call fails** with an error that
  says so, rather than hashing a guess. A request that could not reach the
  server could not have been answered by it either.

The rejected alternative was to record the version in reports and leave the
cache alone. It is cheaper, and it leaves the failure in place: a replayed
answer from the old version, printed beside fresh ones from the new version,
with nothing to tell them apart.

The eval now records the version in every run-history entry and in
`evals/REPORT.md`, and a comparison that spans two versions says so:

```
  NOTE: the previous run was answered by Ollama 0.34.1, this one by
  Ollama 0.34.2. A runtime update alone moves answers (M17), so what
  changed above is not only the effect of any other change.
```

Each change has a test, and each test was seen failing first. Two check that
different versions give different keys and that the client hashes the version
the server reports (a fake server says `9.9.9`, which no default could produce).
One checks the comparison's note. And the tests that drive the eval with the
model replaced now also replace the version request: the whole suite passes
with Ollama unreachable, which is how it was checked.

**Not covered:** a GPU driver update also changes the code that does the
arithmetic, and Ollama's version does not say which driver ran. That was not
measured, and it is not in the key.

---

## Step 4: one runtime, measured from nothing

With the version in every key, nothing stored before could be replayed, so
this was a full measurement from nothing: every clause reading, every set of
facts, every pick and every answer, all from Ollama 0.34.2. Three samples per
question, the first in the file's order and the other two shuffled.

### What it cost

The first measurement after every Ollama update will look like this one:

| Piece | Minutes |
|---|---:|
| Synthetic policy, 40 clause readings | 5 |
| Main set, sample 1 (facts and answers) | 11.5 |
| Main set, samples 2 and 3 | 7.5 + 8 |
| Held-out batch 1, three samples | 11 |
| Held-out batch 2, three samples | 9.5 |
| Star policy, 80 clause readings | 9.7 |
| Star questions, three samples, in three pieces | 9.9 + 6.9 + 6.1 |
| **Total** | **about 85** |

About 4 of those minutes went on the version request itself, before
Failure 71 (below) was found and fixed. Even so, that is nearly three times what this change
was expected to cost when it was chosen (25 to 30 minutes, an estimate that
counted the answers and forgot that the clause readings and facts are keyed
too). Two pieces ran past the ten
minutes each piece was meant to stay under and finished in the background.

### Results

| | M16 closing (two Ollama versions mixed) | M17 (one version, every answer fresh) |
|---|---|---|
| Synthetic main set | 34/40 | 34/40 |
| Held-out batch 1 | 10/16 | 11/16 |
| Held-out batch 2 | 12/13 | 11/13 |
| Star, right verdicts | 7/10 | 7/10 |
| Star, citing the deciding clause | 5/10 | 5/10 |
| Questions whose three samples disagree | 10 of 79 | **3 of 79** |
| Detection integrity, every sample | 1.000 | 1.000 |

The headline numbers barely moved. What moved is the last row: on one runtime,
with every answer from that runtime, 76 of 79 questions gave the same verdict
in all three samples, asked in three different orders. The three that did not:
`initial-waiting-period` (conditional, conditional, covered; wrong every time),
`dental-no-accident` (right once in three), and `ho-icu-within-cap` (right once
in three).

The held-out questions M16 lost came back: scuba diving, the early
pre-hospitalisation bill and the short procedure are right in all three
samples, and every scuba answer cites the hazardous-sports exclusion (4.3).
Three others went the other way (`ho2-drunk-scooter`, `ho-icu-within-cap`,
`ho-dental-accident`). The held-out batches together are 22 of 29, as on
M16's second day, against 24 of 29 at the end of M14.

**Reading.** On a fixed runtime this system is close to deterministic, and a
majority of three mostly confirms that. What a sample can still show is the
handful of questions whose answer depends on what was asked before them, and
the shuffled order is what lets it show them. The large movements between
measurements came from the runtime changing underneath, and now that the
runtime is in the key, a replayed answer can no longer carry one runtime's
verdict into another's measurement.

### Failure 71: a cost called small without measuring it

**Setup.** The version request was written to run before every cached call,
with a docstring saying it "costs little beside a model call of seconds".

**What happened.** The measurement in Step 4 ran with it, and then the tests
that use the live model were run. One of them times a cache hit against a
model call, and failed:

```
AssertionError: cache appears not to be hit: cold=2.59s warm=0.81s
```

The answer came from the cache (the test's other assertions passed); the
lookup itself had become most of a second slower.

**Why.** Timing the request on its own found two costs, neither of them the
request:

| Version request | Time |
|---|---:|
| new HTTP client, `localhost` | 430 ms |
| new HTTP client, `127.0.0.1` | 170 ms |
| new client, no TLS context, `127.0.0.1` | 14 ms |
| one client reused, `localhost` | 6 ms |

Building an `httpx` client prepares a TLS context, about 0.2 s, even for a plain
`http://` address that will never use it. And `localhost` costs about 0.25 s
more, because Windows tries the IPv6 address first and Ollama listens only on
IPv4.

**The fix.** The version is remembered for a minute:

```python
# api/app/llm/client.py
_VERSION_TTL_SECONDS = 60.0
_version: tuple[float, str] | None = None

async def runtime_version() -> str:
    global _version
    now = time.monotonic()
    if _version is not None and now - _version[0] < _VERSION_TTL_SECONDS:
        return _version[1]
    ...
    _version = (now, version)
    return version
```

That trades exactness for speed, and the trade is written down in the code:
the only answer that can be filed under an old version is one generated in the
first minute after an Ollama update, by a process that was already running. A
test pins the minute with a fake clock and a fake server.

**What it showed about everything else.** Every model call builds the same
kind of client, so every call has always paid about 0.4 s of setup. Beside an
answer of several seconds that is small, and it is left alone here: noted, not
measured further. It did inflate Step 4's timings, by roughly 0.4 s on each of
about 500 calls, about 4 minutes of the 85.

The lesson is the one Failure 53 recorded about a number in a comment: a claim
about cost is a measurement or it is a guess. This one was caught because a
test written with the client, in the project's first commit, measured the
property instead of assuming it.

### Failure 72: a test that emptied the evals' cache

**Setup.** To check Failure 71's fix, every test was run, including the eight
that need the live model, from the repository root rather than from `api/`.

**What happened.** The next report run found almost nothing stored. Everything
Step 4 had generated, about 85 minutes of answers, was gone, and so was every
older entry. The cache held 93 entries, the oldest written at the minute the
live tests ran.

**Why.** One live test checks that a second identical call is served from the
cache, and it starts from an empty one:

```python
# api/tests/test_llm.py
async def test_cache_returns_identical_result_and_skips_the_model():
    cache.clear()
```

The cache's location is relative to the working directory:

```python
# api/app/llm/cache.py
_CACHE_PATH = Path("data/llm_cache.db")
```

Run from `api/`, as the README says, that is `api/data/llm_cache.db`, a file
only the tests use. Run from the repository root, where the eval scripts run,
it is the evals' own cache. The test suite already moved the app's database
and upload folder to a temporary directory for exactly this reason (`DB_PATH`
and `UPLOAD_DIR` in `api/tests/conftest.py`); the response cache was never
included.

**What it cost.** The measurement's results were not lost: they are in the run
history and the logs. The stored answers behind them were, so `evals/REPORT.md`
was generated from a fresh sample (20 minutes) instead of a replay. The older
entries could no longer be reached by the current code, because they have no
Ollama version in their key, but checking out an older commit would have
replayed them; that is gone too.

**The fix.** Every test now gets a private cache, the same way it gets a
private database:

```python
# api/tests/conftest.py
@pytest.fixture(autouse=True, scope="session")
def _private_llm_cache():
    from app.llm import cache
    cache._CACHE_PATH = _TMP / "llm_cache.db"
    cache._conn = None
    yield
```

and the test that empties a cache first checks whose it is:

```python
    assert "ipce-tests-" in str(cache._CACHE_PATH), f"refusing to clear {cache._CACHE_PATH}"
```

Checked the way it went wrong: all 256 tests run from the repository root, and
the evals' cache held 187 entries before and 187 after.

**The lesson.** A path relative to the working directory means a different
file depending on where a command is typed. A destructive test has to check
its own target, not trust that it is being run from the right place.

---

## Closing out M17

### How it connects to what was already there

Majority of three (Concept 34, M12) assumed each sample was a new draw. M17
measured what actually varies from one sample to the next, and found three
layers, in decreasing size:

1. **The runtime** (Failure 70, Concept 45): an Ollama update moved 10 of 69
   verdicts. Now in the cache key, beside the KV cache precision (Concept 41),
   and recorded in every report and history entry.
2. **The order of the questions** (Concept 43 through Step 1): 3 of 79. Now
   varied deliberately, one order per sample.
3. **A freshly loaded server** (Step 2): none measured.

The first layer is the one that had been read as noise. It is not noise: it is
a different system answering, and the fix is to keep it out of comparisons, not
to average over it.

### Where this leaves the numbers

All on Ollama 0.34.2 and `qwen2.5:7b-instruct-q4_K_M`, every answer generated
for this measurement: main set 34/40, held-out batches 11/16 and 11/13
(together about 76%), Star 7/10 verdicts with 5/10 citing the deciding clause,
and detection integrity 1.000 in every sample.

**Still open:**

- whether the version explains all ten of M16's day-to-day differences: shown
  only by elimination, not by running 0.34.1 again;
- a GPU driver update changes the arithmetic too, and is not in the key;
- the first measurement after each Ollama update costs about 80 minutes;
  Ollama's automatic updates are what decide when that happens;
- every model call spends about 0.4 s building an HTTP client and trying IPv6
  for `localhost` (Failure 71);
- everything M16 left open, including the unexplained drop from 38/40
  (Failure 69).

---

## Check it yourself

```bash
curl http://localhost:11434/api/version                        # the runtime the keys now carry
api/.venv/Scripts/python.exe -m pytest api/tests -m "not llm"  # passes with Ollama switched off
python evals/run_scenario_eval.py --heldout 2 --repeats 3      # three samples, three orders
```

**Question to sit with:** after an Ollama update, the first eval run finds
nothing in the cache and regenerates everything. Someone proposes keeping the
old answers reachable, to save the 85 minutes, by leaving the version out of
the key for runs that only compare prompts. What would a prompt comparison
across the update then measure, and how would its report look different from
an honest one?

<details>
<summary>Answer</summary>

It would measure the prompt change and the runtime change together, and could
not separate them. The cases the prompt change touched would be answered fresh
by the new runtime; the cases it did not touch would replay the old runtime's
answers. The report would call the first group "regenerated" and the second
"replayed - cannot have moved", which is true of the stored bytes and false of
the system: under the new runtime some of the replayed cases would answer
differently. The comparison's central promise (Concept 29), that a replayed
case cannot have moved, would hold for the cache and not for the model.

An honest report either regenerates everything, which is what the key forces,
or says which runtime answered each case. The note printed when a comparison
spans two versions is the second half of that.
</details>

---

# M18 — The first thing a user sees

## The starting position

Every milestone up to this point was measured through an eval harness: a JSON
answer key, a score, a report. That is the right way to know whether a change
helped, and it has one blind spot. **A number in a report is not the thing the
user looks at.** The eval scores verdicts and citations; it never renders the
dashboard, never reads a clause card, and never sees the sentence printed under
an answer.

So the app was opened and driven end to end, by hand, on a real wording: Star
Health's Arogya Sanjeevani, uploaded through the interface the way a user would,
then asked real questions. The pipeline worked — 80 clauses analysed, an answer
in about forty seconds. Four things on the screen were wrong, and no eval in the
repository could have seen any of them:

1. The **second-highest-ranked clause in the whole policy** said *"The policy
   will never pay for any medical expenses you incur."* The policy does not say
   that, and nothing in it means that.
2. A clause card reported that the policy's annexure of non-payable items was
   *"written at grade 94 reading level"*. There is no grade 94.
3. A correct answer about a hernia, citing the right clause and quoting it
   exactly, was shown under a red banner reading **"Not verified. Treat this
   answer as unreliable."**
4. A settled answer — *not covered*, on a waiting period with ten months left to
   run — was printed under the heading **"To answer this properly, it would need
   to know"**, followed by the claimant's age.

These four share a shape worth naming before the detail. Each is a place where
the system **said something false while every component worked exactly as
designed**. No exception was raised, no test failed, no metric moved. That is
what makes them a category: they are not bugs in the sense of code doing
something other than what it says, they are bugs in what the code was told to
say.

---

## Failure 73: the worst clause in the policy, and the policy never said it

### What was on the screen

The dashboard ranks clauses by impact — how likely a clause is to decide a claim
against you. Rank 2 on Star Health's wording, above every real exclusion in the
document, was a card whose plain-English rewrite read:

> The policy will never pay for any medical expenses you incur.

That is a repudiation of the entire contract. It is also not in the policy. Here
is the clause the card was built from, exactly as the segmenter produced it:

```
The Company shall not be liable to make any payments under this Policy
in respect of any expenses what so ever incurred by the Insured Person
in connection with or in respect of;
S T A N D A R D  E X C L U S I O N S
```

Read the ending again: **"in connection with or in respect of;"**. The sentence
stops there. It has a subject, a verb, and no object. On the page, its object is
the numbered list that follows — twenty-two exclusions, 1 through 22, each one a
thing the company will not pay for.

The segmenter cut the fragment off from its list and handed it to the analysis
stage as a clause in its own right. The model was asked, in effect, "what does
this clause exclude?" — and the only faithful answer to a sentence whose object
has been removed is "everything it names", which here is *any expenses what so
ever incurred by the Insured Person*.

**The model read the fragment correctly. The fragment was the lie.**

Then the scorer did its job on that reading. An exclusion carries the heaviest
type weight in the impact formula (`TYPE_WEIGHT[EXCLUSION] = 0.95` in
`api/app/pipeline/score.py`), and the model rated an exclusion of all medical
expenses as maximally severe and maximally likely to bite. The product put it
second in the document. Every stage after the bad boundary behaved correctly,
and **correctness downstream of a bad boundary does not repair the error, it
gives it authority.**

### First question: is this one document's quirk?

Before writing a rule, the obvious thing to establish is whether this is Star's
typesetting or how IRDAI wordings are written. Every unnumbered clause in every
available wording — three real, three synthetic — was printed with its ending
character:

```
star-arogya-sanjeevani.pdf    80 segments,  12 unnumbered
hdfc-optima-secure.pdf       101 segments,   2 unnumbered
niva-reassure-2.pdf          143 segments,   0 unnumbered
synthetic-health-policy.pdf   40 segments,   1 unnumbered
synthetic-hostile-policy.pdf  40 segments,   1 unnumbered
synthetic-mini-policy.pdf      4 segments,   0 unnumbered
```

Two of the three real wordings open their exclusions with exactly this
construction, and nothing else among the 408 segments looks like it:

```
Star:  "...incurred by the Insured Person in connection with or in respect of;"

HDFC:  "The Company shall not make payment for any claim in respect of any
        Insured Person caused by, arising from or attributable to any of the
        following unless expressly stated to the contrary in the Policy:"
```

Both are unnumbered. Both end on a colon or a semicolon. Both are completed by
the numbered list beneath them. That is a **drafting pattern**, not a quirk, and
it has a signature a rule can read: *no clause number of its own, and a sentence
that does not end.*

### The fix

In `api/app/pipeline/segment.py`, a pass that runs after boundaries are found:

```python
    merged: list[Segment] = []
    for seg in reversed(segments):
        if merged and not seg.number and seg.text.rstrip().endswith((":", ";")):
            follower = merged[0]
            follower.text = raw_text[seg.char_start : follower.char_end]
            follower.char_start = seg.char_start
            follower.page_start = seg.page_start
            follower.bboxes = seg.bboxes + follower.bboxes
            merged[0:1] = _split_oversized(follower, raw_text)
            continue
        merged.insert(0, seg)
    return merged
```

The list is walked **backwards** because a lead-in attaches to what comes *after*
it, so the follower has to already be in hand by the time the fragment is
reached. Walking forwards would mean carrying a pending fragment through the
loop, which is the same logic with an extra variable.

Requiring the fragment to be unnumbered is what stops the rule eating real
clauses. A numbered clause that happens to introduce a list of its own —
"4. The following are excluded:" — keeps its number, its identity and its own
entry.

### Three alternatives, and why each was rejected

- **Delete the fragment.** It is real text from the document. Deleting text
  because the pipeline finds it awkward is how a tool starts lying by omission,
  and this one is the operative verb phrase of twenty-two exclusions.

- **Attach it to all twenty-two exclusions.** It does govern all of them. But
  every one of those clauses would then carry a duplicated preamble, inflating
  its length, its measured reading difficulty and its share of the context
  window — for no gain, because exclusions 2 through 22 already read correctly
  alone ("21. Any expenses incurred on Domiciliary Hospitalization and OPD
  treatment", filed under a section headed EXCLUSIONS).

- **Ask the model to notice fragments.** This is stage 2, which contains no LLM
  on purpose. Clause boundaries must be reproducible run to run, and a model
  asked the same question twice does not answer identically twice. Rejected on
  the architecture, not on the result.

### Why the join is a slice and not a concatenation

`follower.text = raw_text[seg.char_start : follower.char_end]` looks like a long
way to write `seg.text + follower.text`. It is not the same thing.

Every clause in this system carries `char_start` and `char_end` into the original
document, and one invariant holds throughout the pipeline:

```python
raw_text[seg.char_start : seg.char_end] == seg.text
```

That invariant is what makes a citation **provable** — it is how a quoted span is
traced back to a position in the PDF the user uploaded. A concatenation of two
strings produces text that appears nowhere in the document at those offsets, and
the invariant would break silently, taking the grounding guarantee with it.

Taking the slice keeps it exact, and has a useful side effect: anything sitting
*between* the two pieces — such as the section heading in the middle — comes back
along with it, which is what the page shows anyway.
`api/tests/test_segment.py` asserts this invariant over every clause of every
test document, and it still holds after the merge.

### Failure 73b: a guard that switched the fix off exactly where it was needed

The first version of the merge refused to join when the result would exceed
`MAX_CLAUSE_CHARS` (3,000), with this reasoning written in the comment:

> Skip rather than re-split: a lead-in long enough to breach the cap is not a
> lead-in.

Measured, the fix then worked on Star and did **nothing at all** on HDFC:

```
idx=48 num=''  chars=200   <- the lead-in
idx=49 num='1' chars=2997  <- the clause it introduces
merged length would be: 3198
```

HDFC's first exclusion had **already been split** by `_split_oversized` to sit
just under the cap. Adding a 200-character lead-in to a 2,997-character clause
overflows by 198, the guard fired, and the merge was skipped — on one of only two
documents the rule exists for.

Notice what the comment claims and what the code tested. The comment reasons
about the **lead-in's** length; the code tested the **merged** length. They are
different claims, and the fragment is 200 characters either way. Writing the
comment before the condition would have caught it.

The repair reuses the splitting the pipeline already does:

```python
            # HDFC's first exclusion already fills 2,997 of the 3,000-character
            # cap, so refusing to merge over the cap would have left that
            # document's lead-in stranded - the exact case this exists for.
            # Re-splitting keeps the lead-in on the first piece and moves the
            # overflow into a continuation, which is what the cap is for.
            merged[0:1] = _split_oversized(follower, raw_text)
```

The lead-in stays with the first piece; the overflow becomes a continuation.
**A guard that protects an invariant should re-establish the invariant, not
abandon the work it was guarding.** Skipping was the cheaper branch to write;
splitting was two more lines and actually did the job.

---

## Failure 74: nine headings the segmenter could not read

### The setup

While measuring Failure 73, one other Star segment stood out — 36 characters, no
clause number, and its entire content was:

```
S T A N D A R D  C O N D I T I O N S
```

That is a section heading set with its letters held apart. It is a typesetting
effect; a reader sees the words STANDARD CONDITIONS. PDF extraction returns the
characters where they sit, spaces and all.

`segment.py` decides whether a line is a section heading partly by looking for
known policy vocabulary inside it:

```python
    stripped = line.text.strip()
    if len(stripped) <= 70 and stripped.isupper():
        if _RE_SECTION_HEAD.match(stripped):
            return True
        if _RE_SECTION_WORD.search(stripped.lower()):
            return True
```

`_RE_SECTION_WORD` looks for whole words — `\bconditions?\b`, `\bexclusions?\b`
— and "s t a n d a r d  c o n d i t i o n s" contains no such word. The test
failed, and every letter-spaced heading in the document was treated as ordinary
text.

Searching all three real wordings for lines whose tokens are mostly single
characters showed the scale of it:

```
star-arogya-sanjeevani.pdf
  'S T A N D A R D  D E F I N I T I O N S'
  'P O L I C Y  W O R D I N G S'
  'S P E C I F I C  D E F I N I T I O N S'
  'S T A N D A R D  E X C L U S I O N S'
  'S P E C I F I C  E X C L U S I O N S'
  'S T A N D A R D  C O N D I T I O N S'
  'S P E C I F I C  C O N D ITION S'
  'A N N E X U R E  -  A'
  'L I S T  O F  I N S U R A N C E  O M B U D S M A N'
hdfc-optima-secure.pdf   (none)
niva-reassure-2.pdf      (none)
```

**All nine of Star's major headings**, and not one in the other two documents.
One of the nine became a clause of its own; the rest were swallowed into
whichever clause they sat beside — which is why the fragment in Failure 73 ended
in `S` rather than in `;`.

### Concept 46: reading what the page says, not what the file contains

A PDF does not store words. It stores glyphs and the positions they are painted
at. The gap between two letters and the gap between two words are the same kind
of thing — a distance — and text extraction turns distances into spaces by
comparing them against a threshold derived from the font.

Letter-spaced display type defeats that threshold, because its letter gaps are
deliberately wider than the threshold expects. Every letter gap becomes a space,
and the word is shattered.

The information is not lost, though, and this is the part worth keeping. **The
typesetter faced the same problem and solved it the only way available: by making
the word gaps bigger still.** Star's headings use one space between letters and
two between words. That second gap survives extraction, and recovering the words
is simply reading it:

```python
def _despaced(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 4 or sum(len(t) == 1 for t in tokens) / len(tokens) <= 0.6:
        return text
    return " ".join("".join(word.split()) for word in re.split(r"\s{2,}", text.strip()))
```

The guard on the first line is what makes this safe to run over every line of
every document: **no real sentence is mostly one-letter words.** Ordinary text
fails the test and is returned untouched, so this cannot corrupt a clause.

Splitting on the *word* gap rather than trying to detect runs of single letters
has an unplanned benefit. One of Star's headings extracts irregularly:

```
'S P E C I F I C  C O N D ITION S'  ->  'SPECIFIC CONDITIONS'
```

`C O N D ITION S` is a mess — three differently-spaced pieces. But the double
space before it is intact, so the word boundary is found, and everything inside
that boundary is joined regardless of how it was spaced internally. A rule built
on the letter gaps would have failed here. **The boundary carried the
information; the letters were noise.**

### What it changed

Star's section list, before and after:

```
before:  ['Policy Wordings', '3 - DEFINITIONS', 'STAR HEALTH AND ALLIED
          INSURANCE COMPANY LIMITED', '4 - COVERAGE', '5 - EXCLUSIONS',
          '6 - CONDITIONS', 'TABLE OF BENEFITS']

after:   [..., 'STANDARD DEFINITIONS', 'SPECIFIC DEFINITIONS',
          'STANDARD EXCLUSIONS', 'SPECIFIC EXCLUSIONS', 'STANDARD CONDITIONS',
          'SPECIFIC CONDITIONS', 'ANNEXURE - A', ...]
```

Six headings recovered, one junk clause removed, and every clause now files under
a more precise section than "5 - EXCLUSIONS". Star went from 80 segments to 79.

**Niva's wording and all three synthetic policies changed by not one segment.**
That was the check that mattered: none of them is letter-spaced, so a correct fix
had to leave them completely alone. It did.

---

## Failure 75: a reading level that does not exist

### What was on the screen

A clause card for Star's annexure of non-payable items carried the note:

> written at grade 94 reading level

**Flesch–Kincaid, from scratch.** It estimates the US school grade needed to read
a text on one pass, from two measurements: how long the sentences are, and how
many syllables the words have. Long sentences are hard to hold in your head at
once; long words tend to be the Latinate ones. Grade 8 is ordinary newspaper
prose; grade 20 is dense legal drafting. The formula, in
`api/app/pipeline/score.py`:

```python
    grade = 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59
```

It is used here because legal writing scores high on both terms at once — long
subordinate sentences built from Latinate vocabulary — and that is precisely the
obscuring this project exists to measure.

### The cause

The annexure is 145 lines of item names — BABY FOOD, BELTS/BRACES, OXYGEN
CYLINDER (FOR USAGE OUTSIDE THE HOSPITAL) — and **not one of them ends in a full
stop.** Sentences were counted by splitting on `[.!?]`, which on a text
containing no terminator returns exactly one piece. So:

```
words=221  sentences=1  ->  0.39 * 221 = 86.2  ->  grade 93.7
```

The formula was applied to something that is not prose. A list has no sentences,
so "words per sentence" is not a property it possesses, and dividing by one
invented sentence produces a number from a scale that tops out near 20.

It is worth being precise about what did *not* go wrong. The ranking was
unaffected, because the buriedness signal was already clamped:

```python
        grade = flesch_kincaid_grade(seg.text)
        reading = min(max((grade - 8.0) / 12.0, 0.0), 1.0)
```

Grade 94 and grade 20 both yield `reading = 1.0`. **The score was right and the
displayed number was nonsense** — which is exactly why no eval caught it. The
harness scores verdicts and rankings; this number only ever reached a human eye.

### A wrong fix, measured before it was written

The obvious repair is to count line breaks as sentence ends. It was checked
against the data first, and abandoned. PDF extraction preserves the document's
visual line wrapping, so ordinary prose is full of line breaks that mean nothing:

```
grade=12.4  words=125  sentences=10  lines=31   "1. Hospitalization: The Company..."
```

Ten sentences, thirty-one lines. Counting lines there would cut a real clause's
measured difficulty by roughly a third for a typesetting reason. **That fix trades
a wrong number on 2 segments for a wrong number on 400.**

What separates the two cases is not the presence of line breaks — both have them
— but the **complete absence of any sentence**:

```python
    units = [s for s in _RE_SENTENCE.split(text) if s.strip()]
    # A list of item names ends no sentence, so splitting on full stops makes
    # the whole block one: Star's annexure of 145 non-payable items came out at
    # 221 words per "sentence" and reported grade 94, a number that does not
    # exist. Where a text terminates nothing, the unit a reader takes in at once
    # is the line. Line breaks are not used otherwise, because in wrapped prose
    # they fall wherever the measure ran out and mean nothing.
    if not _RE_SENTENCE.search(text):
        units = [line for line in text.split("\n") if line.strip()]
    sentences = max(len(units), 1)
```

Where a text ends sentences, they are the unit and line breaks are ignored
entirely. Where it ends none, it is not prose, and the line is the unit.

### What it changed, measured across all 408 segments

Eight segments in the six documents have no sentence terminator at all:

```
LIST I (Star, 221 words on 145 lines)     93.7 -> 8.1    the bug
HDFC 2.8 (a 41-word list on 29 lines)     34.4 -> 18.9   also absurd, also fixed
HDFC (a), 45 words on 3 lines             21.9 -> 10.2   a regression, kept
Star / HDFC exclusion lead-ins            21.0 -> 4.4    no longer exist (Failure 73)
Niva List II / III (one line each)        unchanged      one line either way
```

**The regression is real and is being kept deliberately.** HDFC's definition of
"acute condition" is genuine prose that happens to omit its final full stop. It
is now measured as three short lines rather than one long sentence — grade 10,
where 22 is nearer the truth. It is one segment in 408; its effect is one input
to a buriedness signal weighted 0.35 which is then clamped; and the alternative
fix damaged 400 segments to rescue it.

Writing the regression down here, with its size, is the point. A fix recorded as
"fixed the grade bug" would leave the next person to rediscover this by accident.

---

## Failure 76: an exact quotation, marked unreliable

### What was on the screen

Asked *"I need surgery for an inguinal hernia that was found last month. I bought
this policy 14 months ago and it is my first health insurance."*, the system
answered **not_covered**, citing Star's specified-disease waiting period — the
right verdict, from the right clause. Above it sat a red banner:

> **Not verified.** Some wording quoted below could not be found in your policy.
> Treat this answer as unreliable and check the clauses yourself.

### Why the check said that

Star's clause 2 is lettered A to E. The model quoted the heading, A, D and E, and
skipped B and C. B is about enhancement of the sum insured; C is about overlap
with the pre-existing waiting period. Neither bears on the question asked.

`api/app/grounding.py` required a quotation to appear in the clause as **one
unbroken substring**. Every word the model printed was the clause's own, in the
clause's own order, but the run was broken twice. Checked piece by piece against
the stored clause text with a cursor that only moves forward:

```
found=0    len=61   | 2. specified disease / procedure waiting period -code excl 02
found=62   len=301  | a. expenses related to the treatment of the following listed...
found=645  len=145  | d. the waiting period for listed conditions shall apply even...
found=791  len=222  | e. if the insured person is continuously covered without any...
```

All four present, ascending, never overlapping. The quotation was honest. The
check had no way to express what it was.

### Concept 47: provenance and meaning are different questions

This is the most delicate change in the milestone, because it **loosens the one
mechanism the whole project rests on.** The project's rule is that a fabricated
citation must be *impossible*, not merely unlikely, and two layers make that
true: clause ids are `enum`-constrained during decoding, so a made-up address
cannot be sampled at all; and quoted text must be found in the clause it is
attributed to. It is layer two being relaxed here, so the reasoning needs to be
exact.

The distinction that makes it safe is between two questions a checker might be
asked:

- **Provenance** — did these words come from this clause? Decidable by looking.
  It is the only question this check has ever answered.
- **Meaning** — does the quotation fairly represent the clause? That requires
  judging whether the skipped material mattered, which is a reading, not a
  lookup. No substring test has ever been able to do it.

A contiguous-substring check was never verifying meaning, and it is worth being
blunt about why. A model could already quote one true sentence and omit the
qualifying sentence immediately after it. **Truncation is elision with only one
gap**, and truncation always passed. What contiguity actually bought was a limit
on how far apart the kept pieces could sit — a weaker guarantee than it appears
to be, and not the guarantee anyone thought they were relying on.

So the replacement keeps provenance exact and adds three constraints that
preserve the parts of contiguity worth keeping:

```python
    pieces = [p for p in (normalize(line) for line in quote.split("\n")) if p]
    # One piece would have passed the contiguous check already; reaching here
    # with one piece means it genuinely is not in the clause.
    if len(pieces) < 2:
        return False

    haystack = normalize(source_text)
    cursor = 0
    for piece in pieces:
        if len(piece) < MIN_QUOTE_CHARS:
            return False
        found = haystack.find(piece, cursor)
        if found == -1:
            return False
        cursor = found + len(piece)
    return True
```

- **Split only at the model's own line breaks.** It may elide where it chose to
  break a line, never mid-sentence, so it cannot assemble a new sentence out of
  phrases gathered from across the clause.
- **Every piece at least `MIN_QUOTE_CHARS` (25).** An answer cannot be stitched
  from fragments too short to mean anything alone. This is the bar the
  whole-quote check has always applied, now applied to each piece.
- **A forward-only cursor.** Pieces must appear in document order and may not
  overlap, so a condition cannot be lifted above the sentence that qualified it,
  and no passage can be quoted twice to pad out the others.

The strict contiguous test still runs **first**, and the elision path is reached
only after it fails. An unbroken quotation is checked exactly as it always was:

```python
    # The strictest reading first, so an unbroken quotation is still checked
    # exactly as it always was, and the elision path is only ever a fallback.
    if normalized_quote in normalize(source_text):
        return True, ""
    if _verify_elided(quote, source_text):
        return True, ""
    return False, "quote does not appear in the cited clause"
```

Run against the real failing citation and three attacks on it:

```
real elided quote      -> (True, '')
reordered (D before A) -> (False, 'quote does not appear in the cited clause')
one line invented      -> (False, 'quote does not appear in the cited clause')
short stitched frags   -> (False, 'quote does not appear in the cited clause')
```

### Why a false alarm is not a safe failure

It is tempting to file this under "the check was being conservative", and treat
conservative as harmless. It is not harmless, and the reason generalises well
beyond this project.

A warning is only useful for as long as people believe it. This banner tells a
reader to distrust the answer and go and read the clauses themselves. If it
fires on answers that are exactly right, readers learn within a handful of uses
that it means nothing — and then it is still there, still firing, on the day the
model **does** fabricate a quotation.

**A check that cries wolf does not degrade into a stricter check. It degrades
into no check at all**, while continuing to look like a check in the code and in
the interface. That same argument already appears in this file as the reason a
trailing reference tag is stripped before comparison, and it is why this one was
worth doing carefully rather than leaving alone.

---

## Failure 77: asking for facts about an answer it had already decided

### What was on the screen

Under the same hernia answer — a settled *not covered*, decided by a waiting
period with ten months still to run — the interface printed:

> **To answer this properly, it would need to know**
> — your age
> — whether this condition existed before you bought the policy
> — whether you were admitted overnight
> — how soon you told the insurer

Two separate things are wrong there, with two different causes.

### Part one: one list, two audiences, opposite meanings

The list comes from `DECISIVE_FACTS` — the facts that *can* decide an outcome
under an Indian health policy — filtered to those the person did not state:

```python
    missing = [k for k in DECISIVE_FACTS if facts.get(k) in (None, "", "unknown")]
```

That single list has two consumers, and they want opposite things from it.

**In the reasoning prompt** it appears under the heading *"NOT STATED by the
person (do not assume values for these)"*, and listing every unstated decisive
fact there is exactly right. It is what lets the model tell "the policy is
silent on this" apart from "the person didn't mention it" — two situations that
lead to different verdicts, and the reason `insufficient_information` can be a
first-class answer at all.

**In the interface** the same list ran under a heading asserting that the answer
above it was not proper.

When the verdict genuinely is `insufficient_information`, that heading is the
entire point: those facts are the reason there is no answer, and asking for them
is the useful thing to do. When a verdict *was* reached, the same words tell the
reader that a correct, decided answer is incomplete — and nothing on the screen
distinguishes the two situations.

There is a documented reason the two lists are built from one source: they were
once two lists that drifted apart, and the interface ended up offering "body
system" and "estimated cost inr" as information it needed. So the fix is not to
split the list again. It is to let the **heading** depend on the verdict, since
the verdict is the thing that actually differs:

```tsx
          <p className="label mb-1.5">
            {result.verdict === 'insufficient_information'
              ? 'To answer this, it would need to know'
              : "You didn't mention these, and they can affect a claim"}
          </p>
```

Nothing is hidden from the reader — the same facts are still listed. The claim
attached to them is now one the system can support.

### Part two: a fact stated twice and read as unknown

`pre_existing_condition` should never have been on that list at all. The question
says the hernia was **found last month**, and that the policy was **bought 14
months ago**. A condition found one month ago, under cover held for fourteen,
began thirteen months *into* the policy. It is not pre-existing, and the
description says so — just not in those words.

The extractor returned `"unknown"`, and this sentence in the fact prompt is why:

```
For pre_existing_condition, answer "unknown" unless the description makes it
clear either way. "Unknown" is the honest answer far more often than not.
```

**This project has been here before.** An earlier milestone rewrote that exact
sentence to make the extractor read harder. It worked on the pre-existing
sentences and cost **six unrelated main-set cases** — an ICU rate, a breach of
law, pre-hospitalisation expenses — because the fact prompt feeds every reasoning
prompt, so changing what it says about one field changes answers everywhere. It
was reverted and recorded as open.

The reason it kept going wrong is that it was being treated as a **reading**
problem. It is not. *"Found last month"* against *"bought 14 months ago"* is a
**subtraction**, and this pipeline's governing principle is that the model never
does arithmetic: stages 2 and 4 contain no LLM at all, unit conversion is done in
Python, waiting periods are compared in Python, and `age_at_policy_start` is
already derived rather than inferred for exactly this reason.

So the condition's age becomes a duration like every other duration in the
schema — a value and a unit, reported, not judged:

```python
        # HOW LONG THE CONDITION HAS BEEN KNOWN, as a value and a unit, like
        # every other duration here. Whether a condition is pre-existing is
        # often not stated at all - it is the comparison between this and
        # `time_since_policy_start`. "Found last month" on a policy bought 14
        # months ago is a condition that arose 13 months INTO cover, and the
        # model was answering "unknown" to it, because working that out is
        # subtraction. The model reports the sentence; `derive_pre_existing`
        # below does the comparison.
        "condition_known_for_value": {"type": ["integer", "null"]},
        "condition_known_for_unit": {
            "type": ["string", "null"],
            "enum": ["days", "weeks", "months", "years", None],
        },
```

and the comparison happens in code, in `api/app/pipeline/scenario.py`:

```python
def derive_pre_existing(facts: dict[str, Any]) -> dict[str, Any]:
    if facts.get("pre_existing_condition") != "unknown":
        return facts
    known = condition_known_days(facts)
    policy = policy_age_days(facts)
    if known is None or policy is None or known == policy:
        return facts
    return facts | {"pre_existing_condition": "yes" if known > policy else "no"}
```

Two deliberate narrowings, both following a principle this project arrived at
earlier — that a check which overrides a model must act only on positive evidence
for the other reading, because its two kinds of mistake do not cost the same:

- **It only fills an `"unknown"`.** Someone who states outright that a condition
  is long-standing has given evidence no arithmetic should overrule. A wrong
  `"yes"` denies a claim that should be paid; a wrong `"no"` pays one that should
  not. Neither is cheap, so the correction acts only where the model declined to
  answer at all.
- **It only fires when both durations were stated.** One duration is not a
  comparison, and supplying the missing one would be exactly the invention the
  extraction prompt spends its whole length forbidding.

The "unknown is honest" sentence is **left standing**. That is the difference
from the attempt that was reverted: this change does not ask the model to judge
harder, it gives it one more thing to copy down.

---

## Failure 78: four prompts to add one field, and what each one broke

The schema field and the Python comparison were right on the first attempt and
never changed. **The prompt took four versions**, and each version broke
something that had nothing to do with pre-existing conditions. That sequence is
the most transferable thing in this milestone, so it is recorded in full.

Throughout, the rule for deciding whether something is real: **three fresh
samples on the new prompt and three on committed `main`.** A single sample of
this model proves nothing — one case in this very set, `seven-years-at-59`,
flipped between runs with no code change at all.

### v45 — the example that stole an age

The first version taught the new field with this example, among others:

```
- "a hernia that was found last month"  -> condition_known_for_value: 1, condition_known_for_unit: months
```

The check then failed a case with nothing to do with pre-existing conditions:

```
BAD eval  copay-applies-emergency  age=None
```

The sentence is *"I took this policy out when I was 70 and I am 72 now. I was
admitted with pneumonia last month."*

```
v45 (new field):  age=None  age=None  age=None
v44 (baseline):   age=72    age=72    age=72
```

Unanimous both ways, so not noise. The phrase "last month" appears in both
sentences doing different work: in the example it dates the *condition*, in the
eval sentence it dates the *admission*. Teaching the model to reach for a new
field on "last month" pulled its attention onto that phrase, and `age: 72` —
which nothing in the new block mentions — was dropped on the floor.

### v46 — a fix that named too many fields

The repair spelled out what the field is not:

```
This field never takes a value away from another one. It is not an age, it is
not how long the policy has been held, and it is not when a hospital stay
happened. ...
```

`copay-applies-emergency` recovered, 3/3. Two other cases broke, 3/3 each:

```
63-held-eleven      age_at_policy_start=52          (63 - 11, computed)
ped-waiting-served  time_since_policy_start=None    (the stated 5 years, lost)
```

Checked against baseline: `ped-waiting-served` and `63-held-eleven` both pass
3/3 on `main`, so both were genuine regressions. A third case in the same run,
`70-bought-at-55`, failed on **both** prompts 3/3 — a pre-existing failure, not
a regression, and worth separating out before drawing any conclusion.

The cause is visible in the wording. Naming `age` and `how long the policy has
been held` in one breath, as things the new field is *not*, put those two fields
side by side in the model's attention — and it responded by relating them. That
is how `63 - 11 = 52` appears in a field the prompt has always said to leave null
when unstated.

### v47 — concrete examples instead of abstract prohibitions

The abstract denial was replaced with the thing the rest of this prompt already
does everywhere: worked examples on the exact collision.

```
Time measured from the POLICY still goes in time_since_policy_start, and time
measured from a HOSPITAL STAY is not this field either:

- "hospitalised for it 5 years after taking the policy out"
     -> time_since_policy_start_value: 5, unit: years; condition_known_for: null, null
- "I was admitted with pneumonia last month"
     -> condition_known_for: null, null (this dates the admission, not the condition)
```

`ped-waiting-served` recovered 3/3. `63-held-eleven` still returned 52, 3/3.

### v48 — restating a rule the prompt already had

The age subtraction is not a new problem; the prompt has always said so, in the
ages section: *"Fill in only what was actually said; the arithmetic linking them
is done afterwards, in code."* The new block had quietly weakened it by
introducing one more duration to relate. So the closing line restates the
existing rule rather than inventing one for this field:

```
This field never takes a number away from another field, and never creates one.
Time measured from the POLICY still goes in time_since_policy_start, time
measured from a HOSPITAL STAY is not this field either, and no field anywhere in
this schema is ever filled by subtracting one number in the sentence from
another - if it was not said, it is null:
```

All six watched cases clean, 3/3 each:

```
v48-condition-known-for
  ped-waiting-served           ['ok', 'ok', 'ok']
  63-held-eleven               ['ok', 'ok', 'ok']
  copay-applies-emergency      ['ok', 'ok', 'ok']
  hernia-found-last-month      ['ok', 'ok', 'ok']
  diabetes-longer-than-policy  ['ok', 'ok', 'ok']
  scan-after-discharge         ['ok', 'ok', 'ok']
```

### Concept 48: a prompt is not a list of independent rules

The pattern across those four versions is one thing, seen four times: **adding an
instruction for one field competes for attention with every other field, and the
fields it damages are the ones nothing warned you to look at.** An example about
hernias cost an age. A denial mentioning two other fields caused a subtraction
between them.

It is the same mechanism as the earlier reversion recorded in this log, where a
change aimed at pre-existing conditions broke an ICU rate — and it is why that
attempt was abandoned rather than debugged. What made this one survivable was
not better prompt wording. It was:

1. **A check that covers the fields you are not changing.** The collisions were
   found in `copay-applies-emergency` and `63-held-eleven`, cases about ages and
   policy durations, because the harness scores every field on every sentence
   rather than the one being worked on.
2. **Three samples, and a baseline run of the same three.** Of the four
   suspicious failures, two were real regressions, one was noise, and one was a
   pre-existing failure. Acting on the single-sample list would have meant
   chasing two phantoms.
3. **Restating an existing rule rather than writing a new one.** Three of the
   four versions added new prohibitions and each added a new interaction. The
   version that worked pointed back at a rule the prompt already contained.

Two honest notes about what this evidence supports. `copay-applies-emergency`
and `ped-waiting-served` are **eval** sentences, not held-out ones, and the v46
and v47 wordings were written while looking at their failures — they are evidence
the collisions were closed, not that the fix generalises. The v48 line was
written from the prompt's own existing rule rather than from
`63-held-eleven`'s output, and `63-held-eleven` is held out; but having watched
it fail, the cleanest reading is that the held-out group as a whole carries the
generalisation claim, not that one case.

### The stale answer key

Four pre-existing sentences in the answer key came back `"unknown"`:

```
BAD heldout  back-pain-began-later   pre_existing_condition=unknown
BAD heldout  asthma-before           pre_existing_condition=unknown
BAD tuned    ho-knee-replacement     pre_existing_condition=unknown
BAD tuned    ho-kidney-stone         pre_existing_condition=unknown
```

The obvious reading is that this milestone broke them. Before accepting it, the
same six sentences were run against committed `main` with every change stashed:

```
prompt version: v44-picked-clauses-covered-means-in-full

  BAD back-pain-began-later    want=no       got=unknown
  BAD asthma-before            want=yes      got=unknown
  ok  fever-no-history         want=unknown  got=unknown
  BAD migraines-new            want=no       got=unknown
  BAD ho-knee-replacement      want=no       got=unknown
  BAD ho-kidney-stone          want=no       got=unknown

  1/6
```

**One out of six on the baseline.** Five of those cases were already failing on
committed `main`, before this milestone touched anything.

Two things follow, and both matter more than the six cases.

**The answer key had gone stale, and nothing was watching.** Those expectations
were recorded when they passed; they have since stopped passing; and because
this check is run by hand rather than from the test suite, nobody was told. A
stale expectation is worse than a missing one — it reads as a guarantee, and it
mislabels a pre-existing failure as a regression, which is exactly what it did
here until the baseline was run.

**This is the runtime lesson arriving from a different direction.** An earlier
milestone established that the program doing the arithmetic is part of the model,
and that stored answers must be re-measured when it changes. These expectations
were written under an earlier Ollama and are being read under 0.34.2. **A
committed answer key carries a runtime with it whether or not anyone wrote that
down.**

Those four cases are **not fixed by this milestone and remain open**: the
extractor still will not answer "no" or "yes" to a sentence that states the
relationship in words rather than in two durations. What this milestone fixed is
the case that can be settled by arithmetic — which is the case that should never
have been put to the model in the first place.

---

## Closing out M18

### How it connects to what was already there

Everything before this milestone was measured through the eval harness, and the
harness is good at what it measures: verdicts, citations, clause rankings,
fact extraction. What it has never done is *look at the product*. It does not
render a dashboard, does not read a clause card, and does not see the sentence
printed beneath an answer.

All five failures here were found in about twenty minutes of using the app by
hand, and every one of them had been shipping for milestones. That is not a
criticism of the harness — it is a statement about coverage. **An eval measures
the thing you pointed it at. Everything else is unmeasured, and unmeasured is
not the same as working.**

Three of the five were failures of the *boundary between stages*, which is the
structural lesson worth carrying forward:

- **Failure 73** — the segmenter handed the analyzer a sentence fragment, and the
  analyzer did a faithful job on it. Neither stage was wrong. The contract
  between them was: stage 2 promises "this is a clause", and it delivered
  something that was not one.
- **Failure 75** — the scorer handed the interface a number from a formula that
  did not apply, and clamped it internally so the ranking stayed correct. The
  clamp hid the problem from every consumer except the one that printed it raw.
- **Failure 77** — one list served the reasoning prompt and the interface, which
  needed opposite things from it. The data was right for both; the heading was
  right for only one.

The two that were not boundary failures were failures of a check being **wrong
about its own job**: the quote verifier answering a question about meaning when
it could only answer one about provenance (Failure 76), and the fact extractor
being asked to do arithmetic it had been told everywhere else not to do
(Failure 77, part two).

### What it made possible

The lead-in merge and the heading recovery are the first segmentation rules in
this project derived from a **pattern measured across documents** rather than
from one document's failure. Both were checked against all six available
wordings before being written, and both were required to leave the other
documents untouched — which they did, to the segment.

That is the shape a rule needs to have here. Star's wording is one insurer's
house style; a rule tuned to it that moves Niva's or HDFC's boundaries is not a
fix, it is a second bug waiting for a different upload.

### Where this leaves the numbers

**Segmentation and scoring** (no model involved, so these are exact):

```
                              segments   offset drift   oversize
star-arogya-sanjeevani.pdf     80 -> 79        0            0
hdfc-optima-secure.pdf        101 -> 101       0            0
niva-reassure-2.pdf           143 -> 143       0            0
synthetic-health-policy.pdf    40 -> 40        0            0
synthetic-hostile-policy.pdf   40 -> 40        0            0
synthetic-mini-policy.pdf        4 -> 4        0            0
```

Star loses the letter-spaced heading that had become a clause, and its exclusion
lead-in merges into exclusion 1. HDFC's lead-in merges the same way without
changing the segment count, because the overflow moves into a continuation.
**Every other document is untouched**, and the offset invariant holds on all
408 segments.

Highest reported reading grade in each document, before and after:

```
star-arogya-sanjeevani.pdf   93.7 -> 28.8
hdfc-optima-secure.pdf       34.4 -> 21.0
```

**Fact extraction**, `evals/check_fact_extraction.py`, Ollama 0.34.2, one fresh
uncached sample per sentence:

```
                v44 (main)   v48 (this milestone)
eval               18/18            18/18
```

The held-out comparison is **partial and is reported as such**. The full
baseline run was interrupted by the machine running out of memory after 36 of
its 44 sentences, so four third-person age sentences (`son-9-tonsils`,
`wife-58`, `grandmother-90`, `mother-took-it-at-61`) have a v48 result and no
v44 result. All four pass on v48 and none is related to anything this milestone
changed.

Every case with a result on both prompts:

```
  70-bought-at-55          v44 BAD    v48 ok     fixed
  63-held-eleven           v44 ok     v48 ok
  back-pain-began-later    v44 BAD    v48 BAD    open, see below
  asthma-before            v44 BAD    v48 BAD    open
  migraines-new            v44 BAD    v48 BAD    open
  ho-knee-replacement      v44 BAD    v48 BAD    open
  ho-kidney-stone          v44 BAD    v48 BAD    open
  fever-no-history         v44 ok     v48 ok
  seven-years-at-59        v44 ok     v48 BAD    noise, see below
  all other held-out       v44 ok     v48 ok
```

Plus the six sentences written for the new field, which have no v44 result
because the field does not exist there. All six pass, and the three watched most
closely pass three samples out of three.

`seven-years-at-59` is recorded as noise rather than a regression on direct
evidence: **the same prompt, v45, produced both outcomes on two different runs**
of the same sentence. It has since flickered on v46 and v48 as well. This is the
reason single-sample results in this project are never acted on alone.

The five "open" rows are the pre-existing-condition sentences whose relationship
is stated in words rather than in two durations. **Four of the five were already
failing on committed `main`** before this milestone started, which is the finding
written up under Failure 78 as the stale answer key.

### What is deliberately not fixed

Three things measured here are being left alone, with reasons, so that finding
them again does not read as a discovery:

1. **HDFC's "acute condition" definition now reports grade 10 where 22 is nearer
   the truth.** It is prose that omits its final full stop, so the new rule
   treats it as a list. One segment in 408; the alternative fix damaged 400.
2. **The extractor still will not read a stated pre-existing relationship**
   ("diagnosed long before I bought this policy"). It was failing before this
   milestone and it fails now. The arithmetic case is fixed; the reading case is
   not, and two previous attempts at the reading case cost unrelated answers.
3. **`age` is still offered as a fact the reader did not mention, on policies
   with no age-dependent term in them.** Deciding which unstated facts could
   actually have changed a given verdict needs a link from each fact to the
   clauses that turn on it, which nothing in the pipeline currently produces.
   The false claim attached to the list is fixed; the list is not filtered.

### Check it yourself

Segmentation and scoring, no model needed:

```bash
cd api && python -m pytest -m "not llm" -q     # 264 tests
```

To see Failure 73 directly, segment a real wording and look for an unnumbered
clause whose text ends in a semicolon. There should not be one — it should have
become the opening of the clause below it.

The fact extractor, about twenty-five minutes and one fresh model call per
sentence:

```bash
python evals/check_fact_extraction.py
```

**A question to sit with before reading on.** The elided-quote check (Failure 76)
accepts a quotation assembled from several exact runs of one clause, in order,
each at least 25 characters, with material skipped between them. A reader might
reasonably object: *surely skipping material can change what a clause means, so
this check now passes quotations that mislead?*

<details>
<summary>Answer</summary>

It can, and it did before the change too. A contiguous quotation can stop
immediately before the sentence that qualifies it — truncation is elision with
the gap at the end — and truncation always passed. So the objection is really an
objection to what a substring check is, not to this change.

The useful move is to be precise about the question the check answers.
**Provenance** — did these words come from this clause? — is decidable by
looking, and is what the check has always done. **Meaning** — is the quotation
fair? — requires judging whether the skipped material mattered, which is a
reading. No substring test has ever been able to answer it, and one that appears
to is more dangerous than one that admits it cannot.

What the three constraints preserve is the part of contiguity that was doing
real work: pieces must be substantial, in document order, and non-overlapping,
so a condition cannot be lifted above the sentence qualifying it and nothing can
be stitched from scattered phrases. The elision remains visible in the quotation
the reader is shown, which is the honest place for the meaning question to be
settled.
</details>

---

# M19 — One clause list, built in one place

## The starting position

Stage 5, the scenario simulator, reasons over a list of `ShortlistClause`
objects: one per analysed clause, carrying the clause's text, its type, its
impact score, and the structured numbers extracted at analysis time (waiting
periods in days, co-pay percentage, room-rent caps, cover windows). Two
different pieces of code built that list:

- **The scenario endpoint**, `api/app/routers/scenarios.py`, read clauses and
  their analyses from SQLite and copied them into `ShortlistClause` one field
  at a time, decoding the JSON columns on the way:

  ```python
  ShortlistClause(
      clause_id=citation_id,
      ref=clause.id,
      number=clause.number,
      clause_type=analysis.clause_type,
      text=clause.text,
      impact_score=analysis.impact_score,
      waiting_periods_days=json.loads(analysis.waiting_periods_json or "[]"),
      exceptions=json.loads(analysis.exceptions_json or "[]"),
      copay_percent=analysis.copay_percent,
      # ... eight more fields ...
  )
  ```

- **The scenario eval**, `evals/run_scenario_eval.py` (`build_clauses`), ran the
  four pipeline stages in memory and copied the *in-memory* analysis objects
  into `ShortlistClause`, field by field, the same seventeen fields again.

Both built the same object from different sources, and nothing checked that
they agreed. Every time a field was added to the analysis (the co-pay age, the
ICU caps, the cover window all arrived this way) someone had to remember to add
it in both places. Forget the eval's copy, and the eval quietly measures a
system that differs from the one users get: the endpoint would reason with
the new field and the eval without it, and the eval's numbers would describe
neither.

This had already gone wrong once in a smaller way. The citation ids (the
policy's own clause numbers, made unique: `"10"`, `"10#2"`) used to be computed
separately too, and the two computations disagreed on repeated numbers. That
was fixed by pulling one function, `citation_ids`, out of both. The rest of
the constructor was never given the same treatment.

## Concept 49: a deep module, and the deletion test

A **module** here means anything with an interface and an implementation. That
could be a function, a class or a file. Its **interface** is everything a
caller has to know to use it correctly: the arguments, and also the ordering
rules, the error cases and the assumptions it makes.

A module is **deep** when a lot of behaviour sits behind a small interface.
It is **shallow** when the interface is nearly as complicated as what is
behind it. A shallow module moves complexity around. A deep one absorbs it.

The seventeen-field constructor calls were the opposite of deep. Each caller
had to know every field, where it lived in its source, and which ones needed
JSON decoding. The knowledge was spread across every place that built the list.

The **deletion test** is a quick way to tell the two apart. Imagine deleting
the module. If complexity disappears, it was a pass-through and the deletion is
an improvement. If complexity reappears in every caller, the module was earning
its keep. Apply it to the new function below: delete `load_clauses` and the
sorting, the filtering, the id assignment and all seventeen fields come back in
the endpoint *and* in the eval. That is a module worth having.

## The change

One function, `load_clauses`, in `api/app/pipeline/run.py`, next to the query
it builds on:

```python
def load_clauses(session: Session, doc_id: str) -> list[ShortlistClause]:
    """Every analysed clause of a document, as the scenario step receives it."""
    rows = sorted(
        ((c, a) for c, a in clause_rows(session, doc_id) if a is not None),
        key=lambda row: row[0].order_idx,
    )
    ids = citation_ids([(clause.number, clause.order_idx) for clause, _ in rows])
    return [ShortlistClause(clause_id=citation_id, db_id=clause.id, ...)
            for (clause, analysis), citation_id in zip(rows, ids)]
```

It owns four jobs that used to be spread across both callers:

1. sorting into document order (the query has no `ORDER BY`)
2. leaving out clauses the model failed on
3. assigning citation ids
4. decoding the JSON columns

The endpoint shrank to one call, plus the HTTP errors that are genuinely its
own job (404 for no such document, 409 for an unfinished one or one with
nothing analysed).

**The eval now goes through the database too.** `build_clauses` runs the real
`process_document`, the same function an upload triggers, against an
in-memory SQLite database, then reads the clauses back with `load_clauses`:

```python
engine = create_engine("sqlite://", poolclass=StaticPool)
SQLModel.metadata.create_all(engine)
# ... add a Document row ...
await process_document(doc_id, str(pdf), engine=engine)
with Session(engine) as session:
    ...
    return load_clauses(session, doc_id)
```

The eval now covers the save-and-read-back as well, which it never did before.
A field that is saved wrong, or decoded wrong, shows up in the eval's numbers.
`StaticPool` makes every connection share one in-memory database. Without
it, each new connection to `sqlite://` would get a fresh, empty database.

`process_document` never raises: in the app it runs as a background task with
nobody waiting, so it records a failure on the document row instead. In the
eval somebody *is* waiting, so `build_clauses` checks the row's status and
raises with the recorded error if the pipeline failed.

**Two smaller changes came with it:**

- **`ref` was dead.** It was a field on `ShortlistClause` that nothing in the
  repository ever read. The eval filled it with something different from the
  endpoint (`"synthetic-health-policy:12"` versus a database id), which looked
  like drift but harmed nothing. It is now `db_id` and is actually used.
- **The endpoint's `meta` dict went away.** It had built a separate lookup of
  `(db id, number, heading, page)` per citation id, to map the model's
  citations back to rows for display. `ShortlistClause` now carries `heading`
  and a 1-based `page` itself, so the endpoint looks citations up in the list
  it already has.

## Concept 50: a seam, and when one is real

A **seam** is a place where behaviour can change without editing the code at
that place. The `engine` parameter on `process_document` is one:

```python
async def process_document(
    doc_id: str, pdf_path: str, engine: Engine = engine
) -> None:
```

The upload endpoint passes nothing and gets the app's database. The eval
passes an in-memory one. Nothing inside `process_document` knows which one it
has.

A useful rule: **one implementation behind a seam is a hypothetical seam; two
is a real one.** A parameter that only ever receives one value is indirection
with no payoff. This one has two real callers passing two different
databases, so it earns its place.

### Why a parameter, and not an environment variable

The test suite already points the app at a throwaway database by setting
`DB_PATH` in `api/tests/conftest.py` before anything imports the app. The
same trick was considered for the eval and rejected, because of import order.
The app reads its settings **once, when `app.config` is first imported**, and
creates the database engine when `app.db` is first imported. Two of the eval
scripts, `evals/check_determinism.py` and `evals/real/measure_prompt.py`,
import app code *before* they import `run_scenario_eval`. By the time
`build_clauses` could set `DB_PATH`, the setting would already be read, and the
eval would silently write its test policy into the developer's real
`data/app.db`. A parameter has no import order.

## A reversed decision

The top of `api/app/pipeline/run.py` used to promise the opposite:

> none of them knows the database exists. That is what lets the eval harness
> run the identical pipeline with no web server and no database at all.

The first half is still true: `ingest`, `segment`, `analyze` and `score` are
still pure functions of their inputs, and still usable without a database. The
second half is deliberately reversed. Running the stages without a database
meant the eval had to build its clause list by hand, and that hand-built copy
was exactly the problem. The docstring now says why the eval goes through the
database on purpose.

## Failure 79: an architecture review that overstated its evidence

This change started from an automated architecture review of the repository.
The review reported that the endpoint and the eval had **"already drifted"**,
citing two differences:

- the `ref` values differed
- the endpoint kept clauses where `analysis is not None`, while the eval
  kept clauses present in both `analyses` and `scored`

Reading the code before changing it showed that neither was a live bug:

- **`ref` was never read**, so its two values could not disagree about anything.
- **The filters are equivalent.** The pipeline only saves an analysis row
  when the clause has *both* an analysis and a score:

  ```python
  analysis = analyses.get(key)
  sc = scored.get(key)
  if analysis is None or sc is None:
      continue
  ```

  So "has an analysis row" in the database means exactly "in `analyses` and in
  `scored`" in memory.

The honest case for the change was weaker and still sufficient. There was no
bug yet. There was a structure that made the next bug easy, and one fix of the
same shape (`citation_ids`) already on record. **A refactor's justification
should be checked against the code before it is acted on**, including when the
justification comes from a tool.

## Failure 80: thirty-six tests broken by a field's position

The first version put the new fields (`db_id`, `heading`, `page`) together
where `ref` had been, all three required. The loader's own test passed, and
the full suite then failed 36 tests with one error:

```
TypeError: ShortlistClause.__init__() missing 2 required positional arguments: 'text' and 'impact_score'
```

Many tests in `api/tests/test_scenario.py` build clauses positionally:

```python
ShortlistClause("2.4", "t:1", "2.4", "coverage", DAY_CARE_TEXT, 1.0)
```

That is `clause_id, ref, number, clause_type, text, impact_score`. Inserting
two required fields after the second position shifted every argument after it
by two, so `"coverage"` landed in `heading` and the call ran out of
arguments. A dataclass's constructor is positional by default, so **field
order is part of its interface**.

The fix kept the interface where callers already stood. `db_id` took `ref`'s
place as the second field, so the positional `"t:1"` now means a database id,
which is what it had always stood for. `heading` and `page` moved to the end,
with defaults (`""` and `0`), alongside the other optional fields.

## How it was verified

The test suite only covers pieces of this change. So the change was also
checked end to end, against the real golden policy:

1. **Before touching any code**, the old `build_clauses` was run and its output
   saved as JSON. The fields that were being renamed or added were left out.
2. **After the change**, the new `build_clauses` was run the same way.
3. The two files were compared **byte for byte: identical**. All 40 clauses
   had the same values in the same order. The second run made zero model calls
   (every analysis came from the cache), so any difference would have been the
   code's, not the model's.

Also checked: no `data/app.db` was created by the eval run, and the whole
non-model test suite passes (265 tests).

A new test, `api/tests/test_load_clauses.py`, seeds an in-memory database and
pins what the loader promises:

- clauses come back in document order even when inserted out of order
- a repeated clause number becomes `"10#2"`
- a clause the model failed on is left out
- the JSON columns come back as lists
- pages are 1-based

## What is still not covered

The endpoint's success path (calling `run_scenario`, then mapping citations
back to rows and saving the `ScenarioRun`) still only runs under the live
model test. That test is skipped by `-m "not llm"`. The loader made that path
much smaller, but testing it without a model would need a way to substitute
the model in `run_scenario`, which is separate work.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_load_clauses.py -v
.venv/Scripts/python -m pytest -q -m "not llm"
```

**Predict before you look:** in `tests/test_load_clauses.py`, the clauses are
added to the database in the order 2, 1, 0, and clause 2 has no analysis.
Clauses 0 and 1 are both numbered `"10"`. Which clause gets `"10"` and which
gets `"10#2"`, and why can't the answer depend on the insert order?

<details>
<summary>Answer</summary>

Clause 0 gets `"10"` and clause 1 gets `"10#2"`. `load_clauses` sorts by
`order_idx`, the clause's position in the document, before assigning ids, and
`citation_ids` gives the unsuffixed number to the first occurrence in document
order. The order rows were inserted in, or the order SQLite happens to return
them without an `ORDER BY`, never reaches the id assignment. That matters
because the ids are what the model cites: if they depended on storage order,
the same policy could give the same citation two different meanings.
</details>

---

## M19, part 2: one duration rule, one meaning of "stated"

### The starting position

The scenario simulator's first step asks the model to read the person's
question into a flat dictionary of facts (`FACTS_SCHEMA` in
`api/app/pipeline/scenario.py`). Durations come back as two fields each, a
number and a unit, because converting units is arithmetic and the project keeps
arithmetic out of the model:

```
time_since_policy_start_value: 14      time_since_policy_start_unit: "months"
expense_timing_value: 120              expense_timing_unit: "days"
condition_known_for_value: 1           condition_known_for_unit: "months"
```

Python then turns each pair into days. It did that in **three functions with
the same four lines**, differing only in which keys they read:

```python
def policy_age_days(facts):
    value = facts.get("time_since_policy_start_value")
    unit = facts.get("time_since_policy_start_unit")
    if not value or value <= 0 or unit not in _DAYS_PER_UNIT:
        return None
    return value * _DAYS_PER_UNIT[unit]

def expense_offset_days(facts): ...     # same, "expense_timing_*"
def condition_known_days(facts): ...    # same, "condition_known_for_*"
```

A second rule was copied too: **what counts as a fact the person actually
gave.** Three places answered it, each with its own tuple:

- `run_scenario` listed the missing decisive facts: `facts.get(k) in (None, "", "unknown")`
- the reasoning prompt listed them again the same way, and listed the known
  facts with a *different* tuple: `v not in (None, "", [], "unknown")`
- the scenario endpoint filtered the facts it shows the reader: `v not in (None, "", "unknown")`

Three copies of a rule are three chances to change one and not the others. A
wrong duration conversion here changes which waiting periods have been served,
and a wrong "missing" list tells a reader the system needs information they
already gave.

### The change

One conversion, parameterised by the name every duration already shares:

```python
def duration_days(facts: dict[str, Any], name: str) -> int | None:
    value = facts.get(f"{name}_value")
    unit = facts.get(f"{name}_unit")
    if not value or value <= 0 or unit not in _DAYS_PER_UNIT:
        return None
    return value * _DAYS_PER_UNIT[unit]
```

One meaning of "stated", and the missing list built from it:

```python
def stated(value: Any) -> bool:
    return value not in (None, "", [], "unknown")

def missing_facts(facts: dict[str, Any]) -> list[str]:
    return [k for k in DECISIVE_FACTS if not stated(facts.get(k))]
```

`run_scenario` and the prompt call `missing_facts`; the prompt and the
endpoint call `stated`. The tuples differed only by `[]`. No extracted fact
is ever a list (the facts schema contains no arrays), so the merge changes
nothing, and the byte-for-byte check below confirms it.

### What was considered and rejected

A **`Facts` class** (a value object wrapping the dictionary, with the derived
numbers as properties) was the first idea. It fails the deletion test
(Concept 49). Most readers of the facts need raw keys:

- the reduction check reads `sum_insured_value`, `room_is_icu` and `age`
- the prompt prints every stated fact
- the endpoint returns the dictionary
- the eval compares it against an answer key

So each of them would reach through the class to the dictionary inside it. A
class that its callers have to see through is not hiding anything.

A **typed class with one field per fact** was rejected faster. It would be a
21-field list maintained by hand beside `FACTS_SCHEMA`, which is exactly the
kind of parallel list the first half of M19 removed.

### A known gap, deliberately left open

The architecture review that proposed this change also pointed at a
behaviour. `derive_pre_existing` can settle, by comparing two durations, that
a condition began *after* the policy did. But the waiting-period check never
receives that answer. Its line for a pre-existing-diseases clause still says:

```
concerns ONLY an illness the person already had when the policy began.
Unless the description says this one had begun by then, it is not the
reason for this claim.
```

This is true but hedged, when code already knows the answer. Passing the
derived fact into `waiting.evaluate` would change the prompt, and a prompt
change in this project has to be measured, not assumed. Before building it,
the latest scenario report was checked for a wrong answer this could explain.
None of the six verdict misses is one. The only pre-existing-condition miss,
`ped-waiting-served`, is a condition that *was* pre-existing, answered by
citing the cosmetic-surgery exclusion. So the change was not made. It becomes
worth making when a case shows the hedge costing an answer.

### How it was verified

A refactor that promises "no behaviour change" should be checked on the
behaviour, not just the tests:

1. **Before any edit**, the reasoning prompt was rendered for every scenario
   case: the 40 main cases and both held-out batches, 69 in all. The script
   used the real facts extracted for each case and the real computed
   waiting-period, reduction and window blocks. It saved each prompt with its
   missing-facts list.
2. **After the edit**, the same script ran again.
3. The two files were **byte-for-byte identical**. The second run made zero
   model calls, so the facts were the same stored answers both times.

The whole non-model test suite passes (265). The existing tests of the
policy-age conversion now call `duration_days` through
`partial(duration_days, name="time_since_policy_start")`, so their assertions
did not change.

### Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_scenario.py -q -k "policy_age or pre_existing or not_stated"
```

**Predict before you look:** `duration_days({"expense_timing_value": 0,
"expense_timing_unit": "days"}, "expense_timing")` — is the answer `0` or
`None`, and why does the difference matter for a waiting period?

<details>
<summary>Answer</summary>

`None`. A value of zero is treated as unstated (`not value` is true for 0).
For the policy's age this is deliberate: nobody claims on a policy held for
zero days, so a 0 is far more likely to be the model filling a slot than the
person saying it. And a `None` makes every waiting period come back UNKNOWN,
which can lead to an honest "insufficient information". A 0 would mark every
waiting period as not served and refuse the claim on a number nobody said.
</details>

---

# M20 — The reasoning request, built in one place

## The starting position

The scenario simulator's reasoning step sends the model two things:

- **the messages**: a system prompt, then a user prompt listing the
  situation, the facts, the computed blocks and every clause under its id
  (`### clause_id=4.1 ...`)
- **the schema** the answer must fit, where the `clause_id` field is an
  `enum` of exactly those ids

That pairing is the project's central grounding mechanism. Ollama enforces
the enum while decoding, so a citation of a clause that is not in the prompt
cannot be generated at all (see the entries on constrained decoding). **But
this only holds if the prompt and the enum are built from the same clause
list.** Build them from two lists and the guarantee quietly lapses.

Inside the app they were built together, in `reason()` in
`api/app/pipeline/scenario.py`. Two problems sat around it:

1. **Code outside `reason()` rebuilt the pair by hand.** `evals/check_determinism.py`
   imported the private `_reasoning_schema` and the prompt renderer
   separately, in two places, and assembled its own messages.
2. **The prompt module imported from the pipeline that imports it.**
   `render_reasoning_request` in `api/app/llm/prompts.py` did its own lookups
   (which facts are stated, which are missing, which words the question
   shares with a clause) by importing them from `scenario.py`, inside the
   function body:

   ```python
   from app.pipeline.scenario import missing_facts, shared_words, stated
   ```

## Concept 51: a circular import, and why it was hidden inside a function

Python runs a module's top-level code the first time it is imported, and
the module's names only exist once that run finishes. `scenario.py` imports
`prompts.py` at the top. If `prompts.py` also imported `scenario.py` at the
top, loading either one would start loading the other before the first had
finished. One of them would then look for a name that doesn't exist yet,
and fail with `ImportError: cannot import name ... (most likely due to a
circular import)`.

Putting the import **inside the function** avoids the crash, because it only
runs when the function is called, and by then both modules have finished
loading. So it works. But it is a sign of a design problem rather than a
fix. The two modules depend on each other, so neither can be understood,
changed or tested on its own. Here the cause was that the prompt module was
doing two jobs: formatting text, and deciding things (what counts as
stated, which clauses share words) that belong to the pipeline.

## The change

**One public function returns the pair.** It lives in `scenario.py`:

```python
def reasoning_request(scenario, facts, clauses, computed, *, nudge=None):
    """The messages for the reasoning step, and the schema its answer must fit."""
    known = {k: v for k, v in facts.items() if stated(v)}
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user", "content": render_reasoning_request(
            scenario, known, missing_facts(facts),
            shared_words(scenario, clauses, facts), clauses,
            waiting.render(computed.waiting), ...)},
    ]
    if nudge:
        messages.append({"role": "user", "content": nudge})
    return messages, _reasoning_schema([c.clause_id for c in clauses])
```

`reason()` shrank to building that pair and sending it:

```python
messages, schema = reasoning_request(scenario, facts, clauses, computed, nudge=nudge)
return await client.complete_json(messages, schema, use_cache=use_cache)
```

**The prompt module only formats.** `render_reasoning_request` now takes the
known facts, the missing list and the shared words as arguments. The
decisions are made in `scenario.py` and arrive as data. `prompts.py` imports
nothing from the pipeline, so the cycle is gone rather than hidden.

**The evals and tests call the same function.** `check_determinism.py`
builds its requests through `reasoning_request` and no longer imports the
private `_reasoning_schema`. Six tests that checked the prompt text by
calling the renderer directly now go through a small helper that calls
`reasoning_request`, which is the interface the pipeline uses.

## Failure 81: a docstring that said the check measured what the eval sends

While moving `check_determinism.py` onto the new function, one of its
docstrings turned out to be wrong. The **sequence check** runs the first few
eval questions in order, twice, and compares the answers. Its docstring
said:

> That is exactly the operation `run_scenario_eval.py` performs, and exactly
> the property its numbers depend on.

The requests it built looked like this:

```python
render_reasoning_request(case["scenario"], {}, considered, "", "")
```

Empty facts and empty computed blocks: no waiting-period, window or
reduction lines, and no list of what the person did not state. The eval's
real prompts carry all of those. So the check measured whether *shorter*
prompts were stable, and the docstring claimed it measured the eval's.

This is the same kind of problem as the rest of M19 and M20: a second copy
that drifted from the first while its description stayed the same. Nothing
tested it, because a docstring cannot fail.

**What was done, and what was not.** The check now builds its requests with
`reasoning_request`, passing empty facts and an empty `Computed`. That
reproduces the old prompts exactly, and the docstring now says plainly that
these are not the eval's prompts. Making the check send the eval's real
prompts was deliberately left for later. It changes *what is measured*, so
its results would stop being comparable with earlier runs, and it needs a
live run to set a new baseline. Changing how code is built and changing
what an eval measures belong in separate steps, so that each can be checked
on its own.

## How it was verified

The model call was intercepted, so nothing was sent to Ollama, and every
request the code *would* send was recorded:

- for each of the 69 scenario cases (the main set and both held-out
  batches): the reasoning request, with and without a retry nudge (138
  requests)
- both determinism checks: the single-prompt model check and the sequence
  check over three cases, two passes (7 requests)

That recording was made once before the change and once after. The two
were **byte-for-byte identical**: every system prompt, user prompt and schema.
The whole non-model test suite passes (265).

## Check it yourself

```bash
grep -n "app.pipeline" api/app/llm/prompts.py     # prints nothing: no cycle
cd api && .venv/Scripts/python -m pytest tests/test_scenario.py -q
```

**Predict before you look:** suppose someone later adds a clause filter
*after* the prompt is rendered but *before* the schema is built. For
example, dropping definitions from the enum because "nobody cites a
definition". What becomes possible that was impossible before, and which
check would still catch it?

<details>
<summary>Answer</summary>

The prompt would show clauses the enum no longer allows. The model could
still *read* a definition, but could no longer cite it, so an answer that
rests on one would be pushed to cite something else. The reverse filter is
worse. An enum listing clauses the prompt doesn't show lets the model cite
a clause it was never given, which is the exact failure the enum exists to
prevent. In that case the verbatim quote check (`app/grounding.py`) is the
remaining defence: a quote from a clause the model never saw will almost
never match its stored text, so the answer is flagged unverified rather
than shown as fact. That is why the pair is built in one function. The
second check should not have to catch what the first was supposed to make
impossible.
</details>

---

# M21 — Another policy covering the same claim

## The starting position

One case in the scenario eval's main set, `other-policy-contribution`, asks:

> "I hold another health policy with a different insurer and I have sent the
> same hospital bill to both. I have held this policy four years."

The synthetic policy answers it in clause 6.6:

> **6.6 Contribution.** If at the time of a claim the Insured Person holds any
> other policy of indemnity covering the same risk, the Company shall not be
> liable to pay more than its rateable proportion of the claim, and the Insured
> Person shall disclose particulars of such other policy to the Company.

So the expected verdict is `conditional`: the claim is paid, but this policy
pays only its share. The case must cite 6.6.

The run history (`evals/run-history.json`, which records every eval run
locally) showed the case answered `conditional`, then `covered`, then
`conditional` on Ollama 0.34.2. On Ollama 0.35.0 it was `insufficient_information`
five times out of five. It looked like a runtime upgrade had broken it.

The history said otherwise. Even in the runs where the verdict was
`conditional`, the report listed the case as citing `2.1 (missing 6.6)`.
**Clause 6.6 had never been cited, on any runtime.** The right verdicts had been
right for the wrong reason, and the new runtime only made the miss visible as a
wrong verdict.

## Reading what the model was given

Before changing anything, the actual request was rebuilt and printed. Three
findings:

1. **Clause 6.6 was in the prompt.** The synthetic policy's 40 clauses fit in
   the context window whole, so nothing was cut.
2. **Nothing pointed at it.** The prompt has a "WORDS THIS QUESTION SHARES WITH
   A CLAUSE" section (Concept 35), which lists unusual words the question and a
   clause have in common. The question says "another health policy",
   "different insurer", "same hospital bill"; the clause says "other policy of
   indemnity", "same risk", "rateable proportion", "Contribution". The only
   shared word is "policy", which is everywhere in the document, so nothing was
   listed.
3. **The missing-facts list pulled the model off course.** The prompt's "NOT
   STATED" section listed age, pre-existing condition, hospitalised and hours
   since admission, none of which matter here. The model answered:

   > "...the situation does not specify whether the claim is for a pre-existing
   > disease, cosmetic surgery, or any other excluded treatment. Additionally,
   > the policyholder has sent the same hospital bill to another insurer, which
   > may affect the claim..."

   It noticed the other insurer and never connected it to 6.6.

## The runtime moved twice in a week

This investigation crossed two automatic Ollama updates: 0.34.2 to 0.35.0, then
0.35.0 to 0.35.1. Two consequences, both following from Concept 45 (the runtime
is part of the model):

- **The response cache emptied itself, in effect.** The cache key includes the
  Ollama version, so every stored answer from the old version is a miss. The
  first sign was a "quick" check making 40 live model calls.
- **A comparison spread across days can mix a code change with a runtime
  change.** So every measurement in this milestone was taken on one version,
  0.35.1, back to back: the old code first, the new code immediately after.

## Test cases written before the fix

A fix designed while staring at one case is tuned to that case (Concept 36).
So before any code was written, a third held-out batch was added,
`evals/golden/scenarios-heldout-3.json`, with three cases:

| Case | Situation | Expected |
|---|---|---|
| `ho3-group-cover-shares-bill` | employer's group insurance paid part, claiming the rest here | `conditional`, cite 6.6 |
| `ho3-two-policies-appendix` | two policies, claiming an appendix operation on both | `conditional`, cite 6.6 |
| `ho3-switched-from-old-insurer` | left an old insurer last year, no longer holds it | `covered`, must **not** cite 6.6 |

The first two say the same thing as the main-set case in different words. The
third mentions another insurer but must not bring in the clause, because 6.6
applies only to another policy held "at the time of a claim". These were never
used to choose or adjust the fix.

## The real policies say something different

The fix needed code to find "the clause about another policy" in any policy, not
just the synthetic one. Before writing that pattern, it was run over the three
real insurer wordings the project measures against (reading and segmenting
only, no model). None of them has a "Contribution" clause. All three carry
IRDAI's standard **"Multiple Policies"** clause (Star's clause 4, Niva's
6.1.11, HDFC's inside one oversized segment). That clause says something
different: the insured person may choose which policy settles the claim first.
It does not limit the insurer to a share.

This shaped the design. Code can reliably tell *which* clause is about another
policy, because the subject is named in the wording. It cannot reliably tell
*what that clause decides*, because insurers decide it differently. So the
line code adds names the clause and leaves the reading to the model. This is
the project's usual division of labour: code does the lookups, the model reads
the language.

## The first version: a new fact

**A new fact.** `FACTS_SCHEMA` in `api/app/pipeline/scenario.py` gained one
nullable yes/no field, `another_policy_covers_this_claim`, and the extraction
prompt (`FACTS_SYSTEM` in `api/app/llm/prompts.py`) a short section:

```
ANOTHER POLICY: another_policy_covers_this_claim is true only when the person
says they hold another health policy NOW that covers this same claim. A policy
they used to have, and have left, is false. Nothing said about other insurance
is null.
```

**A pattern for the clause**, matching both wordings by their shared subject:

```python
_ANOTHER_POLICY = re.compile(
    r"\brateable\b|\bcontribution\b|\b(?:other|multiple) (?:polic(?:y|ies)|insurance)\b",
    re.IGNORECASE,
)
```

It matches only 6.6 in the synthetic policy and exactly one clause in each real
one.

**A line in the reasoning prompt**, only when the person said so and a clause
matched:

```
ANOTHER POLICY ALSO COVERS THIS CLAIM (the person said so).
- clause 6.6 sets what happens when another policy covers the same claim.
  Read it: it decides how this claim is shared between the policies.
```

It sits after the payment-reduction section, because like those it bears on how
much this policy pays rather than whether the claim is payable. The same clause
ids are added to `named_by_code`, so on a policy too long to show whole, the
clause-picking step can never drop it.

`PROMPT_VERSION` became `v50-another-policy`.

## Failure 82: the target fixed, the new cases not

Single runs, both on Ollama 0.35.1:

| | old code (v49) | new code (v50) |
|---|---:|---:|
| main set, verdicts right | 32/40 | 31/40 |
| held-out batch 1 | 0.750 | 0.688 |
| held-out batch 2 | 0.923 | 1.000 |
| held-out batch 3 (new) | 0/3 | 0/3 |
| citation recall | 0.688 | 0.812 |
| made-up quotes | 0 | 0 |

Resampled five times, the target case went from wrong 5/5 to **right 5/5,
citing 6.6**. The mechanism works.

The new held-out batch stayed at 0 of 3, for three different reasons, two of
which have nothing to do with this change:

- **`ho3-group-cover-shares-bill`: the fact was not read.** "My employer's group
  health insurance paid part of my hospital bill" came back with
  `another_policy_covers_this_claim: null`, so the line never appeared. The
  fact-extraction check (`evals/check_fact_extraction.py`) found the same: of
  four new sentences written for the field, "my old policy lapsed" and "no
  other insurance" were read right, and both phrasings of "two insurers, one
  bill" were missed in at least one run.
- **`ho3-two-policies-appendix`: the line worked, the answer was still wrong.**
  The model wrote "the fact that another policy covers this claim (clause 6.6)
  does not affect the outcome", then refused the claim under the
  pre-existing-disease waiting period (3.2) for an appendix operation nobody
  described as pre-existing.
- **`ho3-switched-from-old-insurer`: the fact was read right** (old policy
  lapsed, so no line), and the model again applied the pre-existing-disease
  waiting period, this time to dengue.

The last two are the weakness recorded at the end of M19 part 2: the
waiting-period line for the pre-existing-diseases clause hedges, and the model
over-applies that clause. M19 left it open "until a case shows it costs an
answer". Two cases now do.

## Failure 83: three breaks, and what actually caused them

In the single runs, three cases that are normally right came back wrong under
the new code: `cosmetic-exclusion`, `no-preauth-cashless` and
`ayush-private-clinic`. Resampling settled whether they were real. On the old
code, all three were right **five times out of five**. On the new code they
were wrong four or five times out of five.

**None of these cases mentions another policy, so the new line never
appeared.** The structured facts extracted for them were identical between old
and new code. One thing did differ: the free-text `notes` field.

| Case | `notes` under the old prompt | `notes` under the new prompt |
|---|---|---|
| cosmetic-exclusion | "Procedure was for aesthetic reasons, not due to a pre-existing condition." | "The procedure was a nose job, but there is no mention of when the policy started, the person's age, the cost..." |
| ayush-private-clinic | "treatment at a private clinic that is not government-run and has no Quality Council accreditation" | "Private clinic, not government-run, no Quality Council accreditation" |
| no-preauth-cashless | (the question, ending "I have held the policy four years.") | (the same, without the last sentence) |

To test whether the notes caused the flips, the new code was run with each
case's facts held fixed and only `notes` swapped, three fresh samples each:

| Case | new notes | old notes | no notes |
|---|---:|---:|---:|
| ayush-private-clinic | **0/3** | 3/3 | 3/3 |
| cosmetic-exclusion | 3/3 | 3/3 | 3/3 |
| no-preauth-cashless | 3/3 | 3/3 | **0/3** |

Two different results:

- **`ayush-private-clinic` is a genuine regression, and the notes caused it.**
  The same facts with the old wording are right every time; with the new
  wording, wrong every time.
- **`cosmetic-exclusion` and `no-preauth-cashless` are right every time with
  the new notes when asked on their own**, yet were wrong inside the eval run.
  What differed was the questions asked *before* them. Concept 26 and
  Concept 27 explain why that matters: the runtime reuses the cached state of
  a matching prompt prefix and batches work differently depending on what came
  just before, so a near-tie can break differently. These two are fragile
  cases sitting on a near-tie, not cases this change damaged.

## Concept 52: a free-text field is a side channel

Every structured fact the extractor returns is checked: an enum, a nullable
integer, a yes/no. `notes` is different. It is free text, kept because people
sometimes say things no field captures, and the reasoning prompt shows it
under "FACTS UNDERSTOOD" like any other fact.

A free-text field is a **side channel**: a path information takes that no
schema constrains. Two consequences showed up here:

1. **Any change to the extraction prompt rewrites it for every case.** Adding
   one field about other insurance changed the wording of `notes` on questions
   about nose jobs and Ayurveda. Nothing tested those words, because nothing
   can: there is no right answer for a paraphrase.
2. **The reasoning step is sensitive to its wording.** "a private clinic that
   is not government-run" and "Private clinic, not government-run" mean the
   same thing to a person, and produced opposite verdicts here, every time.

This is Concept 48 (a prompt is not a list of independent rules) arriving
through a side door. Concept 48 is about an instruction for one field competing
with the others. Here, the instruction changed nothing structured and still
moved unrelated answers, through the one field nothing checks.

## Measured properly

Single runs could not settle whether the first version helped, so both
versions were measured by majority of three (Concept 34): each case asked
three times and scored by its most common verdict. The old code ran first,
then the new, back to back on Ollama 0.35.1:

| | old code (v49) | first version (v50) |
|---|---:|---:|
| main set | **34/40** | **30/40** |
| held-out batch 1 | 12/16 | 12/16 |
| held-out batch 2 | 12/13 | 13/13 |
| held-out batch 3 | 0/3 | 0/3 |
| total | **58/72** | **55/72** |

On the main set the first version fixed two cases (`other-policy-contribution`,
`breach-of-law`) and broke six (`ped-waiting-served`, `cosmetic-exclusion`,
`copay-unknown-inception-age`, `no-preauth-cashless`, `ayush-private-clinic`,
`day-care-not-listed`). **It made the system worse overall.** Inside the eval
run, `cosmetic-exclusion` and `no-preauth-cashless` were wrong in the majority
even though they were right when asked on their own. The eval run is what's
measured, so they count.

The route the damage took is the one Failure 83 proved for `ayush-private-clinic`:
the extraction prompt changed, so `notes` changed for every question, so
answers to questions that never mentioned another policy moved.

## The rebuild: a lookup instead of a fact

The fix was kept and the side channel closed. The person's mention of a second
policy is now found **in code, from their own words**, and the extraction
prompt is back to exactly what it was. This is the same move the project
already makes in `correct_misfiled_age`, which checks the person's words for a
mention of the policy starting rather than asking the model.

The pattern in `api/app/pipeline/scenario.py`:

```python
_MENTIONS_ANOTHER_POLICY = re.compile(
    r"\b(?:another|other|second)\s+(?:health\s+)?(?:insurance\s+)?"
    r"(?:polic(?:y|ies)|insurers?|insurance|mediclaim)\b"
    r"|\b(?:two|both|multiple)\s+(?:health\s+)?(?:insurance\s+)?"
    r"(?:polic(?:ies|y)|insurers|insurance companies|companies)\b"
    # Employer and group cover: the usual second policy in India.
    r"|\b(?:group|corporate|company|employer'?s?|office)\s+(?:health\s+)?"
    r"(?:insurance|cover|policy|mediclaim)\b",
    re.IGNORECASE,
)
```

and a sentence-level exclusion, so a policy the person no longer holds, or
never had, does not count:

```python
_NOT_HELD_NOW = re.compile(
    r"\b(?:no|not|never|don't|no longer|used to|previous|old|lapsed?|left|dropped"
    r"|switched|cancell?ed)\b",
    re.IGNORECASE,
)
```

**The pattern was checked against every question in the eval first.** A looser
draft also matched "two weeks after my policy" and "two years into the
policy", which would have put the line into two unrelated questions. The
final version matches exactly three of the 72 questions: `other-policy-contribution`
and the two held-out batch-3 cases about a second policy.

One honest caveat: by then the batch-3 wording had been seen, so the pattern is
not a clean test against those two cases. It was written from general
phrasings (employer and group cover is the usual second policy in India), not
from their exact words.

Because the line is now a lookup, it is labelled as one (Concept 35):

```
THE PERSON MENTIONS ANOTHER POLICY (a lookup on their words, not a judgement).
- clause 6.6 sets what happens when another policy covers the same claim.
  Read it: it decides how this claim is shared between the policies.
```

`PROMPT_VERSION` became `v51-another-policy-lookup`.

### Isolation, proven rather than hoped for

Every request the pipeline sends was recorded for all 72 eval questions on the
old code and on the rebuilt code, with the model call intercepted:

```
cases with any different request: 3
cases whose FACT request differs: 0
```

This has a consequence worth stating carefully. The response cache keys on the
exact request (Concept 29). For the 69 unchanged questions, the rebuilt code
sends **byte-identical requests**, so the measurements already taken on the old
code are measurements of the rebuilt code too. Only the three changed questions
needed new samples, and they got five each.

## The rebuild, measured

| Case | old code | rebuilt (v51), 5 samples |
|---|---|---|
| `other-policy-contribution` | covered, 2 of 3 | **conditional 5/5, cites 6.6** |
| `ho3-group-cover-shares-bill` | covered | covered 5/5, does not cite 6.6 |
| `ho3-two-policies-appendix` | not_covered | not_covered 5/5, does not cite 6.6 |

| | old code | first version | rebuilt |
|---|---:|---:|---:|
| main set | 34/40 | 30/40 | **35/40** |
| held-out batches 1 / 2 / 3 | 12/16 · 12/13 · 0/3 | 12/16 · 13/13 · 0/3 | 12/16 · 12/13 · 0/3 |
| total | 58/72 | 55/72 | **59/72** |

What this does and does not show:

- **It cannot cost an answer to a question that doesn't mention a second
  policy.** The request comparison proves that; it isn't inferred from scores.
- **The case it was built for is fixed, reliably.**
- **It did not carry over to new wording: 0 of 2.** In both held-out cases the
  line is now in the prompt and the model still does not cite 6.6. In
  `ho3-two-policies-appendix`, the pre-existing-disease waiting period also gets
  in the way, as in Failure 82.

It was merged on those terms: a small, contained gain on a case it was tuned
against, with no evidence yet that it generalises. Why the model reads the line
and still ignores the clause is the open question it leaves.

## Closing out M21

### How it connects to what was already there

M21 has the same shape as the project's earlier fixes: the model reads the
language, and code finds the clause and puts it in front of the model.
Waiting periods, reductions and cover windows all work this way. What M21
added is a lesson about **where** the reading happens. Asking the fact
extractor for one more field looked like the obvious place, and it was the
expensive one, because that prompt feeds a free-text field into every
question. Reading the person's words with a fixed pattern did the same job
without touching anything else.

### What it made possible

The request comparison is reusable. Any change that should touch only some
questions can be checked the same way, before any model time is spent: record
every request on old and new code, and confirm that only the intended questions
differ. Where that holds, the unchanged questions don't need re-measuring at
all.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_scenario.py -q -k "another_policy or shared_with_another"
```

**Predict before you look:** a person writes "My wife's office mediclaim
covers me too. Can I claim the rest of the bill from this policy?" Will the
line appear in the prompt? Now try "I used to have another policy but it
lapsed." Which part of the code decides each one?

<details>
<summary>Answer</summary>

The first one fires. "office mediclaim" matches the employer-and-group branch
of `_MENTIONS_ANOTHER_POLICY`, nothing in the sentence matches `_NOT_HELD_NOW`,
and the policy has a clause the `_ANOTHER_POLICY` pattern finds, so the line
names it. The second does not: "another policy" matches, but the same sentence
contains "used to" and "lapsed", so `_NOT_HELD_NOW` excludes it.

The exclusion works sentence by sentence, and that is also its weakness. "I had
another policy. It lapsed last year." splits into two sentences, so the first
one matches with nothing to exclude it, and the line appears for a policy the
person no longer holds. A lookup is cheap and predictable, and its mistakes are
predictable too, which is why its tests list the phrasings it must not match.
</details>

---

# After M21 — Testing the scenario endpoint without a model

## The starting position

The scenario endpoint, `create_scenario` in `api/app/routers/scenarios.py`,
does real work around the model's answer:

- loads the document's clauses (`load_clauses`, from M19)
- calls `run_scenario`
- turns each citation back into a database row, heading and page
- saves the run as a `ScenarioRun` row
- hides facts the person never stated before sending the response

Until now, the only test that reached that work was
`test_scenario_end_to_end_is_grounded`. That test needs a live model, so it is
marked `@pytest.mark.llm` and skipped by the everyday `pytest -m "not llm"`
run. The other endpoint tests cover only the error paths (404, 409, 422). So a
bug in, say, the page arithmetic would pass the everyday suite.

## The change, and the bigger change it replaced

An architecture review of the codebase had proposed something larger: one
shared "model" interface that every test could plug a fake into, replacing
four different ways the tests stub the model today. Looked at closely, two of
those stubs live in one test file, and the other two test the model client
itself, so they have to sit where they are. A shared interface would have
added a layer across the codebase to replace two local fakes. The gap that
mattered was narrower: the endpoint's own work was untested.

So the new test, `test_scenario_answer_is_mapped_back_to_the_policy_and_stored`
in `api/tests/test_api.py`, replaces `run_scenario` itself with a fixed answer:

```python
async def fixed_answer(question, clauses):
    received.extend(clauses)
    return ScenarioResult(
        verdict="not_covered",
        reasoning="Cosmetic surgery is excluded.",
        citations=[Citation("4.1", "shall not be liable", "denies")],
        facts={"procedure": "nose job", "age": None, "pre_existing_condition": "unknown"},
        missing_facts=["age"],
        clauses_considered=len(clauses),
    )

monkeypatch.setattr(scenarios, "run_scenario", fixed_answer)
```

Replacing `run_scenario`, rather than the model underneath it, keeps this a
test of the endpoint and nothing else. Against the seeded test database it
checks that:

- all four analysed clauses reached `run_scenario`, in document order
- the citation came back as clause `4.1`, its database row, its heading, and
  page 1 (pages are stored counting from 0)
- the facts the person never gave (`None`, `"unknown"`) were not shown back
- the run was saved with its verdict and citations

## Seen failing, on purpose

A new test that passes the first time it runs has only shown that it doesn't
fail on correct code. It hasn't shown that it can fail at all. So the endpoint
was broken deliberately, twice, with the test run after each:

1. the page counted from 0 instead of 1: the test failed, `assert 0 == 1`
2. the filter that hides unstated facts removed: the test failed

Both breaks were then undone. A test that can't fail is decoration. This is the
same rule as "each test was seen failing first", applied after the fact
instead of before.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_api.py -q -k mapped_back
```

**Predict before you look:** if `load_clauses` started returning clauses in
the order SQLite happened to store them, which assertion in this test would
catch it, and why does the order matter beyond this test?

<details>
<summary>Answer</summary>

`assert [c.clause_id for c in received] == ["2.1", "3.2", "4.1", "5.1"]`.
The fake records exactly which clauses reached `run_scenario`, in order.
Beyond the test, order decides which repeat of a clause number gets the plain
id and which gets the `#2` suffix (M19), and it decides the order the clauses
appear in the prompt. The model tends to cite the first relevant clause it
reads, so the same policy could produce different answers if the order
drifted.
</details>


---

# M22 — Whether an illness began before the policy

## The starting position

Indian health policies carry a **pre-existing disease waiting period**. In the
synthetic policy it is clause 3.2: an illness the person already had when the
policy began is not covered until the policy has been held for 36 months. An
illness that began *after* the policy started is not touched by it at all.

So whether the condition predates the policy decides many claims, and the
scenario pipeline has a fact for it: `pre_existing_condition`, one of `"yes"`,
`"no"` or `"unknown"`, filled in by the fact-extraction step (stage 5a, an LLM
call with a schema). Two eval cases showed that fact going wrong while the
person had said the answer plainly:

> `ho-copay-derived-age`: "I am 66 and have held this policy for two years. I
> was admitted for a kidney stone, **which I never had before taking the
> policy**."

> `ho-knee-replacement-served`: "I needed a knee replacement thirty months
> after my policy began. **The knee trouble only started after I took the
> policy out.**"

Both came back `pre_existing_condition: "unknown"`. Both claims were then
refused under clause 3.2, which had not been served (two years, and thirty
months, are both short of 36).

The obvious fix was to make the extraction prompt better at this field. Two
earlier milestones (M18 and M21) had shown why that is expensive: the
extraction prompt also writes a free-text `notes` field for every question,
and any change to the prompt rewords those notes for questions that have
nothing to do with the change. In M21 that cost four unrelated answers
(Concept 52, *a free-text field is a side channel*). M21's answer was to leave
the prompt alone and read the person's words with a fixed pattern instead.
M22 does the same.

## A held-out batch, written before the code

Before any code, a fourth held-out batch was written:
`evals/golden/scenarios-heldout-4.json`, six cases that say the same kinds of
thing in other words, plus traps:

| Case | Says | Expected |
|---|---|---|
| `ho4-thyroid-after-cover` | "diagnosed with a thyroid problem after I took out this cover", 20 months held | covered |
| `ho4-asthma-since-childhood` | asthma "well before this policy", two years held | not_covered (3.2) |
| `ho4-gallstones-found-last-year` | found last year, policy held two years | covered |
| `ho4-back-trouble-on-and-off` | history, no dates | insufficient_information |
| `ho4-before-buying-checked-cover` | "Before I bought this policy I checked..." (a trap: no illness) | covered |
| `ho4-acute-no-history` | appendicitis, nothing about history | covered |

The last case measures a problem M22 deliberately does not try to fix: an
illness with no history mentioned, refused under 3.2 anyway. On the code
before M22 (prompt version v51), majority of three samples, this batch scored
**1 of 6**; only the asthma case was right.

## The lookup

In `api/app/pipeline/scenario.py`:

```python
# The start of the policy, as people say it: "I took it out", "buying",
# "my cover started", "this policy".
_START = (
    r"(?:I\s+(?:took(?:\s+(?:it|this|the\s+policy|this\s+policy|this\s+cover|the\s+cover))?\s+out"
    r"|took\s+out\s+(?:this|the|my)\s+(?:policy|cover)|bought|got|started|joined|took)"
    r"|(?:taking|buying|getting|starting)\b"
    r"|(?:my|the|this)\s+(?:policy|cover)\s+(?:started|began)"
    r"|(?:this|the|my)\s+(?:policy|cover)\b)"
)
_STATED_NOT_PRE_EXISTING = re.compile(
    r"\b(?:never|not)\s+(?:had|suffered\s+from)\b[^.;]{0,30}?\bbefore\s+" + _START
    + r"|\b(?:started|began|begun|developed|appeared|arose|diagnosed|found|noticed)\b"
    r"[^.;]{0,40}?\bafter\s+" + _START,
    re.IGNORECASE,
)


def stated_not_pre_existing(scenario: str) -> bool:
    return bool(_STATED_NOT_PRE_EXISTING.search(scenario))
```

Two shapes are recognised: "never had ... before taking the policy" and
"started / was diagnosed ... after I took the policy out". `[^.;]{0,40}?`
means "up to 40 characters that are not the end of a sentence", so the two
halves must be in the same sentence and close together. The pattern was run
against all 78 eval questions and the fact-check sentences before it was used,
to see every question it matches.

It is used at the end of `extract_facts`, and only to fill in a gap:

```python
facts = derive_pre_existing(correct_misfiled_age(facts, scenario))
# After the arithmetic, which is the stronger evidence, and like it, only
# ever filling in an answer the model declined to give.
if facts.get("pre_existing_condition") == "unknown" and stated_not_pre_existing(scenario):
    facts = facts | {"pre_existing_condition": "no"}
```

`derive_pre_existing` (from M18) is the arithmetic route: if the person said
both how long they have had the condition and how long they have held the
policy, the comparison settles it. The lookup runs after it and never
overrides an answer the model or the arithmetic already gave.

## Telling the waiting-period check

Settling the fact was not enough on its own. The waiting-period block of the
reasoning prompt (`api/app/pipeline/waiting.py`, the code that does the date
arithmetic so the model doesn't) did not look at the fact at all. For clause
3.2 it always wrote a hedged line:

```
- clause 3.2: concerns ONLY an illness the person already had when the policy
  began. Unless the description says this one had begun by then, it is not the
  reason for this claim. (Requires 36 months; policy held 2 years, so not yet
  served.)
```

So `evaluate` now takes the settled fact, and each `WaitingCheck` carries it:

```python
def concerns_nothing_here(self) -> bool:
    """A pre-existing diseases bar, for a condition settled as beginning after the policy."""
    return self.pre_existing and self.condition_predates_policy == "no"
```

When that is true, the line becomes firm, and it is listed first:

```
- clause 3.2 (Pre-existing Disease Waiting Period): concerns ONLY an illness
  the person already had when the policy began. This one began after the
  policy did, so this waiting period does not concern this claim
```

## Failure 84: the "yes" direction broke a case that was already right

**What was tried.** The first version read the person's words in both
directions. Besides "began after the policy" → `"no"`, it read "since before
I bought the policy" → `"yes"`. That needed a guard: "Before I bought this
policy I checked that it covered hospital stays" says "before I bought" but
names no illness, so a "yes" only counted in a sentence that also had a verb
like *had*, *diagnosed* or *since*. When the fact was settled as "yes" and the
period not served, the waiting-period line said so plainly: "the person's
condition began before the policy, so this pre-existing disease waiting period
applies to it ... and blocks this claim".

**What happened.** The request comparison (below) showed six of 78 questions
changed. Each was asked five times. The knee case was fixed, five of five. But
`ped-waiting-served`, which had been right, broke five times in five:

> "I have had high blood pressure since before I bought the policy. I was
> hospitalised for it 5 years after taking the policy out."

Expected `covered`: the condition is pre-existing, but five years is past the
36-month wait. The lookup correctly set the fact to "yes". The model then
answered `not_covered`, citing **clause 4.1, the cosmetic surgery exclusion**,
which has nothing to do with high blood pressure.

**Why.** The waiting-period line for this case had not changed: the period was
served, so it still said "no longer applies". What changed was the facts block,
which now showed `pre_existing_condition: yes` instead of `unknown`. A 7B model
seeing "pre-existing: yes" leaned towards refusing, and reached for a clause to
refuse under. Which clause it reached for looks arbitrary. That is the
dangerous part: a true fact made the answer worse, and nothing in the prompt
pointed at the clause it cited.

**The fix.** Drop the "yes" direction entirely. The lookup now only ever says
"this began after the policy". "Yes" stays a judgement for the model, as
before, under the old hedged line. With that change the request comparison
showed 4 of 78 questions changed (the two "yes" cases dropped out), and none
of them was a fact-extraction request.

## Concept 53: a fix that acts in one direction only

A rule that can set a fact either way looks more complete than one that can
only set it one way. Here the two directions had very different costs:

- Saying "no" when the person said "it started after I took the policy out"
  removes a bar the model was wrongly applying. If the lookup misfires, the
  model still sees every clause and the person's own words.
- Saying "yes" adds weight towards refusing, and a 7B model already leans that
  way on this policy. A refusal is the expensive mistake in this domain: a
  person told "not covered" may never file a valid claim.

So the fix covers only the direction that was measured to help, and the other
direction is left as it was. This is the same reasoning as
`derive_pre_existing`, which "only ever fills in an answer the model declined
to give": correct where the evidence is plain, and otherwise don't move.

## Measuring part 1

**Isolation first, by request comparison** (introduced in M21). Every request
the pipeline would send the model is recorded on the old code and on the new,
for all 78 questions, with the model call itself replaced by a recorder (the
fact-extraction answers come from the cache, so this needs no model time).
Comparing the two recordings shows exactly which questions the change can
affect:

- 4 of 78 questions changed: `ho-knee-replacement-served`,
  `ho-copay-derived-age`, `ho4-thyroid-after-cover`,
  `ho4-gallstones-found-last-year`.
- 0 fact-extraction requests changed.

The other 74 questions send byte-identical requests, so they cannot have been
affected and keep their old measurements.

**Then five samples of each changed question** (Ollama 0.35.1):

| Case | Expected | Before | After, 5 samples |
|---|---|---|---|
| `ho-knee-replacement-served` | covered | not_covered | **covered 5/5** |
| `ho-copay-derived-age` | conditional | not_covered | not_covered 5/5 |
| `ho4-thyroid-after-cover` | covered | not_covered | not_covered 5/5 |
| `ho4-gallstones-found-last-year` | covered | not_covered | not_covered 5/5 |

One fixed, nothing broken, three still wrong. The three were the interesting
part.

## Part 2: a line that did not say what it bars

Here is the waiting-period block the thyroid question got after part 1
(policy held 20 months):

```
- Already satisfied, so IRRELEVANT here and not worth citing: 3.1
- clause 3.2 (...): ... This one began after the policy did, so this waiting
  period does not concern this claim
- clause 3.3: requires 24 months, policy held 20 months -> this waiting period
  still applies and blocks treatment covered by THIS clause
- clause 3.4: requires 36 months, policy held 20 months -> this waiting period
  still applies and blocks treatment covered by THIS clause
```

Clause 3.3 is the specified-disease waiting period (cataract, hernia, joint
replacement and others). Clause 3.4 is maternity. Neither concerns a thyroid
problem. But the lines say only "blocks treatment covered by THIS clause",
and the model had to go and read each clause to learn what that treatment is.
With 3.2 settled, it fell back on the next unserved bar in the list. For the
kidney stone it was the same: a 36-month maternity wait was effectively being
read as a bar on everything.

**The change.** Each waiting-period line now names its subject, taken from
the clause heading. In `waiting.py`:

```python
# What the period is for, from the clause heading ("Maternity Waiting
# Period"). M22: a line saying only "blocks treatment covered by THIS
# clause" was read as blocking everything - a 36-month maternity wait was
# given as the reason to refuse unrelated claims on a new policy.
subject: str = ""
```

filled in by `evaluate`:

```python
# The heading repeats the clause number; the line already has it.
subject=re.sub(r"^[\d.]+\s*", "", getattr(clause, "heading", "") or ""),
```

so the line reads `clause 3.4 (Maternity Waiting Period): requires 36
months...`. A policy whose clauses have no headings gets the old line
unchanged.

This one is wide. Every question whose prompt has a waiting-period line
changes. The request comparison showed **29 of 78** questions changed (still
0 fact requests), so it needed a full measurement: majority of three samples
on every set, run as separate jobs because a background command is stopped
after 30 minutes.

## The result

Majority of three samples, Ollama 0.35.1, `PROMPT_VERSION =
"v53-waiting-subject"`:

| Set | v51 (before M22) | M22 |
|---|---:|---:|
| main set | 35/40 | **36/40** |
| held-out batch 1 | 12/16 | **13/16** |
| held-out batch 2 | 12/13 | 12/13 |
| held-out batch 3 | 0/3 | 0/3 |
| held-out batch 4 (written for M22) | 1/6 | 1/6 |
| total | 60/78 | **62/78** |

Every one of the 29 changed questions that is now wrong was already wrong
before M22. The main set's four misses (`initial-waiting-period`,
`copay-applies-emergency`, `intoxication-injury`, `day-care-not-listed`) are
the same misses as before. One held-out-1 miss, `ho-short-procedure` (2 of 3
`conditional`), is not among the changed questions: its request is
byte-identical to v51, so that is sampling noise, not this change.

**One cost, in citations rather than verdicts.** The regenerated report's
false-citation rate went from 0.000 to 0.200. The cause is one changed
question:

> `copay-unknown-inception-age`: "I am 68 and I was hospitalised for a stroke.
> I don't remember when I took the policy out."

Expected `insufficient_information`, and it still gets that, three times in
three. But its answer key forbids citing clause 5.3, the co-payment for people
aged 60 or more *when the policy began*. The person's age at the start is
unknown, so the co-payment can't be applied. Before M22 the answer cited 3.2,
3.3 and 3.4. Now it cites 3.2, 3.3 and 5.3, with this reasoning:

> "...the age at the time of policy inception is not stated, which is relevant
> for the 20% co-payment clause. Therefore, the decision cannot be made with
> the given information."

Read on its own, that reasoning is sound: it names 5.3 as relevant and
undecided, and doesn't apply it. But the answer key was written before the
run and says "don't cite 5.3", and changing a key after seeing an answer is
how an eval stops measuring anything. So it stays counted as a false citation
and a cost of part 2. It also shows that a wording change in one block moves
which clauses the model cites from other blocks.

## Failure 85: the batch written for this milestone did not move

The honest headline is the last row of the table. **The batch written to test
M22 scored 1 of 6 before and 1 of 6 after.** The gains are on cases that were
read while building it (the knee case) and on main-set cases.

What the five failing cases now cite, from the stored answers:

| Case | Fact now | Answer | Cites |
|---|---|---|---|
| `ho4-thyroid-after-cover` | pre-existing: no | not_covered | 3.3 Specified Disease |
| `ho4-gallstones-found-last-year` | pre-existing: no | not_covered | 4.4 Breach of Law |
| `ho4-back-trouble-on-and-off` | unknown | not_covered | 4.4 Breach of Law |
| `ho4-before-buying-checked-cover` | unknown | not_covered | 3.3 Specified Disease |
| `ho4-acute-no-history` | unknown | not_covered | 3.3 Specified Disease |

The facts are right where M22 meant them to be: thyroid and gallstones are now
settled as "no" (gallstones by the M18 arithmetic: known for one year, policy
held for two). The pre-existing bar is no longer cited. The model refuses
anyway, and moves to another clause:

- **3.3** is the specified-disease period, and its line now says "Specified
  Disease Waiting Period". That name doesn't say *which* diseases. Thyroid,
  malaria and appendicitis are not on the list, but the line doesn't say so,
  and the model doesn't go and check.
- **4.4** excludes treatment connected with a breach of law. Nothing in either
  question mentions one.

**What this shows.** On a policy held for under three years, this model leans
towards refusing, and each fix that removes one reason for refusing reveals
the next. Naming the subject helped where the subject was obviously unrelated
(maternity). Where the name is a category ("specified disease"), the line
needs the list itself, and that is a separate change for a separate
milestone. The batch shows exactly what it was written to show: whether the
fix generalises. It does not, yet.

## Part 3: fixing the two costs

Parts 1 and 2 left two problems: a new forbidden citation, and a held-out
batch that didn't move. Both were investigated before anything was changed.

### The forbidden citation was a scoring question

Every citation the model returns carries an **effect**, picked from a fixed
list: `denies`, `delays`, `reduces`, `requires`, `permits`. The answer for
`copay-unknown-inception-age` cited 5.3 with effect `requires`, meaning "this
clause needs information that is missing". It did not say `reduces`. The
system's own reductions block had told the model:

```
- CANNOT TELL: clause 5.3: a 20% co-payment applies if the person had reached
  60 at inception, but their age when the policy STARTED was not stated, so
  this cannot be determined
```

The case's own `why` agrees: "Two things are unknown and both matter. The
co-payment keys on age AT INCEPTION..." And the eval file defines
`must_not_cite` as "a clause a correct answer must NOT **rely on** - a
co-payment for someone who was 58 at inception". Naming 5.3 as undecided is
not relying on it.

So the scorer changed, not the answer key. In `evals/run_scenario_eval.py`:

```python
def relied_on(citations, forbidden: set[str]) -> list[str]:
    """The forbidden clauses an answer cites AGAINST the claim. ..."""
    return sorted({c.clause_id for c in citations
                   if c.clause_id in forbidden and c.effect in ("denies", "delays", "reduces")})
```

**This is a rule changed after seeing an answer,** which Part 2 said not to
do for a single case. What makes it acceptable here is the check made first:
every stored answer, for all 78 questions, was replayed from the cache (no
model time) and scored under both rules. Exactly one answer moved,
`copay-unknown-inception-age`. A rule that had quietly forgiven other real
false citations would have shown up in that list. A test pins both sides: a
`requires` citation of a forbidden clause is not counted, a `reduces` one is.

### The batch: the model was copying the line

Reading the reasoning of the five held-out-4 refusals showed the mechanism
directly. The thyroid answer said:

> "...this waiting period still applies and blocks treatment covered by this
> clause. The claim is not covered."

That is the waiting-period line, copied word for word. For malaria the model
wrote "specified diseases, including malaria"; malaria is not on the list. The
two breach-of-law (4.4) citations were mix-ups: the reasoning discussed
hazardous sports or the pre-existing period under 4.4's number. The quote
check had already marked both answers *unverified*, so they were left alone.

**The obvious fix was unsafe, and a check showed it before any code.** The
idea was to tell the model "none of the question's words appear in this
clause's list", using the existing word lookup (`named_in_question`, which
lists words the person used that a clause also uses). The lookup was run
against every question where the specified-disease or maternity period is
unserved. For `maternity-too-early`:

> "I gave birth 14 months after buying the policy. Will the delivery be paid
> for?"

it found **nothing**. The clause says "maternity, childbirth ... two
deliveries", and "gave birth" / "delivery" don't match those words. A line
saying "not named" would have cleared a claim that must be refused.

**What was built instead** leaves the judgement with the model and changes
only how the line opens. This is the same lesson as the pre-existing line:
the first words of a line are the ones acted on. In
`api/app/pipeline/waiting.py`:

```python
# A period that bars only the treatments its clause lists (IRDAI's "Specified
# disease/procedure waiting period", maternity), recognised the same way. Only
# these get the narrower line; anything not recognised keeps the firm one,
# because calling an all-illness period list-only would pay claims it bars.
_LISTED = re.compile(r"specified|maternity", re.IGNORECASE)
```

and, for an unserved period of that kind whose words the question does not
share:

```
- clause 3.3 (Specified Disease Waiting Period): bars ONLY the treatments
  this clause lists. If this treatment is not one of them, this period is not
  the reason for this claim. (Requires 24 months; policy held 20 months, so
  not yet served.)
```

Two choices in it follow Concept 53, *a fix that acts in one direction only*:

- **Recognise the list-type periods, not the all-illness one.** The opposite
  design would recognise "initial waiting period" and narrow everything else.
  A miss there would tell the model that a 30-day bar on *every* illness
  only bars a list, and wrongly pay a claim. A miss in the chosen design only
  leaves today's over-refusal.
- **Keep the firm line when the question's words appear in the clause.** A
  hysterectomy 18 months in shares "hysterectomy" with clause 3.3, so it is
  probably listed, and the firm "still applies and blocks" line stays.

**Measured.** The request comparison showed 16 of 78 questions changed, with no
fact requests among them. Only those 16 were re-asked, three samples each:

| | Part 2 | Part 3 |
|---|---:|---:|
| the 16 changed questions | 8 right | **10 right** |
| `maternity-too-early` (must stay refused) | not_covered | not_covered 3/3 |
| `ho-hysterectomy-too-soon` (must stay refused) | not_covered | not_covered 3/3 |
| `ho2-cataract-too-soon` (must stay refused) | not_covered | not_covered 3/3 |
| `ho-copay-derived-age` (the kidney stone) | not_covered | **conditional 3/3** |
| `ho4-gallstones-found-last-year` | not_covered | **covered 3/3** |

The kidney-stone case that started M22 is now right. The other 62 questions
send byte-identical requests and keep their measurements, so the totals are:

| Set | v51 | Part 2 | Part 3 |
|---|---:|---:|---:|
| main set | 35/40 | 36/40 | 36/40 |
| held-out 1 | 12/16 | 13/16 | **14/16** |
| held-out 2 | 12/13 | 12/13 | 12/13 |
| held-out 3 | 0/3 | 0/3 | 0/3 |
| held-out 4 | 1/6 | 1/6 | **2/6** |
| total | 60/78 | 62/78 | **64/78** |

**What is still wrong in held-out 4, and why it stops here.** In each of the
four, the model's own judgement fails, not a line the system wrote:

- thyroid (now `conditional`) and appendicitis still say clause 3.3 covers
  them
- malaria: the model reads "Before I bought this policy I checked..." as a
  pre-existing illness, the trap the code no longer falls into
- back trouble: the model decides it is pre-existing, where the history is
  simply unknown

Going further would mean code deciding whether a treatment is on a clause's
list. The `maternity-too-early` check showed the word lookup can't be trusted
to do that.

## Closing out M22

### How it connects to what was already there

M22 follows the shape of every fix since M14:

1. the person's words are read with a fixed pattern (as in M21)
2. a fact is settled in code only where the evidence is plain (as in M18's
   `derive_pre_existing`)
3. the waiting-period arithmetic tells the model the result in a line it
   can't misread

What M22 added:

- a fix can act in one direction only (Concept 53)
- a correct fact can still make the answer worse (Failure 84)
- a line in the prompt has to say what it is about, not just which clause it
  comes from, and the model may copy it word for word into its answer
- a scoring rule can be corrected after seeing an answer only if every stored
  answer is rescored under both rules and the full list of what moved is shown

### What it made possible

The pre-existing-disease fact now reaches the waiting-period check, which
M19 had left as an open gap "until a case shows it costs an answer". The case
turned up, and the gap is closed.

What is left is the model's own judgement: deciding whether a treatment
is on a clause's list, and whether an illness with no stated history predates
the policy. Neither can be handed to code without a lookup that has been
shown to understand "gave birth" as "childbirth".

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_scenario.py tests/test_waiting.py -q -k "pre_existing or new_condition or names_what or list or stays_firm"
```

**Predict before you look:** "I have had high blood pressure since before I
bought the policy." What does `stated_not_pre_existing` return, and what
would the pipeline's `pre_existing_condition` fact be if the model had said
`"unknown"`?

<details>
<summary>Answer</summary>

`False`. "Since before I bought" is the "yes" direction, which this lookup
deliberately does not read (Failure 84). The fact stays `"unknown"`, unless
the person also gave two durations that `derive_pre_existing` can compare.
The waiting-period line for clause 3.2 stays the hedged one, and whether the
condition predates the policy remains the model's judgement.
</details>

---

# M23 — A waiting period that bars only its list

## The problem

Indian health policies have a **specified disease waiting period**: a list of
treatments that are not paid for until the policy has been held for a while.
In the synthetic policy it is clause 3.3:

> "Expenses related to the treatment of cataract, hernia, hysterectomy, benign
> prostatic hypertrophy, joint replacement surgery, and diseases of the ear,
> nose and throat shall be excluded until the expiry of twenty four months of
> continuous coverage from the date of inception of the first policy."

Anything *not* on that list is untouched by it. Appendicitis 18 months into
a policy is not on the list, so 3.3 has nothing to say about it.

The scenario pipeline does the waiting-period arithmetic in code
(`api/app/pipeline/waiting.py`) and tells the reasoning model the result as
one line per period. Since M22, a period whose heading says "specified" or
"maternity" gets a narrower line than the blunt "still applies and blocks":

```
- clause 3.3 (Specified Disease Waiting Period): bars ONLY the treatments
  this clause lists. If this treatment is not one of them, this period is not
  the reason for this claim. (Requires 24 months; policy held 18 months, so
  not yet served.)
```

Two held-out cases were still refused under it: `ho4-acute-no-history`
(appendicitis, 18 months) and `ho4-thyroid-after-cover` (thyroid problem, 20
months). Both are expected `covered`.

## What the model actually did

Reading the stored answer for appendicitis showed something surprising. The
model's own quote was the whole list:

```json
"reasoning": "The claim for appendicitis surgery is not covered because the
  24-month waiting period for 'Specified Disease' as outlined in clause 3.3
  has not been satisfied...",
"deciding_clauses": [{"clause_id": "3.3",
  "quote": "Expenses related to the treatment of cataract, hernia,
            hysterectomy, ... shall be excluded until the expiry of twenty
            four months ...",
  "effect": "denies"}]
```

So the list was in front of it: the clause text is in the prompt, and the
model copied it out. It still applied the period to appendicitis. What it
acted on was the arithmetic in the line ("requires 24 months; policy held 18
months, so not yet served"). It never checked appendicitis against the list,
which sits much further down the prompt.

## Two wordings, tried before any code

Both were tried by patching the line in a scratch script, on the 16 questions
whose prompt contains a list-type line, three fresh answers each, scored by
majority. The current wording scores 10 of those 16.

- **V1:** quote the clause's own text into the line:
  `bars ONLY the treatments this clause lists: "Expenses related to the
  treatment of cataract, hernia, ...". If this treatment is not one of them...`
- **V2:** V1, plus name what the person was treated for beside it:
  `Compare what the person was treated for ("appendicitis surgery") with that
  list.`

| | current | V1 | V2 |
|---|---:|---:|---:|
| the 16 questions | 10 | **11** | 10 |
| `ho4-acute-no-history` (appendicitis) | not_covered | **covered 3/3** | covered 3/3 |
| `ho4-gallstones-found-last-year` | covered | covered | **conditional 3/3** |
| `maternity-too-early` (must stay refused) | not_covered | not_covered | not_covered |

V2 looked like the stronger nudge and was worse: spelling out the comparison
broke gallstones, which had been right. V1 was built. The comparison itself
stays the model's job, because code cannot do it safely: M22 showed that a
word lookup does not match "gave birth" to "childbirth".

## What was built

In `api/app/pipeline/waiting.py`, the list-type line now quotes the clause:

```python
lists = f': "{self.listing}"' if self.listing else ""
return (
    f"{who}: bars ONLY the treatments this clause lists{lists}. If this "
    f"treatment is not one of them, this period is not the reason "
    ...
```

`listing` is the clause text with its heading removed, filled in only when
the period is list-type and the text is short:

```python
# The list is quoted into the line only up to this length. The synthetic
# specified-disease clause is about 290 characters; Star's is about 1,970, and
# quoting that would repeat ~560 tokens outside the clause budget.
_QUOTE_LIST_CHARS = 600
```

The limit exists because the waiting-period lines sit *outside* the clause
token budget (the room the shortlist reserves for clause text). Quoting a
2,000-character clause twice could push the prompt past the context window,
and the model would silently lose the end of it. A clause over the limit keeps
the M22 line, unquoted.

Measuring that limit on the real Star Health policy turned up a second
problem.

## Concept 54: a ligature, and why "Speciﬁed" is not "Specified"

Typesetting software joins some letter pairs into one glyph so they look
better: "f" followed by "i" becomes "ﬁ". In a PDF, that glyph is often stored
as **one character**, Unicode U+FB01 LATIN SMALL LIGATURE FI, not as the two
characters `f` and `i`. It looks identical on screen. To a program it is a
different string:

```python
>>> "Speciﬁed" == "Specified"
False
>>> len("Speciﬁed"), len("Specified")
(8, 9)
```

Star's specified-disease clause is headed "2. Speciﬁed disease / procedure
waiting period", with the ligature. M22's pattern `specified|maternity`
never matched it. **On the one real policy this project measures, M22's
list-type line had never been used**; Star's specified-disease clause still
got the blunt "blocks" line. The unit tests could not catch it: they were
written with plain ASCII text.

The standard fix is **Unicode normalization**. Unicode defines, for many
characters, an equivalent plainer spelling, and `unicodedata.normalize`
rewrites text into one canonical form. The form called **NFKC**
("compatibility composition") replaces presentation variants with their plain
equivalents: the "ﬁ" ligature becomes `f` + `i`, a full-width "Ａ" becomes
"A", a superscript "²" becomes "2". It does not change ordinary letters or
accents.

This project already used it in one place. The quote checker
(`api/app/grounding.py`, `normalize`) applies NFKC before comparing a
model's quote with a clause, so a quote typed with plain "fi" still matches
the PDF's ligature. The waiting-period code never got the same treatment.
Now it does, in `evaluate`:

```python
# NFKC first: PDFs set "fi" as one ligature character, so Star's
# "Speciﬁed disease" never matched "specified" and got the firm line.
text = " ".join(unicodedata.normalize("NFKC", getattr(clause, "text", "") or "").split())
heading = unicodedata.normalize("NFKC", getattr(clause, "heading", "") or "").strip()
listed = bool(_LISTED.search(f"{heading} {text[:_OPENING_CHARS]}"))
```

The lesson outlives this project: **any text pattern run on PDF text should
run on normalized text.** The same gap almost certainly exists in other
lookups here (the word lookup `named_in_question`, for one); they were left
alone because no measured case yet shows them costing an answer.

## Measuring

**Which questions change.** Every reasoning request was hashed with the model
call replaced by a stub, once on main and once on this branch. On the
synthetic policy, 16 of 78 questions changed: exactly the ones with a
list-type line. On Star, 5 of 10 changed, all from the ligature fix. Every
other request is byte-identical, so it keeps its stored answer and its
measurement.

**The 16 synthetic questions**, majority of three, with the code as built:

| | before (v54) | after (v55) |
|---|---:|---:|
| the 16 changed questions | 10 | **12** |
| `ho4-acute-no-history` (appendicitis) | not_covered | **covered 3/3** |
| `ho4-before-buying-checked-cover` (malaria) | not_covered | **covered 3/3** |
| `maternity-too-early`, `ho-hysterectomy-too-soon`, `ho2-cataract-too-soon` (must stay refused) | not_covered | not_covered 3/3 |
| `ho4-thyroid-after-cover` | conditional | conditional 3/3 |
| `ho4-back-trouble-on-and-off` | not_covered | not_covered 3/3 |

The malaria case ("Before I bought this policy I checked that it covered
hospital stays") was not a target: it had been read as a pre-existing
illness. With the list quoted, the model stopped refusing it. No case that
was right went wrong.

One cost, in citations rather than verdicts: `initial-waiting-period` ("a bad
chest infection two weeks after my policy started") was already wrong
(`covered`, expected `not_covered`), and now also stops citing clause 3.1,
the 30-day initial waiting period, in 3 of 3 answers. Main-set citation recall
in the regenerated report falls from 0.812 to 0.781. With two list-type lines
each ending "this period is not the reason for this claim", the model
answers as if no waiting period applied at all. Left open, along with the
wrong verdict it already had.

| Set | M22 | M23 |
|---|---:|---:|
| main set | 36/40 | 36/40 |
| held-out 1 | 14/16 | 14/16 |
| held-out 2 | 12/13 | 12/13 |
| held-out 3 | 0/3 | 0/3 |
| held-out 4 | 2/6 | **4/6** |
| total | 64/78 | **66/78** |

## Failure 86: the right answer, now for the wrong reason

**Setup.** Star's 5 changed questions, majority of three, on main and on this
branch, Ollama 0.35.1 both times.

**What happened.** All five verdicts were right on both. But
`star-dengue-first-month` ("admitted with dengue fever 20 days after buying
this policy"), which must cite the 30-day initial waiting period (Star's
clause `3#2`), cited it in 3 of 3 answers on main and **0 of 3** on the branch.
It now cites a clause called `c7`:

```json
"reasoning": "The claim for dengue fever is not covered because it falls
  under the 36-month specific waiting period for 'Specified diseases' as per
  clause c7...",
"deciding_clauses": [{"clause_id": "c7", "quote": "Specific Waiting Period
  means a period up to 36 months from the commencement of a health insurance
  policy during which period specified diseases/treatments ... are not
  covered...", "effect": "delays"}]
```

**Why.** `c7` is not a waiting period. It is a run of *definitions* from
Star's definitions section (the end of "Pre-existing Disease", then
"Pre-hospitalization Medical Expenses", then "Specific Waiting Period
means..."), headed only "(cont.)" because it continues a page. The analysis
step (stage 3) typed it `waiting_period` and recorded 36 months for it, so
the waiting-period check gives it the blunt line:

```
- clause c7 ((cont.)): requires 36 months, policy held 20 days -> this
  waiting period still applies and blocks treatment covered by THIS clause
```

On main, the specified-disease clause got the same blunt line, and the model
passed over both to cite the 30-day period. Now that the specified-disease line
correctly says it bars only its list, `c7` is the most forceful bar left, and
the model leans on it.

**What was decided.** The change was kept, and `c7` was left as a known issue
to fix on its own. The misreading of `c7` is an analysis error that existed
before this change; this change only exposed it. Fixing it means changing how
stage 3 types a definitions block, which needs its own measurement. Folding it
in here would have made M23's numbers impossible to attribute.

The general point: **removing one wrong signal can promote the next wrong
signal.** A verdict that stays right is not proof that nothing moved; the
citation is what shows which reason the answer now rests on.

## Closing out M23

### How it connects

M23 is the third step on the same line:

1. M22 parts 1-2: every waiting-period line names its subject.
2. M22 part 3: a list-type period opens by saying it bars only its list.
3. M23: it shows the list.

Each step leaves the judgement ("is this treatment on the list?") with the
model and changes only what the model is shown. Each was tested before it was
built, on the questions it changes.

### What it made possible

M22's line now reaches real policies, through normalization. The
remaining wrong answers in held-out 4 are thyroid (the model still answers
`conditional`) and back trouble (it decides an illness with an unknown
history is pre-existing). Open next: Star's `c7` (Failure 86), and the same
normalization in the other text lookups.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_waiting.py -q -k "quotes or ligature or long_list"
```

**Predict before you look:** `"Speciﬁed".lower() == "specified"` - true or
false? And after `unicodedata.normalize("NFKC", ...)`?

<details>
<summary>Answer</summary>

False, then true. Lowercasing doesn't touch the ligature, because "ﬁ" is
already lowercase; it is just not the two letters `f` and `i`. NFKC replaces
it with them. That's why `re.IGNORECASE` alone never matched Star's heading.
</details>

---

# M24 — A definitions block is not a waiting period

## The problem

Every Star Health question with a policy held under three years was being
told, in the waiting-period block of its reasoning prompt:

```
- clause c7 ((cont.)): requires 36 months, policy held 20 days -> this
  waiting period still applies and blocks treatment covered by THIS clause
```

`c7` is not a waiting period. Star's definitions section is long, so the
segmenter (stage 2) cuts it into pieces of at most `MAX_CLAUSE_CHARS`, at
line breaks, in `_split_oversized` (`api/app/pipeline/segment.py`). Every
piece after the first gets the heading of the original plus " (cont.)".
That's how `c3` to `c7` were made. Each piece is then analysed separately by
the model (stage 3). `c3` to `c6` came back `definition`. `c7` came back
`waiting_period`, with 36 months, because one of the terms it defines is:

> "Specific Waiting Period means a period up to 36 months from the
> commencement of a health insurance policy during which period specified
> diseases/treatments (except due to an Accident) are not covered."

M23 recorded what this cost (Failure 86): once the specified-disease line
stopped saying "blocks", the dengue question cited `c7` as its reason.

## Why the model typed it that way

The classification prompt (`CLASSIFY_SYSTEM` in `api/app/llm/prompts.py`)
resolves overlapping labels with tie-breakers checked in order:

```
TIE-BREAKERS (apply in this order - the more specific label always wins):
1. Does it cap or reduce an amount?              -> sub_limit
2. Does coverage start after a stated period?    -> waiting_period
...
5. Does it define a term?                        -> definition
```

A definition *of* a waiting period hits rule 2 before rule 5. The model
followed the prompt.

## Two places a fix could go

- **The prompt.** Add "a clause that defines terms is a definition, even when
  a term it defines is a waiting period". That is the root cause, but a
  changed analysis prompt is a cache miss for every clause of every policy.
  Every clause would be re-analysed, every scenario prompt could change, and
  the whole eval would have to be measured again to learn what one sentence
  did.
- **The waiting-period check.** Leave the stored type alone and stop a
  definitions block from producing a waiting-period line. Only the line
  changes, and only where the rule fires.

The second was built, after checking the rule cleanly separates the two on
every policy available.

## The rule, measured before it was written

The candidate signal was the word "means", which is how Indian policy
definitions are worded ("Hospital means...", "Accident means..."). Every
clause with a waiting period recorded, on all three policies, with the
number of times its text says "means":

| Policy | Clause | Says "means" |
|---|---|---:|
| synthetic | 3.1, 3.2, 3.3, 3.4 | 0 each |
| Star | `c7` (definitions block) | **8** |
| Star | `2#2` specified disease, `3#2` 30-day | 0 each |
| HDFC | `2.10#2`, `2.12`, `1#2`, `1#4`, `1.5#2`, `1.6#2`, `6` | 0 each |

In `api/app/pipeline/waiting.py`:

```python
# A run of definitions ("X means ...") is not a waiting period, even when one
# term it defines is. ...
_DEFINES = re.compile(r"\bmeans\b", re.IGNORECASE)
_DEFINITIONS_BLOCK = 2
```

and in `evaluate`, before a check is built:

```python
if len(_DEFINES.findall(text)) >= _DEFINITIONS_BLOCK:
    continue
```

Two, not one: a real waiting-period clause could reasonably define one term
in passing. A run of definitions defines several.

The same table turned up another misreading, left for later: HDFC's
pre- and post-hospitalisation windows (`1.5#2`, `1.6#2`: "60 days", "180
days") are typed `waiting_period`. They are the windows for claiming
expenses before and after a stay, not periods before cover starts. They say
"means" zero times, so this rule does not catch them.

## Measuring

**Which questions change.** The request comparison (every reasoning request
hashed with the model stubbed out, before and after) found **0 of 78**
synthetic questions changed (no synthetic clause says "means" twice and
carries a waiting period), and **10 of 10** Star questions, since `c7`'s
line had been in every one.

**Star, majority of three, Ollama 0.35.1:**

| Case | M23 | M24 |
|---|---|---|
| `star-caesarean` (expected not_covered) | conditional | **not_covered** |
| `star-treatment-abroad` (expected not_covered) | not_covered | **conditional** |
| `star-gallstones-no-timing` (expected insufficient_information) | insufficient_information | **conditional** |
| right verdicts | 9/10 | 8/10 |
| answers citing the clause that decides them | 6/10 | 7/10 |

## Failure 87: the score fell, and the change was kept

**What happened.** Removing a false line cost one right verdict overall.
The answers explain why:

- **Treatment abroad** ("hospitalised with pneumonia while on holiday in
  Dubai... held the policy for two years"). Star excludes treatment outside
  India in its clause `22`. Under M23 the answer was `not_covered`, but **no**
  answer cited 22, in three samples of three. It was refused on `c7`'s
  36-month bar, which is false. With that gone, the model answers
  `conditional` (pneumonia is payable, less Star's 5% co-payment), and never
  sees that the exclusion of treatment abroad applies. The right verdict had
  been propped up by a wrong fact.
- **Gallstones, no timing** ("I need my gallbladder removed... How much will
  the policy pay?"). The reasoning is sound. It says gallbladder stones are on
  the specified-disease list, that the policy's age was not stated, and that
  if the period is not served the claim is refused. It then picks
  `conditional` where the answer key says `insufficient_information`. The
  verdict is wrong; the reasoning is not.
- **Caesarean** is now right. With no false 36-month bar on the page, the
  model found Star's maternity exclusion.

Dengue still misses its deciding clause, the 30-day wait. It now cites the
specified-disease period instead and treats dengue as being on its list. That
list is about 1,970 characters, over the 600 M23 allows for quoting, so the
line never shows it.

**Why it was kept.** The line removed was false for every Star question. A
score that depends on a false statement measures luck, not the system, and
protecting it would mean keeping a known wrong fact in every prompt. The
change exposes Dubai's real problem: the model doesn't notice an exclusion of
treatment abroad. That is the next thing to fix, and it can now be measured
honestly.

The general point pairs with Failure 86. There, removing a wrong signal let
the next wrong signal take over. Here, removing a wrong signal took away a
right answer that had rested on it. **A verdict is only as good as the clause
it rests on, so read the citation before counting the verdict.**

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_waiting.py -q -k definitions
```

**Predict before you look:** the synthetic policy's clause 1.2 says
"Pre-existing Disease means any condition...". Does the new rule drop it
from the waiting-period check?

<details>
<summary>Answer</summary>

It never reaches the rule. Clause 1.2 is typed `definition` and records no
waiting period, so `evaluate` skips it at the top of its loop. If it
did record one, it would still stay in: it says "means" once, and the rule
needs two.
</details>

---

# M25 — Treatment outside India

## The problem

Star Health's Arogya Sanjeevani excludes treatment abroad in two short lines,
its exclusion 22:

> "22. Treatment taken outside the geographical limits of India"

The Star question `star-treatment-abroad` asks:

> "I was hospitalised with pneumonia while on holiday in Dubai. I have held
> the policy for two years."

The right answer is `not_covered`, citing 22. Under M24 the model answered
`conditional` (pneumonia is payable, less Star's 5% co-payment), 3 samples of
3. Clause 22 was in the prompt; the model never connected "Dubai" with
"outside the geographical limits of India". The fact extraction step did not
help either: it has no field for where treatment happened, and wrote "Dubai"
only into its free-text `notes`.

Three fixes were considered:

- **A new fact field**, "where treated", in the extraction prompt. Rejected
  for the reason M21 recorded as Concept 52 (*a free-text field is a side
  channel*): any change to that prompt rewords the notes on every question,
  and every answer would have to be measured again.
- **Always point at the clause**, on every question, for any policy with an
  "outside India" clause. Rejected: all ten Star questions would change, and
  a line asking "where was the treatment?" invites `insufficient_information`
  on every question that doesn't say.
- **A word lookup on the question**, M21's pattern: code finds a place abroad
  in the person's words and names the clause; the model judges. Chosen.

## Questions written before the code

`evals/real/scenarios-star-heldout-abroad.json`, four questions written before
any code or run: two treated abroad (Singapore, London) and two traps
(typhoid in Mumbai; "fell ill on a trip to Thailand, flew home, and was
admitted to a hospital in Chennai"). On the M24 code, majority of three: **2
of 4**, the traps right and both abroad questions wrong, never citing 22.

## The lookup

In `api/app/pipeline/scenario.py`, two patterns. One finds the clause:

```python
_ABROAD_CLAUSE = re.compile(r"\boutside\s+(?:the\s+)?(?:geographical\s+limits\s+of\s+)?india\b"
                            r"|\bgeographical\s+(?:limits|scope)\b", re.IGNORECASE)
```

The other finds a place abroad in the question: "abroad", "overseas",
"outside India", and a list of the places Indians most often travel to
(Dubai, Singapore, Thailand, London, the USA, ...). It is a list, so a place
not on it gets no line, which is the behaviour from before this change,
never a wrong one. "Foreign" was left out on purpose: "foreign body removal"
is a medical procedure.

Run against every one of the 87 questions in the repository, it fired on
Dubai and the abroad questions and nowhere else.

## Failure 88: a hedged line changed nothing

The first line, placed right after the waiting-period block:

```
THE PERSON MENTIONS A PLACE OUTSIDE INDIA (a lookup on their words, not a judgement).
- clause 22 is about treatment taken outside India. Decide from the
  description WHERE the treatment itself was taken: if outside India, read
  this clause as deciding the claim; if in India, it does not apply.
```

Result: **no change at all**. Dubai, Singapore and London stayed
`conditional`, 3 samples of 3. Reading the prompt showed why. The line sat
between two firm statements:

```
- No waiting period blocks this claim. Some other clause decides it.
...
- APPLIES: clause 9: a 5% co-payment is taken from EVERY claim this policy pays ...
```

A conditional instruction ("if outside India, read this clause as
deciding") lost to the firm lines around it. M22 had found the same thing:
the model acts on what a line opens with and states plainly, not on what it
asks the model to work out.

The second line opens with the fact the lookup found:

```
- The person names "Singapore", which is outside India. Clause 22 excludes
  treatment taken outside India. Unless the description says the treatment
  itself was taken in India, clause 22 refuses this claim.
```

Singapore, London and Dubai became `not_covered`, citing 22, 3 samples of 3.
**But the Chennai trap was refused**, 3 of 3. The model didn't count
"admitted to a hospital in Chennai" as "the description says the treatment
was taken in India". That's the costly direction: a valid claim told it
will be refused.

## The fallback, and keeping the test honest

The fix follows Concept 53, *a fix that acts in one direction only*. The firm
line is used only when the question names nowhere in India. If it also names
an Indian city, "India", "home", "came back" or "returned", where the
treatment happened is a real judgement, and the soft line is used:

```python
if _MENTIONS_INDIA.search(scenario):
    # Both places named, so where the treatment happened is a judgement.
    ...
```

A miss in that list (a small Indian city, no "came back") gives the firm
line, so its ceiling is marked with a `ponytail:` comment.

The fallback was designed *after* reading the Chennai failure, so Chennai
can no longer measure it honestly. Before writing the fallback's code, three
more questions were added and the file's notes mark Chennai as no longer
clean:

- "I came back from a work trip to Dubai ... admitted to a hospital in Kochi"
  (expected `conditional`)
- "After returning from a holiday in Bali, I developed dengue and was
  hospitalised in Nashik" (expected `conditional`; Nashik is deliberately not
  in the city list, so only "returning" can trigger the fallback)
- "I flew from India to visit my brother in Toronto and was admitted to a
  hospital there" (expected `not_covered`). This one measures the fallback's
  cost: "India" makes the line soft, and the soft line was measured not to work.

## Measuring

Majority of three, Ollama 0.35.1, all eight questions about place, on the
M24 code and on M25:

| Question | M24 | M25 |
|---|---|---|
| Singapore (treated there) | conditional ✗ | not_covered ✓ |
| London (planned surgery there) | conditional ✗ | not_covered ✓ |
| Dubai (`star-treatment-abroad`) | conditional ✗ | not_covered ✓ |
| Toronto ("flew from India") | conditional ✗ | conditional ✗ |
| Mumbai | conditional ✓ | conditional ✓ |
| Thailand, treated in Chennai | conditional ✓ | conditional ✓ |
| Dubai trip, treated in Kochi | conditional ✓ | conditional ✓ |
| Bali holiday, treated in Nashik | conditional ✓ | conditional ✓ |
| **right** | **4/8** | **7/8** |

One blemish: in the Kochi question, 2 of 3 answers also list clause 22 among
their reasons, though the verdict is right.

The request comparison (every reasoning request hashed, model stubbed) shows
**0 of 78** synthetic questions changed and **1 of 10** Star questions
(Dubai), so every other measurement stands. Star overall: right verdicts
8/10 → **9/10**, answers citing the deciding clause 7/10 → **8/10**.

## Closing out M25

M25 is M21's pattern applied to a new fact: read the person's words with a
fixed pattern, name the clause, and leave the judgement to the model. It
added two lessons:

- **A hedged instruction can do literally nothing** next to firm ones
  (Failure 88). The same idea worked when the line opened with the fact.
- **A firm line needs a way out** when the question gives evidence both
  ways, so the firm version is limited to the questions where its mistake
  can't happen.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_scenario.py -q -k abroad
```

**Predict before you look:** "I had a foreign body removed from my eye in
Pune." Which line does the prompt get: firm, soft, or none?

<details>
<summary>Answer</summary>

None. "Foreign" is deliberately not in the abroad pattern and nothing else
in the question names a place abroad, so `abroad_clauses` returns an empty
list and the block isn't rendered. "Pune" only matters once a place abroad
has been found, when it switches the line to the soft one.
</details>

---

# M26 — Quoting a real policy's list

## The problem

M23 made the waiting-period line for a list-type period (specified diseases,
maternity) quote the clause's own list, because the model kept applying the
list to treatments that weren't on it. It quoted only clauses of 600
characters or less:

```python
_QUOTE_LIST_CHARS = 600
```

The synthetic clause is about 290 characters. Star Health's
specified-disease clause, `2#2`, is about 1,970: conditions A to E about
how the period works, then "List of specific diseases/procedures", twenty
items under "24 Months waiting period" and two under "36 Months". So on the
one real policy, the list was never shown in the line.

The question `star-dengue-first-month` ("admitted to hospital with dengue
fever 20 days after buying this policy") had the right verdict,
`not_covered`, for the wrong reason, 3 samples of 3:

> "The claim for dengue fever is not covered because the 24/36-month
> specific disease waiting period for the treatment of dengue fever has not
> yet been served."

Dengue isn't on Star's list. The deciding clause is the 30-day initial
waiting period, `3#2`.

## Why 600 was too cautious

M23 chose 600 to avoid crowding the prompt: the waiting-period lines sit
outside the *clause budget*, the room the shortlist reserves for clause
text. Checking the numbers showed the worry didn't apply:

- The context window (`num_ctx` in `api/app/config.py`) is 28,672 tokens.
- A policy with more than `PICK_ABOVE_TOKENS` (6,000) tokens of clause text
  is narrowed by the picking step first, so a real prompt sits far below
  the window. Star's dengue prompt is around 8,000 tokens.
- And if a prompt ever did overflow, the client (`_check_context` in
  `api/app/llm/client.py`) detects the truncation from Ollama's exact
  token count, `prompt_eval_count`, and raises an error instead of
  returning the answer. A truncated prompt is never answered silently.

Star's whole clause costs about 560 extra tokens. The limit was raised to
2,500 characters:

```python
_QUOTE_LIST_CHARS = 2_500
```

M23's alternative, extracting only the list (from "List of" onward), was
not built. It would be more code to keep two lines of prose out of a prompt
with plenty of room.

## Measuring

The request comparison (every reasoning request hashed, model stubbed)
found **0 of 78** synthetic questions changed, none of the seven
place questions from M25, and **1 of 10** Star questions: dengue, the only
one with an unserved list-type period whose list was over 600 characters.

Majority of three, Ollama 0.35.1:

| | M25 | M26 |
|---|---|---|
| `star-dengue-first-month` verdict | not_covered | not_covered |
| cites the deciding clause `3#2` | 0 of 3 | **3 of 3** |

Star overall: 9 of 10 right verdicts, unchanged; answers citing the
deciding clause **8 → 9 of 10**. The one wrong verdict left is
`star-gallstones-no-timing` (`conditional` where the answer key says
`insufficient_information`, with sound reasoning; see Failure 87).

## The point

A limit chosen to be safe should be checked against the real constraint
once a real case runs into it. 600 was a guess at a risk; the context
window, the picking step and the overflow check together were the actual
protection, and they had room to spare.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_waiting.py -q -k "list"
```

**Predict before you look:** with the limit at 2,500, does the synthetic
policy's specified-disease line change at all compared with M23?

<details>
<summary>Answer</summary>

No. Its clause is about 290 characters, under both limits, so it was quoted
under M23 too. That's why the request comparison found 0 of 78 synthetic
questions changed.
</details>

---

# M27 — HDFC's waiting periods

## The problem

Until M27, scenarios were measured on two documents: the synthetic policy
written for this repository, and Star Health's Arogya Sanjeevani. The
repository fetches a third real wording, HDFC ERGO's **my: Optima Secure**
(`evals/real/sources.json`), the longest of the three, but no question had
ever been asked of it.

Listing the clauses the pipeline treats as waiting periods on HDFC showed a
problem before any question was asked. Two of them were rows of the
plan-comparison table near the end of the PDF:

```
1.5#2 | waiting_period [60]  | 1.5 Pre-Hospitalization 60 days 60 days 60 days 60 days (India only) 60 days 60 days 30 days
1.6#2 | waiting_period [180] | 1.6 Post-Hospitalization 180 days 180 days 180 days 180 days (India only) 180 days 180 days 60 days
```

Those are **cover windows**: how long before admission and after discharge
the policy pays for tests and medicines. A waiting period counts from the day
the *policy* began; a cover window counts from one *hospital stay*. Typed as
waiting periods, they tell every claim in the first six months that a
"180-day waiting period still applies and blocks" it.

(The ids with `#` are repeats of a clause number. HDFC numbers its benefits
1.1 to 1.8 in Section B and again in the annexure table, so the table's 1.6
becomes `1.6#2`. See `citation_ids()` in `api/app/pipeline/scenario.py`.)

## Questions written before the code

`evals/real/scenarios-hdfc-optima-secure.json`, seven questions written from
the policy's own wording, before any code and before any run:

- Three payable claims inside the first six months: typhoid on day 45, dengue
  at four months, and a road accident on day 10. The 30-day initial waiting
  period exempts accidents.
- One refusal: typhoid on day 20, inside the 30-day wait.
- Three about the real windows (clauses 1.5 and 1.6): medicines in the three
  months after discharge (paid), physiotherapy seven months after discharge
  (past the 180 days), and scans 75 days before admission (past the 60 days).

HDFC has no compulsory co-payment, so a payable claim is `covered`, not
`conditional` as on Star.

Two more were added partway through (see below), before any run of them:
a hernia 14 months in, and diabetes held six years before a one-year-old
policy.

**On the code as it stood: 2 of 7 right**, majority of three. All three
payable claims were refused, the accident on day 10 included.

## Step 1: a waiting period says so

The table rows were found by reading, so the first question was how to tell
them apart from a real waiting period without an LLM. Every clause typed
`waiting_period` in all five policies the repository has (three synthetic,
Star, HDFC) was listed with one test: does its heading or text contain the
letters "wait"?

```
WAIT  3.1, 3.2, 3.3, 3.4          synthetic (and the same in the other two)
WAIT  2#2, 3#2, c7                Star
WAIT  2.10#2, 2.12, 1#2, 1#4, 6   HDFC
----  1.5#2, 1.6#2                HDFC's table rows
```

A clean split. In `api/app/pipeline/waiting.py`:

```python
_WAITS = re.compile(r"wait", re.IGNORECASE)
...
if len(_DEFINES.findall(text)) >= _DEFINITIONS_BLOCK or not _WAITS.search(f"{heading} {text}"):
    continue
```

The skip sits beside M24's rule for definitions blocks, in `evaluate()`, so
every caller goes through it. Skipping a clause here removes only its
computed line; the clause's text still reaches the model. The comment marks
the ceiling with `ponytail:`: a real waiting period that never says "wait"
would lose its line.

Nineteen existing tests failed at once. Their fake clauses had empty text, so
none said "wait". A real clause always has text, so the test helper got a
neutral default (`text: str = "Waiting period"`), and one fixture that
abbreviated the IRDAI pre-existing-disease wording got back the sentence it
had dropped: "...then waiting period for the same would be reduced to the
extent of prior coverage."

**Measured: verdicts unchanged, 2 of 7.** Citations improved (citing the
deciding clause went from 0–20% of answers to 40%), and quotes failing the
verbatim check fell from 6–7 a run to 1–2. But nothing that was refused
became paid.

## Failure 89: five more lines said "blocks"

Reading the prompt for the accident question showed why:

```
- clause 2.10#2 (Global Health Cover (Emergency & Planned Treatments) (cont.)): requires 36 months,
  policy held 10 days -> this waiting period still applies and blocks treatment covered by THIS clause
- clause 2.12 (PED waiting period modification): requires 36 months ... still applies and blocks ...
- clause 1#2 (Waiting Periods): requires 36 months ... still applies and blocks ...
- clause 1#4 (Waiting Periods (cont.)): requires 1 month ... blocks ..., EXCEPT where the situation
  falls within this clause's own exception: "claims arising due to an accident"
- clause 6: requires 1 month ... still applies and blocks treatment covered by THIS clause
```

Four firm "blocks" lines and one exception. Only 1#4 was a correct line. The
table rows had been two of seven false signals, and removing them left five.
This is Failure 86 again (removing one wrong signal lets the next one
decide), and it is worth naming as its own failure because of how it was
found: the fix was measured, and the measurement said *nothing moved*.
Without the measurement, the table-row fix would have looked like the whole
answer.

Each of the remaining lines was wrong for a different reason:

| Clause | What it is | Why it read as a bar on everything |
|---|---|---|
| 1#2 | Pre-existing diseases (36 months) **and** the specified-disease list (24 months), together | The segmenter never split them |
| 2.12 | An optional cover that changes the pre-existing wait | Its heading says "PED"; the code looked for "pre-existing" |
| 2.10#2 | 36 months for *planned treatment abroad*, under an optional cover | Nothing marked it as narrow |
| 6 | An add-on: 30 days for asthma, BP, cholesterol, diabetes | Nothing marked it as narrow |

## Step 2: splitting HDFC's lettered exclusions

IRDAI standardises health-policy exclusions and gives each a code: `Excl01`
pre-existing diseases, `Excl02` specified diseases, `Excl03` the 30-day wait,
and so on to `Excl18`. Star numbers these items "1.", "2.", "3.", which the
segmenter already treats as clause starts. HDFC letters them, with the letter
on its own line:

```
a.                                      (bold, its own line)
Pre-Existing Diseases: Code – Excl01    (bold, same row)
i.
Expenses related to the treatment of a pre-existing disease ...
```

The segmenter (`api/app/pipeline/segment.py`) joins a lone marker to the
words beside it, but only recognised "2." or "(a)" as markers, never "a.". So
everything from "1. Waiting Periods" ran on until the 3,000-character cap cut
it into `1#2`, `1#3`, `1#4`.

Accepting every "a." as a clause start would be wrong: HDFC also sets roman
sub-items "i." the same way, and those must stay inside their clause. What
makes a lettered line safe to believe is the code:

```python
_RE_MARKER_ONLY = re.compile(r"^(?:\d+(?:\.\d+)*\.?|\((?:[a-zA-Z]|[ivxlIVXL]+|\d+)\)|[a-z]\.)$")
_RE_EXCL_ITEM = re.compile(r"^[a-z]\.\s+[^.]*?\bCode\s*[–-]?\s*Excl\s*(\d+)", re.IGNORECASE)
...
    if m := _RE_EXCL_ITEM.match(text):
        return f"Excl{int(m.group(1)):02d}", True
```

The code also becomes the clause's **id**. A letter would repeat
(`a`, `a#2`, ...); "Excl01" is unique and printed in the clause, so the id the
model cites is a string the reader can find on the page. Three items print
the code at the *end* of their text ("... thereof. Code – Excl12"); the
`[^.]*?` refuses those, so they stay inside the item above them. That is
harmless so far and marked with `ponytail:`.

Comparing the segmentation of all six PDFs before and after: **only HDFC
changed.** Star, Niva and the three synthetic policies segment exactly as
before. HDFC's Section C became `Excl01`, `Excl02`, `Excl03`, and its
standard exclusions became `Excl04` onwards.

The same step taught the pre-existing check the abbreviation:

```python
_PRE_EXISTING = re.compile(r"(?i:pre[\s-]?existing)|\bPED\b")
```

"PED" is case-sensitive, because lower-case "ped" turns up inside words.

### A side effect: re-analysis

New clauses are new text, so analysis (stage 3, the model reading each clause
in batches of five) ran again. Changing one clause shifts every batch after
it, so most of HDFC was re-read, which took over ten minutes. One batch
failed: clause 92, the insurance ombudsman's address list, made the model's
answer run past its token limit three times. It decides no claim, so it was
left unanalysed and noted.

The re-analysis also moved something that mattered: **Excl02 came back typed
`exclusion` with no waiting period at all.** Its text says "excluded until
the expiry of 24 months", but the analysis model didn't record the 24.

### The ids in the questions

The questions had been written citing the 30-day clause as `1#4`, its id at
the time. After the split it is `Excl03`. Only the id was renamed in the
case file; no expectation changed, and the file's `_why` says so.

### Two more questions

The split exists to separate the pre-existing bar from the specified-disease
bar, and none of the seven questions involved either. Two were added before
any run of them: hernia 14 months in (`not_covered`, citing `Excl02`) and
diabetes held six years before a one-year-old policy (`not_covered`, citing
`Excl01`). On the old code both were refused, 3 of 3. That proves little: the
old code refused nearly everything, and their ids didn't exist yet.

**Measured after step 2: 4 of 9.** Medicines-after-discharge became right.
Hernia became wrong ("covered"), because Excl02 had lost its period and got no
line.

## Step 3: a period under a benefit bars only that benefit

2.10#2 and 6 are waiting periods for one benefit, not for the policy. A
question across all five policies: where does each waiting period sit?

```
'SECTION 3 - WAITING PERIODS'               synthetic 3.1–3.4
'SECTION 1 - COVER AND EXCLUSIONS'          synthetic mini 3.2
'STANDARD EXCLUSIONS'                       Star 2#2, 3#2
'SECTION C. WAITING PERIOD AND EXCLUSIONS'  HDFC Excl01, Excl03
'SECTION B. BENEFITS'                       HDFC 2.10#2, 2.12
'SECTION D. GENERAL TERMS AND CLAUSES'      HDFC 6 (the annexure, which the segmenter files under D)
```

Every general period sits in a section named for waiting periods or
exclusions. So a period anywhere else is treated like a list-type period (M22,
M23): its line says it "bars ONLY the treatments this clause lists" and quotes
the clause.

```python
_GENERAL_SECTION = re.compile(r"wait|exclu", re.IGNORECASE)
...
section = getattr(clause, "section_path", "") or ""
listed = (bool(_LISTED.search(f"{heading} {text[:_OPENING_CHARS]}"))
          or bool(section) and not _GENERAL_SECTION.search(section))
```

An unknown (empty) section changes nothing. The ceiling, marked
`ponytail:`, is a general period in an oddly named section, which would get
the narrow line and could pay claims it bars.

**Measured: 6 of 9.** Typhoid on day 45, dengue at four months and the
accident on day 10 all became right, 3 of 3. Diabetes became wrong.

### Failure 87 again: diabetes had been right for a false reason

Diabetes had been refused under the false 2.10#2 "blocks" line. With that
line gone, the model read the hedged pre-existing line:

```
- clause Excl01 (...): concerns ONLY an illness the person already had when the policy
  began. Unless the description says this one had begun by then, it is not the reason
  for this claim. (Requires 36 months; policy held 1 year, so not yet served.)
```

and paid, reasoning that "clause 2.12 ... states that the Pre-existing Disease
Waiting Period has been modified to 12 months". Clause 2.12 says the modified
period is "as stipulated in the Policy Schedule"; the 12 was invented.

## Step 4: two more lines

**Reading a period off a clause headed as one.** Excl02 has a heading that
says what it is ("Specified Disease/Procedure waiting period") and a text that
says how long ("24 months"). Reading that number is extraction, not
judgement:

```python
if not required and _HEADED_WAIT.search(heading):
    required = sorted({int(n) * _DAYS_PER[u.lower()] for n, u in _DURATION.findall(text)})
```

It only runs when analysis gave no period. Listed across all five policies,
it fires on exactly one clause: Excl02, giving 720 days.

**A firm line for an illness settled as older than the policy.** The facts
for the diabetes question already said `pre_existing_condition: yes` (six
years against one). M22 had kept the hedged line for "yes", and the reason
matters. M22's Failure 84 was a *served* wait: blood pressure held since
before a policy five years old, refused under an unrelated clause once the
facts block said "yes". In M22 the firm line for an *unserved* wait had in
fact fixed its question, 5 of 5; it was dropped along with the lookup that set
"yes". So the firm line returns, for unserved periods only:

```python
if (self.pre_existing and self.condition_predates_policy == "yes"
        and self.status is WaitingStatus.NOT_SERVED):
    return (
        f"{who}: concerns ONLY an illness the person already had when the "
        f"policy began. This one had begun by then, and the policy has been "
        f"held {_human(self.held_days)} of the {requires} required, so this "
        f"waiting period still applies and blocks this claim"
    )
```

M22's test, which pinned "yes keeps the hedge", was rewritten to pin the new
decision: "unknown" keeps the hedge; "yes" with an unserved wait is firm.

The request comparison found **0 of 95** questions on the other sets changed.

**Measured: 6 of 9.** Hernia and diabetes became right, 3 of 3. But typhoid
on day 20 and dengue at four months, both right before, went wrong.

## Failure 90: a long quote drowned the line beside it

Excl02's new line quoted the whole clause, about 2,400 characters: five
conditions (i. to v.) and then the list. In the dengue answer the model
applied the list to dengue ("a 24-month waiting period ... applies"). In the
typhoid-on-day-20 prompt the correct 30-day line came *after* that quote:

```
- clause Excl02 (...): bars ONLY the treatments this clause lists: "i. Expenses related to
  the treatment of the listed Conditions ... [2,400 characters] ..."
- clause Excl03 (c. 30-day waiting period: Code – Excl03): requires 1 month, policy held
  20 days -> this waiting period still applies and blocks treatment covered by THIS clause,
  EXCEPT where the situation falls within this clause's own exception: "claims arising due
  to an accident". ...
```

Two fixes were considered.

- **Put periods that bar everything before periods that bar a list**, in
  `render()`, which already orders lines because "the model tends to cite the
  first bar it reads". Tried, and rejected *without a model run*: the request
  comparison showed it changed **25** requests in the other sets. That is a
  re-measurement of a quarter of everything for one HDFC question.
- **Quote only the list**, from "List of ..." onward, when a clause has one.
  M26's code comment had named this as the upgrade "if a real policy shows
  that costs answers". The synthetic clauses don't contain "List of", so they
  can't change. Chosen.

```python
_LIST_STARTS = re.compile(r"list\s+of\b", re.IGNORECASE)
...
listing = body[m.start():] if (m := _LIST_STARTS.search(body)) else body
```

The request comparison: **1 of 95** changed, Star's dengue question, whose
2#2 clause also says "List of specific diseases/procedures".

**Measured: HDFC 7 of 9.** Dengue became right again. Typhoid on day 20 stayed
wrong, 3 of 3, with this reasoning:

> "The 20 days since you bought the policy is not long enough to trigger the
> waiting periods listed in the policy."

The line it was given says "requires 1 month, policy held 20 days -> this
waiting period still applies and blocks". The model read the comparison
backwards. The same line was answered correctly before Excl02 gained its
line, so a busier prompt seems to tip it. One untried idea: state both sides
in the same unit ("requires 30 days, policy held 20 days"). That changes the
wording for every short wait in every set, so it is left for its own
milestone.

**And Star's dengue** kept its right verdict, `not_covered`, but stopped
citing the 30-day clause 3#2 (3 of 3 under M26, 0 of 3 now). It now gives the
pre-existing exclusion as the reason, for a fever that began 20 days into
cover. Quoting only the list helped HDFC's dengue and hurt Star's. That trade
was kept on purpose: HDFC's case is a claim a person would act on (paid, with
a false warning that a 24-month wait applies), while Star's is the wrong
reason for a right refusal.

## Results

HDFC, majority of three, Ollama 0.35.1:

| Question | Expected | Before M27 | M27 |
|---|---|---|---|
| typhoid, day 45 | covered | not_covered ✗ | covered ✓ |
| dengue, 4 months | covered | not_covered ✗ | covered ✓ |
| typhoid, day 20 | not_covered | not_covered ✓ | covered ✗ |
| accident, day 10 | covered | not_covered ✗ | covered ✓ |
| medicines after discharge | covered | not_covered ✗ | covered ✓ |
| physiotherapy 7 months after | not_covered | conditional ✗ | conditional ✗ |
| tests 75 days before | not_covered | not_covered ✓ | not_covered ✓ |
| hernia, 14 months | not_covered | not_covered ✓ | not_covered ✓ |
| diabetes, held before | not_covered | not_covered ✓ | not_covered ✓ |
| **right** | | **4/9** | **7/9** |

The "before" column is honest but flattering: its four right answers are all
refusals from a system that refused almost everything. Typhoid on day 20 was
right before for the same reason.

The other sets: **0 of 95** requests changed except Star's dengue question.
Synthetic 36/40 and held-out 30/38 stand. Star: 9 of 10 verdicts, unchanged;
citing the deciding clause 9 → **8 of 10**.

Still open on HDFC:

- Typhoid on day 20 is paid (Failure 90).
- Physiotherapy seven months after discharge is `conditional`, not
  `not_covered`: clause 1.6 states the 180 days, but analysis never recorded
  it as a cover window, so no window line is computed for it.
- Clause 92 (the ombudsman list) is unanalysed.

## Closing out M27

M27 is the first time the pipeline met a second real policy's structure, and
most of what broke was **reading**, not reasoning: table rows typed as
waiting periods, lettered exclusions never split, an abbreviation, optional
covers with nothing to mark them narrow. Each fix was a fixed rule checked
across all five policies before it was written: "says wait", "has an IRDAI
code", "sits in a waiting-period section", "headed as a waiting period". Each
was confirmed to touch only HDFC with the request comparison.

Two lessons repeat from earlier milestones, and that repetition is the
lesson:

- **Measure each fix alone** (Failure 89). The first fix was correct and moved
  nothing; only the measurement showed there were five more false lines.
- **A right answer can rest on a false line** (Failure 87). Diabetes was right
  until the false line under it was removed.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_waiting.py tests/test_segment.py -q
```

**Predict before you look:** HDFC's item "i. Treatment for Alcoholism, drug or
substance abuse ... thereof. Code – Excl12" has an IRDAI code. Does the
segmenter start a new clause for it?

<details>
<summary>Answer</summary>

No. `_RE_EXCL_ITEM` needs the code inside the item's *title*: `[^.]*?` allows
no full stop between the letter and "Code". Here the code comes after
"thereof.", at the end of the text, so the line doesn't match, and the item
stays inside the clause above it (Excl11). That is the `ponytail:` ceiling in
`segment.py`.
</details>

---

# M28 — Two explanations that measurement refuted

## The problem

After M27, one HDFC question in the costly direction was still wrong. On HDFC
ERGO's Optima Secure, `hdfc-typhoid-20-days` asks:

> "I was admitted to hospital with typhoid 20 days after buying this policy.
> It is my first health insurance."

The policy excludes any illness in the first 30 days (IRDAI code Excl03, the
"30-day waiting period"), so the answer is `not_covered`. The system answered
`covered`, 3 samples of 3. The waiting-period line it was given (built by
`describe()` in `api/app/pipeline/waiting.py`) was correct:

```
- clause Excl03 (c. 30-day waiting period: Code – Excl03): requires 1 month, policy
  held 20 days -> this waiting period still applies and blocks treatment covered by
  THIS clause, EXCEPT where the situation falls within this clause's own exception:
  "claims arising due to an accident". Decide from the description whether it does
```

and the model's reasoning said:

> "The 20 days since you bought the policy is not long enough to trigger the
> waiting periods listed in the policy."

M28 tested two explanations for that, one at a time. Both were wrong. M28
keeps no code change; what it keeps is three new questions and the evidence.

## Questions written first

Typhoid on day 20 had been read closely while forming the explanations, so it
could no longer measure them fairly. Three questions were added to
`evals/real/scenarios-hdfc-optima-secure.json` before any code or run:

- malaria, policy bought 25 days ago (`not_covered`, citing Excl03)
- gastroenteritis, three weeks in (`not_covered`, citing Excl03)
- pneumonia, 40 days in (`covered`: the 30 days are served)

On the M27 code, majority of three: malaria and pneumonia right,
**gastroenteritis paid**, the same failure as typhoid.

## Explanation 1: the units

`_human()` writes each number in its natural unit: 30 days becomes "1 month",
20 days stays "20 days". The model reasoned as if 20 days hadn't reached the
bar, so perhaps it was failing the conversion. This module exists to keep
arithmetic away from the model, so a line that makes it compare months with
days is a plausible leak.

**The change:** when the time held is in days, state the requirement in days
too ("requires 30 days, policy held 20 days"). Lines with the policy held in
whole months stay unchanged.

**Request comparison:** 9 questions changed: 2 synthetic, Star's dengue, and
6 on HDFC.

**Result:**

| | M27 | units |
|---|---|---|
| HDFC typhoid day 20, gastroenteritis 3 weeks | paid ✗ | paid ✗ |
| HDFC malaria 25 days | refused ✓ | **paid ✗** |
| Star dengue cites the 30-day clause 3#2 | 0 of 3 | 3 of 3 |

With the line now reading "requires 30 days, policy held 25 days -> this
waiting period still applies and blocks", the malaria answer said:

> "The policy does not have any waiting periods or exclusions that apply to
> malaria."

So the units were not the problem. The model wasn't misreading the
comparison. It wasn't acting on the line at all.

## Explanation 2: the order

Reading the whole waiting-period block for those prompts showed where the
30-day line sat:

```
- clause 2.10#2 (...): bars ONLY the treatments this clause lists: "..." If this
  treatment is not one of them, this period is not the reason for this claim. ...
- clause Excl02 (...): bars ONLY the treatments this clause lists: "List of ..." ...
- clause Excl03 (...): requires 1 month, policy held 20 days -> ... still applies and blocks ...
- clause 6: bars ONLY the treatments this clause lists: "..." ... not the reason ...
- clause 2.12 (...): concerns ONLY an illness the person already had ...
- clause Excl01 (...): concerns ONLY an illness the person already had ...
```

One firm line among five that each end "...is not the reason for this claim".
`render()` already orders these lines on the principle that "the model tends
to cite the first bar it reads", so the second explanation was: put periods
that bar *everything* before periods that bar only a list. The units change
was reverted first, so the two were measured separately.

```python
def first(check: WaitingCheck) -> tuple[bool, bool, bool]:
    return (not check.named, check.listed, check.pre_existing)
```

**Request comparison:** 32 questions changed across all sets, because the
synthetic policy and Star both have list-type periods. All 32 were measured
on both codes, majority of three (the M27 answers replayed from the cache):

| Set (changed questions only) | M27 | ordering |
|---|---|---|
| Synthetic main (12) | 10 | 10 |
| Synthetic held-out 1–3 (7) | 4 | 4 |
| Synthetic held-out 4 (4) | 3 | **1** |
| Star (2) | 1 | 1 |
| HDFC (7) | 5 | **4** |
| **Total** | **23** | **20** |

Typhoid on day 20 and gastroenteritis stayed paid. Pneumonia at 40 days,
whose 30 days are served, was now refused. On held-out 4, two payable claims
(`ho4-before-buying-checked-cover`, `ho4-acute-no-history`) were refused,
citing the pre-existing diseases clause 3.2, which their answer key forbids.
Star's dengue cited 3#2 again, as it had under the units change.

Ordering was reverted too.

## Failure 91: two plausible explanations, both wrong

Each explanation came from reading the prompt and the answer, and each
matched the evidence available when it was formed:

- The model's own words ("not long enough to trigger") pointed at the
  comparison, so the units looked guilty. A rewritten comparison changed
  nothing, and the next answer ("no waiting periods ... apply") showed the
  line wasn't being used at all.
- The block's layout pointed at position. Moving the line up changed the
  answers of other questions far more than of the ones it was aimed at.

A 7B model's reasoning text describes its answer; it is not a trace of how
the answer was reached. "Not long enough to trigger" sounds like a unit
mistake, but the same verdict came back once there was no unit left to get
wrong. The only reliable test of an explanation was the one used here:
change exactly one thing, run the request comparison to know which questions
it can affect, and measure all of them.

What is still known, and not explained: on HDFC, a correct firm 30-day line
loses to a block of narrower lines. Before M27 gave Excl02 a line, typhoid on
day 20 was refused, 3 of 3. So the clearest remaining lead is the *number* of
"not the reason" lines around the firm one, not their order. That is not
tested here.

## Results

No code changed in M28. HDFC, majority of three, on the M27 code:
**9 of 12** (M27's 7 of 9, plus malaria and pneumonia right and
gastroenteritis wrong). Every other set is exactly as M27 left it.

## Check it yourself

```bash
python evals/run_scenario_eval.py --policy samples/real/hdfc-optima-secure.pdf \
  --cases evals/real/scenarios-hdfc-optima-secure.json \
  --only hdfc-typhoid-20-days,hdfc-gastroenteritis-three-weeks --repeats 3
```

**Predict before you look:** the units change made Star's dengue question cite
the 30-day clause again, 3 of 3. Was that a reason to keep it?

<details>
<summary>Answer</summary>

Not on its own. It was one citation gained against one HDFC verdict lost
(malaria at 25 days, paid). A wrong verdict is the costlier error: a person
inside the 30-day wait would be told they're covered. The citation gain is
recorded here as evidence that the same-unit wording helps Star, for any
later change that also fixes what it broke on HDFC.
</details>

---

# M29 — HDFC's 180-day window after discharge

## The problem

Health policies pay for treatment around a hospital stay only within stated
**cover windows**: tests and medicines for so many days *before admission*,
follow-up care for so many days *after discharge*. Deciding whether an
expense fell inside a window is a comparison of two numbers, so since M11 it
has been done in code (`api/app/pipeline/window.py`), not by the model. The
model is given a line such as:

```
WHEN THE EXPENSES FELL (arithmetic, not opinion).
- OUTSIDE THE WINDOW: clause 2.3 pays only for expenses within 90 days after
  discharge. These were 120 days after discharge, so clause 2.3 does NOT pay for them.
```

The code needs two operands. The *expense* side comes from fact extraction
(`expense_timing_value`, `_unit`, `_anchor`). The *window* side,
`cover_window_days` and `cover_window_anchor`, comes from analysis: the model
reads each clause once when the policy is uploaded and records the window it
states.

On HDFC ERGO's Optima Secure, clause 1.6 says:

> "Such expenses shall be indemnified if the same were incurred upto 180 days
> unless otherwise specified in the Policy Schedule, immediately post the date
> of discharge from the Hospital."

Analysis recorded no window for it. So `hdfc-medicines-seven-months-after`
(physiotherapy seven months after discharge, expected `not_covered`) got no
window line at all, and was answered `conditional`, 3 of 3. The facts were
right: `expense_timing: 7 months, after_discharge`.

## Questions written first

That question had been read while finding the problem, so two fresh ones were
added to `evals/real/scenarios-hdfc-optima-secure.json` before any code:

- medicines five months after discharge for a kidney infection (`covered`,
  inside 180 days)
- a follow-up scan 200 days after discharge for a fractured hip
  (`not_covered`)

On the code before M29, majority of three: five months right, **the scan
`conditional`**.

## The fix: read a window stated in figures

The same move as M27's fallback for a waiting period: when analysis gave a
clause nothing, read the number off the clause's own words. A window stated
in figures has a recognisable shape: a number of days, then within one
sentence "prior to / preceding / before ... admission" or "post / following /
after ... discharge".

Before writing it, the pattern was run over every clause in five policies:

```
synthetic 2.2, 2.3; Star 4, 5    windows from analysis; pattern finds nothing (numbers spelt out: "sixty days")
HDFC 1.5    coverage        analysis: 60 before_admission   pattern: 60 before
HDFC 1.6    coverage        analysis: none                  pattern: 180 after
HDFC 2.14   definition      analysis: none                  pattern: 30 before
HDFC 2.15   waiting_period  analysis: none                  pattern: 60 after
```

2.14 and 2.15 are optional covers that *change* the window for one plan
("Modification of Post-Hospitalization expenses days from 180 days ... to 60
days ... This option is inbuilt in Optima Lite plan"). Read as windows, 2.15
would tell every claim 61 to 180 days after discharge that it is outside a
60-day window. A cover window is a property of a benefit, so the fallback only
reads coverage clauses, which leaves 1.6 alone.

In `api/app/pipeline/window.py`:

```python
_STATED_WINDOW = re.compile(
    r"\b(\d+)\s*days?\b[^.]{0,80}?\b(?:(?:prior\s+to|preceding|before)\s+(?:the\s+)?(?:date\s+of\s+)?admission"
    r"|(?:post|following|after)\s+(?:the\s+)?(?:date\s+of\s+)?discharge)", re.IGNORECASE)
...
        if not window and getattr(clause, "clause_type", "") == "coverage":
            text = unicodedata.normalize("NFKC", getattr(clause, "text", "") or "")
            if m := _STATED_WINDOW.search(" ".join(text.split())):
                window = int(m[1])
                clause_anchor = (Anchor.AFTER_DISCHARGE if "discharge" in m[0].lower()
                                 else Anchor.BEFORE_ADMISSION)
```

`[^.]{0,80}?` allows up to 80 characters within the same sentence between the
number and the anchor: 1.6's "unless otherwise specified in the Policy
Schedule, immediately" is about 60.

**Request comparison:** 0 of 95 questions outside HDFC changed, and 3 of 14 on
HDFC: the three whose facts place an expense after discharge. The
five-months question is not among them; its facts didn't place the medicines
after discharge, so no window line is raised (and it was already right).

## Measuring

Majority of three, Ollama 0.35.1:

| Question | Before | M29 |
|---|---|---|
| scan, 200 days after (fresh) | conditional ✗ | **not_covered ✓** |
| medicines, 5 months after (fresh) | covered ✓ | covered ✓ |
| medicines, 3 months after | covered ✓ | covered ✓ |
| physiotherapy, 7 months after | conditional ✗ | conditional ✗ |

HDFC overall: **11 of 14**.

## Failure 92: a firm, correct line, and an invented co-payment

Physiotherapy at seven months now gets the right line:

```
- OUTSIDE THE WINDOW: clause 1.6 pays only for expenses within 180 days after
  discharge. These were 210 days after discharge, so clause 1.6 does NOT pay for them.
```

and the model still answered `conditional`:

> "You are covered for physiotherapy, but the claim will be reduced by a 20%
> co-payment. ... clause 1.24 states that 'No co-payment shall apply if Insured
> Person from Tier 2 avails a treatment in Tier 1.' ..."

There is no 20% anywhere in the prompt. The code's own reductions block,
which lists every co-payment and cap that applies, was empty for this
question. Clause 1.24 is HDFC's premium-tier clause, and its only sentence
about co-payment says when there is *none*. The model built a co-payment from
the word and let it outrank a firm, correct line.

This is not the problem M29 set out to fix (the window line was missing), and
the question had been read closely while fixing it, so tuning the prompt for
it here would fit a known answer. It is recorded and left open. The fresh
200-day scan, with the same line, is refused correctly, 3 of 3. So the line
works, and something about this question pulls the model to 1.24.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_window.py -q
```

**Predict before you look:** HDFC's 2.15 says "Post-hospitalization medical
expenses shall be indemnified only if the same were incurred upto 60 days
immediately post the date of discharge". The pattern matches it. Why does
the fallback never give 2.15 a 60-day window?

<details>
<summary>Answer</summary>

The fallback only reads clauses typed `coverage`, and analysis typed 2.15
`waiting_period`. That restriction is deliberate: 2.15 is an option for one
plan (Optima Lite), and treating its 60 days as the policy's window would
refuse claims between 61 and 180 days after discharge on every other plan.
The test `test_a_window_stated_in_figures_is_read_off_a_coverage_clause`
pins it.
</details>

---

# M30 — Cleanups, part 1: ligatures in the word lookups

## The problem

Some PDFs store "fi", "fl" and "ff" as one character each (a *ligature*;
Concept 54 explains them from scratch). M23 found that Star's "Speciﬁed
disease" clause never matched the pattern `specified`, and fixed it with
Unicode NFKC normalisation inside the waiting-period check
(`api/app/pipeline/waiting.py`). That left every other piece of code that
pattern-matches policy text unprotected.

The one that matters is the word lookup in `api/app/pipeline/scenario.py`.
`named_in_question` and `shared_words` find clauses that use an unusual word
from the person's question ("hernia", "caesarean") and make sure those
clauses are in the prompt. Both build their word lists with `_stems`:

```python
for word in re.findall(r"[a-z]{5,}", text.lower()):
```

`[a-z]` keeps runs of plain letters only, and U+FB01 is not one of them. So
"ﬁstula" splits into "stula", and a question about fistula surgery never
meets the clause that names it.

## Measured before choosing where to fix it

Counting the characters NFKC would change, on every PDF the evals use:

| PDF | changed characters |
|---|---|
| the three synthetic policies | 0 |
| HDFC Optima Secure | 0 |
| Niva ReAssure 2.0 | 119 (fi, fl, ff) |
| Star Arogya Sanjeevani | 100 (fi, fl, no-break space) |

So no fix can change a synthetic or HDFC answer. Only Star has eval
questions among the two affected policies.

## Failure 93: the right normalisation at the wrong layer

The obvious root-cause fix was to normalise once, where text enters the
system: in `ingest.py`, before character offsets are assigned, so the rule
`raw_text[line.char_start:line.char_end] == line.text` still holds. One
change, and no lookup anywhere would ever need to remember ligatures again.
Segmentation was checked first and came out identical on all six PDFs (same
clause count, numbers and headings).

Then Star was measured, majority of three samples, against `main`:

| Star | main | ingest fix |
|---|---|---|
| verdicts right | 9/10 | 8/10 |
| caesarean | right 3/3 | wrong 2/3 |
| dengue cites its waiting period 3#2 | 0/3 | 3/3 |

Comparing the stored clause analyses explained it. Normalising at ingest
changes the text **the model reads**, not just the text the code matches.
Every Star clause containing a ligature became a new prompt and was analysed
afresh, and four of 79 came back different. The costly one was the Table of
Benefits row capping room rent at "2% of the Sum Insured subject to maximum
of Rs.5000/- per day". On `main` it is a `sub_limit` with both caps. After
the change it is `coverage` with no caps at all.

That is not a bug in normalisation. It is a 7B model answering a slightly
different input slightly differently, which is what such models do. But the
lesson is general: **a change to shared input reaches every consumer of that
input, including the model.** The cleanup was meant for the code's lookups.
Fixing it upstream also re-rolled the model's reading of 38 clauses, and that
was a much larger change than intended. Reverted.

## The fix

Normalise where the matching happens, and nowhere else:

```python
# api/app/pipeline/scenario.py, in _stems
text = unicodedata.normalize("NFKC", text)
out: dict[str, str] = {}
for word in re.findall(r"[a-z]{5,}", text.lower()):
```

The other three patterns that read clause text were checked and left alone:
"outside India" / "geographical limits", "rateable / contribution / other
policy", and "Annexure N". None of them contains an f-ligature pair, so a
ligature cannot hide them. Normalising there would protect against nothing
seen.

The test, `test_a_ligature_in_the_policy_does_not_hide_a_named_word` in
`api/tests/test_scenario.py`, puts "ﬁstula" in a clause and asks about
fistula surgery. It fails without the line above and passes with it.

## Measuring

The model's input must now be byte-identical to `main`'s. That is
checkable without reading any answer: the cache key is a hash of the exact
request, so an unchanged request replays and a changed one calls the model.
Running both Star sets on the branch made **0 fresh model calls** for sample
1 (17 questions). Nothing the eval sends changed, so no score can move.

What it buys is the next real policy. A question naming a treatment that the
policy prints with a ligature will now find its clause.

## An environment change found on the way

Ollama had updated itself from 0.35.1 to 0.40.0. Because the runtime version
is part of every cache key (see Failure 70), none of the stored answers
applied any more, and every recorded score was from a version that was no
longer running. The decision was to finish the project on 0.35.1 rather than
re-measure everything. Ollama 0.35.1 now runs from the standalone Windows
build (`ollama-windows-amd64.zip` from the v0.35.1 GitHub release), which
cannot update itself, with the same `OLLAMA_FLASH_ATTENTION=1` and
`OLLAMA_KV_CACHE_TYPE=q8_0`. Re-running Star on `main` under it reproduced
the recorded results exactly (9/10, the same two citation misses).

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_scenario.py -q -k ligature
curl -s localhost:11434/api/version     # must say 0.35.1
```

**Predict before you look:** the grounding check (`app/grounding.py`) also
compares model text with policy text, and it already normalises with NFKC.
Why can it normalise both sides freely, when normalising at ingest broke
things?

<details>
<summary>Answer</summary>

The grounding check normalises only its own copies, for one comparison, and
throws them away. Nothing it does reaches a prompt. Normalising at ingest
replaced the stored text itself, and the stored text is what the model is
shown. Where the change is made decides who sees it.
</details>

---

# M30 — Cleanups, part 2: HDFC's annexures, found inside "Contact Us"

## The problem

M27 recorded that one HDFC clause, the insurance ombudsman's address list,
"made the model's answer run past its token limit three times" during
analysis, and left it unanalysed because an address list decides no claim.

Looking at that clause properly showed the note had the cause wrong. The
clause was the third length-split piece of `2. Contact Us`, and it did start
with ombudsman addresses. But 573 characters in, it ran straight into:

```
Annexure B- Items for which Coverage is not available in the Policy (Non-Medical Expenses)
S. NO. ITEM
1
BABY FOOD
...
68
VASOFIX SAFETY
```

followed by the start of the plan chart (Annexure C). HDFC's three annexures
(A, the ombudsman list; B, 68 non-payable items; C, the plan chart) had all
been segmented as continuation pieces of "Contact Us".

Sending the piece to the model once, alone, showed what happened. It typed
the clause `exclusion` and began copying the items into its `exceptions`
array. Then it repeated the same ten items over and over until it hit its
output limit:

```
"BIRTH CERTIFICATE", "CERTIFICATE CHARGES", "COURIER CHARGES", ...,
"WALKING AIDS CHARGES", "NEBULISATION KIT", "ATTENDANT CHARGES",
"ANY KIT WITH NO DETAILS MENTIONED", "BIRTH CERTIFICATE", ...
```

This is why the client's retry was useless here. On a truncated answer it
doubles `num_predict` (1,600, then 3,200, then 6,400 tokens), which is right
when an answer is merely long. A model stuck in a loop fills any limit. (The
stored analysis on `main` shows a later attempt did eventually get through,
so the "unanalysed" note was also out of date. The segmentation defect
underneath it was not.)

## Why the segmenter missed the headings

`api/app/pipeline/segment.py` starts a new section on a line that is either
larger than body text and bold, or all capitals and shaped like a section
heading:

```python
_RE_SECTION_HEAD = re.compile(
    r"^(?:SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE)\s+[\dIVXL]+\b"
)
```

HDFC's headings fail all three tests. They are body-size type (11.04pt, the
same as the text), in mixed case ("Annexure"), and labelled with letters
(A, B, C) where the pattern accepts only digits and roman numerals.

## Measured before writing the rule

Every line starting with "Annexure", across all six PDFs the evals use:

| PDF | page | bold | line |
|---|---|---|---|
| HDFC | 0 | no | Annexure A / B / C (contents page) |
| HDFC | 15 | no | Annexure B to this Policy incurred in relation to a claim ... |
| HDFC | 31 | no | ANNEXURE B and also available at www.hdfcergo.com. |
| HDFC | 45 | **yes** | Annexure A |
| HDFC | 48 | **yes** | Annexure B- Items for which Coverage is not available ... |
| HDFC | 49 | **yes** | Annexure C - Plan Chart: |
| Niva | 19 | **yes** | Annexure I - The expenses that are not covered ... |
| Niva | 21 | **yes** | Annexure II - List of Insurance Ombudsmen |
| Star | 8 | no | Annexure-A. The list of expenses that are |

Every real heading is bold. Every mention inside running text is not. Weight
separates them perfectly, and the wording alone could not: "Annexure B to
this Policy" starts exactly like "Annexure B- Items".

## The fix

```python
# api/app/pipeline/segment.py
_RE_ANNEXURE_HEAD = re.compile(r"^annexure[\s-]+(?:[a-z]|[ivxl]+|\d+)\b", re.I)

def _looks_like_section(line: Line, body_size: float) -> bool:
    if line.size >= body_size * SECTION_SIZE_RATIO and line.bold:
        return True
    if line.bold and _RE_ANNEXURE_HEAD.match(line.text.strip()):
        return True
    ...
```

Comparing segmentation before and after on all six PDFs:

- **Synthetic (×3) and Star: byte-identical.** None of their eval requests
  can change.
- **HDFC:** "Contact Us" drops from 2,985 characters to 664. Annexure A
  (two pieces), B (the 68 items, nothing else) and C become their own
  sections. The 20 plan-chart rows after it ("1.1 Hospitalization ...")
  were filed under "SECTION D. GENERAL TERMS AND CLAUSES" and are now under
  "Annexure C - Plan Chart", which is what they are.
- **Niva:** "List IV" and the ombudsman list move out of clause 6.2.9
  ("Assignment") into Annexures I and II. Niva has no eval questions.

The test `test_a_bold_annexure_heading_starts_a_section_and_a_mention_does_not`
in `api/tests/test_segment.py` rebuilds the HDFC arrangement line by line:
a bold heading that must split, and a plain-type "Annexure B to this
Policy" mention that must not.

## The cost: a re-roll of HDFC, measured

The analysis prompt identifies each clause by its position in the document
(`### CLAUSE id=92`) and names its section. Splitting one clause into three
moves every later clause's position by two, and the plan-chart rows changed
section. So about 23 HDFC clauses were analysed afresh, the same kind of
change that cost a verdict in part 1 (Failure 93). It had to be measured.

The clause analyses that changed, matched on text since ids moved: five
plan-chart rows. Two of them (1.5 and 1.6, the 60- and 180-day windows) came
back with different `waiting_periods_days`. That cannot reach a waiting-
period line: `waiting.py` drops any clause that never says "wait", the rule
M27 added for exactly these rows. The other three are type flips on HDFC's
optional benefits (2.3, 2.4, 2.5). All 115 clauses are now analysed; Annexure
B is an `exclusion`.

HDFC, majority of three, `main` against the branch:

| HDFC | main | branch |
|---|---|---|
| verdicts right | 11/14 | 12/14 |
| gastroenteritis-three-weeks | wrong 3/3 | right 3/3 |
| typhoid-20-days | wrong 3/3 | wrong 2/3 |
| malaria-25-days | right 3/3 | right 2/3 |
| failed quotations per sample | 10-12 | 5-9 |

No case got worse by majority. Every HDFC request changed (the clause ids
moved), so this is honestly a re-roll, and one that came out ahead. The
gain on gastroenteritis is not claimed as caused by the annexures. Which of
the changed inputs moved it was not isolated, and with every request
changed, nothing could isolate it. What the measurement does establish is
that the correct segmentation costs nothing.

## Check it yourself

```bash
cd api
.venv/Scripts/python -m pytest tests/test_segment.py -q -k annexure_heading
```

**Predict before you look:** the client doubles `num_predict` when an answer
is cut off mid-JSON. For which kind of over-long answer does that help, and
for which is it wasted?

<details>
<summary>Answer</summary>

It helps when the answer is simply long, for example a clause with many
genuine exceptions: a bigger limit lets it finish. It is wasted on a
repetition loop. A model repeating the same ten items has no end to reach,
so it fills 1,600 tokens, then 3,200, then 6,400, and fails three times
having spent the most time on the attempts that could never succeed. The
cure for a loop is a better input (here, a clause that is one list rather
than three unrelated things), not a larger limit.
</details>
