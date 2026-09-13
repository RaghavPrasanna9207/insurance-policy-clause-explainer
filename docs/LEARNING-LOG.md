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

The full system is in `DESIGN.md`, including a list of forbidden anti-patterns that `CLAUDE.md` points at.

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

The rule is written into `CLAUDE.md` because it is the sort of thing that erodes silently: *the policy's words are always Source Serif 4; everything the app says is always Instrument Sans; never mix them.*

---

## Concept 12: Colour must never be the only signal

Severity uses exactly two hues:

- **Oxide red `#A8321E`** — can deny or void your claim
- **Ochre `#946A22`** — reduces what you are paid

Deliberately **not** a red/amber/green traffic light. A traffic light implies a scale from bad to good, but "reduces your payout" is not a midpoint between "denied" and "covered" — it is a **different kind of harm**. Encoding it as a middle state would misrepresent it.

And the hard rule, from `DESIGN.md`:

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

Reasonable-looking, and wrong. On a phone the severity colour and its text label survived, but **the numeral disappeared entirely** — which is precisely the rule `DESIGN.md` states as non-negotiable, violated by the same person who wrote it, in the same week.

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

This matters more than it sounds. `DESIGN.md` forbids a specific list of visual patterns, and there is no way to check a built interface against that list except by seeing it. A typecheck passes on a page with an invisible progress bar. A unit test passes on a layout with 500px of dead space. Neither can tell you the severity numeral vanished on a phone.

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

Files added: `web/src/index.css` (design tokens), `clauseMeta.ts`, `api.ts`, `types.ts`, `components/Chrome.tsx`, `components/RiskCard.tsx`, `components/ClausePanel.tsx`, `routes/Upload.tsx`, `routes/Policy.tsx`, `scripts/screenshots.mjs`, plus `DESIGN.md` at the repo root.

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

`CLAUDE.md` states the governing principle: *deterministic where possible, LLM only where language understanding is genuinely required.* Stage 2 (segmentation) and stage 4 (scoring) follow it — I specifically refused to let the model compute buriedness because that would be arithmetic. **Stage 5 had quietly broken the same rule**, and I had been calling the result a model limitation.

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

This project's governing rule is stated in `CLAUDE.md`: *deterministic where
possible, LLM only where language understanding is genuinely required.*

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
