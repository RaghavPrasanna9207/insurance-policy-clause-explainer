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
