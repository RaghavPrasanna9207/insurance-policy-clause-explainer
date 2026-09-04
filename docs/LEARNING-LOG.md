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
