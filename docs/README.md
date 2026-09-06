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

---

## Related documents

- **[../DESIGN.md](../DESIGN.md)** - the design system: aesthetic direction, the four-face typography with its semantic rule, the colour tokens, and the list of forbidden anti-patterns.
- **[../CLAUDE.md](../CLAUDE.md)** — the architecture, the deliberate omissions (no vector DB, no task queue, no LLM in the scoring path), and the conventions this project follows.
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
