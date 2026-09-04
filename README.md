# Insurance Policy Clause Explainer

Upload an Indian health insurance policy PDF. Find out what could get your claim denied — before you file one.

> **Status: in development.** The full pipeline and REST API work end to end — upload a policy PDF and get back a ranked list of the clauses that could cost you money. Clause classification scores macro-F1 **1.000** on the golden set. The web UI and scenario simulator are still to come. See [Roadmap](#roadmap).

---

## The problem

Policyholders make uninformed decisions because insurance documents are written in legal and technical language that **obscures** the exclusions and conditions deciding claim outcomes.

That word — *obscures* — is what this project takes seriously. A generic "chat with your PDF" wrapper does not address it, because it can only answer questions you already knew to ask. The clauses that get claims denied are the ones you never thought to ask about: a 36-month waiting period, a room-rent cap that silently shrinks every associated charge, a 24-hour notice requirement buried on page 34 behind a defined term.

So this tool inverts the interaction. **It reads the policy and tells you what will get your claim denied, before you ask.**

## What it produces

**1. A risk-ranked clause map.** Every clause typed (coverage / exclusion / condition / sub-limit / waiting period / definition / procedural), rewritten in plain English, and scored for claim-denial impact.

**2. A scenario simulator.** *"I had knee surgery 8 months after buying this"* → covered / not covered / conditional, with the exact clauses that decide it.

---

## Design

```
PDF ──> [1 ingest] ──> [2 segment] ──> [3 analyze] ──> [4 score] ──> SQLite
        PyMuPDF        rules-based      LLM batched     pure math
        text+bbox      NO LLM           schema-locked   NO LLM
        +char offsets                   + cached

scenario ──> [5a extract facts] ──> [5b shortlist] ──> [5c reason] ──> [5d verify]
             LLM, schema              pure filter      LLM, id-enum     substring check
```

**Deterministic wherever possible; a model only where language understanding is genuinely required.** This is what makes a 7B model running on a laptop GPU reliable enough to build on. Stages 2 and 4 contain no LLM at all — the model is never asked to do arithmetic, count pages, or decide where a paragraph ends.

### Grounding is structural, not advisory

A wrong answer here costs someone a real claim, so hallucination is prevented by construction rather than discouraged by prompt wording:

**1. Citation IDs are `enum`-constrained** to exactly the clauses supplied in that prompt. Ollama enforces JSON-Schema enums at the *decoding* level — the sampler masks out every token that could not continue a valid document under the schema. A fabricated citation is therefore unrepresentable in the output grammar, not merely unlikely.

**2. Verbatim span check.** Any text the model quotes must appear as a substring of the cited clause's stored source. Failures are surfaced as *unverified*, never rendered as fact.

Underpinning both: every clause stores character offsets into the document's extracted text, with the invariant `raw_text[start:end] == clause.text` asserted by tests. A slice cannot drift from what it is a slice of, the way a copy can.

### Choices that look like omissions

- **No vector DB, no embeddings, no BM25.** A policy's decision-relevant clauses number ~60–80 ≈ 6k tokens — they all fit in the model's 32k context. Showing it every clause that could matter beats retrieval on recall, at zero infrastructure cost.
- **No task queue.** `BackgroundTasks` + polling. Single-user local app.
- **No cloud API.** Runs entirely on a local Ollama model. Your policy document never leaves your machine.

---

## Running it

**Requirements:** Python 3.12+, [Ollama](https://ollama.com), ~6GB free VRAM or RAM.

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M

cd api
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt                   # macOS/Linux

.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Then open <http://localhost:8000/health> to confirm Ollama is reachable and the model is present, or <http://localhost:8000/docs> for the API.

### Tests

```bash
cd api
.venv/Scripts/python.exe -m pytest -v          # all tests
.venv/Scripts/python.exe -m pytest -m "not llm"  # skip tests needing Ollama
```

The test suite hits a real Ollama server rather than a mock for the LLM tests, on purpose: the claim under test — that the sampler cannot emit a value outside our enum — is a claim about Ollama's behaviour. A mock would only test our belief about it.

### Test data

`evals/golden/build_synthetic_policy.py` generates a 39-clause IRDAI-style policy spanning all seven clause types, plus a ground-truth label file. The PDF and the labels come from the same source, so they cannot drift apart.

It also generates a **hostile variant**: the identical policy typeset with one font, no bold, and clause numbers run inline — the way many machine-generated policy PDFs actually arrive. The segmenter must extract the same 39 clauses from both. That fixture exists because the original test suite passed 14/14 while being circular: the thresholds had been tuned on the very document they were tested against.

No real insurer policy wordings are committed to this repository.

---

## Documentation

**[`docs/LEARNING-LOG.md`](docs/LEARNING-LOG.md)** explains what each part of the system does, why it is built that way, and every non-obvious failure hit along the way — written to be read cold, without any other context. [`docs/README.md`](docs/README.md) is the reading-order index.

[`CLAUDE.md`](CLAUDE.md) records the architecture and conventions.

---

## Roadmap

| | Stage | Status |
|---|---|---|
| M0 | Ollama client: constrained JSON, retries, content-addressed cache | ✅ |
| M1 | Ingest + segment — offsets and clause boundaries, no LLM | ✅ |
| M2 | Analyze + score — classification, deterministic impact formula | ✅ macro-F1 1.000 |
| M3 | REST API — upload, poll, ranked clauses | ✅ |
| M4 | Web UI — risk dashboard, clause explorer | |
| M5 | Scenario simulator | |
| M6 | Eval report | |

Evaluation is the point, not an afterthought: model and prompt changes in this project are justified by measured numbers on the golden set, never by impressions.

---

## Not advice

This tool produces an automated reading of a document. It is **not legal or financial advice**, it is not a substitute for reading your policy, and it may be wrong. Always confirm anything that matters with your insurer in writing.

## Licence

MIT
