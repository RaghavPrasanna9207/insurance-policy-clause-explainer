# CLAUDE.md — Insurance Policy Clause Explainer

## What this project is

Upload an Indian (IRDAI) health insurance policy PDF. Get back:

1. **A risk-ranked clause map** — every clause typed, rewritten in plain English, and scored for how likely it is to get a claim denied.
2. **A scenario simulator** — "I had knee surgery 8 months after buying this" → covered / not covered / conditional, with the exact clauses that decide it.

The problem being solved is that insurance documents *obscure* the exclusions and conditions that decide claim outcomes. So this tool inverts the usual interaction: rather than answering questions the user already knew to ask, **it reads the policy and tells them what will get their claim denied, before they ask.**

---

## Teaching mode — THE RULE THAT OUTRANKS THE OTHERS

**This project has two deliverables: the working app, and Raghav's understanding of it.**

The goal is that by the end, he can explain every part of this system to someone else without looking at the code. **When a choice is between fast and comprehensible, take comprehensible.**

### After every meaningful step, explain — never skip this:

- **What I did** — the change, in one or two lines.
- **Why this way** — the reasoning, *and what was rejected*. A decision presented without its alternatives teaches nothing.
- **New concept** — when something appears for the first time (constrained decoding, FastAPI `BackgroundTasks`, SQLModel session lifecycle, Flesch–Kincaid), explain it properly from scratch. Never just name-drop it.
- **What broke and why** — real errors, the root cause, and why the fix actually works. **Failures are the most valuable teaching material in this project. Never hide one, never quietly patch around one.** If I take a wrong turn, he sees the wrong turn and the correction.
- **Check it yourself** — a command he can run to watch it work, or a question that tests whether the idea landed.

### Supporting rules

- **`docs/LEARNING-LOG.md`** accumulates these entries in order. Chat scrolls away and context gets compacted; the log is what survives. Append to it as work proceeds.

### THE LOG MUST STAND ALONE — this is a hard requirement

Raghav has said explicitly: some days he will have time to code but not to read the explanations. **He may only sit down to learn this project after it is finished.** So the log cannot be a set of footnotes to a conversation that no longer exists.

Write every entry for a reader who **never saw the chat**, arriving cold, months later:

- **Never reference the conversation.** No "as I mentioned", "the failure above", "we decided earlier". If a fact matters, restate it in the entry.
- **Restate the problem before the solution.** An entry that opens mid-thought is useless to a cold reader.
- **Quote the code being discussed**, inline, with its file path. Don't assume the reader has the file open, or that the file still looks the same.
- **Keep `docs/README.md` as the reading-order index** — where to start, what each entry covers, what to read if you only have twenty minutes.
- **Concepts get taught from scratch, once, in the entry where they first appear**, and are cross-referenced by name afterwards.
- **Record failures with their full setup:** what was tried, what happened, why, what fixed it. A bare "this broke" teaches nothing later.

The test: if the whole log were printed and handed to a stranger with the repo, could they understand why this system is built the way it is? If not, it is not finished.
- **Milestone recaps.** At each milestone boundary, recap how the new piece connects to what already exists and what it made possible.
- **Comments explain *why*, not *what*.** Good: `# batch of 5: fits context with room for the schema, and one bad clause only poisons 5`. Useless: `# loop over clauses`.
- **No code dumps.** Walk through large files section by section.
- **Ask him to predict.** Occasionally ask what a piece of code will do before explaining it. Recall beats re-reading.

Where a concept has real transferable depth — why constrained decoding makes hallucinated citations *impossible* rather than merely unlikely, why segmentation must never touch an LLM — go deeper than the immediate task requires. Those ideas outlive this project.

---

## Architecture, and the reasoning behind it

```
PDF ──> [1 ingest] ──> [2 segment] ──> [3 analyze] ──> [4 score] ──> SQLite
        PyMuPDF        rules-based      LLM batched     pure math
        text+bbox      NO LLM           schema-locked   NO LLM
        +char offsets                   + cached

scenario ──> [5a extract facts] ──> [5b shortlist] ──> [5c reason] ──> [5d verify]
             LLM, schema              pure filter      LLM, id-enum     substring check
```

**The governing principle: deterministic where possible, LLM only where language understanding is genuinely required.** This is what makes a local 7B model reliable enough to trust. Stages 2 and 4 contain no LLM at all — the model is never asked to do arithmetic, count pages, or decide where a paragraph ends.

### Grounding — the core correctness mechanism

This domain punishes hallucination with real financial harm, so grounding is structural, not advisory:

1. **Citation IDs are `enum`-constrained** to exactly the clauses supplied in that prompt. Ollama enforces JSON-schema enums at the decoding level (verified on this machine — see below), so a fabricated citation is *unrepresentable in the output grammar*. Not "discouraged by the prompt" — impossible.
2. **Verbatim span check.** Any text the model quotes is normalized and must appear as a substring of the cited clause's stored source. Failure → one retry → otherwise shown in the UI as *unverified*. Never silently rendered as fact.

### Choices that look like omissions but are deliberate

Do not "helpfully" add these back:

- **No vector DB, no embeddings, no BM25.** A policy's decision-relevant clauses (exclusions / conditions / waiting periods / sub-limits) number ~60–80 ≈ 6k tokens — they all fit in qwen2.5's 32k context. Showing the model every clause that could matter beats retrieval on recall *and* costs zero infrastructure.
- **No Celery/Redis.** `BackgroundTasks` + polling. Single-user local app; a queue would be infrastructure with no reader.
- **No ORM migrations in v1.** SQLite, recreate on schema change.

---

## Verified environment facts

Probed on this machine before any code was written. These drive the design:

| Fact | Value |
|---|---|
| Ollama | 0.33.3, `qwen2.5:7b-instruct-q4_K_M` (4.7GB) |
| Hardware | RTX 4060 Laptop, 8GB VRAM / 15.7GB RAM — model fits fully in VRAM |
| Warm throughput | ~50 tok/s; ~1.5s per clause batched → 200-clause policy in 2–5 min |
| Cold start | ~27s model load — warm the model on API startup |
| **`enum` enforcement** | **Confirmed.** An unconstrained field invented `"Limitation"`; an enum-constrained one returned exactly `exclusion`. |

That last row is load-bearing for the whole grounding design. **Every categorical LLM output field must carry an `enum`.**

---

## Conventions

- **Backend** `api/` — FastAPI · SQLModel · PyMuPDF · httpx · pytest. Python 3.12.
- **Frontend** `web/` — Vite · React · TypeScript · Tailwind.
- **Evals** `evals/` — the portfolio differentiator. Anyone can ship an LLM wrapper; the eval harness shows engineering judgment. Model and prompt changes are justified by eval numbers, never by vibes.
- **Prompts** live in `api/app/llm/prompts.py` with a `PROMPT_VERSION`. Bump it when a prompt changes — it labels eval reports, the run history and stored analyses. It is **not** part of the cache key: the cache hashes the exact prompt text, schema, model and options, so a reworded prompt never gets a stale answer and an unchanged one keeps its stored answer. That is what lets an eval isolate a change to the cases it actually touched.
- **Never commit real insurer policy PDFs** (licensing). Evals run on a synthetic policy authored for this repo; real wordings are fetched by script.

## Non-negotiable: this is not advice

Every user-facing surface carries a clear "not legal or financial advice" disclaimer. `insufficient_information` is a first-class verdict — the system must be able to say *"your document doesn't address this"* rather than guessing. A confident wrong answer here costs someone a real claim.

## Design System

Always read [DESIGN.md](DESIGN.md) before making any visual or UI decision. Fonts, colours, spacing, radius, motion, and the aesthetic direction are defined there. Do not deviate without explicit approval.

The two rules most easily broken by accident:

1. **The policy's own words are always set in Source Serif 4; everything the app says is always set in Instrument Sans.** The typeface encodes which voice is speaking. Never mix them.
2. **Severity is never carried by colour alone.** Every severity indicator pairs its hue with a numeral and a text label.

DESIGN.md also lists forbidden anti-patterns (gradients, glassmorphism, centered heroes, bubble radius, Inter). Treat that list as hard constraints, and flag any code that violates it.
