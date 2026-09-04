# Design System — Insurance Policy Clause Explainer

## Product Context

- **What this is:** Upload an Indian health insurance policy PDF; get back every clause typed, rewritten in plain English, and ranked by how badly it could hurt a claim.
- **Who it's for:** Ordinary Indian policyholders. Not insurance professionals. Often anxious, low trust in insurers, reading on a laptop.
- **Space/industry:** Consumer fintech / insurtech, IRDAI-regulated health cover.
- **Project type:** Local-first web app (document reader + ranked report).

## The memorable thing

> **"I can see what they were hiding."**

Every design decision serves this one feeling. Not "this is pretty", not "this is friendly" — **exposure**. The product's entire premise is that policies *obscure* the clauses that decide claims, so the emotional payoff is being shown what was hidden.

Concretely, this means:

- `buriedness` is **visible in the UI**, not just computed in the backend. Every risk card carries a **"Why you'd miss this"** line assembled from real signals: how far into the document it sits, its Flesch-Kincaid reading grade, how many other clauses it points at, how many defined terms it leans on.
- The dashboard leads with **a count and a verdict**, not a greeting.
- Page numbers and clause numbers are always shown. "Page 34" is part of the accusation.
- Copy names consequences plainly and never reassures beyond what the clause says.

## Aesthetic Direction

- **Direction:** Editorial-Documentary. Specifically a **critical edition** — source text on one side, the editor's gloss on the other, with an apparatus of notes.
- **Decoration level:** Intentional. Hairline rules, a double-rule masthead, small-caps section labels, mono marginalia. No gradients, no glass, no blobs, no shadows doing decorative work.
- **Mood:** A serious printed instrument. Closer to an inspection notice or an annotated contract than to a SaaS dashboard. Calm, exact, slightly severe.

**Why this direction:** the product *is* a document arguing with another document. The critical-edition metaphor is already latent in the data model — the API returns `source_text` alongside `char_start`/`char_end`, so the original wording and its exact location are always available. The design should make that structural fact visible rather than inventing a decorative theme on top of it.

## Typography

Four faces, each doing a **semantic** job. This is the load-bearing idea of the system.

| Role | Font | Rationale |
|---|---|---|
| **Display / masthead** | **Instrument Serif** | Editorial authority, high contrast. Sharp rather than luxurious. |
| **Body / UI / plain language** | **Instrument Sans** | Same superfamily as the display face, so headings and interface cohere. This is *the app talking to you*. |
| **Verbatim policy text** | **Source Serif 4** | A document serif. The original wording should *look like* the original. |
| **Data / clause numbers / scores** | **JetBrains Mono** | Tabular figures. Reads as instrument output, not prose. |

**The rule that matters:** the policy's own words are ALWAYS set in Source Serif 4; everything the app says is ALWAYS set in Instrument Sans. The typeface itself encodes which voice is speaking, so a side-by-side view needs no "original wording" label. Never mix these.

- **Loading:** Google Fonts, `display=swap`, weights limited to what is used (see `web/index.html`).
- **Scale** (base 16px, ratio ≈1.25):

| Token | Size | Use |
|---|---|---|
| `xs` | 12px | Mono labels, small caps, footnotes |
| `sm` | 14px | Secondary text, table cells |
| `base` | 16px | Body, plain language |
| `lg` | 20px | Card titles |
| `xl` | 25px | Section headings |
| `2xl` | 31px | Page headings |
| `3xl` | 39px | Masthead |

- **Measure:** plain-language prose capped at ~68ch. Long legal text is hard enough without a 120-character line.

## Color

- **Approach:** Restrained. Paper, ink, and exactly two alarm hues.

### Light (default)

| Token | Hex | Use |
|---|---|---|
| `paper` | `#F7F4EE` | Page ground. Warm and archival, never clinical white. |
| `surface` | `#FFFDF8` | Cards, raised areas |
| `ink` | `#1A1815` | Primary text. Warm near-black, never `#000`. |
| `muted` | `#6B6459` | Secondary text |
| `rule` | `#DDD6C8` | Hairlines, borders |
| `oxide` | `#A8321E` | **Can deny or void your claim** — exclusion, condition |
| `ochre` | `#946A22` | **Reduces what you are paid** — sub-limit, waiting period |

### Dark

Dark mode is **ink**, not a grey dashboard: a warm near-black ground with paper-coloured text, so the document metaphor survives the inversion.

| Token | Hex |
|---|---|
| `paper` | `#14130F` |
| `surface` | `#1C1A15` |
| `ink` | `#ECE6DA` |
| `muted` | `#9A9184` |
| `rule` | `#33302A` |
| `oxide` | `#E06A52` |
| `ochre` | `#D3A24B` |

Accents are lightened and slightly desaturated for the dark ground; the same hue family is preserved so severity reads identically in both modes.

### The colourblind rule

**Severity is never carried by hue alone.** Every severity indicator pairs its colour with a numeral (the impact score) and a text label ("Can void your claim"). A risk product that fails a colourblind user is a broken risk product.

Two semantic hues, not a red/amber/green traffic light — traffic lights are both visually cheap and semantically wrong here, since "reduces your payout" is not a middle state between denied and covered. It is a different kind of harm.

## Spacing

- **Base unit:** 4px
- **Density:** Comfortable in prose, compact in data
- **Scale:** `2xs` 2 · `xs` 4 · `sm` 8 · `md` 12 · `base` 16 · `lg` 24 · `xl` 32 · `2xl` 48 · `3xl` 64

## Layout

- **Approach:** Grid-disciplined for the app, editorial for the masthead.
- **Masthead:** left-aligned, like a report cover. Product name in Instrument Serif, a thin double rule, then a mono key-value strip (policy · pages · clauses · analysed). **Never a centered hero.**
- **Max content width:** 1240px. Prose column ~68ch.
- **Border radius: 2px, everywhere.** Documents do not have rounded corners, and uniform bubble-radius is the loudest AI-generated tell there is. The only exception is a pill-shaped filter chip at 9999px, used sparingly and deliberately as contrast.
- **Borders over shadows.** Structure is drawn with 1px rules in `rule`, not with soft drop shadows.

## Motion

- **Approach:** Minimal-functional. This is an anxiety product; bouncy or playful motion would be actively wrong.
- **Permitted:** progress bar movement, 120ms hover/focus transitions, a short fade-in when a list first renders.
- **Forbidden:** scroll-driven choreography, spring easing, staggered card cascades, anything that draws attention to the interface rather than the content.
- **Easing:** enter `ease-out`, exit `ease-in`, move `ease-in-out`. Duration: micro 80ms, short 160ms, medium 240ms.
- Respect `prefers-reduced-motion` and disable all non-essential transitions.

## Anti-patterns — never ship these

Explicitly forbidden in this project, because they are what "AI-generated design" looks like:

- Purple or violet gradients, or gradients as an accent anywhere
- Glassmorphism, backdrop blur, frosted panels
- A centered hero with a large centered heading and a subtitle beneath
- Three-column feature grids with icons in coloured circles
- Uniform large border-radius on every element
- Gradient buttons as the primary call to action
- Inter, Roboto, Poppins, Montserrat, or `system-ui` as a display or body face
- Emoji used as interface iconography
- Decorative drop shadows standing in for structure

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-04 | Editorial-documentary / critical-edition direction | The product is a document arguing with another document; the metaphor matches the data model, which already pairs verbatim source text with character offsets. |
| 2026-09-04 | Serif for policy text, sans for app voice | Makes the side-by-side legible without labels. Typography does semantic work instead of decorating. |
| 2026-09-04 | 2px border radius | Documents have square corners; uniform bubble-radius is the clearest AI-design tell. |
| 2026-09-04 | Two semantic hues, not a traffic light | "Reduces your payout" is a different kind of harm from "denies your claim", not a middle state. |
| 2026-09-04 | Dark mode included (user decision) | Night reading matters; built on CSS variables from day one so it costs little. Dark is "ink", not grey, to preserve the document metaphor. |
| 2026-09-04 | `buriedness` surfaced in the UI as "Why you'd miss this" | Follows directly from the memorable thing, "I can see what they were hiding". The score was already computed; hiding it wasted the project's most distinctive idea. |
