import { useState } from 'react'

import { api } from '../api'
import type { ScenarioResponse, Verdict } from '../types'

/**
 * How each verdict is presented.
 *
 * Note there is no green. The palette is paper, ink and two alarm hues, so a
 * good outcome is shown by the ABSENCE of an alarm colour rather than by a
 * reassuring one. That is deliberate: this tool reads a document, it does not
 * approve claims, and a green tick would promise a certainty it cannot give.
 *
 * `insufficient_information` is styled as a real answer, not an error. The
 * system being able to say "your document does not settle this" is a feature
 * of the design, and presenting it as a failure would teach people to distrust
 * the one honest thing it does.
 */
const VERDICTS: Record<
  Verdict,
  { label: string; tone: string; rule: string; blurb: string }
> = {
  not_covered: {
    label: 'Not covered',
    tone: 'text-oxide',
    rule: 'border-oxide',
    blurb: 'On the wording below, this policy would not pay.',
  },
  conditional: {
    label: 'It depends',
    tone: 'text-ochre',
    rule: 'border-ochre',
    blurb: 'Cover is possible, but something has to hold for it to be paid.',
  },
  covered: {
    label: 'Likely covered',
    tone: 'text-ink',
    rule: 'border-ink',
    blurb: 'Nothing in this policy appears to block it, on what you described.',
  },
  insufficient_information: {
    label: "Your policy doesn't settle this",
    tone: 'text-muted',
    rule: 'border-rule-strong',
    blurb: 'Rather than guess, here is what would be needed to answer.',
  },
}

const EFFECT_LABEL: Record<string, string> = {
  denies: 'Blocks it',
  delays: 'Not yet',
  reduces: 'Cuts the amount',
  requires: 'You must',
  permits: 'Allows it',
}

// Concrete rather than generic. Each names a situation the policy actually has
// something to say about, so a first-time user sees the tool working instead of
// discovering the shape of a good question by trial and error.
const EXAMPLES = [
  'I have had diabetes for years and was hospitalised for it 8 months after buying this policy.',
  'My sum insured is 5 lakh and I stayed in a room costing 9,000 rupees a night.',
  'I was admitted in an emergency and only told the insurer four days later.',
  'I bought this policy at 67 and I am claiming three years later.',
]

const FACT_LABELS: Record<string, string> = {
  time_since_policy_start_value: 'how long you have held the policy',
  age: 'your age',
  pre_existing_condition: 'whether this condition existed before you bought the policy',
  hospitalised: 'whether you were admitted overnight',
  hours_since_admission: 'how soon you told the insurer',
}

export function ScenarioPanel({ docId }: { docId: string }) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ScenarioResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function ask(scenario: string) {
    if (scenario.trim().length < 8) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.scenario(docId, scenario))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="mt-16 max-w-[1040px] border-t border-rule pt-6">
      <h3 className="label">Check your own situation</h3>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed">
        Describe what happened, in your own words. This checks it against every clause in your
        policy and shows you the ones that decide the answer.
      </p>

      <form
        onSubmit={(event) => {
          event.preventDefault()
          void ask(text)
        }}
        className="mt-5"
      >
        <label htmlFor="scenario" className="sr-only">
          Your situation
        </label>
        <textarea
          id="scenario"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={3}
          placeholder="I had knee surgery 8 months after buying this policy…"
          className="w-full resize-y rounded-sm border border-rule-strong bg-surface p-3 text-[15px] leading-relaxed outline-none transition-colors duration-150 placeholder:text-muted focus:border-ink"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={busy || text.trim().length < 8}
            className="rounded-sm border border-ink bg-ink px-4 py-2 text-[14px] font-medium text-paper transition-opacity duration-150 hover:opacity-85 disabled:opacity-40"
          >
            {busy ? 'Checking your policy…' : 'Check this'}
          </button>
          <span className="text-[12px] text-muted">
            Reads every clause. Takes up to a minute on a local model.
          </span>
        </div>
      </form>

      {!result && !busy && (
        <div className="mt-5">
          <p className="label mb-2">Or try one of these</p>
          <div className="flex flex-col gap-1.5">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setText(example)
                  void ask(example)
                }}
                className="max-w-[76ch] border-l-2 border-rule pl-3 py-1 text-left text-[13px] leading-relaxed text-muted transition-colors duration-150 hover:border-ink hover:text-ink"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <p className="mt-5 border-l-2 border-oxide bg-oxide-soft py-2 pl-3 text-[13px] text-oxide">
          {error}
        </p>
      )}

      {result && <Answer result={result} />}
    </section>
  )
}

function Answer({ result }: { result: ScenarioResponse }) {
  const verdict = VERDICTS[result.verdict] ?? VERDICTS.insufficient_information

  return (
    <article className="rise mt-7 border-t border-rule pt-6">
      <div className={`border-l-2 ${verdict.rule} pl-4`}>
        <h4 className={`font-display text-[30px] leading-tight ${verdict.tone}`}>
          {verdict.label}
        </h4>
        <p className="mt-1 text-[13px] text-muted">{verdict.blurb}</p>
      </div>

      {/* Shown before the reasoning, not after. If any quotation could not be
          found in the policy, the reader needs to know that BEFORE they read
          an explanation built on it. */}
      {!result.verified && (
        <p className="mt-5 border-l-2 border-oxide bg-oxide-soft py-2.5 pl-3 text-[13px] leading-relaxed text-oxide">
          <span className="font-medium">Not verified.</span> Some wording quoted below could not
          be found in your policy. Treat this answer as unreliable and check the clauses
          yourself.
        </p>
      )}

      <p className="mt-5 max-w-[68ch] text-[15px] leading-relaxed">{result.reasoning}</p>

      {result.missing_facts.length > 0 && (
        <div className="mt-5 max-w-[68ch]">
          <p className="label mb-1.5">To answer this properly, it would need to know</p>
          <ul className="space-y-1">
            {result.missing_facts.map((fact) => (
              <li key={fact} className="flex gap-2 text-[13px] leading-relaxed text-muted">
                <span aria-hidden className="text-rule-strong">
                  —
                </span>
                {FACT_LABELS[fact] ?? fact.replace(/_/g, ' ')}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.citations.length > 0 && (
        <div className="mt-7">
          <p className="label mb-3">
            The clauses that decide it · checked against {result.clauses_considered} clauses
          </p>
          <div className="space-y-4">
            {result.citations.map((citation, index) => (
              <div
                key={`${citation.clause_id}-${index}`}
                className={`border-l-2 pl-4 ${
                  citation.verified ? 'border-rule-strong' : 'border-oxide'
                }`}
              >
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="tnum font-mono text-[12px] text-muted">
                    {citation.number || citation.clause_id}
                  </span>
                  <span className="text-[14px] font-medium">
                    {citation.heading.replace(/^[\d.]+\s*/, '') || 'Clause'}
                  </span>
                  <span className="label">{EFFECT_LABEL[citation.effect] ?? citation.effect}</span>
                  <span className="font-mono text-[11px] text-muted">page {citation.page}</span>
                </div>

                {/* The policy's own words, in the document serif. Same rule as
                    the clause panel: the typeface says whose voice this is. */}
                <blockquote className="verbatim mt-2 max-w-[72ch] bg-sunk px-3 py-2">
                  {citation.quote}
                </blockquote>

                {!citation.verified && (
                  <p className="mt-1.5 text-[12px] text-oxide">
                    Could not verify this quotation: {citation.unverified_reason}.
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </article>
  )
}
