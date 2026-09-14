import { useEffect, useState } from 'react'

import { api } from '../api'
import { BAND_STYLES, CLAUSE_META, buriedLabel, buriedReasons } from '../clauseMeta'
import type { ClauseDetail } from '../types'
import { stripNumber } from './RiskCard'

/**
 * The clause reading view: our explanation beside the policy's own words.
 *
 * The typographic rule does the labelling here. Everything the
 * app says is set in Instrument Sans; the verbatim policy text is set in
 * Source Serif 4 via the `.verbatim` class. Because the two voices look
 * different, the panel needs no "original wording" caption - the reader can
 * see which is which. That is typography doing semantic work rather than
 * decorating.
 */
export function ClausePanel({ clauseId, onClose }: { clauseId: string; onClose: () => void }) {
  const [clause, setClause] = useState<ClauseDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    setClause(null)
    setError(null)
    api
      .clause(clauseId)
      .then((data) => live && setClause(data))
      .catch((err) => live && setError(err.message))
    return () => {
      live = false
    }
  }, [clauseId])

  // Escape closes the panel. A reading surface that traps the keyboard is a
  // reading surface people stop trusting.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-ink/25"
      />
      <section className="relative flex h-full w-full max-w-[720px] flex-col border-l border-rule-strong bg-paper shadow-none">
        {error && <p className="p-6 text-[14px] text-oxide">{error}</p>}
        {!clause && !error && (
          <p className="label p-6">Loading clause…</p>
        )}
        {clause && <Body clause={clause} onClose={onClose} />}
      </section>
    </div>
  )
}

function Body({ clause, onClose }: { clause: ClauseDetail; onClose: () => void }) {
  const meta = CLAUSE_META[clause.clause_type]
  const band = BAND_STYLES[meta.band]
  const reasons = buriedReasons(clause)

  return (
    <>
      <header className={`border-b-2 ${band.border} bg-surface px-7 pb-5 pt-6`}>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="label">{clause.section_path || 'Clause'}</p>
            <div className="mt-1.5 flex flex-wrap items-baseline gap-x-3">
              {clause.number && (
                <span className="tnum font-mono text-[15px] text-muted">{clause.number}</span>
              )}
              <h2 className="font-display text-[26px] leading-tight">
                {stripNumber(clause.heading, clause.number) || meta.label}
              </h2>
            </div>
            <p className={`mt-2 text-[13px] font-medium ${band.text}`}>{meta.consequence}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="label shrink-0 rounded-sm border border-rule px-2.5 py-1 transition-colors duration-150 hover:border-rule-strong hover:text-ink"
          >
            Close
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-7 py-6">
        <Section label="In plain English">
          <p className="max-w-[68ch] text-[15px] leading-relaxed">{clause.plain_language}</p>
          {clause.what_it_means && (
            <p className="mt-3 max-w-[68ch] border-l-2 border-rule-strong pl-3 text-[14px] leading-relaxed text-muted">
              {clause.what_it_means}
            </p>
          )}
        </Section>

        {/* The evidence. Set in the document serif so it reads as the policy
            speaking, with its exact location stated: the reader can go and
            check this against the PDF themselves. */}
        <Section label={`The policy's own words · page ${clause.page}`}>
          <blockquote className="verbatim border-l-2 border-rule-strong bg-sunk py-3 pl-4 pr-3">
            {clause.source_text}
          </blockquote>
          <p className="mt-2 font-mono text-[11px] text-muted">
            characters {clause.char_start.toLocaleString()}–{clause.char_end.toLocaleString()} of
            the extracted document
          </p>
        </Section>

        {reasons.length > 0 && (
          <Section label="Why you'd miss this">
            <p className="mb-2 text-[13px] font-medium text-ink">
              {buriedLabel(clause.buriedness)}
            </p>
            <ul className="space-y-1">
              {reasons.map((reason) => (
                <li key={reason} className="flex gap-2 text-[13px] leading-relaxed text-muted">
                  <span aria-hidden className="text-rule-strong">
                    —
                  </span>
                  {reason}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {(clause.monetary_limits.length > 0 ||
          clause.time_windows.length > 0 ||
          clause.triggers.length > 0) && (
          <Section label="Extracted details">
            <Facts title="Limits" values={clause.monetary_limits} />
            <Facts title="Deadlines" values={clause.time_windows} />
            <Facts title="Applies when" values={clause.triggers} />
          </Section>
        )}

        <Section label="How this was scored">
          <dl className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-4">
            <Stat label="Impact" value={clause.impact_score.toFixed(0)} />
            <Stat label="Likelihood" value={`${clause.likelihood}/5`} />
            <Stat label="Severity" value={`${clause.severity}/5`} />
            <Stat label="Reading grade" value={clause.reading_grade.toFixed(0)} />
          </dl>
          <p className="mt-3 max-w-[68ch] text-[12px] leading-relaxed text-muted">
            Likelihood and severity are judged by a language model running on this machine.
            Reading grade and how buried the clause is are measured directly from the document,
            not guessed.
          </p>
        </Section>
      </div>
    </>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="mb-7 border-t border-rule pt-4 first:border-t-0 first:pt-0">
      <h3 className="label mb-2.5">{label}</h3>
      {children}
    </section>
  )
}

function Facts({ title, values }: { title: string; values: string[] }) {
  if (values.length === 0) return null
  return (
    <div className="mb-3 last:mb-0">
      <p className="text-[12px] text-muted">{title}</p>
      <ul className="mt-1 flex flex-wrap gap-1.5">
        {values.map((value) => (
          <li
            key={value}
            className="rounded-sm border border-rule bg-surface px-2 py-0.5 font-mono text-[12px]"
          >
            {value}
          </li>
        ))}
      </ul>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="tnum mt-0.5 font-mono text-[17px]">{value}</dd>
    </div>
  )
}
