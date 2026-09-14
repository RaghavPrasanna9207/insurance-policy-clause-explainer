import { BAND_STYLES, CLAUSE_META, buriedLabel, buriedReasons } from '../clauseMeta'
import type { ClauseSummary } from '../types'

/**
 * One clause on the risk list.
 *
 * Structure is drawn with rules and a left severity bar, not with drop shadows
 * or a rounded card. The whole page should read as an annotated document, and
 * floating cards would fight that.
 */
export function RiskCard({
  clause,
  rank,
  onOpen,
}: {
  clause: ClauseSummary
  rank?: number
  onOpen: (clause: ClauseSummary) => void
}) {
  const meta = CLAUSE_META[clause.clause_type]
  const band = BAND_STYLES[meta.band]
  const reasons = buriedReasons(clause)

  return (
    <article
      className={`group border-b border-rule border-l-2 ${band.border} bg-surface transition-colors duration-150 hover:bg-sunk`}
    >
      <button
        type="button"
        onClick={() => onOpen(clause)}
        className="flex w-full items-start gap-5 px-4 py-4 text-left"
      >
        {/* Rank in the margin, like a numbered item in a printed schedule. */}
        {rank !== undefined && (
          <span className="tnum mt-0.5 w-6 shrink-0 font-mono text-[13px] text-muted">
            {String(rank).padStart(2, '0')}
          </span>
        )}

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            {clause.number && (
              <span className="tnum font-mono text-[12px] text-muted">{clause.number}</span>
            )}
            <h3 className="font-medium text-[15px] leading-snug text-ink">
              {stripNumber(clause.heading, clause.number) || meta.label}
            </h3>
          </div>

          {/* The severity signal: hue AND text AND a number, together.
              The number appears inline below `sm` because the rail is hidden
              at that width - and the design rules require severity to always carry a
              numeral, so hiding the rail must not take the score with it. */}
          <p className={`mt-1 flex items-baseline gap-2 text-[12px] font-medium ${band.text}`}>
            {meta.consequence}
            <span className="tnum font-mono opacity-70 sm:hidden">
              · impact {clause.impact_score.toFixed(0)}
            </span>
          </p>

          <p className="mt-2 max-w-[68ch] text-[14px] leading-relaxed text-ink/90">
            {clause.plain_language}
          </p>

          {/* The product's promise, made concrete. Not "buriedness: 0.47" but
              the actual reasons this clause was easy to walk past. */}
          {reasons.length > 0 && (
            <p className="mt-2.5 max-w-[68ch] text-[12px] leading-relaxed text-muted">
              <span className="label mr-1.5">Why you'd miss this</span>
              {reasons.join(' · ')}
            </p>
          )}
        </div>

        <ImpactRail score={clause.impact_score} buriedness={clause.buriedness} band={band.text} />
      </button>
    </article>
  )
}

/**
 * The right-hand rail: impact score over a buriedness reading.
 *
 * The bar is a plain filled rule, not a rounded progress pill. The number is
 * always shown beside it, so the bar is reinforcement rather than the only
 * carrier of the value.
 */
function ImpactRail({
  score,
  buriedness,
  band,
}: {
  score: number
  buriedness: number
  band: string
}) {
  return (
    // `band` sets the text colour on the container, so the bar's `bg-current`
    // picks up the severity hue. Previously the colour was applied only to the
    // numeral, leaving the bar plain ink and throwing away the reinforcement.
    <div className={`hidden w-[148px] shrink-0 pt-0.5 sm:block ${band}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="label">Impact</span>
        <span className="tnum font-mono text-[17px] leading-none">{score.toFixed(0)}</span>
      </div>
      {/* A visible track, so the fill reads as a proportion rather than as a
          stray rule. `bg-current/15` keeps the track in the same hue family. */}
      <div className="mt-1.5 h-[3px] w-full bg-current/15">
        <div className="h-full bg-current" style={{ width: `${Math.max(score, 2)}%` }} />
      </div>
      <p className="mt-2 text-right text-[11px] leading-tight text-muted">
        {buriedLabel(buriedness)}
      </p>
    </div>
  )
}

/** Headings arrive as "4.7 Non-Medical Expenses"; the number is shown
 *  separately in mono, so strip it to avoid printing it twice. */
export function stripNumber(heading: string, number: string): string {
  if (!heading) return ''
  const trimmed = number && heading.startsWith(number) ? heading.slice(number.length) : heading
  return trimmed.replace(/^[\s.—-]+/, '').trim()
}
