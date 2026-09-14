import type { ClauseSummary, ClauseType } from './types'

/**
 * How each clause type is presented.
 *
 * Two semantic hues, never a red/amber/green traffic light. "Reduces what you
 * are paid" is a different KIND of harm from "can deny your claim", not a
 * middle state between denied and covered, and colouring it as a halfway point
 * would misrepresent it.
 *
 * `consequence` exists to satisfy the design rule that severity is never
 * carried by colour alone: every indicator pairs its hue with this text and
 * with the numeric impact score. A risk product that fails a colourblind
 * reader is a broken risk product.
 */
export type Band = 'deny' | 'reduce' | 'neutral'

interface Meta {
  /** What this type is, in the reader's words rather than the policy's. */
  label: string
  band: Band
  /** The text half of the severity signal. Always rendered alongside the hue. */
  consequence: string
}

export const CLAUSE_META: Record<ClauseType, Meta> = {
  exclusion: {
    label: 'Never covered',
    band: 'deny',
    consequence: 'Will not be paid',
  },
  condition: {
    label: 'Something you must do',
    band: 'deny',
    consequence: 'Can void your claim',
  },
  sub_limit: {
    label: 'Payout cap',
    band: 'reduce',
    consequence: 'Reduces what you are paid',
  },
  waiting_period: {
    label: 'Not covered yet',
    band: 'reduce',
    consequence: 'Delays your cover',
  },
  coverage: {
    label: 'Covered',
    band: 'neutral',
    consequence: 'What the policy pays for',
  },
  definition: {
    label: 'Defined term',
    band: 'neutral',
    consequence: 'Changes what other clauses mean',
  },
  procedural: {
    label: 'Admin',
    band: 'neutral',
    consequence: 'No direct effect on a claim',
  },
}

export const BAND_STYLES: Record<Band, { text: string; bg: string; border: string }> = {
  deny: { text: 'text-oxide', bg: 'bg-oxide-soft', border: 'border-oxide' },
  reduce: { text: 'text-ochre', bg: 'bg-ochre-soft', border: 'border-ochre' },
  neutral: { text: 'text-muted', bg: 'bg-sunk', border: 'border-rule-strong' },
}

/** Types the reader is here to find. Used for the headline count. */
export const RISKY_TYPES: ClauseType[] = [
  'exclusion',
  'condition',
  'sub_limit',
  'waiting_period',
]

/**
 * Turn the computed buriedness signals into reasons a person can read.
 *
 * This is the product's whole promise made concrete. The backend already
 * measures WHY a clause is easy to overlook - how deep into the document it
 * sits, how hard the sentence is, how many other clauses it sends you to, how
 * many defined terms it leans on. Showing only the blended score would waste
 * that: "buriedness 0.47" tells a reader nothing, while "sits in the last
 * third of the document, reads at grade 17" tells them exactly what happened.
 *
 * Thresholds are set so that a typical clause produces one or two reasons
 * rather than four. A note that fires on everything stops being information.
 */
export function buriedReasons(clause: ClauseSummary): string[] {
  const reasons: string[] = []

  if (clause.position_signal >= 0.55) {
    reasons.push(`buried on page ${clause.page}, deep in the document`)
  }
  if (clause.reading_signal >= 0.45) {
    reasons.push(`written at grade ${Math.round(clause.reading_grade)} reading level`)
  }
  if (clause.crossref_signal >= 0.25) {
    reasons.push('sends you to other clauses to understand it')
  }
  if (clause.jargon_signal >= 0.34) {
    reasons.push('relies on terms defined elsewhere in the policy')
  }

  return reasons
}

/** Coarse label for the buriedness number, so it reads as a judgement. */
export function buriedLabel(buriedness: number): string {
  if (buriedness >= 0.55) return 'Very easy to miss'
  if (buriedness >= 0.35) return 'Easy to miss'
  if (buriedness >= 0.2) return 'Somewhat hidden'
  return 'Plainly stated'
}
