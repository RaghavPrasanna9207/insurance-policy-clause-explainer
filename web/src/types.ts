/** Mirrors api/app/schemas.py. Kept hand-written rather than generated: the
 *  surface is small, and an explicit type file is where the frontend states
 *  what it actually depends on. */

export type ClauseType =
  | 'coverage'
  | 'exclusion'
  | 'condition'
  | 'sub_limit'
  | 'waiting_period'
  | 'definition'
  | 'procedural'

export type DocStatusValue =
  | 'pending'
  | 'ingesting'
  | 'segmenting'
  | 'analyzing'
  | 'scoring'
  | 'ready'
  | 'failed'

export interface DocumentCreated {
  id: string
  filename: string
  status: DocStatusValue
}

export interface DocumentStatus {
  id: string
  status: DocStatusValue
  stage_detail: string
  progress: number
  page_count: number
  clause_count: number
  error: string | null
}

export interface ClauseSummary {
  id: string
  order_idx: number
  number: string
  heading: string
  section_path: string
  page: number

  clause_type: ClauseType
  plain_language: string
  what_it_means: string
  impact_score: number

  /** Why this clause is easy to miss. Each 0-1. */
  buriedness: number
  position_signal: number
  reading_signal: number
  crossref_signal: number
  jargon_signal: number
  reading_grade: number

  triggers: string[]
  monetary_limits: string[]
  time_windows: string[]
}

export interface ClauseDetail extends ClauseSummary {
  /** The policy's own words, verbatim. Always rendered in Source Serif 4. */
  source_text: string
  char_start: number
  char_end: number
  page_start: number
  page_end: number
  likelihood: number
  severity: number
}

export interface TypeCount {
  clause_type: ClauseType
  count: number
}

export interface DocumentSummary {
  id: string
  filename: string
  status: DocStatusValue
  page_count: number
  clause_count: number
  type_counts: TypeCount[]
  top_risks: ClauseSummary[]
  unanalysed_count: number
}

export type Verdict = 'covered' | 'not_covered' | 'conditional' | 'insufficient_information'

export interface Citation {
  clause_id: string
  clause_db_id: string | null
  number: string
  heading: string
  page: number
  effect: string
  quote: string
  /** False when the quoted text could not be found in the cited clause. */
  verified: boolean
  unverified_reason: string
}

export interface ScenarioResponse {
  id: string
  scenario: string
  verdict: Verdict
  reasoning: string
  citations: Citation[]
  missing_facts: string[]
  facts: Record<string, unknown>
  /** False if ANY citation failed verification. */
  verified: boolean
  clauses_considered: number
}
