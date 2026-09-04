import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api } from '../api'
import { CLAUSE_META, RISKY_TYPES } from '../clauseMeta'
import { ClausePanel } from '../components/ClausePanel'
import { Disclaimer, Masthead, MetaStrip, Page } from '../components/Chrome'
import { RiskCard } from '../components/RiskCard'
import type { ClauseSummary, ClauseType, DocumentStatus, DocumentSummary } from '../types'

const POLL_MS = 1200

export function Policy() {
  const { id = '' } = useParams()
  const [status, setStatus] = useState<DocumentStatus | null>(null)
  const [summary, setSummary] = useState<DocumentSummary | null>(null)
  const [clauses, setClauses] = useState<ClauseSummary[] | null>(null)
  const [filter, setFilter] = useState<ClauseType | null>(null)
  const [openClause, setOpenClause] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Poll until the pipeline finishes. The backend returns 202 on upload and
  // does the work in the background, so this is how the UI learns it is done.
  useEffect(() => {
    if (!id) return
    let live = true
    let timer: number

    const tick = async () => {
      try {
        const next = await api.status(id)
        if (!live) return
        setStatus(next)
        if (next.status === 'ready' || next.status === 'failed') return
        timer = window.setTimeout(tick, POLL_MS)
      } catch (err) {
        if (live) setError(err instanceof Error ? err.message : 'Could not reach the server')
      }
    }

    void tick()
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [id])

  // Load results once, when the document becomes ready.
  useEffect(() => {
    if (status?.status !== 'ready') return
    void api.summary(id, 10).then(setSummary).catch(() => undefined)
  }, [status?.status, id])

  useEffect(() => {
    if (status?.status !== 'ready') return
    void api
      .clauses(id, { type: filter, sort: 'impact' })
      .then(setClauses)
      .catch(() => undefined)
  }, [status?.status, id, filter])

  const riskyCount = useMemo(
    () =>
      summary?.type_counts
        .filter((t) => RISKY_TYPES.includes(t.clause_type))
        .reduce((total, t) => total + t.count, 0) ?? 0,
    [summary],
  )

  const closePanel = useCallback(() => setOpenClause(null), [])

  if (error) {
    return (
      <>
        <Masthead />
        <Page>
          <p className="border-l-2 border-oxide bg-oxide-soft py-3 pl-4 text-[14px] text-oxide">
            {error}
          </p>
          <Link to="/" className="mt-4 inline-block text-[14px] underline">
            Start again
          </Link>
        </Page>
      </>
    )
  }

  if (!status || (status.status !== 'ready' && status.status !== 'failed')) {
    return <Processing status={status} />
  }

  if (status.status === 'failed') {
    return (
      <>
        <Masthead />
        <Page>
          <h2 className="font-display text-[30px]">That document could not be read</h2>
          <p className="mt-3 max-w-[60ch] text-[15px] leading-relaxed text-muted">
            {status.error ?? 'The file could not be processed.'}
          </p>
          <Link
            to="/"
            className="mt-6 inline-block rounded-sm border border-ink px-4 py-2 text-[14px]"
          >
            Try another policy
          </Link>
        </Page>
      </>
    )
  }

  return (
    <>
      <Masthead>
        <Link
          to="/"
          className="label rounded-sm border border-rule px-2.5 py-1 transition-colors duration-150 hover:border-rule-strong hover:text-ink"
        >
          New policy
        </Link>
      </Masthead>

      <Page>
        <MetaStrip
          items={[
            { label: 'Document', value: summary?.filename ?? '—' },
            { label: 'Pages', value: status.page_count },
            { label: 'Clauses', value: status.clause_count },
            { label: 'Could cost you', value: riskyCount },
          ]}
        />

        {/* The headline is a verdict and a count, not a greeting. The product's
            promise is showing the reader what was hidden, so the page opens by
            saying how much of it there is. */}
        <section className="mt-10">
          <h2 className="max-w-[24ch] font-display text-[38px] leading-[1.1] tracking-[-0.015em]">
            {riskyCount} clauses in this policy could reduce or deny a claim.
          </h2>
          <p className="mt-3 max-w-[64ch] text-[15px] leading-relaxed text-muted">
            Ranked by what each one could cost you, not by the order your insurer put them in.
            Every entry shows the policy's own wording and the page it is on.
          </p>
        </section>

        {summary && summary.unanalysed_count > 0 && (
          <p className="mt-6 border-l-2 border-ochre bg-ochre-soft py-2 pl-3 text-[13px] text-ochre">
            {summary.unanalysed_count} clause
            {summary.unanalysed_count === 1 ? '' : 's'} could not be read, so this list may be
            incomplete.
          </p>
        )}

        <div className="mt-9 max-w-[1040px] flex flex-wrap items-center justify-between gap-4 border-b border-rule pb-3">
          <h3 className="label">Every clause, worst first</h3>
          <TypeFilter
            counts={summary?.type_counts ?? []}
            active={filter}
            onChange={setFilter}
          />
        </div>

        <div className="rise max-w-[1040px]">
          {clauses?.map((clause, index) => (
            <RiskCard
              key={clause.id}
              clause={clause}
              rank={filter ? undefined : index + 1}
              onOpen={(c) => setOpenClause(c.id)}
            />
          ))}
          {clauses?.length === 0 && (
            <p className="py-8 text-[14px] text-muted">No clauses of this type in this policy.</p>
          )}
          {!clauses && <p className="label py-8">Loading clauses…</p>}
        </div>

        <Disclaimer className="mt-10 max-w-[70ch]" />
      </Page>

      {openClause && <ClausePanel clauseId={openClause} onClose={closePanel} />}
    </>
  )
}

function TypeFilter({
  counts,
  active,
  onChange,
}: {
  counts: { clause_type: ClauseType; count: number }[]
  active: ClauseType | null
  onChange: (type: ClauseType | null) => void
}) {
  // Risky types first: the filter should lead with what the reader came for.
  const ordered = [...counts].sort((a, b) => {
    const rank = (t: ClauseType) => (RISKY_TYPES.includes(t) ? 0 : 1)
    return rank(a.clause_type) - rank(b.clause_type) || b.count - a.count
  })

  return (
    <div className="flex flex-wrap gap-1.5">
      <Chip active={active === null} onClick={() => onChange(null)}>
        All
      </Chip>
      {ordered.map(({ clause_type, count }) => (
        <Chip
          key={clause_type}
          active={active === clause_type}
          onClick={() => onChange(active === clause_type ? null : clause_type)}
        >
          {CLAUSE_META[clause_type].label}
          <span className="tnum ml-1.5 opacity-60">{count}</span>
        </Chip>
      ))}
    </div>
  )
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-sm border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.08em] transition-colors duration-150 ${
        active
          ? 'border-ink bg-ink text-paper'
          : 'border-rule text-muted hover:border-rule-strong hover:text-ink'
      }`}
    >
      {children}
    </button>
  )
}

/**
 * The two-minute wait.
 *
 * Shows the real stage and a real count, because the backend reports both. A
 * spinner would be a lie of omission here: the user has handed over a document
 * and has no way to tell a working model from a hung one.
 */
function Processing({ status }: { status: DocumentStatus | null }) {
  const progress = Math.round((status?.progress ?? 0) * 100)

  return (
    <>
      <Masthead />
      <Page>
        <div className="max-w-[62ch] pt-6">
          <p className="label">Reading your policy</p>
          <h2 className="mt-3 font-display text-[34px] leading-tight">
            {status?.stage_detail ?? 'Starting up'}
          </h2>

          <div className="mt-7 flex items-center gap-4">
            <div className="h-[3px] flex-1 bg-sunk">
              <div
                className="h-full bg-ink transition-[width] duration-500 ease-out"
                style={{ width: `${Math.max(progress, 1)}%` }}
              />
            </div>
            <span className="tnum font-mono text-[13px] text-muted">{progress}%</span>
          </div>

          <p className="mt-6 text-[14px] leading-relaxed text-muted">
            Each clause is read individually by a model running on your machine. That is slower
            than sending the whole document at once, and measurably more accurate, so it is worth
            the wait.
          </p>

          <Disclaimer className="mt-8" />
        </div>
      </Page>
    </>
  )
}
