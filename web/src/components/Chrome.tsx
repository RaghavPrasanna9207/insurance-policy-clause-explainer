import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

/**
 * Masthead: left-aligned, like the title block of a printed report.
 *
 * Deliberately NOT a centered hero with a big heading and a subtitle beneath.
 * That layout is the clearest signal of a generated design, and it is also
 * wrong for this product: a report leads with its identity and its metadata,
 * not with a pitch.
 */
export function Masthead({ children }: { children?: React.ReactNode }) {
  return (
    <header className="double-rule mb-8">
      <div className="mx-auto flex max-w-[1240px] items-end justify-between gap-6 px-6 pb-3 pt-7">
        <div>
          <Link to="/" className="group inline-block">
            <h1 className="font-display text-[34px] leading-none tracking-[-0.01em]">
              Policy Reader
            </h1>
          </Link>
          <p className="mt-1.5 max-w-[52ch] text-[13px] leading-snug text-muted">
            Finds the clauses your insurer would rather you read last.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3 pb-1">
          {children}
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

/**
 * Metadata strip under the masthead: key-value pairs in mono, the way a
 * document header states its own particulars. Replaces the "stat card grid"
 * that a dashboard would reach for here.
 */
export function MetaStrip({
  items,
}: {
  items: { label: string; value: string | number }[]
}) {
  return (
    <dl className="flex flex-wrap items-baseline gap-x-8 gap-y-2 border-b border-rule pb-3">
      {items.map((item) => (
        <div key={item.label} className="flex items-baseline gap-2">
          <dt className="label">{item.label}</dt>
          <dd className="tnum font-mono text-[13px] text-ink">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function ThemeToggle() {
  const [dark, setDark] = useState(
    () => typeof document !== 'undefined' && document.documentElement.classList.contains('dark'),
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('theme', dark ? 'dark' : 'light')
    } catch {
      /* private browsing; the in-memory toggle still works for this session */
    }
  }, [dark])

  return (
    <button
      type="button"
      onClick={() => setDark((value) => !value)}
      aria-pressed={dark}
      className="rounded-sm border border-rule px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.11em] text-muted transition-colors duration-150 hover:border-rule-strong hover:text-ink"
    >
      {dark ? 'Paper' : 'Ink'}
    </button>
  )
}

/**
 * The disclaimer. Present on every screen, not tucked into a footer nobody
 * scrolls to. This is a tool that reads a legal document and ranks financial
 * risk; the limits of what it is have to travel with it.
 */
export function Disclaimer({ className = '' }: { className?: string }) {
  return (
    <p
      className={`border-l-2 border-rule-strong pl-3 text-[12px] leading-relaxed text-muted ${className}`}
    >
      <span className="font-medium text-ink">This is not legal or financial advice.</span> It
      is an automated reading of your document and it can be wrong. It is not a substitute for
      reading your policy. Confirm anything that matters with your insurer in writing.
    </p>
  )
}

export function Page({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-[1240px] px-6 pb-24">{children}</div>
}
