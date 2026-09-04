import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '../api'
import { Disclaimer, Masthead, Page } from '../components/Chrome'

/**
 * Upload.
 *
 * Left-aligned and asymmetric: the explanation column sits beside the drop
 * target rather than a centered card floating in the middle of the viewport.
 * A centered hero is the default shape of a generated landing page, and it
 * would also misrepresent this screen, which is a form, not a pitch.
 */
export function Upload() {
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  async function send(file: File | undefined) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const created = await api.upload(file)
      navigate(`/policy/${created.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setBusy(false)
    }
  }

  return (
    <>
      <Masthead />
      <Page>
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_420px] lg:gap-16">
          <div>
            <h2 className="max-w-[20ch] font-display text-[42px] leading-[1.08] tracking-[-0.015em]">
              Your policy is not hiding one thing. It is hiding about forty.
            </h2>
            <p className="mt-5 max-w-[60ch] text-[15px] leading-relaxed text-ink/90">
              Insurers write cover in the first ten pages and the conditions that undo it in the
              last thirty. This reads the whole document, translates every clause into plain
              English, and ranks them by how much each one could cost you.
            </p>

            <ol className="mt-8 max-w-[58ch] space-y-3 border-t border-rule pt-5">
              {[
                ['Reads the PDF', 'Every clause, with its exact page and position.'],
                [
                  'Sorts them by what they cost you',
                  'Exclusions and notice deadlines first, admin last.',
                ],
                [
                  'Shows the original wording',
                  'Beside every explanation, so you can check it yourself.',
                ],
              ].map(([title, body], index) => (
                <li key={title} className="flex gap-3">
                  <span className="tnum mt-[3px] font-mono text-[12px] text-muted">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <div>
                    <p className="text-[14px] font-medium">{title}</p>
                    <p className="text-[13px] leading-relaxed text-muted">{body}</p>
                  </div>
                </li>
              ))}
            </ol>

            <p className="mt-8 max-w-[58ch] text-[13px] leading-relaxed text-muted">
              Everything runs on your own machine against a local model. Your policy is never
              uploaded to anyone.
            </p>
          </div>

          <div>
            <div
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                void send(event.dataTransfer.files?.[0])
              }}
              className={`border-2 border-dashed p-8 transition-colors duration-150 ${
                dragging ? 'border-ink bg-sunk' : 'border-rule-strong bg-surface'
              }`}
            >
              <p className="label">Your policy document</p>
              <p className="mt-3 text-[15px] leading-relaxed">
                Drop the PDF here, or choose it from your computer.
              </p>

              <input
                ref={inputRef}
                type="file"
                accept="application/pdf,.pdf"
                className="sr-only"
                onChange={(event) => void send(event.target.files?.[0])}
              />
              <button
                type="button"
                disabled={busy}
                onClick={() => inputRef.current?.click()}
                className="mt-5 w-full rounded-sm border border-ink bg-ink px-4 py-2.5 text-[14px] font-medium text-paper transition-opacity duration-150 hover:opacity-85 disabled:opacity-50"
              >
                {busy ? 'Reading your policy…' : 'Choose a PDF'}
              </button>

              <p className="mt-3 font-mono text-[11px] text-muted">PDF · up to 25 MB</p>

              {error && (
                <p className="mt-4 border-l-2 border-oxide bg-oxide-soft py-2 pl-3 text-[13px] text-oxide">
                  {error}
                </p>
              )}
            </div>

            <p className="mt-5 text-[13px] leading-relaxed text-muted">
              A 40-clause policy takes about two minutes to read. The model runs locally, so the
              speed depends on your machine, not on a queue.
            </p>

            <Disclaimer className="mt-6" />
          </div>
        </div>

        <WhatItLooksFor />
      </Page>
    </>
  )
}

/**
 * The four clause types that cost people money, named up front.
 *
 * This exists because the page had a large empty lower half, and empty space
 * at the bottom of a first screen reads as unfinished. Rather than padding it
 * with decoration, it sets expectations: the reader learns the vocabulary the
 * report will use, and sees that "denies your claim" and "shrinks your payout"
 * are tracked as different kinds of harm before they meet the results.
 */
function WhatItLooksFor() {
  const items: { term: string; body: string; band: 'deny' | 'reduce' }[] = [
    {
      term: 'Exclusions',
      body: 'Treatment the policy will never pay for, however long you hold it.',
      band: 'deny',
    },
    {
      term: 'Conditions',
      body: 'Deadlines and duties that can void an otherwise valid claim.',
      band: 'deny',
    },
    {
      term: 'Payout caps',
      body: 'Room-rent limits and co-payments that quietly shrink what you receive.',
      band: 'reduce',
    },
    {
      term: 'Waiting periods',
      body: 'Cover that only begins months or years after you buy the policy.',
      band: 'reduce',
    },
  ]

  return (
    <section className="mt-20 border-t border-rule pt-6">
      <h3 className="label">What it looks for</h3>
      <dl className="mt-5 grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((item) => (
          <div
            key={item.term}
            className={`border-l-2 pl-3 ${
              item.band === 'deny' ? 'border-oxide' : 'border-ochre'
            }`}
          >
            <dt
              className={`text-[14px] font-medium ${
                item.band === 'deny' ? 'text-oxide' : 'text-ochre'
              }`}
            >
              {item.term}
            </dt>
            <dd className="mt-1 text-[13px] leading-relaxed text-muted">{item.body}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-6 max-w-[70ch] text-[13px] leading-relaxed text-muted">
        Definitions, covered benefits and administrative clauses are read and listed too, but
        they rank below anything that can cost you money.
      </p>
    </section>
  )
}
