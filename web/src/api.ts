import type {
  ClauseDetail,
  ClauseSummary,
  ClauseType,
  DocumentCreated,
  DocumentStatus,
  DocumentSummary,
} from './types'

/** Requests go to /api and Vite proxies them to the backend (vite.config.ts),
 *  so the browser only ever sees one origin and no base URL is baked in. */
const BASE = '/api'

class ApiError extends Error {
  // Declared and assigned explicitly rather than as a constructor parameter
  // property: this project's tsconfig sets `erasableSyntaxOnly`, which forbids
  // TypeScript syntax that has no plain-JavaScript equivalent.
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(BASE + path, { signal })
  if (!response.ok) throw await toError(response)
  return response.json() as Promise<T>
}

async function toError(response: Response): Promise<ApiError> {
  // FastAPI puts the human-readable reason in `detail`. Surfacing it verbatim
  // is why the upload route bothers to distinguish "not a PDF" from "empty
  // file" - a generic "request failed" would waste that.
  let detail = `Request failed (${response.status})`
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') detail = body.detail
  } catch {
    /* non-JSON error body; keep the status-based message */
  }
  return new ApiError(detail, response.status)
}

export const api = {
  async upload(file: File): Promise<DocumentCreated> {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(`${BASE}/documents`, {
      method: 'POST',
      body: form,
    })
    if (!response.ok) throw await toError(response)
    return response.json()
  },

  status: (id: string, signal?: AbortSignal) =>
    get<DocumentStatus>(`/documents/${id}/status`, signal),

  summary: (id: string, top = 10) =>
    get<DocumentSummary>(`/documents/${id}/summary?top=${top}`),

  clauses: (id: string, opts: { type?: ClauseType | null; sort?: 'impact' | 'document' } = {}) => {
    const params = new URLSearchParams()
    if (opts.type) params.set('clause_type', opts.type)
    params.set('sort', opts.sort ?? 'impact')
    return get<ClauseSummary[]>(`/documents/${id}/clauses?${params}`)
  },

  clause: (clauseId: string) => get<ClauseDetail>(`/clauses/${clauseId}`),
}

export { ApiError }
