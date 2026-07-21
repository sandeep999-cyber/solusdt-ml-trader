import type { SeriesResponse, InferenceRangeResponse, InferenceResult, CheckpointListResponse, CheckpointSelectResponse } from './types'

// Use relative URLs when served from the same origin, or fall back to explicit
// localhost so Vite dev-server proxy (port 5173 → 8000) works.
const BASE = location.origin.startsWith('http://localhost:517') ? 'http://localhost:8000' : ''

const FETCH_TIMEOUT_MS = 120_000

async function fetchWithTimeout(url: string, timeoutMs: number = FETCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const r = await fetch(url, { signal: controller.signal })
    return r
  } finally {
    clearTimeout(timer)
  }
}

export async function fetchSeries(symbol = 'SOLUSDT', limit?: number): Promise<SeriesResponse> {
  const params = new URLSearchParams({ symbol })
  if (limit !== undefined) params.set('limit', String(limit))
  const r = await fetchWithTimeout(`${BASE}/series?${params}`)
  if (!r.ok) throw new Error(`Series fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchInferenceRange(symbol = 'SOLUSDT', limit?: number): Promise<InferenceRangeResponse> {
  const params = new URLSearchParams({ symbol })
  if (limit !== undefined) params.set('limit', String(limit))
  const r = await fetchWithTimeout(`${BASE}/inference/range?${params}`)
  if (!r.ok) throw new Error(`Inference range fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchInferenceDetail(symbol: string, timestamp: string): Promise<InferenceResult> {
  const r = await fetchWithTimeout(`${BASE}/inference?symbol=${symbol}&timestamp=${encodeURIComponent(timestamp)}`)
  if (!r.ok) throw new Error(`Inference detail fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const r = await fetchWithTimeout(`${BASE}/checkpoints/list`)
  if (!r.ok) throw new Error(`Checkpoint list fetch failed: ${r.status}`)
  return r.json()
}

export async function selectCheckpoint(runName: string): Promise<CheckpointSelectResponse> {
  const body = new URLSearchParams({ run_name: runName })
  const r = await fetch(`${BASE}/checkpoints/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!r.ok) throw new Error(`Checkpoint select failed: ${r.status}`)
  return r.json()
}

export async function uploadCheckpoint(file: File): Promise<CheckpointSelectResponse> {
  const body = new FormData()
  body.append('file', file)
  const r = await fetch(`${BASE}/checkpoints/upload`, {
    method: 'POST',
    body,
  })
  if (!r.ok) throw new Error(`Checkpoint upload failed: ${r.status}`)
  return r.json()
}

export async function reloadEngine(): Promise<void> {
  await fetch(`${BASE}/reload`, { method: 'POST' })
}
