import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import type { Bar, InferenceResult, DecisionEntry } from './types'
import { fetchSeries, fetchInferenceRange, fetchInferenceDetail } from './api'
import PriceChart from './components/PriceChart'
import InfoPanel from './components/InfoPanel'
import Scrubber from './components/Scrubber'
import CheckpointSelector from './components/CheckpointSelector'

export default function App() {
  const [bars, setBars] = useState<Bar[]>([])
  const [inferences, setInferences] = useState<InferenceResult[]>([])
  const [cursorTimestamp, setCursorTimestamp] = useState<string | null>(null)
  const [playbackSpeed, setPlaybackSpeed] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [detailTs, setDetailTs] = useState<string | null>(null)
  const [detail, setDetail] = useState<InferenceResult | null>(null)
  const [activeRunName, setActiveRunName] = useState<string | null>(null)
  const fetchIdRef = useRef(0)

  const loadData = useCallback(() => {
    setLoading(true)
    setError(null)
    const DEFAULT_LIMIT = 500
    Promise.all([fetchSeries('SOLUSDT', DEFAULT_LIMIT), fetchInferenceRange('SOLUSDT', DEFAULT_LIMIT)])
      .then(([series, infRange]) => {
        setBars(series.bars)
        setInferences(infRange.results)
        if (infRange.results.length > 0) {
          const first = infRange.results[0].timestamp
          setCursorTimestamp(first)
          setDetailTs(first)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Unknown error'
        setError(msg)
      })
      .finally(() => setLoading(false))
  }, [])

  const handleCheckpointChanged = useCallback(() => {
    setActiveRunName(null)
    loadData()
  }, [loadData])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Fetch prediction details when cursor changes (skip during playback)
  useEffect(() => {
    if (!cursorTimestamp || playbackSpeed > 0) return
    const id = ++fetchIdRef.current
    setDetailTs(cursorTimestamp)
    fetchInferenceDetail('SOLUSDT', cursorTimestamp).then(r => {
      if (fetchIdRef.current === id) setDetail(r)
    }).catch(console.error)
  }, [cursorTimestamp, playbackSpeed])

  // Index maps for O(1) lookups
  const { inferenceByTs, indexByTs } = useMemo(() => {
    const byTs = new Map<string, InferenceResult>()
    const byIdx = new Map<string, number>()
    for (let i = 0; i < inferences.length; i++) {
      byTs.set(inferences[i].timestamp, inferences[i])
      byIdx.set(inferences[i].timestamp, i)
    }
    return { inferenceByTs: byTs, indexByTs: byIdx }
  }, [inferences])

  // Only pass decisions within the current bar range to the chart
  const allDecisions: DecisionEntry[] = useMemo(() => {
    if (bars.length === 0 || inferences.length === 0) return []
    const barStart = bars[0].timestamp
    const barEnd = bars[bars.length - 1].timestamp
    return inferences
      .filter(i => i.timestamp >= barStart && i.timestamp <= barEnd)
      .map(i => ({ timestamp: i.timestamp, decision: i.decision }))
  }, [inferences, bars])

  const currentInference = useMemo(
    () => detail ?? (detailTs ? inferenceByTs.get(detailTs) ?? null : null),
    [detail, detailTs, inferenceByTs]
  )

  const cursorIndex = useMemo(
    () => cursorTimestamp ? (indexByTs.get(cursorTimestamp) ?? -1) : -1,
    [cursorTimestamp, indexByTs]
  )

  const handleCursorChange = useCallback((ts: string) => {
    setCursorTimestamp(ts)
    setPlaybackSpeed(0)
  }, [])

  // Playback
  useEffect(() => {
    if (playbackSpeed === 0 || inferences.length === 0) return
    const ms = Math.round(1000 / playbackSpeed)
    const list = inferences
    const idxMap = indexByTs
    const id = setInterval(() => {
      setCursorTimestamp(prev => {
        if (!prev) return list[0]?.timestamp ?? null
        const idx = idxMap.get(prev)
        if (idx === undefined || idx >= list.length - 1) return prev
        return list[idx + 1].timestamp
      })
    }, ms)
    return () => clearInterval(id)
  }, [playbackSpeed, inferences, indexByTs])

  // Keyboard controls — ref-stabilised to avoid listener churn
  const inferencesRef = useRef(inferences)
  const indexByTsRef = useRef(indexByTs)
  const cursorTsRef = useRef(cursorTimestamp)
  inferencesRef.current = inferences
  indexByTsRef.current = indexByTs
  cursorTsRef.current = cursorTimestamp

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const list = inferencesRef.current
      const idxMap = indexByTsRef.current
      const curTs = cursorTsRef.current
      if (e.key === 'ArrowLeft') {
        const curIdx = curTs ? (idxMap.get(curTs) ?? -1) : -1
        if (curIdx > 0) {
          setCursorTimestamp(list[curIdx - 1].timestamp)
          setPlaybackSpeed(0)
        }
      } else if (e.key === 'ArrowRight') {
        const curIdx = curTs ? (idxMap.get(curTs) ?? -1) : -1
        if (curIdx >= 0 && curIdx < list.length - 1) {
          setCursorTimestamp(list[curIdx + 1].timestamp)
          setPlaybackSpeed(0)
        }
      } else if (e.key === ' ') {
        e.preventDefault()
        setPlaybackSpeed(s => (s === 0 ? 1 : 0))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading replay data…</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <div className="loading" style={{ color: '#f87171' }}>
          <p>Failed to load replay data</p>
          <p style={{ fontSize: '0.8rem', marginTop: 8 }}>{error}</p>
          <p style={{ marginTop: 16 }}>
            Is the server running? Try <code>launch.bat serve</code> in a terminal.
          </p>
          <button onClick={loadData} style={{ marginTop: 12, padding: '8px 16px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="toolbar">
        <h1>Teacher Model — Replay</h1>
        <CheckpointSelector
          activeRunName={activeRunName}
          onCheckpointChanged={handleCheckpointChanged}
        />
        <span className="toolbar-info">
          {bars.length} bars loaded &middot; {inferences.length} visible
        </span>
      </header>

      <PriceChart
        bars={bars}
        cursorTimestamp={cursorTimestamp}
        inference={currentInference}
        allDecisions={allDecisions}
        onCursorChange={handleCursorChange}
        playbackSpeed={playbackSpeed}
      />

      <InfoPanel
        bars={bars}
        inferences={inferences}
        currentInf={cursorTimestamp ? (inferenceByTs.get(cursorTimestamp) ?? null) : null}
      />

      <Scrubber
        inferences={inferences}
        cursorTimestamp={cursorTimestamp}
        cursorIndex={cursorIndex}
        onCursorChange={handleCursorChange}
        playbackSpeed={playbackSpeed}
        onPlaybackChange={setPlaybackSpeed}
      />
    </div>
  )
}
