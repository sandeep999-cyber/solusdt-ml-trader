import { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import type { InferenceResult } from '../types'

const SPEEDS = [0, 1, 5, 20]

interface Props {
  inferences: InferenceResult[]
  cursorTimestamp: string | null
  cursorIndex: number
  onCursorChange: (ts: string) => void
  playbackSpeed: number
  onPlaybackChange: (speed: number) => void
}

export default function Scrubber({
  inferences,
  cursorTimestamp,
  cursorIndex,
  onCursorChange,
  playbackSpeed,
  onPlaybackChange,
}: Props) {
  const maxIdx = useMemo(() => inferences.length - 1, [inferences.length])
  const maxIdxRef = useRef(maxIdx)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const lastFireRef = useRef(0)
  const inferencesRef = useRef(inferences)
  const dragIndexRef = useRef<number | null>(null)
  const effectiveIndexRef = useRef(cursorIndex)
  const onCursorRef = useRef(onCursorChange)
  const onPlaybackRef = useRef(onPlaybackChange)
  const playbackRef = useRef(playbackSpeed)

  inferencesRef.current = inferences
  maxIdxRef.current = maxIdx
  onCursorRef.current = onCursorChange
  onPlaybackRef.current = onPlaybackChange
  playbackRef.current = playbackSpeed

  const effectiveIndex = dragIndex ?? cursorIndex
  effectiveIndexRef.current = effectiveIndex
  const fillPct = maxIdx > 0 ? (effectiveIndex / maxIdx) * 100 : 0

  // Window-level mouseup to catch releases outside the slider element
  useEffect(() => {
    if (dragIndex === null) return
    const handle = () => {
      const idx = dragIndexRef.current
      if (idx !== null) {
        const list = inferencesRef.current
        if (idx >= 0 && idx < list.length) {
          onCursorRef.current(list[idx].timestamp)
        }
      }
      setDragIndex(null)
      dragIndexRef.current = null
    }
    window.addEventListener('mouseup', handle)
    window.addEventListener('touchend', handle)
    return () => {
      window.removeEventListener('mouseup', handle)
      window.removeEventListener('touchend', handle)
    }
  }, [dragIndex])

  const handleSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = Number(e.target.value)
    setDragIndex(idx)
    dragIndexRef.current = idx
    const now = Date.now()
    if (now - lastFireRef.current >= 50) {
      lastFireRef.current = now
      const list = inferencesRef.current
      if (idx >= 0 && idx < list.length) {
        onCursorRef.current(list[idx].timestamp)
      }
    }
  }, [])

  const handlePrev = useCallback(() => {
    const idx = effectiveIndexRef.current
    if (idx > 0) {
      const list = inferencesRef.current
      onCursorRef.current(list[idx - 1].timestamp)
    }
  }, [])

  const handleNext = useCallback(() => {
    const idx = effectiveIndexRef.current
    if (idx < maxIdxRef.current) {
      const list = inferencesRef.current
      onCursorRef.current(list[idx + 1].timestamp)
    }
  }, [])

  const togglePlay = useCallback(() => {
    onPlaybackRef.current(playbackRef.current === 0 ? 1 : 0)
  }, [])

  const handleSpeedChange = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    onPlaybackRef.current(Number(e.currentTarget.dataset.speed))
  }, [])

  const isPlaying = playbackSpeed > 0

  return (
    <div className="scrubber">
      <div className="scrubber-controls">
        <button
          className="scrub-btn"
          onClick={handlePrev}
          disabled={cursorIndex <= 0}
          title="Previous bar (←)"
        >
          ◀
        </button>
        <button
          className={`scrub-btn play-btn ${isPlaying ? 'playing' : ''}`}
          onClick={togglePlay}
          title="Play/Pause (Space)"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button
          className="scrub-btn"
          onClick={handleNext}
          disabled={cursorIndex >= maxIdx}
          title="Next bar (→)"
        >
          ▶
        </button>
        <div className="speed-selector">
          {SPEEDS.map(s => (
            <button
              key={s}
              className={`speed-btn ${playbackSpeed === s ? 'active' : ''}`}
              data-speed={s}
              onClick={handleSpeedChange}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <input
        type="range"
        className="timeline"
        min={0}
        max={maxIdx}
        value={Math.max(0, effectiveIndex)}
        onChange={handleSlider}
        style={{ '--fill': `${fillPct}%` } as React.CSSProperties}
        title="Drag to scrub"
      />

      <div className="scrubber-info">
        <span className="cursor-info">
          {effectiveIndex >= 0 ? `${effectiveIndex + 1} / ${inferences.length}` : '—'}
        </span>
        <span className="timestamp-info">
          {cursorTimestamp ?? '—'}
        </span>
      </div>
    </div>
  )
}
