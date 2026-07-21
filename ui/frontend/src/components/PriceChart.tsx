import { useRef, useEffect } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  type SeriesMarker,
} from 'lightweight-charts'
import type { Bar, InferenceResult, DecisionEntry } from '../types'

interface Props {
  bars: Bar[]
  cursorTimestamp: string | null
  inference: InferenceResult | null
  allDecisions: DecisionEntry[]
  onCursorChange: (ts: string) => void
  playbackSpeed: number
}

function toTime(ts: string): Time {
  return (new Date(ts).getTime() / 1000) as Time
}

const COLORS = {
  bg: '#1e1e2e',
  text: '#cdd6f4',
  grid: '#313244',
  up: '#a6e3a1',
  down: '#f38ba8',
  accent: '#89b4fa',
  yellow: '#f9e2af',
  lavender: '#b4befe',
  surface: '#181825',
}

export default function PriceChart({ bars, cursorTimestamp, inference, allDecisions, onCursorChange, playbackSpeed }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const windowRef = useRef<ISeriesApi<'Area'> | null>(null)
  const predRef = useRef<ISeriesApi<'Line'> | null>(null)
  const predUpperRef = useRef<ISeriesApi<'Line'> | null>(null)
  const predLowerRef = useRef<ISeriesApi<'Line'> | null>(null)

  const barsRef = useRef<Bar[]>([])
  const onCursorRef = useRef<(ts: string) => void>(() => {})

  // Create chart
  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.text,
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: {
        borderColor: COLORS.grid,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: COLORS.grid },
    })

    const candles = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderUpColor: COLORS.up,
      borderDownColor: COLORS.down,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    })

    // Input window highlight (blue — what model sees)
    const win = chart.addAreaSeries({
      lineColor: COLORS.accent,
      topColor: 'rgba(137, 180, 250, 0.25)',
      bottomColor: 'rgba(137, 180, 250, 0.05)',
      lineWidth: 2,
      priceLineVisible: false,
    })

    // Prediction ghost (yellow line + dashed band — what model predicts)
    const pred = chart.addLineSeries({
      color: COLORS.yellow,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
    })

    const predUpper = chart.addLineSeries({
      color: 'rgba(249, 226, 175, 0.35)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })

    const predLower = chart.addLineSeries({
      color: 'rgba(249, 226, 175, 0.35)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })

    chartRef.current = chart
    candleRef.current = candles
    windowRef.current = win
    predRef.current = pred
    predUpperRef.current = predUpper
    predLowerRef.current = predLower

    barsRef.current = bars
    onCursorRef.current = onCursorChange

    chart.subscribeClick(param => {
      if (param.time && param.point && barsRef.current.length > 0) {
        const numericTime = Number(param.time)
        const b = barsRef.current
        let closest = b[0]
        let bestDiff = Infinity
        for (let i = 0; i < b.length; i++) {
          const diff = Math.abs(new Date(b[i].timestamp).getTime() / 1000 - numericTime)
          if (diff < bestDiff) { bestDiff = diff; closest = b[i] }
        }
        onCursorRef.current(closest.timestamp)
      }
    })

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    const observer = new ResizeObserver(handleResize)
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Candle data
  useEffect(() => {
    if (!candleRef.current) return
    const data: CandlestickData[] = bars.map(b => ({
      time: toTime(b.timestamp),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    candleRef.current.setData(data)
  }, [bars])

  // Model input window highlight (blue band — what the model "sees")
  useEffect(() => {
    const s = windowRef.current
    if (!s || !inference) {
      s?.setData([])
      return
    }
    const windowBars = bars.filter(
      b => b.timestamp >= inference.window_start && b.timestamp <= inference.window_end
    )
    if (windowBars.length === 0) {
      s.setData([])
      return
    }
    s.setData(
      windowBars.map(b => ({ time: toTime(b.timestamp), value: b.close }))
    )
  }, [inference, bars])

  // Prediction ghost (yellow filled band — what the model predicts)
  useEffect(() => {
    if (!predRef.current || !inference) {
      predRef.current?.setData([])
      predUpperRef.current?.setData([])
      predLowerRef.current?.setData([])
      return
    }
    const pts = inference.predicted_future_state
    predRef.current.setData(
      pts.map(p => ({ time: toTime(p.timestamp), value: p.price }))
    )
    predUpperRef.current?.setData(
      pts.map(p => ({ time: toTime(p.timestamp), value: p.upper }))
    )
    predLowerRef.current?.setData(
      pts.map(p => ({ time: toTime(p.timestamp), value: p.lower }))
    )
  }, [inference])

  // Decision + cursor markers
  useEffect(() => {
    const s = candleRef.current
    if (!s) return

    const cursorNumeric = cursorTimestamp ? toTime(cursorTimestamp) : null

    const markers: SeriesMarker<Time>[] = []

    // Cursor marker takes precedence
    if (cursorNumeric) {
      markers.push({
        time: cursorNumeric,
        position: 'inBar' as const,
        shape: 'circle' as const,
        color: COLORS.accent,
      })
    }

    // Decision markers (skip if same time as cursor)
    for (const d of allDecisions) {
      const t = toTime(d.timestamp)
      if (d.decision === 'flat') continue
      if (cursorNumeric !== null && t === cursorNumeric) continue
      const isLong = d.decision === 'long'
      markers.push({
        time: t,
        position: isLong ? 'belowBar' as const : 'aboveBar' as const,
        shape: isLong ? 'arrowUp' as const : 'arrowDown' as const,
        color: isLong ? COLORS.up : COLORS.down,
      })
    }

    s.setMarkers(markers)
  }, [cursorTimestamp, allDecisions])

  // Auto-pan viewport to follow the cursor during playback
  useEffect(() => {
    if (playbackSpeed === 0 || !cursorTimestamp || !chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visible = timeScale.getVisibleLogicalRange()
    if (!visible) return
    const cursorTime = new Date(cursorTimestamp).getTime() / 1000
    const start = Number(visible.from)
    const end = Number(visible.to)
    const span = end - start
    const edge = span * 0.15
    if (cursorTime < start + edge || cursorTime > end - edge) {
      const half = span / 2
      timeScale.setVisibleLogicalRange({
        from: cursorTime - half,
        to: cursorTime + half,
      })
    }
  }, [cursorTimestamp, playbackSpeed])

  return (
    <div style={{ position: 'relative' }}>
      <div
        ref={containerRef}
        className="chart-container"
        style={{ width: '100%', height: 400, cursor: 'crosshair' }}
      />
      {inference && (
        <div className="chart-legend">
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: COLORS.accent }} />
            Model sees ({bars.filter(b => b.timestamp >= inference.window_start && b.timestamp <= inference.window_end).length} bars)
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: COLORS.yellow }} />
            Predicts ({inference.predicted_future_state.length} steps)
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: COLORS.up }} />
            Long
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: COLORS.down }} />
            Short
          </span>
        </div>
      )}
    </div>
  )
}
