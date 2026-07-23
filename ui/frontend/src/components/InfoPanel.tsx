import { useRef, useEffect, useMemo, useCallback } from 'react'
import {
  createChart,
  ColorType,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts'
import type { Bar, InferenceResult } from '../types'

interface Props {
  bars: Bar[]
  inferences: InferenceResult[]
  currentInf: InferenceResult | null
}

const COLORS = {
  bg: '#1e1e2e',
  text: '#cdd6f4',
  grid: '#313244',
  uncertainty: '#f9e2af',
  surprise: '#b4befe',
  accent: '#89b4fa',
  long: '#a6e3a1',
  short: '#f38ba8',
  flat: '#6c7086',
  realized_vol: '#94e2d5',
}

function toTime(ts: string): Time {
  return (new Date(ts).getTime() / 1000) as Time
}

function RealizedVolMiniChart({
  data,
  inferences,
  currentInf,
}: {
  data: { timestamp: string; realized_vol: number }[]
  inferences: InferenceResult[]
  currentInf: InferenceResult | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const lineRef = useRef<ISeriesApi<'Line'> | null>(null)
  const cursorLineRef = useRef<ISeriesApi<'Line'> | null>(null)

  // Build uncertainty data indexed by timestamp for overlay
  const uncertaintyByTs = useMemo(() => {
    const map = new Map<string, number>()
    for (const inf of inferences) {
      map.set(inf.timestamp, inf.uncertainty)
    }
    return map
  }, [inferences])

  const seriesData = useMemo(
    () => data.map(d => ({ time: toTime(d.timestamp), value: d.realized_vol })),
    [data]
  )

  const uncertaintyData = useMemo(
    () => data
      .filter(d => uncertaintyByTs.has(d.timestamp))
      .map(d => ({ time: toTime(d.timestamp), value: uncertaintyByTs.get(d.timestamp)! })),
    [data, uncertaintyByTs]
  )

  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 100,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.text,
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      timeScale: {
        borderColor: COLORS.grid,
        visible: false,
      },
      rightPriceScale: {
        borderColor: COLORS.grid,
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      crosshair: {
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
      handleScroll: false,
      handleScale: false,
    })

    const volLine = chart.addLineSeries({
      color: COLORS.realized_vol,
      lineWidth: 2,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      title: 'Realized Vol',
    })

    const uncLine = chart.addLineSeries({
      color: COLORS.uncertainty,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      title: 'Predicted Unc',
    })

    const cursorLine = chart.addLineSeries({
      color: COLORS.accent,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    lineRef.current = volLine
    cursorLineRef.current = cursorLine

    // Store uncLine ref for updates
    ;(chartRef as any)._uncLine = uncLine

    const observer = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!lineRef.current) return
    lineRef.current.setData(seriesData)
    const uncLine = (chartRef as any)?._uncLine as ISeriesApi<'Line'> | undefined
    if (uncLine && uncertaintyData.length > 0) {
      uncLine.setData(uncertaintyData)
    }
  }, [seriesData, uncertaintyData])

  useEffect(() => {
    if (!cursorLineRef.current) return
    if (currentInf) {
      const rv = data.find(d => d.timestamp === currentInf.timestamp)
      if (rv) {
        cursorLineRef.current.setData([
          { time: toTime(rv.timestamp), value: rv.realized_vol },
        ])
      }
    } else {
      cursorLineRef.current.setData([])
    }
  }, [currentInf, data])

  // Find current realized vol value
  const currentRv = currentInf ? data.find(d => d.timestamp === currentInf.timestamp) : null

  return (
    <div className="mini-chart">
      <div className="mini-chart-header">
        <span className="mini-chart-label" style={{ color: COLORS.realized_vol }}>Realized Vol</span>
        <span className="mini-chart-value">
          {currentRv ? currentRv.realized_vol.toFixed(4) : '—'}
        </span>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: 100 }} />
    </div>
  )
}

function MiniChart({
  data,
  color,
  label,
  cursorValue,
  formatValue,
}: {
  data: InferenceResult[]
  color: string
  label: string
  cursorValue: InferenceResult | null
  formatValue: (r: InferenceResult) => number
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const lineRef = useRef<ISeriesApi<'Line'> | null>(null)
  const cursorLineRef = useRef<ISeriesApi<'Line'> | null>(null)

  const seriesData = useMemo(
    () => data.map(r => ({ time: toTime(r.timestamp), value: formatValue(r) })),
    [data, formatValue]
  )

  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 100,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.text,
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      timeScale: {
        borderColor: COLORS.grid,
        visible: false,
      },
      rightPriceScale: {
        borderColor: COLORS.grid,
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      crosshair: {
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
      handleScroll: false,
      handleScale: false,
    })

    const line = chart.addLineSeries({
      color,
      lineWidth: 2,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
    })

    const cursorLine = chart.addLineSeries({
      color: COLORS.accent,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    lineRef.current = line
    cursorLineRef.current = cursorLine

    const observer = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [color])

  useEffect(() => {
    if (!lineRef.current) return
    lineRef.current.setData(seriesData)
  }, [seriesData])

  useEffect(() => {
    if (!cursorLineRef.current) return
    if (cursorValue) {
      cursorLineRef.current.setData([
        { time: toTime(cursorValue.timestamp), value: formatValue(cursorValue) },
      ])
    } else {
      cursorLineRef.current.setData([])
    }
  }, [cursorValue, formatValue])

  return (
    <div className="mini-chart">
      <div className="mini-chart-header">
        <span className="mini-chart-label" style={{ color }}>{label}</span>
        <span className="mini-chart-value">
          {cursorValue !== null ? formatValue(cursorValue).toFixed(4) : '—'}
        </span>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: 100 }} />
    </div>
  )
}

export default function InfoPanel({ bars, inferences, currentInf }: Props) {
  const fmtUncertainty = useCallback((r: InferenceResult) => r.uncertainty, [])
  const fmtSurprise = useCallback((r: InferenceResult) => r.surprise, [])

  // Build realized vol data from bars (indexed by timestamp)
  const realizedVolData = useMemo(() => {
    const byTs = new Map<string, InferenceResult>()
    for (const inf of inferences) {
      byTs.set(inf.timestamp, inf)
    }
    return bars
      .filter(b => b.realized_vol != null && byTs.has(b.timestamp))
      .map(b => ({
        timestamp: b.timestamp,
        realized_vol: b.realized_vol!,
      }))
  }, [bars, inferences])

  return (
    <div className="info-panel">
      {/* Realized Volatility chart — uses bars data directly */}
      {realizedVolData.length > 0 && (
        <RealizedVolMiniChart
          data={realizedVolData}
          inferences={inferences}
          currentInf={currentInf}
        />
      )}

      <MiniChart
        data={inferences}
        color={COLORS.uncertainty}
        label="Predicted Uncertainty"
        cursorValue={currentInf}
        formatValue={fmtUncertainty}
      />
      <MiniChart
        data={inferences}
        color={COLORS.surprise}
        label="Surprise"
        cursorValue={currentInf}
        formatValue={fmtSurprise}
      />

      {currentInf && (
        <div className={`info-badge decision-${currentInf.decision}`}>
          {currentInf.decision.toUpperCase()}
        </div>
      )}
    </div>
  )
}
